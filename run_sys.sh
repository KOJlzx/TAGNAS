#!/usr/bin/env bash
# =========================================================================
# Async distributed training system launcher
#
# Launches 4 processes:
#   rank 0 → coordinator    (CPU, global cache routing + metrics aggregation)
#   rank 1 → graph_worker   (CPU, runs the GNN encoder)
#   rank 2 → text_worker_0  (GPU:0, runs the LM encoder)
#   rank 3 → text_worker_1  (GPU:1, runs the LM encoder)
#
# Two phases:
#   search  → GNN architecture search, produces genotype.json
#   retrain → loads genotype.json, retrains with the fixed architecture
#
# Common tuning notes:
#   train node count ≈ 3783 (PubMed); batches per epoch = ceil(3783 / batch_size)
#   graph-batch-size too large → too few batches per epoch (undertrained)
#   text-chunk-size is usually kept small (LM forward pass is heavier)
# =========================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Base paths and environment
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active) -- aligned with run_clip.sh/run_ddp.sh, so that
# `bash run_sys.sh` works whether or not the caller pre-activated it; a fresh shell would
# otherwise fall back to a system python without torch/omegaconf and fail on import.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Select the GPUs to use (2 cards, one each for text_worker_0 and text_worker_1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# ──────────────────────────────────────────────────────────────────────
# HuggingFace network settings (consistent with run_deepspeed.sh)
# Offline mode: skip all Hub network calls once the model/tokenizer are cached locally.
# Defaults to true -- newer transformers versions still call list_repo_templates to
# fetch the chat template from the Hub even when the model is cached; without network
# access this raises ConnectTimeout, which then hangs/crashes the post-training
# eval_link_pred_distributed.py evaluation. Offline mode reads straight from the local
# cache instead. Set HF_OFFLINE=false if the model isn't cached yet.
# ──────────────────────────────────────────────────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# ──────────────────────────────────────────────────────────────────────
# Tunable parameters (override via environment variables, e.g. PHASE=retrain bash run_sys.sh)
# ──────────────────────────────────────────────────────────────────────
PHASE="${PHASE:-search}"                                  # search / retrain / both
CONFIG_PATH="${CONFIG_PATH:-configs/clip-graph/inductive-causal/pubmed/gassobase.yaml}"
DATASET="${DATASET:-reddit}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/model/distributed}"
# time-breakdown profiling: non-empty PROFILE adds --profile; PROFILE_OUT sets the JSON output path.
PROFILE="${PROFILE:-}"
PROFILE_OUT="${PROFILE_OUT:-}"

# Epoch counts (shared by graph/text sides by default; can also be set separately via GRAPH_EPOCHS/TEXT_EPOCHS)
SEARCH_EPOCHS="${SEARCH_EPOCHS:-100}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-50}"
GRAPH_EPOCHS="${GRAPH_EPOCHS:-100}"                          # empty falls back to SEARCH_EPOCHS
TEXT_EPOCHS="${TEXT_EPOCHS:-50}"

# Batch sizes (affect training-step density)
GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-1024}"                # node count / this value = graph batches per epoch
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-64}"
GRAPH_K_HOP="${GRAPH_K_HOP:-1}"
# GNN message-passing layer count (overrides gnn_params.num_layers in the YAML).
# Empty string keeps the YAML setting. For large bipartite graphs (e.g. twitter),
# NUM_LAYERS=1 is recommended to avoid over-smoothing.
# Note: this should usually match GRAPH_K_HOP (subgraph sampling hops), otherwise
# the extra sampled hops go to waste.
NUM_LAYERS="${NUM_LAYERS:-1}"
# Number of text workers (= number of GPUs used); world_size = 2 + this value. Empty keeps the default of 2.
# Used for the multi-GPU scalability test (test4), set to the GPU count; CUDA_VISIBLE_DEVICES
# must list at least that many cards.
NUM_TEXT_WORKERS="${NUM_TEXT_WORKERS:-}"

# Ablation switch: true → degrade to "sync push" (bidirectional lock-step).
# Graph/text sides use the same batch order and each step waits for the other side's
# update before advancing. Automatically clamps NUM_TEXT_WORKERS=1 and sets
# TEXT_CHUNK_SIZE=GRAPH_BATCH_SIZE (both sides' batches must match).
# Default false = the existing async bidirectional push (drop-on-miss).
SYNC_PUSH="${SYNC_PUSH:-false}"
if [[ "${SYNC_PUSH}" == "true" ]]; then
    NUM_TEXT_WORKERS=1
    TEXT_CHUNK_SIZE="${GRAPH_BATCH_SIZE}"
fi

# Cache capacity: drop-on-miss requires that the opposite-side embeddings for every
# target node in a batch are all cache hits. If capacity < total node count, FIFO
# eviction keeps the hit rate perpetually low and parameters never get updated.
# For large graphs (e.g. goodreads-crime with 422K nodes), be sure to set both of
# these to >= the total node count.
GRAPH_CACHE_CAPACITY="${GRAPH_CACHE_CAPACITY:-600000}"
TEXT_CACHE_CAPACITY="${TEXT_CACHE_CAPACITY:-600000}"

# Randomly use only this fraction (0,1] of train nodes for training alignment: shrinks
# the "seed node set pushed into the cache" → cache fills instantly, drop-on-miss drops
# sharply, convergence speeds up (strongly recommended for large graphs like goodreads,
# e.g. 0.1). Sampled once by seed and held fixed for the whole run, consistent across
# graph/text sides; only affects training — val/test still use the full split.
# Empty = use all train nodes (default, backward compatible).
TRAIN_NODE_FRACTION="${TRAIN_NODE_FRACTION:-0.1}"

# Ports (change these two to avoid conflicts when running multiple instances in parallel)
DIST_PORT="${DIST_PORT:-29590}"
RPC_PORT="${RPC_PORT:-29591}"

# Artifact write frequency
METRICS_INTERVAL="${METRICS_INTERVAL:-10}"
CKPT_INTERVAL="${CKPT_INTERVAL:-600}"

# Safety limits (prevent hangs)
# MAX_RUNTIME is the global default; SEARCH/RETRAIN_MAX_RUNTIME can override it
# separately (not cumulative in "both" mode). Leave empty for no limit.
# When running a single phase, MAX_RUNTIME alone is enough.
MAX_RUNTIME="${MAX_RUNTIME:-3000}"                                # seconds, global default; empty = unlimited
SEARCH_MAX_RUNTIME="${SEARCH_MAX_RUNTIME:-3000}"              # search phase only, overrides MAX_RUNTIME
RETRAIN_MAX_RUNTIME="${RETRAIN_MAX_RUNTIME:-3000}"            # retrain phase only, overrides MAX_RUNTIME
# Cap by "graph-side successful update steps" (= fixed search workload). Empty = unlimited.
# Used by the test5 ablation to equalize workload before comparing wall-clock time.
MAX_GRAPH_STEPS="${MAX_GRAPH_STEPS:-}"

# Validation-set early stopping
VAL_EVAL_INTERVAL="${VAL_EVAL_INTERVAL:-100}"              # validation eval interval (seconds)
VAL_EVAL_MIN_UPDATES="${VAL_EVAL_MIN_UPDATES:-20}"        # minimum updates before the first validation
VAL_PATIENCE="${VAL_PATIENCE:-5}"                         # consecutive no-improvement rounds → early stop
VAL_MIN_DELTA="${VAL_MIN_DELTA:-0.01}"                    # a round only resets the patience if AUC gains > this (0 = any gain)
VAL_EARLY_STOP="${VAL_EARLY_STOP:-true}"                  # true/false
VAL_EVAL_ON_TEST="${VAL_EVAL_ON_TEST:-}"                  # non-empty = run periodic eval on the test split with greedy decoding (comparable metric)
TEXT_DDP="${TEXT_DDP:-}"                                  # non-empty = use real DDP (bucketing + comm overlap) instead of the hand-written all_reduce

# Node-selection strategy (which seed nodes a batch covers; cache/drop-on-miss policy unchanged).
TEXT_SHARD_NODES="${TEXT_SHARD_NODES:-1}"                 # empty = every text worker shuffles the whole train set (old behaviour)
GRAPH_CACHE_SAMPLING="${GRAPH_CACHE_SAMPLING:-1}"         # empty = graph seeds sampled blindly, so a cold cache drops nearly every batch
LAG_INJECT_STEPS="${LAG_INJECT_STEPS:-0}"                    # embedding-lag injection (lag.py): producer batches
LAG_INJECT_SECONDS="${LAG_INJECT_SECONDS:-0}"                # ... and/or seconds
LAG_INJECT_DIRECTION="${LAG_INJECT_DIRECTION:-both}"         # both | graph_to_text | text_to_graph
TRAIN_ENTRY="${TRAIN_ENTRY:-bin/async_coexec_train.py}"         # entry script; bin/staleness_sim.py = staleness experiment B
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"                    # extra arguments appended to the entry's command line (word-split)
SIM_OMP_THREADS="${SIM_OMP_THREADS:-5}"                      # CPU threads per bin/staleness_sim.py process (experiment B runs ~10 at once)
[[ "${TRAIN_ENTRY}" == bin/staleness_sim.py && -z "${OMP_NUM_THREADS:-}" ]] && export OMP_NUM_THREADS="${SIM_OMP_THREADS}"
STALENESS_STATS="${STALENESS_STATS:-}"                      # non-empty = count-only staleness stats (lag_G/lag_T, coverage)
STALENESS_COVER_GATE="${STALENESS_COVER_GATE:-0.98}"         # lag_G counted once C_G covers this share of training nodes
VAL_EVAL_ALWAYS="${VAL_EVAL_ALWAYS:-}"                      # non-empty = validate periodically even with VAL_EARLY_STOP=false
TEXT_TOKEN_BUDGET="${TEXT_TOKEN_BUDGET:-0}"                 # > 0: length-bucketed text chunks of <= this many padded tokens
TEXT_MAX_CHUNK_ROWS="${TEXT_MAX_CHUNK_ROWS:-1024}"          # row cap of such a chunk
TEXT_PROBE_WORST_CASE="${TEXT_PROBE_WORST_CASE:-}"          # non-empty = memory probe: first chunk carries the longest text
TEXT_GRAD_CKPT="${TEXT_GRAD_CKPT:-}"                        # non-empty = activation checkpointing in the text LM
GRAPH_PUSH_EXTRA_MAX="${GRAPH_PUSH_EXTRA_MAX:-0}"           # per graph batch, push up to N exact by-product embeddings besides the seeds (0 = off)
GRAPH_CACHE_EXPLORE_EVERY="${GRAPH_CACHE_EXPLORE_EVERY:-0}"  # every Nth graph batch covers never-seen nodes; +22% coverage rate, -16pp graph hit rate
VAL_EARLY_STOP_MIN_TEXT_UPDATES="${VAL_EARLY_STOP_MIN_TEXT_UPDATES:-1}"  # block val early stop until the text tower has actually trained
# k-step distributed (local_avg) = TAGNAS's default sync strategy (since 2026-06-28). Only
# takes effect with multiple workers (>=2); with 1 worker / sync_push it's automatically inert.
# Sweet spot K=16 / warmup=30 (measured on the pubmed K-sweep: ~2x throughput, equal quality).
# To fall back to per-step all_reduce: set TEXT_LOCAL_AVG=0.
TEXT_LOCAL_AVG="${TEXT_LOCAL_AVG:-1}"                     # k-step enabled by default; =0 falls back to per-step
TEXT_SYNC_INTERVAL="${TEXT_SYNC_INTERVAL:-16}"          # local_avg's K (batches between two weighted averages)
TEXT_SYNC_WARMUP="${TEXT_SYNC_WARMUP:-30}"             # local_avg warmup sync count (average every step for the first N times)
LINK_NEG_SAMPLING="${LINK_NEG_SAMPLING:-auto}"            # default/type_aware/auto

# Ablation switches (default empty = full system)
GRAPH_DEVICE="${GRAPH_DEVICE:-}"                       # w/o heterog placement: e.g. cuda:0 (graph tower shares a GPU with text worker 0)
BLOCK_ON_MISS="${BLOCK_ON_MISS:-}"                     # w/o drop-on-miss: non-empty = block and wait on a miss (with timeout) instead of skipping
BLOCK_ON_MISS_TIMEOUT="${BLOCK_ON_MISS_TIMEOUT:-}"     # block timeout in seconds (default 30)

# Adaptive pruning (dynamic search-space reduction, only takes effect in the search phase)
USE_ADAPTIVE_PRUNING="${USE_ADAPTIVE_PRUNING:-true}"     # true/false
PRUNING_THRESHOLD="${PRUNING_THRESHOLD:-0.20}"             # candidates with weight below this threshold get pruned
PRUNING_CHECK_INTERVAL="${PRUNING_CHECK_INTERVAL:-100}"     # check once every N arch updates
PRUNING_MIN_UPDATES="${PRUNING_MIN_UPDATES:-100}"          # warmup arch updates before pruning starts
PRUNING_MIN_KEEP="${PRUNING_MIN_KEEP:-2}"                  # minimum candidates kept per layer/group
PRUNING_MODE="${PRUNING_MODE:-relative}"                   # absolute (raw softmax weight) / relative (fraction of the current group max)
PRUNING_THRESHOLD_ABS="${PRUNING_THRESHOLD_ABS:-0.0}"      # absolute floor delta of max(delta, rho*p_max); 0.0 = disabled
PRUNING_GROUPS="${PRUNING_GROUPS:-op, norm, act}"            # comma-separated groups allowed to be pruned (recommended: op only) op / norm / act

# retrain-phase warm-start (in "both" mode, paths are auto-derived from search artifacts; symmetric for graph+text)
RETRAIN_WARM_START="${RETRAIN_WARM_START:-true}"            # true/false
WARM_START_GRAPH_CKPT="${WARM_START_GRAPH_CKPT:-}"          # set explicitly when running retrain alone; can be left empty for "both"
WARM_START_TEXT_CKPT="${WARM_START_TEXT_CKPT:-}"            # same as above

# Required for the retrain phase (auto-generated by the search phase)
GENOTYPE_PATH="${GENOTYPE_PATH:-}"

# ──────────────────────────────────────────────────────────────────────
# Port-in-use check (avoid EADDRINUSE from leftover processes)
# ──────────────────────────────────────────────────────────────────────
for port in "${DIST_PORT}" "${RPC_PORT}"; do
    if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
        echo "[WARN] Port ${port} is already in use, please clean up any leftover process first:"
        echo "       lsof -i:${port}   # check what's using it"
        echo "       kill -9 <PID>     # kill it"
        echo "       or use different ports: DIST_PORT=30590 RPC_PORT=30591 bash run_sys.sh"
        exit 1
    fi
done

# ──────────────────────────────────────────────────────────────────────
# Assemble the launch command
# ──────────────────────────────────────────────────────────────────────
CMD=(
    python "${TRAIN_ENTRY}"
    --phase "${PHASE}"
    --config-path "${CONFIG_PATH}"
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --search-epochs "${SEARCH_EPOCHS}"
    --retrain-epochs "${RETRAIN_EPOCHS}"
    --graph-batch-size "${GRAPH_BATCH_SIZE}"
    --text-chunk-size "${TEXT_CHUNK_SIZE}"
    --graph-k-hop "${GRAPH_K_HOP}"
    --graph-cache-capacity "${GRAPH_CACHE_CAPACITY}"
    --text-cache-capacity "${TEXT_CACHE_CAPACITY}"
    --dist-port "${DIST_PORT}"
    --rpc-port "${RPC_PORT}"
    --metrics-interval-seconds "${METRICS_INTERVAL}"
    --checkpoint-interval-seconds "${CKPT_INTERVAL}"
    --val-eval-interval-seconds "${VAL_EVAL_INTERVAL}"
    --val-eval-min-updates "${VAL_EVAL_MIN_UPDATES}"
    --val-patience "${VAL_PATIENCE}"
    --val-min-delta "${VAL_MIN_DELTA}"
)

# Validation-set early-stop switches
[[ -n "${VAL_EVAL_ON_TEST}" ]] && CMD+=(--val-eval-on-test)
[[ -n "${TEXT_DDP}" ]] && CMD+=(--text-ddp)
[[ "${TEXT_LOCAL_AVG}" != "0" ]] && CMD+=(--text-local-avg)   # k-step by default; TEXT_LOCAL_AVG=0 falls back to per-step
[[ -n "${TEXT_SYNC_INTERVAL}" ]] && CMD+=(--text-sync-interval "${TEXT_SYNC_INTERVAL}")
[[ -n "${TEXT_SYNC_WARMUP}" ]] && CMD+=(--text-sync-warmup "${TEXT_SYNC_WARMUP}")
[[ -n "${GRAPH_DEVICE}" ]] && CMD+=(--graph-device "${GRAPH_DEVICE}")           # w/o heterog placement ablation
[[ -n "${BLOCK_ON_MISS}" ]] && CMD+=(--block-on-miss)                            # w/o drop-on-miss ablation
[[ -n "${BLOCK_ON_MISS_TIMEOUT}" ]] && CMD+=(--block-on-miss-timeout "${BLOCK_ON_MISS_TIMEOUT}")
[[ -z "${TEXT_SHARD_NODES}" ]] && CMD+=(--no-text-shard-nodes)                   # old per-worker full-set shuffle
[[ -z "${GRAPH_CACHE_SAMPLING}" ]] && CMD+=(--no-graph-cache-sampling)           # old blind seed sampling
CMD+=(--graph-cache-explore-every "${GRAPH_CACHE_EXPLORE_EVERY}")
CMD+=(--graph-push-extra-max "${GRAPH_PUSH_EXTRA_MAX}")
CMD+=(--val-negative-sampling "${LINK_NEG_SAMPLING}")                          # online monitor = final eval's negatives
[[ -n "${TEXT_GRAD_CKPT}" ]] && CMD+=(--text-grad-checkpointing)
CMD+=(--text-token-budget "${TEXT_TOKEN_BUDGET}" --text-max-chunk-rows "${TEXT_MAX_CHUNK_ROWS}")
[[ -n "${TEXT_PROBE_WORST_CASE}" ]] && CMD+=(--text-probe-worst-case)
[[ -n "${VAL_EVAL_ALWAYS}" ]] && CMD+=(--val-eval-always)
# shellcheck disable=SC2206
[[ -n "${EXTRA_TRAIN_ARGS}" ]] && CMD+=(${EXTRA_TRAIN_ARGS})
[[ -n "${STALENESS_STATS}" ]] && CMD+=(--staleness-stats --staleness-cover-gate "${STALENESS_COVER_GATE}")
CMD+=(--lag-inject-steps "${LAG_INJECT_STEPS}" --lag-inject-seconds "${LAG_INJECT_SECONDS}" --lag-inject-direction "${LAG_INJECT_DIRECTION}")
CMD+=(--val-early-stop-min-text-updates "${VAL_EARLY_STOP_MIN_TEXT_UPDATES}")
if [[ "${VAL_EARLY_STOP}" == "false" ]]; then
    CMD+=(--no-val-early-stop)
fi

# Adaptive pruning switch + hyperparameters
if [[ "${USE_ADAPTIVE_PRUNING}" == "true" ]]; then
    CMD+=(--use-adaptive-pruning)
else
    CMD+=(--no-use-adaptive-pruning)
fi
CMD+=(--pruning-threshold "${PRUNING_THRESHOLD}")
CMD+=(--pruning-check-interval-updates "${PRUNING_CHECK_INTERVAL}")
CMD+=(--pruning-min-updates-before-start "${PRUNING_MIN_UPDATES}")
CMD+=(--pruning-min-candidates-per-layer "${PRUNING_MIN_KEEP}")
CMD+=(--pruning-threshold-mode "${PRUNING_MODE}")
CMD+=(--pruning-threshold-abs "${PRUNING_THRESHOLD_ABS}")
CMD+=(--pruning-groups "${PRUNING_GROUPS}")

# Search sharpening (addresses root cause of alpha under-convergence): arch_lr + arch
# softmax temperature annealing. Optional, defaults to no behavior change.
[[ -n "${ARCH_LR:-}" ]] && CMD+=(--graph-arch-lr "${ARCH_LR}")
[[ -n "${ARCH_TEMP_START:-}" ]] && CMD+=(--graph-arch-temp-start "${ARCH_TEMP_START}")
[[ -n "${ARCH_TEMP_END:-}" ]] && CMD+=(--graph-arch-temp-end "${ARCH_TEMP_END}")

# retrain warm-start switch
if [[ "${RETRAIN_WARM_START}" == "true" ]]; then
    CMD+=(--retrain-warm-start)
else
    CMD+=(--no-retrain-warm-start)
fi

# Optional parameters
[[ -n "${GRAPH_EPOCHS}" ]]          && CMD+=(--graph-epochs "${GRAPH_EPOCHS}")
[[ -n "${TEXT_EPOCHS}" ]]           && CMD+=(--text-epochs "${TEXT_EPOCHS}")
[[ -n "${NUM_LAYERS}" ]]            && CMD+=(--num-layers "${NUM_LAYERS}")
[[ -n "${NUM_TEXT_WORKERS}" ]]      && CMD+=(--num-text-workers "${NUM_TEXT_WORKERS}")
[[ "${SYNC_PUSH}" == "true" ]]      && CMD+=(--sync-push)
[[ -n "${MAX_RUNTIME}" ]]           && CMD+=(--max-runtime-seconds "${MAX_RUNTIME}")
[[ -n "${SEARCH_MAX_RUNTIME}" ]]    && CMD+=(--search-max-runtime-seconds "${SEARCH_MAX_RUNTIME}")
[[ -n "${RETRAIN_MAX_RUNTIME}" ]]   && CMD+=(--retrain-max-runtime-seconds "${RETRAIN_MAX_RUNTIME}")
[[ -n "${MAX_GRAPH_STEPS}" ]]       && CMD+=(--max-graph-steps "${MAX_GRAPH_STEPS}")
[[ -n "${TRAIN_NODE_FRACTION}" ]]   && CMD+=(--train-node-fraction "${TRAIN_NODE_FRACTION}")
[[ -n "${GENOTYPE_PATH}" ]]         && CMD+=(--genotype-path "${GENOTYPE_PATH}")
[[ -n "${WARM_START_GRAPH_CKPT}" ]] && CMD+=(--warm-start-graph-ckpt "${WARM_START_GRAPH_CKPT}")
[[ -n "${WARM_START_TEXT_CKPT}" ]]  && CMD+=(--warm-start-text-ckpt "${WARM_START_TEXT_CKPT}")
# Anti contrastive-collapse: WD=0.01 suppresses embedding collapse, EPS=1e-8 stabilizes
# the search (aligned with DDP/DeepSpeed; empty = keep YAML defaults 0.0/1e-4)
[[ -n "${WD:-}" ]]                  && CMD+=(--weight-decay "${WD}")
[[ -n "${EPS:-}" ]]                 && CMD+=(--adam-eps "${EPS}")
# time-breakdown profiling
[[ -n "${PROFILE}" ]]               && CMD+=(--profile)
[[ -n "${PROFILE_OUT}" ]]           && CMD+=(--profile-out "${PROFILE_OUT}")
[[ -n "${SEED:-}" ]]                && CMD+=(--seed "${SEED}")

# ──────────────────────────────────────────────────────────────────────
# Print startup info and run
# ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " Async distributed training launch"
echo "=========================================="
echo " Phase        : ${PHASE}"
echo " Config       : ${CONFIG_PATH}"
echo " Dataset      : ${DATASET}"
echo " Output dir   : ${OUTPUT_DIR}"
echo " Epochs       : search=${SEARCH_EPOCHS}  retrain=${RETRAIN_EPOCHS}"
[[ -n "${GRAPH_EPOCHS}" ]] && echo "   graph_epochs=${GRAPH_EPOCHS}"
[[ -n "${TEXT_EPOCHS}" ]]  && echo "   text_epochs =${TEXT_EPOCHS}"
echo " Batch sizes  : graph=${GRAPH_BATCH_SIZE}  text=${TEXT_CHUNK_SIZE}"
[[ -n "${TRAIN_NODE_FRACTION}" ]] && echo " Train nodes  : fraction=${TRAIN_NODE_FRACTION} (only this fraction of train nodes is randomly used for alignment; val/test still use the full set)"
echo " Graph k-hop  : ${GRAPH_K_HOP}  num_layers=${NUM_LAYERS:-<from YAML>} (= GNN k_hop)"
if [[ "${SYNC_PUSH}" == "true" ]]; then
    echo " Push mode    : SYNC (bidirectional lock-step; num_text_workers=1, text_chunk=graph_batch=${GRAPH_BATCH_SIZE})"
else
    echo " Push mode    : ASYNC (bidirectional, drop-on-miss) [default]"
fi
echo " Ports        : dist=${DIST_PORT}  rpc=${RPC_PORT}"
echo " GPUs         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " Val early-stop: ${VAL_EARLY_STOP}  patience=${VAL_PATIENCE}  min_delta=${VAL_MIN_DELTA}  interval=${VAL_EVAL_INTERVAL}s"
echo " Node selection: text_shard=${TEXT_SHARD_NODES:-off}  graph_cache_sampling=${GRAPH_CACHE_SAMPLING:-off}  explore_every=${GRAPH_CACHE_EXPLORE_EVERY}"
echo " Eval negs    : ${LINK_NEG_SAMPLING}"
echo " Adaptive pruning: ${USE_ADAPTIVE_PRUNING}  mode=${PRUNING_MODE}  threshold=${PRUNING_THRESHOLD}  groups=${PRUNING_GROUPS}  interval=${PRUNING_CHECK_INTERVAL}u  warmup=${PRUNING_MIN_UPDATES}u  min_keep=${PRUNING_MIN_KEEP}"
echo " Retrain warm-start: ${RETRAIN_WARM_START}  graph=${WARM_START_GRAPH_CKPT:-<auto>}  text=${WARM_START_TEXT_CKPT:-<auto>}"
echo " Max runtime  : global=${MAX_RUNTIME:-<none>}s  search=${SEARCH_MAX_RUNTIME:-<none>}s  retrain=${RETRAIN_MAX_RUNTIME:-<none>}s"
echo "=========================================="
echo "[cmd] ${CMD[*]}"
echo

# Temporarily disable set -e to capture the exit code: torch RPC/distributed teardown
# occasionally returns nonzero (noise), but the training artifacts
# (search_graph.pt + genotype.json) have actually already been written. Letting set -e
# kill the script here would skip the evaluation below, leaving
# link-prediction-eval-gasso.csv missing.
set +e
"${CMD[@]}"
TRAIN_EXIT_CODE=$?
set -e

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
    echo "[WARN] Training process exited with a nonzero code (${TRAIN_EXIT_CODE}); this may just be distributed-teardown noise."
    echo "       If training artifacts are found below, evaluation continues; if truly absent, the evaluation step is skipped automatically."
fi

# ──────────────────────────────────────────────────────────────────────
# Automatically run link-prediction evaluation after training completes
# ──────────────────────────────────────────────────────────────────────
# Find this run's output directory (pick the most recently modified one)
# Note the `|| true`: head -1 closes the pipe as soon as it's read, and ls continuing
# to write hits SIGPIPE (141); under pipefail+set -e this would silently kill the
# script (more likely to trigger the more run-* directories there are), leaving the
# evaluation step never reached.
RUN_DIR=$(ls -dt "${OUTPUT_DIR}"/run-*/ 2>/dev/null | head -1 || true)

if [[ -z "${RUN_DIR}" ]]; then
    echo "[WARN] No training artifact directory found, skipping evaluation"
    exit 0
fi

# Strip the trailing /
RUN_DIR="${RUN_DIR%/}"

echo
echo "=========================================="
echo " Link-Prediction evaluation"
echo "=========================================="
echo " Run dir  : ${RUN_DIR}"
echo " Dataset  : ${DATASET}"
echo "=========================================="

# Evaluate the search phase (if artifacts exist)
if [[ -f "${RUN_DIR}/search/search_graph.pt" && -f "${RUN_DIR}/search/genotype.json" ]]; then
    echo
    echo "[eval] Evaluating search phase (final)..."
    python bin/eval_link_pred_distributed.py \
        --run-dir "${RUN_DIR}" \
        --phase search \
        --config-path "${CONFIG_PATH}" \
        --dataset-name "${DATASET}" \
        --data-dir "data/${DATASET}" \
        --negative-sampling "${LINK_NEG_SAMPLING}" \
    || echo "[WARN] search phase (final) evaluation failed, continuing..."

    if [[ -f "${RUN_DIR}/search/search_graph_best.pt" && -f "${RUN_DIR}/search/genotype_best.json" ]]; then
        echo
        echo "[eval] Evaluating search phase (best val-AUC)..."
        python bin/eval_link_pred_distributed.py \
            --run-dir "${RUN_DIR}" \
            --phase search \
            --use-best \
            --config-path "${CONFIG_PATH}" \
            --dataset-name "${DATASET}" \
            --data-dir "data/${DATASET}" \
            --negative-sampling "${LINK_NEG_SAMPLING}" \
        || echo "[WARN] search phase (best) evaluation failed, continuing..."
    fi
fi

# Evaluate the retrain phase (if artifacts exist)
if [[ -f "${RUN_DIR}/retrain/retrain_graph.pt" && -f "${RUN_DIR}/retrain/genotype.json" ]]; then
    echo
    echo "[eval] Evaluating retrain phase (final)..."
    python bin/eval_link_pred_distributed.py \
        --run-dir "${RUN_DIR}" \
        --phase retrain \
        --config-path "${CONFIG_PATH}" \
        --dataset-name "${DATASET}" \
        --data-dir "data/${DATASET}" \
        --negative-sampling "${LINK_NEG_SAMPLING}" \
    || echo "[WARN] retrain phase (final) evaluation failed, continuing..."

    if [[ -f "${RUN_DIR}/retrain/retrain_graph_best.pt" && -f "${RUN_DIR}/retrain/genotype_best.json" ]]; then
        echo
        echo "[eval] Evaluating retrain phase (best val-AUC)..."
        python bin/eval_link_pred_distributed.py \
            --run-dir "${RUN_DIR}" \
            --phase retrain \
            --use-best \
            --config-path "${CONFIG_PATH}" \
            --dataset-name "${DATASET}" \
            --data-dir "data/${DATASET}" \
            --negative-sampling "${LINK_NEG_SAMPLING}" \
        || echo "[WARN] retrain phase (best) evaluation failed, continuing..."
    fi
fi

echo
echo "[done] Training + evaluation all complete"
echo "  Training artifacts: ${RUN_DIR}"
echo "  Evaluation results: ${RUN_DIR}/link-prediction-eval-gasso.csv"

# =========================================================================
# Common usage examples
# =========================================================================
#
# 1. Run a default architecture search (auto-evaluates after training)
#    bash run_sys.sh
#
# 2. Quick smoke test with just 2 epochs
#    SEARCH_EPOCHS=2 MAX_RUNTIME=120 bash run_sys.sh
#
# 3. Set graph/text epochs separately
#    GRAPH_EPOCHS=5 TEXT_EPOCHS=10 bash run_sys.sh
#
# 4. Small batch for more thorough training (recommended)
#    GRAPH_BATCH_SIZE=64 TEXT_CHUNK_SIZE=64 bash run_sys.sh
#
# 5. Retrain phase (using the genotype produced by search)
#    PHASE=retrain GENOTYPE_PATH=model/distributed/search/genotype.json bash run_sys.sh
#
# 6. Run search + retrain together
#    PHASE=both bash run_sys.sh
#
# 7. Change ports on conflict
#    DIST_PORT=30590 RPC_PORT=30591 bash run_sys.sh
#
# 8. Use a single GPU only
#    CUDA_VISIBLE_DEVICES=0 bash run_sys.sh
#
# 9. Check training status (in a separate terminal)
#    tail -f model/distributed/run-*/search/metrics.jsonl | jq
#    cat model/distributed/run-*/search/manifest.json | jq
#
# 10. Run evaluation only (no training), against existing artifacts
#     python bin/eval_link_pred_distributed.py \
#       --run-dir model/distributed/run-XXXXX \
#       --data-dir data/pubmed
#
# =========================================================================
