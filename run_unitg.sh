#!/usr/bin/env bash
# =========================================================================
# B5: UniTG-style baseline launcher (corresponds to bin/unitg_train.py)
#
#   Deep reversible GNN (RevGNN, 112 layers by default) trained with inter-layer
#   flow parallelism across the visible GPUs, while the text encoder is fine-tuned
#   concurrently in the pipeline bubbles and periodically synchronised into the
#   GNN input features (UniTG, PVLDB'26, ported to link prediction).
#
# Usage examples:
#   bash run_unitg.sh
#   DATASETS=pubmed CUDA_VISIBLE_DEVICES=0,1,3 bash run_unitg.sh
#   DATASETS=trex CONFIG_PATH=configs/clip-graph/inductive-causal/pubmed/base-linkbert.yaml bash run_unitg.sh
#
# Environment variables (defaults used if unset):
#   - DATASETS:        comma-separated dataset names under data/ (default pubmed)
#   - CONFIG_PATH:     YAML providing lm_name / lr / data args (default base.yaml = distilgpt2)
#   - WORLD_SIZE:      number of pipeline stages = GPUs (default: count of CUDA_VISIBLE_DEVICES)
#   - MASTER_PORT:     rendezvous port (default 30500; keep < 32768, the ephemeral range starts there)
#   - NUM_LAYERS / HIDDEN / MICROBATCHES / MAX_NODES_PER_MICROBATCH: deep-GNN + pipeline knobs
#   - OBJECTIVE: contrastive (default; same graph-text InfoNCE as B1-B3/TAGNAS) | link (UniTG task loss, reference only)
#   - LM_OBJECTIVE: contrastive (default; bidirectional: the LM aligns to the GNN table, the GNN to the LM table) | link (LM anchored on its own text-pair InfoNCE, one-directional LM->GNN)
#   - CONTRASTIVE_NODES / GNN_LR / GNN_DROPOUT / GNN_WEIGHT_DECAY / LM_LR / LM_BATCH_SIZE / LM_MAX_LENGTH / LM_NUM_EXITS
#   - NUM_EPOCHS:      nominal epoch budget (default 1000)
#   - MAX_RUNTIME:     wall-clock cap in seconds (default 3000; also the Δt schedule budget)
#   - VAL_EVAL_INTERVAL / VAL_EVAL_MIN_EPOCHS / EARLY_STOPPING_PATIENCE (or EARLY_STOP_PATIENCE)
#   - VAL_EVAL_ON_TEST: non-empty = monitor the test split (as test1-2 do for the honest speedup)
#   - SYNC_SCHEDULE / SYNC_TIGHTEN / SYNC_UNIT / NO_INITIAL_SYNC
#   - NO_LAZY_TUNER / NO_BUBBLE_INTERLEAVE: ablation switches; ENCODE_EXIT=full|group (exit used for the synced table)
#   - NO_EVAL_AT_START: default 1 = do not score the untrained model at epoch 0 (so the early-stopping
#     patience only counts from real training checkpoints, like the other systems)
#   - PROFILE / PROFILE_OUT: per-iteration telemetry for experiments/04_time_breakdown.sh
#   - LINK_NEG_SAMPLING: default/type_aware/auto (default auto)
#   - SEED, NO_POST_EVAL, OUTPUT_ROOT, RUN_NAME, PYTHON
#   - TEXT_SUBSAMPLE / EARLY_STOP_METRIC: accepted for parity with run_clip.sh (recorded only)
# =========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# HuggingFace endpoint + offline mode (models already cached -> skip Hub network calls).
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Three GPUs by default = three pipeline stages. Works on Default and Exclusive_Process cards
# (one process per card).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
IFS=',' read -ra _GPU_POOL <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${WORLD_SIZE:-${#_GPU_POOL[@]}}"

PYTHON="${PYTHON:-python3}"

DATASETS="${DATASETS:-pubmed}"
CONFIG_PATH="${CONFIG_PATH:-./configs/clip-graph/inductive-causal/pubmed/base.yaml}"
MASTER_PORT="${MASTER_PORT:-30500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model/runpytorch}"
RUN_NAME="${RUN_NAME:-}"

NUM_LAYERS="${NUM_LAYERS:-112}"
HIDDEN="${HIDDEN:-224}"
MICROBATCHES="${MICROBATCHES:-}"                   # default = WORLD_SIZE (M = P)
MAX_NODES_PER_MICROBATCH="${MAX_NODES_PER_MICROBATCH:-100000}"
GNN_LR="${GNN_LR:-5e-4}"
GNN_DROPOUT="${GNN_DROPOUT:-0.2}"
GNN_WEIGHT_DECAY="${GNN_WEIGHT_DECAY:-0.01}"
LM_LR="${LM_LR:-}"
LM_BATCH_SIZE="${LM_BATCH_SIZE:-64}"
OBJECTIVE="${OBJECTIVE:-contrastive}"
LM_OBJECTIVE="${LM_OBJECTIVE:-contrastive}"
CONTRASTIVE_NODES="${CONTRASTIVE_NODES:-1024}"
LM_MAX_LENGTH="${LM_MAX_LENGTH:-128}"
TEXT_PER_NODE="${TEXT_PER_NODE:-joined}"       # joined = UniTG artifact; tagnas = the exact text TAGNAS feeds
TAGNAS_TEXT_CONFIG="${TAGNAS_TEXT_CONFIG:-configs/clip-graph/inductive-causal/pubmed/gassobase.yaml}"  # TAGNAS YAML defining that text
LM_DYNAMIC_PADDING="${LM_DYNAMIC_PADDING:-}"   # non-empty = pad each LM batch to its longest row (as TAGNAS)
NO_LM_AUTOCAST="${NO_LM_AUTOCAST:-}"           # non-empty = run the LM in fp32 (as TAGNAS) instead of bf16
LM_TRAIN_NODE_FRACTION="${LM_TRAIN_NODE_FRACTION:-}"  # e.g. 0.1 = TAGNAS's TRAIN_NODE_FRACTION default; empty = all
LM_PROBE_WORST_CASE="${LM_PROBE_WORST_CASE:-}"  # memory probes: first LM batch carries the pool's longest text
LM_NUM_EXITS="${LM_NUM_EXITS:-4}"

NUM_EPOCHS="${NUM_EPOCHS:-1000}"
MAX_RUNTIME="${MAX_RUNTIME:-3000}"
VAL_EVAL_INTERVAL="${VAL_EVAL_INTERVAL:-60}"
VAL_EVAL_MIN_EPOCHS="${VAL_EVAL_MIN_EPOCHS:-5}"
# experiments/01_end_to_end.sh passes both spellings in EPOCHS mode; either works here.
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-${EARLY_STOP_PATIENCE:-10}}"
VAL_EVAL_ON_TEST="${VAL_EVAL_ON_TEST:-}"
MIN_LM_TEXTS_BEFORE_STOP="${MIN_LM_TEXTS_BEFORE_STOP:-0}"   # -1 = train text pool size; 0 = no guard
SYNC_SCHEDULE="${SYNC_SCHEDULE:-0.3,0.2,0.1,0.02}"
SYNC_TIGHTEN="${SYNC_TIGHTEN:-20}"
SYNC_UNIT="${SYNC_UNIT:-auto}"
NO_INITIAL_SYNC="${NO_INITIAL_SYNC:-}"
NO_LAZY_TUNER="${NO_LAZY_TUNER:-}"
ENCODE_EXIT="${ENCODE_EXIT:-full}"
NO_BUBBLE_INTERLEAVE="${NO_BUBBLE_INTERLEAVE:-}"
NO_EVAL_AT_START="${NO_EVAL_AT_START-1}"          # empty string = evaluate the untrained model at epoch 0
PROFILE="${PROFILE:-}"
PROFILE_OUT="${PROFILE_OUT:-}"
LINK_NEG_SAMPLING="${LINK_NEG_SAMPLING:-auto}"
NO_POST_EVAL="${NO_POST_EVAL:-}"
TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE:-}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-}"

CMD=(
  "${PYTHON}" bin/unitg_train.py
  --datasets "${DATASETS}"
  --config-path "${CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --world-size "${WORLD_SIZE}"
  --master-port "${MASTER_PORT}"
  --num-layers "${NUM_LAYERS}"
  --hidden-channels "${HIDDEN}"
  --max-nodes-per-microbatch "${MAX_NODES_PER_MICROBATCH}"
  --halo-hops "${HALO_HOPS:-0}"
  --gnn-lr "${GNN_LR}"
  --gnn-dropout "${GNN_DROPOUT}"
  --gnn-weight-decay "${GNN_WEIGHT_DECAY}"
  --lm-batch-size "${LM_BATCH_SIZE}"
  --objective "${OBJECTIVE}"
  --lm-objective "${LM_OBJECTIVE}"
  --contrastive-nodes "${CONTRASTIVE_NODES}"
  --lm-max-length "${LM_MAX_LENGTH}"
  --text-per-node "${TEXT_PER_NODE}"
  --tagnas-text-config "${TAGNAS_TEXT_CONFIG}"
  --lm-num-exits "${LM_NUM_EXITS}"
  --encode-exit "${ENCODE_EXIT}"
  --num-epochs "${NUM_EPOCHS}"
  --max-runtime-seconds "${MAX_RUNTIME}"
  --val-eval-interval-seconds "${VAL_EVAL_INTERVAL}"
  --val-eval-min-epochs "${VAL_EVAL_MIN_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --min-lm-texts-before-stop "${MIN_LM_TEXTS_BEFORE_STOP}"
  --sync-schedule "${SYNC_SCHEDULE}"
  --sync-tighten-epochs "${SYNC_TIGHTEN}"
  --sync-unit "${SYNC_UNIT}"
  --negative-sampling "${LINK_NEG_SAMPLING}"
)
[[ -n "${MICROBATCHES}" ]]        && CMD+=(--microbatches "${MICROBATCHES}")
[[ -z "${MICROBATCHES}" ]]        && CMD+=(--microbatches "${WORLD_SIZE}")
[[ -n "${RUN_NAME}" ]]            && CMD+=(--run-name "${RUN_NAME}")
[[ -n "${LM_LR}" ]]               && CMD+=(--lm-lr "${LM_LR}")
[[ -n "${SEED:-}" ]]              && CMD+=(--seed "${SEED}")
[[ -n "${VAL_EVAL_ON_TEST}" ]]    && CMD+=(--val-eval-on-test)
[[ -n "${NO_INITIAL_SYNC}" ]]     && CMD+=(--no-initial-sync)
[[ -n "${NO_LAZY_TUNER}" ]]       && CMD+=(--no-lazy-tuner)
[[ -n "${LM_DYNAMIC_PADDING}" ]]  && CMD+=(--lm-dynamic-padding)
[[ -n "${NO_LM_AUTOCAST}" ]]      && CMD+=(--no-lm-autocast)
[[ -n "${LM_TRAIN_NODE_FRACTION}" ]] && CMD+=(--lm-train-node-fraction "${LM_TRAIN_NODE_FRACTION}")
[[ -n "${LM_PROBE_WORST_CASE}" ]]    && CMD+=(--lm-probe-worst-case)
[[ -n "${NO_BUBBLE_INTERLEAVE}" ]] && CMD+=(--no-bubble-interleave)
[[ -n "${NO_EVAL_AT_START}" ]]    && CMD+=(--no-eval-at-start)
[[ -n "${PROFILE}" ]]             && CMD+=(--profile)
[[ -n "${PROFILE_OUT}" ]]         && CMD+=(--profile-out "${PROFILE_OUT}")
[[ -n "${NO_POST_EVAL}" ]]        && CMD+=(--no-post-eval)
[[ -n "${TEXT_SUBSAMPLE}" ]]      && CMD+=(--text-subsample-fraction "${TEXT_SUBSAMPLE}")
[[ -n "${EARLY_STOP_METRIC}" ]]   && CMD+=(--early-stop-metric "${EARLY_STOP_METRIC}")

echo "=========================================="
echo " B5 UniTG-style baseline launch"
echo "=========================================="
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  WORLD_SIZE(stages)=${WORLD_SIZE}  MASTER_PORT=${MASTER_PORT}"
echo " DATASETS=${DATASETS}  CONFIG_PATH=${CONFIG_PATH}"
echo " Objective: gnn=${OBJECTIVE} lm=${LM_OBJECTIVE} (contrastive_nodes=${CONTRASTIVE_NODES})"
echo " GNN: RevGNN layers=${NUM_LAYERS} hidden=${HIDDEN} microbatches=${MICROBATCHES:-${WORLD_SIZE}} lr=${GNN_LR} wd=${GNN_WEIGHT_DECAY}"
echo " LM : text=${TEXT_PER_NODE} dyn_pad=${LM_DYNAMIC_PADDING:-off} fp32=${NO_LM_AUTOCAST:-off} batch=${LM_BATCH_SIZE} max_len=${LM_MAX_LENGTH} exits=${LM_NUM_EXITS} lazy=$([[ -n "${NO_LAZY_TUNER}" ]] && echo off || echo on) bubble=$([[ -n "${NO_BUBBLE_INTERLEAVE}" ]] && echo off || echo on)"
echo " Budget: epochs=${NUM_EPOCHS} max_runtime=${MAX_RUNTIME}s val_interval=${VAL_EVAL_INTERVAL}s patience=${EARLY_STOPPING_PATIENCE} on_test=${VAL_EVAL_ON_TEST:-no}"
echo " Sync  : schedule=${SYNC_SCHEDULE} tighten=${SYNC_TIGHTEN} unit=${SYNC_UNIT} initial=$([[ -n "${NO_INITIAL_SYNC}" ]] && echo off || echo on)"
echo " Eval negs: ${LINK_NEG_SAMPLING}"
echo "------------------------------------------"
echo "[cmd] ${CMD[*]}"
echo

"${CMD[@]}"
