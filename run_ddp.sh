#!/usr/bin/env bash
# =========================================================================
# bin/ddp_train.py single-node multi-GPU DDP training launcher
#
# Uses mp.spawn to launch WORLD_SIZE ranks, each rank holds the full graph, and
# DistributedSampler splits along the text-batch dimension. Underlying data flows
# through PubmedGraphTextDataModule, supporting any dataset laid out as
# data/<name>/raw/{graph,node-data,texts}.csv.
#
# Run specific datasets:
#   DATASETS="goodreads-crime,amazon-baby" bash run_ddp.sh
#
# Run all available datasets under data/:
#   DATASETS_ALL=true bash run_ddp.sh
#
# Change epochs / lr / GPU count:
#   NUM_EPOCHS=50 LEARNING_RATE=5e-5 WORLD_SIZE=2 bash run_ddp.sh
# =========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ──────────────────────────────────────────────────────────────────────
# Tunable parameters (override via environment variables)
# ──────────────────────────────────────────────────────────────────────

# GPU selection (DDP rank 0 -> first card, rank 1 -> second card, and so on)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# HuggingFace endpoint + offline mode (model already cached -> skip Hub network calls,
# avoiding a ConnectTimeout from mirrors that don't support the list_repo_templates tree
# API). Set HF_OFFLINE=false if the model isn't cached yet.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Dataset (choose one):
#   DATASETS="a,b,c"  explicit list (comma-separated)
#   DATASETS_ALL=true run all available datasets under data/
DATASETS="${DATASETS:-pubmed}"
DATASETS_ALL="${DATASETS_ALL:-false}"

# Training hyperparameters
CONFIG_PATH="${CONFIG_PATH:-./configs/clip-graph/inductive-causal/pubmed/gassobase.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model/runpytorch}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WORLD_SIZE="${WORLD_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # for large graphs (goodreads-crime, etc.) 24 workers x 2 ranks has huge startup overhead and is deadlock-prone; 4 is enough
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-1}"
# If unset, falls back to data.init_args.batch_size in the YAML; if set, overrides via CLI
BATCH_SIZE="${BATCH_SIZE:-16}"
# GNN k_hop / layer count (gnn_params.num_layers). Empty string keeps the YAML value,
# a number overrides via CLI. For large bipartite graphs (e.g. twitter), NUM_LAYERS=1
# is recommended to avoid over-smoothing.
NUM_LAYERS="${NUM_LAYERS:-}"
# Training wall-clock cap (seconds). Empty = run to completion by epoch; if set, stops
# once the time is up (used for test3 equalized budget).
MAX_RUNTIME="${MAX_RUNTIME:-}"
# Use only this fraction (0,1] of text per epoch for alignment (re-sampled randomly
# each epoch). Empty = use all text.
TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE:-}"
# Convergence/early-stop monitoring metric: auc = val link-prediction AUC stops
# improving; loss = val loss stops decreasing. Empty = use the python default (auc).
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-}"
# time-breakdown profiling: non-empty PROFILE adds --profile; PROFILE_OUT sets the JSON output path.
PROFILE="${PROFILE:-}"
PROFILE_OUT="${PROFILE_OUT:-}"
# Non-empty = skip the post-training link-prediction post-eval (not needed for profiling stress tests).
NO_POST_EVAL="${NO_POST_EVAL:-}"
# Override AdamW eps (empty = keep YAML's 1e-4; use 1e-8 to fix search plateaus, aligned with DeepSpeed).
EPS="${EPS:-}"
# Override weight_decay (empty = keep YAML's 0.0; use 0.01 to suppress contrastive collapse, aligned with DeepSpeed, weights group only).
WD="${WD:-}"
# Recompute val-AUC every N seconds and write val_timeline.csv (dense time-to-accuracy sampling; used by test1-2). Empty = disabled.
VAL_EVAL_INTERVAL="${VAL_EVAL_INTERVAL:-}"
# Non-empty = run periodic sampling link prediction on the test split instead (test1-2 uses the test split for an honest speedup comparison). Empty = val.
VAL_EVAL_ON_TEST="${VAL_EVAL_ON_TEST:-}"

# ──────────────────────────────────────────────────────────────────────
# Assemble the command
# ──────────────────────────────────────────────────────────────────────

CMD=(
    python bin/ddp_train.py
    --config-path "${CONFIG_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --num-epochs "${NUM_EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --world-size "${WORLD_SIZE}"
    --master-port "${MASTER_PORT}"
    --num-workers "${NUM_WORKERS}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --checkpoint-frequency "${CHECKPOINT_FREQUENCY}"
)

if [[ -n "${BATCH_SIZE}" ]]; then
    CMD+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${NUM_LAYERS}" ]]; then
    CMD+=(--num-layers "${NUM_LAYERS}")
fi

if [[ -n "${MAX_RUNTIME}" ]]; then
    CMD+=(--max-runtime-seconds "${MAX_RUNTIME}")
fi

if [[ -n "${TEXT_SUBSAMPLE}" ]]; then
    CMD+=(--text-subsample-fraction "${TEXT_SUBSAMPLE}")
fi

if [[ -n "${EARLY_STOP_METRIC}" ]]; then
    CMD+=(--early-stop-metric "${EARLY_STOP_METRIC}")
fi

if [[ -n "${PROFILE}" ]]; then
    CMD+=(--profile)
fi
if [[ -n "${PROFILE_OUT}" ]]; then
    CMD+=(--profile-out "${PROFILE_OUT}")
fi
if [[ -n "${SEED:-}" ]]; then
    CMD+=(--seed "${SEED}")
fi
if [[ -n "${NO_POST_EVAL}" ]]; then
    CMD+=(--no-post-eval)
fi
if [[ -n "${EPS}" ]]; then
    CMD+=(--adam-eps "${EPS}")
fi
if [[ -n "${WD}" ]]; then
    CMD+=(--weight-decay "${WD}")
fi
if [[ -n "${VAL_EVAL_INTERVAL}" ]]; then
    CMD+=(--val-eval-interval-seconds "${VAL_EVAL_INTERVAL}")
fi
[[ -n "${VAL_EVAL_ON_TEST}" ]] && CMD+=(--val-eval-on-test)

if [[ "${DATASETS_ALL}" == "true" ]]; then
    CMD+=(--datasets-all)
else
    CMD+=(--datasets "${DATASETS}")
fi

# ──────────────────────────────────────────────────────────────────────
# Print + run
# ──────────────────────────────────────────────────────────────────────

echo "=========================================="
echo " GraphNAS DDP training launch"
echo "=========================================="
echo " GPUs         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " world_size   : ${WORLD_SIZE}"
echo " master_port  : ${MASTER_PORT}"
if [[ "${DATASETS_ALL}" == "true" ]]; then
    echo " Datasets     : <all available>"
else
    echo " Datasets     : ${DATASETS}"
fi
echo " Config       : ${CONFIG_PATH}"
echo " Epochs       : ${NUM_EPOCHS}"
echo " Learning rate: ${LEARNING_RATE}"
echo " Num workers  : ${NUM_WORKERS}"
echo " Batch size   : ${BATCH_SIZE:-<from YAML>}"
echo " Num layers   : ${NUM_LAYERS:-<from YAML>}  (= GNN k_hop)"
echo " Early-stop   : patience=${EARLY_STOPPING_PATIENCE}"
echo " Checkpoint   : every ${CHECKPOINT_FREQUENCY} epochs"
echo "=========================================="
echo "[cmd] ${CMD[*]}"
echo

exec "${CMD[@]}"
