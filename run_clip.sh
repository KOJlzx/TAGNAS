#!/usr/bin/env bash
# =========================================================================
# CLIP-Graph single-node multi-GPU DDP launcher (corresponds to bin/congrat_train.py)
#
# Usage examples:
#   bash run_clip.sh
#   DATASETS=pubmed,trex-textinit GNN_TYPES=GAT,GCN WORLD_SIZE=2 bash run_clip.sh
#   DATASETS_ALL=true WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_clip.sh
#
# Available environment variables (defaults used if unset):
#   - DATASETS:        comma-separated dataset names (equivalent to --datasets)
#   - DATASETS_ALL:    true/false (true is equivalent to --datasets-all)
#   - GNN_TYPES:       comma-separated GNN types (equivalent to --gnn-types)
#   - CONFIG_PATH:     config file path (equivalent to --config-path)
#   - NUM_EPOCHS:      training epochs (equivalent to --num-epochs)
#   - LEARNING_RATE:   learning rate (equivalent to --learning-rate)
#   - WORLD_SIZE:      number of DDP processes (equivalent to --world-size, usually = GPU count)
#   - MASTER_PORT:     DDP rendezvous port (default 29500, change on conflict)
#   - NUM_WORKERS:     DataLoader workers (equivalent to --num-workers)
#   - EARLY_STOPPING_PATIENCE: early-stopping patience (equivalent to --early-stopping-patience)
#   - CHECKPOINT_FREQUENCY:     save frequency (equivalent to --checkpoint-frequency)
#   - NUM_LAYERS:      GNN layer count / k_hop (equivalent to --num-layers, overrides gnn_params.num_layers in the YAML)
#                      for large bipartite graphs (e.g. twitter), 1 is recommended to avoid over-smoothing; empty string keeps the YAML value
#   - PYTHON:          Python interpreter command (default python)
# =========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active) -- aligned with run_ddp.sh, to avoid import torch
# failing when invoked outside the venv
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Activate venv (if not already active) -- consistent with run_ddp.sh; avoids falling
# back to the system python (no torch) when invoked from an external script.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# HuggingFace endpoint + offline mode (model already cached -> skip Hub network calls,
# avoiding a ConnectTimeout from mirrors that don't support the list_repo_templates tree
# API). Set HF_OFFLINE=false if the model isn't cached yet.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Two GPUs by default; for more, set both WORLD_SIZE and CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PYTHON="${PYTHON:-python3}"

DATASETS="${DATASETS:-pubmed}"                         # e.g. pubmed,trex-textinit
DATASETS_ALL="${DATASETS_ALL:-false}"            # true/false
GNN_TYPES="${GNN_TYPES:-GAT}"                    # e.g. GAT,GCN,SAGE,GIN,LINEAR
CONFIG_PATH="${CONFIG_PATH:-./configs/clip-graph/inductive-causal/pubmed/base.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model/runpytorch}"

NUM_EPOCHS="${NUM_EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WORLD_SIZE="${WORLD_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # for large graphs (amazon-baby / goodreads-crime, etc.) 24 workers x 2 ranks has huge startup overhead and is prone to deadlock/OOM; 4 is enough
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-1}"
# If unset, falls back to data.init_args.batch_size in the YAML; if set, overrides via CLI
BATCH_SIZE="${BATCH_SIZE:-16}"
# GNN k_hop / layer count (gnn_params.num_layers). Empty string keeps the YAML value,
# a number overrides via CLI. For large bipartite graphs (e.g. twitter), NUM_LAYERS=1
# is recommended to avoid over-smoothing.
NUM_LAYERS="${NUM_LAYERS:-}"
# Training wall-clock cap (seconds). Empty = run to completion by epoch; if set, stops
# once the time is up (used for test2 stress testing / equalized budget).
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
# Override AdamW eps (empty = keep YAML's 1e-4; use 1e-8 to fix training plateaus, aligned with DeepSpeed).
EPS="${EPS:-}"
# Override weight_decay (empty = keep YAML's 0.0; use 0.01 to suppress contrastive collapse, aligned with DeepSpeed).
WD="${WD:-}"
# Recompute val-AUC every N seconds and write val_timeline.csv (dense time-to-accuracy sampling; used by test1-2). Empty = disabled.
VAL_EVAL_INTERVAL="${VAL_EVAL_INTERVAL:-}"
# Non-empty = run periodic sampling link prediction on the test split instead (test1-2 uses the test split for an honest speedup comparison). Empty = val.
VAL_EVAL_ON_TEST="${VAL_EVAL_ON_TEST:-}"

CMD=(
  "${PYTHON}" bin/congrat_train.py
  --gnn-types "${GNN_TYPES}"
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
elif [[ -n "${DATASETS}" ]]; then
  CMD+=(--datasets "${DATASETS}")
fi

echo "=========================================="
echo " CLIP-Graph DDP launch"
echo "=========================================="
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " WORLD_SIZE=${WORLD_SIZE}"
echo " MASTER_PORT=${MASTER_PORT}"
echo " CONFIG_PATH=${CONFIG_PATH}"
echo " DATASETS_ALL=${DATASETS_ALL}"
if [[ -n "${DATASETS}" ]]; then
  echo " DATASETS=${DATASETS}"
fi
echo " GNN_TYPES=${GNN_TYPES}"
echo " NUM_EPOCHS=${NUM_EPOCHS}"
echo " LEARNING_RATE=${LEARNING_RATE}"
echo " NUM_WORKERS=${NUM_WORKERS}"
echo " BATCH_SIZE=${BATCH_SIZE:-<from YAML>}"
echo " NUM_LAYERS=${NUM_LAYERS:-<from YAML>}  (= GNN k_hop)"
echo " EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
echo " CHECKPOINT_FREQUENCY=${CHECKPOINT_FREQUENCY}"
echo "------------------------------------------"
echo "[cmd] ${CMD[*]}"
echo

"${CMD[@]}"
