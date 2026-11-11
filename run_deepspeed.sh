#!/usr/bin/env bash
# =========================================================================
# TAGNAS Baseline 4 (B4): ZeRO-Offload GNAS launcher
#
# Uses the deepspeed CLI to launch bin/zero_offload_train.py:
#   - PyTorch DDP -> DeepSpeed ZeRO Stage 2 + CPU offload_optimizer
#   - Forces FP32 (fp16/bf16 both disabled in configs/deepspeed/ds_config_b4_offload.json)
#   - rank-0 prints average time for three phases (LM / GNN / Backward+Step) + Peak GPU Memory
#
# Multi-dataset sweeps are driven by a shell for-loop; each dataset spawns its own
# independent deepspeed process group.
# =========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active) -- aligned with run_clip.sh/run_ddp.sh, so that
# `bash run_deepspeed.sh` works whether or not the caller pre-activated it.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ──────────────────────────────────────────────────────────────────────
# HuggingFace network settings
# ──────────────────────────────────────────────────────────────────────
# HF endpoint (used if a network download is ever needed).
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# Offline mode: skip all Hub network calls once the model/tokenizer are cached locally.
# Defaults to true -- newer transformers versions still call list_repo_templates to
# fetch the chat template from the Hub even when the model is cached, which can hang
# behind restrictive network setups. Offline mode reads straight from the local cache
# and sidesteps that call. Set HF_OFFLINE=false if the model isn't cached yet.
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# ──────────────────────────────────────────────────────────────────────
# Tunable parameters (override via environment variables: NUM_GPUS=2 DATASETS=pubmed bash run_deepspeed.sh)
# ──────────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-2}"
# Warning: DeepSpeed's --num_gpus=N ignores the specific card indices in
#    CUDA_VISIBLE_DEVICES and forces use of the first N physical cards (0..N-1),
#    which on a shared machine can collide with cards other people are using (and
#    wastes the cards you actually specified).
#    Fix: take the first NUM_GPUS physical card indices from CUDA_VISIBLE_DEVICES and
#    pass them explicitly via deepspeed --include=localhost:<indices>; then unset CVD
#    so --include takes effect using physical indices.
_CVD_IN="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra _GPU_POOL <<< "${_CVD_IN}"
if (( NUM_GPUS > ${#_GPU_POOL[@]} )); then
    echo "[ERROR] NUM_GPUS=${NUM_GPUS} exceeds the number of cards listed in CUDA_VISIBLE_DEVICES (${#_GPU_POOL[@]}: ${_CVD_IN})." >&2
    exit 1
fi
INCLUDE_GPUS="$(IFS=,; echo "${_GPU_POOL[*]:0:NUM_GPUS}")"   # e.g. 0,1,4
unset CUDA_VISIBLE_DEVICES

# Single dataset / multiple datasets (comma-separated) / full discovery: DATASETS_ALL=true
DATASETS="${DATASETS:-pubmed}"
DATASETS_ALL="${DATASETS_ALL:-false}"

NUM_EPOCHS="${NUM_EPOCHS:-30}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
CKPT_FREQ="${CKPT_FREQ:-1}"
# GNN k_hop / layer count (gnn_params.num_layers). Empty string keeps the YAML value,
# a number overrides via CLI. For large bipartite graphs (e.g. twitter), NUM_LAYERS=1
# is recommended to avoid over-smoothing.
NUM_LAYERS="${NUM_LAYERS:-}"
# Data batch_size (overrides data.init_args.batch_size in the YAML).
# Warning: GASSO contrastive requires >= 2; at 1 the cross-entropy is always 0 and
# training doesn't work at all.
# Default 4. Empty string keeps the YAML value (not recommended).
BATCH_SIZE="${BATCH_SIZE:-16}"
# Training wall-clock cap (seconds). Empty = run to completion by epoch; if set, stops
# once the time is up (used for test3 equalized budget).
MAX_RUNTIME="${MAX_RUNTIME:-}"
# Use only this fraction (0,1] of text per epoch for alignment (re-sampled randomly
# each epoch). Empty = use all text.
TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE:-}"
# Convergence/early-stop monitoring metric: auc = val link-prediction AUC stops
# improving; loss = val loss stops decreasing. Empty = use the python default (auc).
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-}"
# Non-empty = skip the post-training link-prediction post-eval (not needed for
# throughput stress tests, and ZeRO-3 sharded checkpoints can't be evaluated directly).
NO_POST_EVAL="${NO_POST_EVAL:-}"
# Resume training: reuse an existing run_dir and continue from its checkpoint
# (metrics/log are appended, directory is not recreated). Empty = normal fresh training.
#   RESUME_RUN_DIR=model/runpytorch/gasso_<ds>_b4_zero_<ts>  RESUME_TAG=epoch_23 (auto-picks latest if unset)
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_TAG="${RESUME_TAG:-}"
# time-breakdown profiling: non-empty PROFILE adds --profile; PROFILE_OUT sets the JSON output path.
PROFILE="${PROFILE:-}"
PROFILE_OUT="${PROFILE_OUT:-}"
# Recompute val-AUC every N seconds and write val_timeline.csv (dense time-to-accuracy sampling; used by test1-2). Empty = disabled.
# Note: AUC isn't computed under ZeRO-3 (parameters are sharded); this switch only produces data at stage<3.
VAL_EVAL_INTERVAL="${VAL_EVAL_INTERVAL:-}"
# Non-empty = run periodic sampling link prediction on the test split instead (test1-2 uses the test split for an honest speedup comparison). Empty = val.
VAL_EVAL_ON_TEST="${VAL_EVAL_ON_TEST:-}"

CONFIG_PATH="${CONFIG_PATH:-./configs/clip-graph/inductive-causal/pubmed/gassobase.yaml}"
DS_CONFIG="${DS_CONFIG:-./configs/deepspeed/ds_config_b4_offload.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-model/runpytorch}"

# DeepSpeed launcher port (change this to avoid conflicts when running multiple instances in parallel)
MASTER_PORT="${MASTER_PORT:-29500}"
# One ID is inherited by every DeepSpeed rank; each interpreter has its own clock.
export TAGNAS_RUN_ID="${TAGNAS_RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$}"

# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
if [[ ! -f "${DS_CONFIG}" ]]; then
    echo "[ERROR] DeepSpeed config file does not exist: ${DS_CONFIG}"
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Model YAML config does not exist: ${CONFIG_PATH}"
    exit 1
fi
if ! python -c "import deepspeed" 2>/dev/null; then
    echo "[ERROR] deepspeed is not installed. First run: uv add deepspeed   # or pip install deepspeed"
    exit 1
fi

# Confirm fp16/bf16 are both false (B4 forces FP32)
fp16_enabled=$(python -c "import json; print(json.load(open('${DS_CONFIG}')).get('fp16', {}).get('enabled', False))")
bf16_enabled=$(python -c "import json; print(json.load(open('${DS_CONFIG}')).get('bf16', {}).get('enabled', False))")
if [[ "${fp16_enabled}" != "False" || "${bf16_enabled}" != "False" ]]; then
    echo "[WARN] fp16/bf16 in ds_config are not false (fp16=${fp16_enabled}, bf16=${bf16_enabled})"
    echo "       The B4 baseline requires strict FP32; please confirm this is intentional."
fi

# ──────────────────────────────────────────────────────────────────────
# Assemble the deepspeed command
# ──────────────────────────────────────────────────────────────────────
CMD=(
    deepspeed
    --include="localhost:${INCLUDE_GPUS}"
    --master_port="${MASTER_PORT}"
    bin/zero_offload_train.py
    --num-epochs "${NUM_EPOCHS}"
    --learning-rate "${LR}"
    --num-workers "${NUM_WORKERS}"
    --early-stopping-patience "${EARLY_STOP_PATIENCE}"
    --checkpoint-frequency "${CKPT_FREQ}"
    --config-path "${CONFIG_PATH}"
    --deepspeed_config "${DS_CONFIG}"
    --output-root "${OUTPUT_ROOT}"
)

if [[ "${DATASETS_ALL}" == "true" ]]; then
    CMD+=(--datasets-all)
else
    CMD+=(--datasets "${DATASETS}")
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

if [[ -n "${NO_POST_EVAL}" ]]; then
    CMD+=(--no-post-eval)
fi

if [[ -n "${BATCH_SIZE}" ]]; then
    CMD+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${RESUME_RUN_DIR}" ]]; then
    CMD+=(--resume-run-dir "${RESUME_RUN_DIR}")
fi

if [[ -n "${RESUME_TAG}" ]]; then
    CMD+=(--resume-tag "${RESUME_TAG}")
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
if [[ -n "${VAL_EVAL_INTERVAL}" ]]; then
    CMD+=(--val-eval-interval-seconds "${VAL_EVAL_INTERVAL}")
fi
[[ -n "${VAL_EVAL_ON_TEST}" ]] && CMD+=(--val-eval-on-test)

# ──────────────────────────────────────────────────────────────────────
# Print startup info and run
# ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " TAGNAS B4 (ZeRO-Offload GNAS) launch"
echo "=========================================="
echo " GPUs         : ${NUM_GPUS}  (deepspeed --include=localhost:${INCLUDE_GPUS}; physical cards ${INCLUDE_GPUS})"
echo " HF offline   : ${HF_OFFLINE}  (HF_ENDPOINT=${HF_ENDPOINT})"
echo " Datasets     : ${DATASETS}  (all=${DATASETS_ALL})"
echo " Epochs       : ${NUM_EPOCHS}  LR=${LR}"
echo " Batch size   : ${BATCH_SIZE:-<from YAML>}  (GASSO contrastive requires >= 2)"
echo " Num layers   : ${NUM_LAYERS:-<from YAML>}  (= GNN k_hop)"
echo " Config YAML  : ${CONFIG_PATH}"
echo " DS config    : ${DS_CONFIG}"
echo " Output root  : ${OUTPUT_ROOT}"
echo " Master port  : ${MASTER_PORT}"
echo "=========================================="
echo "[cmd] ${CMD[*]}"
echo

"${CMD[@]}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[ERROR] DeepSpeed B4 training exited abnormally (exit code ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

echo
echo "[done] B4 training complete, artifacts in ${OUTPUT_ROOT}/gasso_*_b4_zero_*/"
echo "  - metrics.csv: includes lm_phase_ms / gnn_phase_ms / bw_step_phase_ms / peak_gpu_mem_gb"
echo "  - checkpoints/<tag>/: native DeepSpeed checkpoints (for Stage 3, use zero_to_fp32.py to recover FP32 weights)"
echo "  - best_model.pth: rank 0's FP32 snapshot at the end of the best epoch (complete for Stage 2, reference only for Stage 3)"
