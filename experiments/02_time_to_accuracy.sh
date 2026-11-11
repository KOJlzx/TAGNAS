#!/usr/bin/env bash
# =========================================================================
# test1-2: honest speedup -- iso-quality (time-to-accuracy) + iso-time
#
# Replaces the "TAGNAS 17.7min vs B2 12h = 42x" claim (different stopping criteria, not
# apples-to-apples) with two honest measures, reconstructed from existing training logs into
# 1 curve chart + 2 tables (iso-quality / iso-time).
#
# Two run modes:
#   (default, zero GPU)  analysis only, on existing runs -> charts/tables in seconds, no GPU
#       usage, doesn't compete with in-flight experiments for GPUs.
#       bash experiments/02_time_to_accuracy.sh
#   (optional, needs GPU)  first retrain a "finely sampled" TAGNAS amazon-baby run (only the
#       validation-sampling frequency is tuned; the training config is otherwise identical to
#       the headline run) to get a smoother early curve, then use it to override the TAGNAS
#       row when plotting:
#       RERUN_TAGNAS=1 CUDA_VISIBLE_DEVICES=0,2,7 bash experiments/02_time_to_accuracy.sh
#
# Artifacts land in results/test1-2/<dataset>/<sampling-signature>/: time_to_accuracy.{png,pdf}
#   + iso_quality.{md,tex} + iso_time.{md,tex} + series.csv + runs_used.json + summary.txt;
#   the top-level index.tsv is appended with an index entry.
#   Different dataset / sampling-frequency combos each get their own subdirectory and never
#   overwrite each other (customize the subdirectory name with TAG=...).
# =========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active) -- consistent with the other run_*.sh scripts
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ──────────────────────────────────────────────────────────────────────
# Centralized configuration (overridable via env vars)
# ──────────────────────────────────────────────────────────────────────
DATASET="${DATASET:-pubmed,trex}"
OUT_DIR="${OUT_DIR:-results/test1-2}"
BASELINE="${BASELINE:-B2}"                                  # primary speedup baseline
# 1=periodically sample link prediction on the test split (the honest speedup uses the test
# set for B1/B2/B3/TAGNAS uniformly); empty=fall back to val.
EVAL_ON_TEST="${EVAL_ON_TEST:-1}"
TARGETS="${TARGETS:-0.60 0.70 0.78 0.82 0.86}"             # iso-quality AUC targets (3-layer convergence criterion, raised)
BUDGETS="${BUDGETS:-17.7 60 180 360 540 720}"              # iso-time budgets (minutes)
# Output subdirectory name. Empty = auto-generate based on each run's sampling method (e.g.
# sync_epoch__tagnas_100s); artifacts land in OUT_DIR/<dataset>/<TAG>/ -- different
# dataset/sampling-frequency combos each get their own subdirectory and never overwrite each other.
TAG="${TAG:-}"
PYTHON="${PYTHON:-python3}"

# Optional retrain: non-empty = first retrain a finely sampled TAGNAS run (requires GPU)
RERUN_TAGNAS="${RERUN_TAGNAS:-1}"
# GPUs used for the retrain (num_text_workers=3 -> needs 3; matches the headline run)
RERUN_GPUS="${CUDA_VISIBLE_DEVICES:-1,2,3}"
# The retrain can only run a single dataset (run_sys.sh does not split on commas). DATASET may
# be a comma-separated list of multiple datasets -> take the first one; the plotting stage
# still uses the full DATASET. Use RERUN_DATASET to explicitly pick which one to retrain.
RERUN_DATASET="${RERUN_DATASET:-${DATASET%%,*}}"
# Fine sampling: shrink the validation interval and the first-validation threshold to make the
# early curve denser (all other training config = the headline run)
RERUN_VAL_INTERVAL="${RERUN_VAL_INTERVAL:-50}"
RERUN_VAL_MIN_UPDATES="${RERUN_VAL_MIN_UPDATES:-3}"
RERUN_MAX_RUNTIME="${RERUN_MAX_RUNTIME:-1500}"
RERUN_DIST_PORT="${RERUN_DIST_PORT:-29592}"
RERUN_RPC_PORT="${RERUN_RPC_PORT:-29593}"

# ── fresh retest of B1/B2/B3 (retest all of them; don't reuse old test1 records) ──
# Fixed epoch budget + sample val-AUC every BL_VAL_INTERVAL seconds -> val_timeline.csv
# (the time-to-accuracy curve).
RERUN_BASELINES="${RERUN_BASELINES:-1}"
BL_GPUS="${BL_GPUS:-${CUDA_VISIBLE_DEVICES:-1,2,3}}"   # B1/B2 are DDP, must use the Default GPUs (exclusive 5,6,7 will hang)
BL_WORLD="${BL_WORLD:-3}"                 # B1/B2 WORLD_SIZE / B3 NUM_GPUS
BL_EPOCHS="${BL_EPOCHS:-50}"              # epoch budget (3-layer convergence criterion; paired with early stopping to stop at convergence)
BL_VAL_INTERVAL="${BL_VAL_INTERVAL:-30}"  # sample val-AUC every this many seconds
BL_BATCH="${BL_BATCH:-16}"
BL_PORT_B1="${BL_PORT_B1:-29611}"
BL_PORT_B2="${BL_PORT_B2:-29621}"
BL_PORT_B3="${BL_PORT_B3:-29631}"
BL_PORT_B5="${BL_PORT_B5:-29641}"

mkdir -p "${OUT_DIR}"
RUNS_JSON=""   # leave empty = use the 4 built-in default runs from tools/make_time_to_accuracy.py
B1_RD="NONE"; B2_RD="NONE"; B3_RD="NONE"; B5_RD="NONE"; TAGNAS_RD="NONE"

# After a baseline finishes, capture the run directory it just created (newer than marker, and
# containing metrics.csv); NONE on failure
_capture_newest() {  # marker  glob
  local marker="$1" glob="$2" newest
  newest="$(ls -dt ${glob} 2>/dev/null | head -1 || true)"; newest="${newest%/}"
  if [[ -n "${newest}" && "${newest}" -nt "${marker}" && -f "${newest}/metrics.csv" ]]; then
    echo "${newest}"
  else
    echo "NONE"
  fi
}

# ──────────────────────────────────────────────────────────────────────
# fresh retest of B1/B2/B3 (sequentially on BL_GPUS; fixed epoch + sample val-AUC every
# BL_VAL_INTERVAL seconds)
# ──────────────────────────────────────────────────────────────────────
if [[ -n "${RERUN_BASELINES}" ]]; then
    echo "=========================================="
    echo " [rerun] fresh B1/B2/B3/B5  ds=${RERUN_DATASET}  GPU=${BL_GPUS}  ${BL_EPOCHS}ep  val every ${BL_VAL_INTERVAL}s"
    echo "=========================================="
    export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    # Shared env (consistent across B1/B2/B3; 3 layers = k_hop3 to reproduce the test1
    # criterion; early stopping on val-loss to stop at convergence, avoiding test-AUC early
    # stopping = selecting on the test set)
    _BLC=(DATASETS="${RERUN_DATASET}" NUM_LAYERS="${BL_NUM_LAYERS:-3}" BATCH_SIZE="${BL_BATCH}"
          NUM_EPOCHS="${BL_EPOCHS}" VAL_EVAL_INTERVAL="${BL_VAL_INTERVAL}" VAL_EVAL_ON_TEST="${EVAL_ON_TEST}"
          EARLY_STOPPING_PATIENCE="${BL_PATIENCE:-8}" EARLY_STOP_METRIC="${BL_STOP_METRIC:-loss}"
          CUDA_VISIBLE_DEVICES="${BL_GPUS}")
    MK="$(mktemp)"
    env "${_BLC[@]}" GNN_TYPES=GAT WORLD_SIZE="${BL_WORLD}" MASTER_PORT="${BL_PORT_B1}" bash run_clip.sh
    B1_RD="$(_capture_newest "${MK}" "model/runpytorch/${RERUN_DATASET}_GAT_ddp_*/")"; rm -f "${MK}"
    MK="$(mktemp)"
    env "${_BLC[@]}" WORLD_SIZE="${BL_WORLD}" MASTER_PORT="${BL_PORT_B2}" bash run_ddp.sh
    B2_RD="$(_capture_newest "${MK}" "model/runpytorch/gasso_${RERUN_DATASET}_ddp_*/")"; rm -f "${MK}"
    MK="$(mktemp)"
    env "${_BLC[@]}" NUM_GPUS="${BL_WORLD}" MASTER_PORT="${BL_PORT_B3}" bash run_deepspeed.sh
    B3_RD="$(_capture_newest "${MK}" "model/runpytorch/gasso_${RERUN_DATASET}_b4_zero_*/")"; rm -f "${MK}"
    MK="$(mktemp)"
    # B5 takes the same budget/sampling knobs but not BATCH_SIZE/GNN_TYPES (its text batch is
    # LM_BATCH_SIZE and its architecture is fixed), and NUM_LAYERS stays at the UniTG default --
    # forcing the 3-layer baseline depth on it would not be the system the paper describes.
    env DATASETS="${RERUN_DATASET}" NUM_EPOCHS="${BL_EPOCHS}" VAL_EVAL_INTERVAL="${BL_VAL_INTERVAL}" \
        VAL_EVAL_ON_TEST="${EVAL_ON_TEST}" EARLY_STOPPING_PATIENCE="${BL_PATIENCE:-8}" \
        LM_BATCH_SIZE="${BL_BATCH}" CUDA_VISIBLE_DEVICES="${BL_GPUS}" \
        WORLD_SIZE="${BL_WORLD}" MASTER_PORT="${BL_PORT_B5}" bash run_unitg.sh
    B5_RD="$(_capture_newest "${MK}" "model/runpytorch/unitg_${RERUN_DATASET}_*/")"; rm -f "${MK}"
    echo "[rerun] B1_RD=${B1_RD}"
    echo "[rerun] B2_RD=${B2_RD}"
    echo "[rerun] B3_RD=${B3_RD}"
    echo "[rerun] B5_RD=${B5_RD}"
fi

# ──────────────────────────────────────────────────────────────────────
# Optional: retrain a finely sampled TAGNAS run (faithfully reproduces the headline run's
# training config, only tightening the validation sampling)
# ──────────────────────────────────────────────────────────────────────
if [[ -n "${RERUN_TAGNAS}" ]]; then
    echo "=========================================="
    echo " [rerun] finely sampled TAGNAS ${RERUN_DATASET} (GPU=${RERUN_GPUS})"
    echo "   only tightening val_eval_interval=${RERUN_VAL_INTERVAL}s / min_updates=${RERUN_VAL_MIN_UPDATES};"
    echo "   graph_batch/text_chunk/k_hop/num_layers/cache/train_fraction match the headline run"
    echo "=========================================="

    export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1

    MARKER="$(mktemp)"
    set +e
    env \
      CUDA_VISIBLE_DEVICES="${RERUN_GPUS}" \
      PHASE=search \
      DATASET="${RERUN_DATASET}" \
      CONFIG_PATH="configs/clip-graph/inductive-causal/pubmed/gassobase.yaml" \
      GRAPH_BATCH_SIZE=1024 \
      TEXT_CHUNK_SIZE=64 \
      GRAPH_K_HOP="${RERUN_K_HOP:-3}" \
      NUM_LAYERS="${RERUN_NUM_LAYERS:-3}" \
      NUM_TEXT_WORKERS=3 \
      GRAPH_CACHE_CAPACITY=600000 \
      TEXT_CACHE_CAPACITY=600000 \
      TRAIN_NODE_FRACTION="${RERUN_FRACTION:-1.0}" \
      SEARCH_EPOCHS="${RERUN_SEARCH_EPOCHS:-100000}" \
      GRAPH_EPOCHS="${RERUN_SEARCH_EPOCHS:-100000}" \
      TEXT_EPOCHS="${RERUN_SEARCH_EPOCHS:-100000}" \
      VAL_EVAL_INTERVAL="${RERUN_VAL_INTERVAL}" \
      VAL_EVAL_ON_TEST="${EVAL_ON_TEST}" \
      VAL_EVAL_MIN_UPDATES="${RERUN_VAL_MIN_UPDATES}" \
      VAL_PATIENCE="${RERUN_VAL_PATIENCE:-5}" \
      VAL_EARLY_STOP="${RERUN_VAL_EARLY_STOP:-true}" \
      MAX_RUNTIME="${RERUN_MAX_RUNTIME}" \
      SEARCH_MAX_RUNTIME="${RERUN_MAX_RUNTIME}" \
      LINK_NEG_SAMPLING=auto \
      DIST_PORT="${RERUN_DIST_PORT}" \
      RPC_PORT="${RERUN_RPC_PORT}" \
      ASYNC_WORKER_LOG_DIR="${OUT_DIR:-results/test1-2}/tagnas_ranks" \
      bash run_sys.sh
    RERUN_RC=$?
    set +e

    # Find the run directory produced this time (newer than marker)
    NEWEST="$(ls -dt model/distributed/run-*/ 2>/dev/null | head -1 || true)"
    NEWEST="${NEWEST%/}"
    if [[ -n "${NEWEST}" && "${NEWEST}" -nt "${MARKER}" && -f "${NEWEST}/search/manifest.json" ]]; then
        echo "[rerun] new TAGNAS run: ${NEWEST}"
        TAGNAS_RD="${NEWEST}"
    else
        echo "[rerun][WARN] no valid new TAGNAS run found (rc=${RERUN_RC}); falling back to the built-in default TAGNAS run."
        TAGNAS_RD="NONE"
    fi
    rm -f "${MARKER}"
fi

# ──────────────────────────────────────────────────────────────────────
# Assemble the --runs override for all 5 systems (any missing ones fall back to
# make_time_to_accuracy's old DEFAULT_RUNS)
# ──────────────────────────────────────────────────────────────────────
if [[ "${B1_RD}${B2_RD}${B3_RD}${B5_RD}${TAGNAS_RD}" != "NONENONENONENONENONE" ]]; then
    RUNS_JSON="${OUT_DIR}/runs_override.json"
    "${PYTHON}" - "${RUNS_JSON}" "${B1_RD}" "${B2_RD}" "${B3_RD}" "${B5_RD}" "${TAGNAS_RD}" <<'PY'
import json, os, sys
out, b1, b2, b3, b5, tg = sys.argv[1:7]
runs = {}
# B5 is a "sync" source like B1-B3: it writes metrics.csv (epoch_seconds/val_auc) and the
# preferred val_timeline.csv with the same header, so load_sync reads it unchanged.
for name, rd in (("B1", b1), ("B2", b2), ("B3", b3), ("B5", b5)):
    if rd and rd != "NONE" and os.path.isfile(os.path.join(rd, "metrics.csv")):
        runs[name] = {"kind": "sync", "path": os.path.join(rd, "metrics.csv")}
if tg and tg != "NONE" and os.path.isfile(os.path.join(tg, "search", "manifest.json")):
    runs["TAGNAS"] = {"kind": "tagnas", "path": os.path.join(tg, "search", "manifest.json")}
json.dump(runs, open(out, "w"), indent=2)
print(f"[override] {len(runs)}/5 fresh -> {out}: {list(runs)}")
miss = [s for s in ("B1", "B2", "B3", "B5", "TAGNAS") if s not in runs]
if miss:
    print(f"[override][WARN] these systems have no fresh run and will fall back to the old DEFAULT_RUNS: {miss}")
PY
fi

# ──────────────────────────────────────────────────────────────────────
# Plot the analysis charts/tables (default path; zero GPU)
# ──────────────────────────────────────────────────────────────────────
CMD=(
  "${PYTHON}" tools/make_time_to_accuracy.py
  --out "${OUT_DIR}"
  --dataset "${DATASET}"
  --baseline "${BASELINE}"
  --targets "${TARGETS}"
  --budgets "${BUDGETS}"
)
if [[ -n "${RUNS_JSON}" ]]; then
    CMD+=(--runs "${RUNS_JSON}")
fi
if [[ -n "${TAG}" ]]; then
    CMD+=(--tag "${TAG}")
fi

echo "=========================================="
echo " test1-2 producing charts/tables"
echo "=========================================="
echo " DATASET=${DATASET}  OUT_DIR=${OUT_DIR}  BASELINE=${BASELINE}"
echo " TAG=${TAG:-<auto sampling signature>}  -> artifacts land in ${OUT_DIR}/${DATASET}/<TAG>/"
echo " TARGETS=${TARGETS}"
echo " BUDGETS(min)=${BUDGETS}"
echo " RUNS_JSON=${RUNS_JSON:-<built-in defaults>}"
echo "------------------------------------------"
echo "[cmd] ${CMD[*]}"
echo

"${CMD[@]}"
