#!/usr/bin/env bash
# =========================================================================
# experiments/04_time_breakdown.sh -- per-iteration time breakdown for the four systems + table output
#
# Four execution paradigms (search space / LM / data split are identical, only the execution
# paradigm differs):
#   fixed_gat_ddp : fixed GAT + LM, synchronous DDP            -> run_clip.sh
#   gnas_ddp      : differentiable GNAS supernet + LM, synchronous DDP    -> run_ddp.sh
#   gnas_zero     : gnas_ddp with ZeRO-Offload added      -> run_deepspeed.sh
#   tagnas        : asynchronous bidirectional push (graph on CPU, LM on GPU)-> run_sys.sh
#
# Add --profile to each system to run a "short" segment (synchronous systems run the full
# PROFILE_EPOCHS epochs with early stopping disabled; asynchronous systems run for
# TAGNAS_MAX_RUNTIME seconds), writing the average per-iteration time for 7 buckets into a
# structured JSON:
#   results/test6/breakdown_<system>_<dataset>.json
# At the end, tools/make_breakdown_table.py is called to summarize into an ASCII table + LaTeX
# booktabs table.
#
# 7 time buckets: SyncWait | XferT | BuildBatch | LMComp | GraphComp | Backward | Update
#   - LMComp/GraphComp = forward pass only; Backward = the single backward pass of the merged
#     two towers; Update = the optimizer step
#   - SyncWait: for DDP/ZeRO = the barrier wait timed after backward and before step; for
#     TAGNAS ~ 0 (asynchronous drop-on-miss)
#
# Usage:
#   First get the full pipeline working on pubmed (default):
#     CUDA_VISIBLE_DEVICES=0,1,2 bash experiments/04_time_breakdown.sh
#   Rerun only one system:
#     SYSTEMS_TEST6="gnas_ddp" CUDA_VISIBLE_DEVICES=0,1 bash experiments/04_time_breakdown.sh
#   Switch dataset (large graph):
#     DATASETS_TEST6="goodreads-crime" bash experiments/04_time_breakdown.sh
#   Only produce the table (no retraining, reads existing JSON):
#     SYSTEMS_TEST6=" " bash experiments/04_time_breakdown.sh
#
# WARNING: requires an idle GPU; run nvidia-smi to check GPUs before starting, and specify a
# free GPU with CUDA_VISIBLE_DEVICES.
# =========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ── GPU pool + NCCL + HF offline (same convention as test1) ────────────────────────────
# IFS=', ' splits on both comma and space, tolerating a spaced-out form like "5, 6, 7" (avoids
# a stray space leaking into a GPU id).
IFS=', ' read -ra GPU_POOL <<< "${CUDA_VISIBLE_DEVICES:-5,6,7}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_POOL[@]}}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > ${#GPU_POOL[@]} )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} is invalid or exceeds the number of visible GPUs ${#GPU_POOL[@]} (${GPU_POOL[*]})" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_POOL[*]:0:WORLD_SIZE}")"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
export LINK_NEG_SAMPLING="${LINK_NEG_SAMPLING:-auto}"

# ── configuration (edit centrally here) ────────────────────────────────────────────────
read -ra SYSTEMS_TEST6  <<< "${SYSTEMS_TEST6:-fixed_gat_ddp gnas_ddp gnas_zero unitg tagnas}"
# Datasets: default is space-separated; defensively tr ',' ' ' first to strip commas, fixing
# the old bug where a trailing comma in "ds," would leak into the filename.
read -ra DATASETS_TEST6 <<< "$(echo "${DATASETS_TEST6:-amazon-baby citation pubmed twitter trex reddit}" | tr ',' ' ')"
PROFILE_EPOCHS="${PROFILE_EPOCHS:-2}"          # number of epochs synchronous systems run (short profiling run, discarding the first iteration is enough to stabilize the mean)
BATCH_SIZE="${BATCH_SIZE:-16}"                 # a single batch size shared across the three synchronous systems (fair comparison)
TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE:-0.5}"
TAGNAS_MAX_RUNTIME="${TAGNAS_MAX_RUNTIME:-600}"   # asynchronous systems run for a fixed duration (seconds), sampling enough iterations
SEED="${SEED:-42}"                             # shared random seed (reproducibility + tags the telemetry JSONL)
# Disable early stopping = patience far greater than the epoch count -> runs the full PROFILE_EPOCHS
_BIG_PATIENCE=$((PROFILE_EPOCHS + 100000))

# rendezvous ports (change these when running multiple instances in parallel)
B1_PORT="${B1_PORT:-29710}"; B2_PORT="${B2_PORT:-29720}"; B3_PORT="${B3_PORT:-29730}"
TAGNAS_DIST_PORT="${TAGNAS_DIST_PORT:-29790}"; TAGNAS_RPC_PORT="${TAGNAS_RPC_PORT:-29791}"
B5_PORT="${B5_PORT:-29740}"
# B5 (UniTG) is bounded by wall clock, not epochs: with the default M=3 / 100k-node microbatches
# every dataset here yields ONE mini-batch per epoch, so PROFILE_EPOCHS=2 would leave a single
# recorded graph iteration after warmup. Shrinking the microbatch to manufacture more iterations
# would change the METIS partition and break comparability with the 01/02 B5 runs, so instead the
# run gets the same fixed duration as TAGNAS.
UNITG_MAX_RUNTIME="${UNITG_MAX_RUNTIME:-${TAGNAS_MAX_RUNTIME}}"
UNITG_NUM_EPOCHS="${UNITG_NUM_EPOCHS:-100000}"
UNITG_NUM_LAYERS="${UNITG_NUM_LAYERS:-112}"

OUT_DIR="${OUT_DIR:-results/test6}"
mkdir -p "${OUT_DIR}"
RUNS_TSV="${OUT_DIR}/_runs.tsv"
printf 'dataset\tsystem\trc\tbreakdown_json\n' > "${RUNS_TSV}"

echo "=========================================="
echo " test6 time-breakdown starting"
echo "=========================================="
echo " Systems  : ${SYSTEMS_TEST6[*]:-<none, summary only>}"
echo " Datasets : ${DATASETS_TEST6[*]}"
echo " GPUs     : WORLD_SIZE=${WORLD_SIZE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo " Profile  : epochs=${PROFILE_EPOCHS} batch=${BATCH_SIZE} text_subsample=${TEXT_SUBSAMPLE} (early stopping disabled)"
echo " TAGNAS   : max_runtime=${TAGNAS_MAX_RUNTIME}s"
echo " Out dir  : ${OUT_DIR}"
echo "=========================================="

# ── fault-tolerant single-system run: capture the exit code + verify the breakdown JSON was produced ──────────
run_one() {
    local ds="$1" sys="$2" bj="$3"; shift 3
    local rc
    echo
    echo "############################################################"
    echo "## [${ds}/${sys}] launching: $*"
    echo "############################################################"
    # Clear old artifacts: the legacy bj + this (sys,ds)'s per-iteration JSONL (avoid mixing across runs)
    rm -f "${bj}"
    rm -f "${OUT_DIR}/telemetry/${sys}_${ds}_"*.jsonl 2>/dev/null || true
    if "$@"; then rc=0; else rc=$?; fi
    # Success criterion: either the legacy bj (synchronous systems) or the telemetry JSONL
    # (TAGNAS dual-lane) existing is enough
    if [[ ! -f "${bj}" ]] && ! compgen -G "${OUT_DIR}/telemetry/${sys}_${ds}_"* > /dev/null; then
        echo "## [${ds}/${sys}] WARN: neither ${bj} nor telemetry/${sys}_${ds}_*.jsonl was generated"
        [[ ${rc} -eq 0 ]] && rc=98
    fi
    printf '%s\t%s\t%s\t%s\n' "${ds}" "${sys}" "${rc}" "${bj}" >> "${RUNS_TSV}"
    echo "## [${ds}/${sys}] rc=${rc}  breakdown=${bj}"
}

for ds in "${DATASETS_TEST6[@]}"; do
    echo
    echo "==================== DATASET: ${ds} ===================="
    for sys in "${SYSTEMS_TEST6[@]}"; do
        bj="${OUT_DIR}/breakdown_${sys}_${ds}.json"
        case "${sys}" in
            fixed_gat_ddp)
                run_one "${ds}" "${sys}" "${bj}" \
                    env DATASETS="${ds}" GNN_TYPES=GAT WORLD_SIZE="${WORLD_SIZE}" SEED="${SEED}" \
                        NUM_EPOCHS="${PROFILE_EPOCHS}" EARLY_STOPPING_PATIENCE="${_BIG_PATIENCE}" \
                        BATCH_SIZE="${BATCH_SIZE}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC=auc \
                        MASTER_PORT="${B1_PORT}" NO_POST_EVAL=1 \
                        PROFILE=1 PROFILE_OUT="${bj}" bash run_clip.sh
                ;;
            gnas_ddp)
                run_one "${ds}" "${sys}" "${bj}" \
                    env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" SEED="${SEED}" \
                        NUM_EPOCHS="${PROFILE_EPOCHS}" EARLY_STOPPING_PATIENCE="${_BIG_PATIENCE}" \
                        BATCH_SIZE="${BATCH_SIZE}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC=auc \
                        MASTER_PORT="${B2_PORT}" NO_POST_EVAL=1 \
                        PROFILE=1 PROFILE_OUT="${bj}" bash run_ddp.sh
                ;;
            gnas_zero)
                run_one "${ds}" "${sys}" "${bj}" \
                    env DATASETS="${ds}" NUM_GPUS="${WORLD_SIZE}" SEED="${SEED}" \
                        NUM_EPOCHS="${PROFILE_EPOCHS}" EARLY_STOP_PATIENCE="${_BIG_PATIENCE}" \
                        BATCH_SIZE="${BATCH_SIZE}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC=auc \
                        MASTER_PORT="${B3_PORT}" NO_POST_EVAL=1 \
                        PROFILE=1 PROFILE_OUT="${bj}" bash run_deepspeed.sh
                ;;
            unitg)
                # B5 uses DATASETS (plural), has no BATCH_SIZE (its text batch is LM_BATCH_SIZE),
                # and ignores TEXT_SUBSAMPLE (record-only there) -- so, unlike B1-B3, it profiles
                # on 100% of the texts; the table caption says so.
                run_one "${ds}" "${sys}" "${bj}" \
                    env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" SEED="${SEED}" \
                        NUM_LAYERS="${UNITG_NUM_LAYERS}" LM_BATCH_SIZE="${BATCH_SIZE}" \
                        NUM_EPOCHS="${UNITG_NUM_EPOCHS}" MAX_RUNTIME="${UNITG_MAX_RUNTIME}" \
                        EARLY_STOPPING_PATIENCE="${_BIG_PATIENCE}" \
                        VAL_EVAL_INTERVAL="${UNITG_MAX_RUNTIME}" VAL_EVAL_MIN_EPOCHS=100000 \
                        MASTER_PORT="${B5_PORT}" NO_POST_EVAL=1 \
                        PROFILE=1 PROFILE_OUT="${bj}" bash run_unitg.sh
                ;;
            tagnas)
                run_one "${ds}" "${sys}" "${bj}" \
                    env DATASET="${ds}" NUM_TEXT_WORKERS="${WORLD_SIZE}" PHASE=search SEED="${SEED}" \
                        SEARCH_MAX_RUNTIME="${TAGNAS_MAX_RUNTIME}" MAX_RUNTIME="${TAGNAS_MAX_RUNTIME}" \
                        DIST_PORT="${TAGNAS_DIST_PORT}" RPC_PORT="${TAGNAS_RPC_PORT}" \
                        PROFILE=1 PROFILE_OUT="${bj}" bash run_sys.sh
                ;;
            *)
                echo "[WARN] unknown system '${sys}', skipping (options: fixed_gat_ddp gnas_ddp gnas_zero unitg tagnas)"
                ;;
        esac
    done
done

# ── summarize into tables (dual-lane ASCII + LaTeX + CSV) + overlap figure ─────────────────────
echo
echo "=========================================="
echo " summarizing -> ${OUT_DIR}/breakdown_table.{txt,tex} + breakdown.csv + overlap_*.png"
echo "=========================================="
python3 tools/make_breakdown_table.py \
    --in-dir "${OUT_DIR}" \
    --telemetry-dir "${OUT_DIR}/telemetry" \
    --out-ascii "${OUT_DIR}/breakdown_table.txt" \
    --out-tex "${OUT_DIR}/breakdown_table.tex" \
    --out-csv "${OUT_DIR}/breakdown.csv" \
    --check || echo "[warn] --check did not fully pass, see the verification output above"

# dual-lane overlapping timeline figure (only produced when tagnas has both graph/text JSONL)
python3 tools/make_overlap_figure.py --in-dir "${OUT_DIR}" --telemetry-dir "${OUT_DIR}/telemetry" \
    || echo "[warn] overlap figure was not generated (tagnas telemetry may be missing)"

echo
echo "[done] test6 complete. Artifacts:"
echo "  ${OUT_DIR}/telemetry/<sys>_<ds>_seed<seed>_<worker>.jsonl  (raw per-iteration x per-worker telemetry)"
echo "  ${OUT_DIR}/breakdown.csv                        (dual-lane + effective rows, machine-readable)"
echo "  ${OUT_DIR}/breakdown_table.txt                  (ASCII dual-lane table)"
echo "  ${OUT_DIR}/breakdown_table.tex                  (LaTeX booktabs, ready to paste into the paper)"
echo "  ${OUT_DIR}/overlap_<dataset>.png               (dual-lane overlapping timeline)"
echo "  ${OUT_DIR}/_runs.tsv                            (manifest)"
