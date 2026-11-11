#!/usr/bin/env bash
# =========================================================================
# experiments/06_ablation.sh -- system ablation study (the individual speedup contribution of asynchronous bidirectional Push / online pruning)
#
# Leave-one-out: remove one core component at a time from Full, fix the workload
# (SEARCH_EPOCHS) and remove the runtime cap, measure how "total search time" changes (AUC
# should stay similar). Only runs the async system's (run_sys.sh) 3 configurations:
#   FULL    : online pruning ON  + asynchronous Push        (baseline H, 0%)
#   NOPRUNE : online pruning OFF + asynchronous Push        (w/o online pruning -> +XX%)
#   NOSYNC  : online pruning ON  + synchronous Push (degraded)  (w/o asynchronous Push -> +XXX%)
#
# Metrics: Test AUC (the discrete searched architecture, greedy), total search time (Hours),
# time-increase percentage (vs FULL).
#
# WARNING: test5 measures "time" -> fixed epoch, no cap (use a large SAFETY_MAX to guard
# against runaway runs), letting each configuration finish naturally.
#
# Usage:
#   bash experiments/06_ablation.sh                                              # default citation goodreads-crime, 2000 graph steps (takes a long time)
#   DATASETS_TEST5=pubmed GRAPH_STEPS_T5=200 SAFETY_MAX=1200 bash experiments/06_ablation.sh   # validate first (fewer steps)
#   CUDA_VISIBLE_DEVICES=2,3 bash experiments/06_ablation.sh                     # specify GPUs
# =========================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Available GPU pool (can be set manually: CUDA_VISIBLE_DEVICES=2,3 bash experiments/06_ablation.sh). Default 0,1.
# WORLD_SIZE = TAGNAS's NUM_TEXT_WORKERS (the graph worker runs separately on CPU). Default =
# number of GPUs in the pool;
#   can be explicitly overridden: WORLD_SIZE=2 bash experiments/06_ablation.sh. Note that the
#   NOSYNC (synchronous degradation) row forces run_sys.sh to clamp to 1 worker.
IFS=',' read -ra GPU_POOL <<< "${CUDA_VISIBLE_DEVICES:-5, 6, 7}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_POOL[@]}}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > ${#GPU_POOL[@]} )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} is invalid or exceeds the number of visible GPUs ${#GPU_POOL[@]}" \
         "(CUDA_VISIBLE_DEVICES=${GPU_POOL[*]}). Please list enough GPUs in CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
# Take the first WORLD_SIZE GPUs from the pool as the ones actually used (visible GPU count = used GPU count).
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_POOL[*]:0:WORLD_SIZE}")"

# HuggingFace mirror + offline mode (models are already cached -> skip Hub network calls, avoiding the
# list_repo_templates tree API, which the mirror doesn't support and causes ConnectTimeout). Inherited by the child run_*.sh processes.
# Multi-GPU NCCL: on this machine, GPU groups that cross a PCIe bridge (e.g. 0,1,2 / 0,1,4) can
# deadlock on P2P/IB communication during the first ALLREDUCE (processes spin and hang at
# epoch 0 batch 0). We uniformly disable P2P/IB to take the safe path (no effect on single-GPU
# runs). Can be overridden via env: NCCL_P2P_DISABLE=0 bash testN.sh
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Dataset list: comma- or space-separated both work. Commas are normalized to spaces before
# tokenizing, because otherwise read -ra splitting on whitespace would leave a trailing comma
# in the name (e.g. amazon-baby,), producing a broken path
# data/amazon-baby,/raw/graph.csv -> FileNotFoundError.
read -ra DATASETS_TEST5 <<< "${DATASETS_TEST5:-amazon-baby citation pubmed twitter trex reddit}"
DATASETS_TEST5=("${DATASETS_TEST5[@]//,/}")
# Fixed "workload" = number of successful graph-side update steps (= architecture-search
# steps, one weight/arch update per step).
# All three configurations do the same number of graph updates, comparing their wall-time.
# This is a fair search-work unit (epochs don't work: the synchronous mode is forced to change
# batch/worker, so "the same epochs" != the same amount of computation). Set small for
# validation on pubmed.
GRAPH_STEPS_T5="${GRAPH_STEPS_T5:-2000}"
SAFETY_MAX="${SAFETY_MAX:-86400}"             # runtime safety cap (seconds, default 24h): guards against runaway runs; normally it will end at the graph-step count first, without being truncated
EPOCH_CAP="${EPOCH_CAP:-100000}"             # epoch set very large -> doesn't let epoch stop it early, capped instead by the graph-step count
OUT_DIR="${OUT_DIR:-results/test5}"

mkdir -p "${OUT_DIR}"
RUNS_TSV="${OUT_DIR}/_runs.tsv"
printf 'dataset\ttag\trc\trun_dir\n' > "${RUNS_TSV}"

CURRENT_TRAIN_PID=""

terminate_group() {
    local pid="${1:-}"
    [[ -z "${pid}" ]] && return
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

on_interrupt() {
    trap - INT TERM
    echo
    echo "[interrupt] received an interrupt signal, cleaning up the current training process group..."
    terminate_group "${CURRENT_TRAIN_PID}"
    exit 130
}
trap on_interrupt INT TERM

# run_cell <ds> <tag> -- <cmd...>
run_cell() {
    local ds="$1" tag="$2"; shift 2
    local marker rc newest run_dir
    marker="$(mktemp)"
    echo
    echo "############################################################"
    echo "## [${ds}/${tag}] launching: $*"
    echo "############################################################"
    setsid "$@" &
    CURRENT_TRAIN_PID=$!
    wait "${CURRENT_TRAIN_PID}"; rc=$?
    CURRENT_TRAIN_PID=""

    newest="$(ls -dt model/distributed/run-* 2>/dev/null | head -1 || true)"
    newest="${newest%/}"
    if [[ -n "${newest}" && "${newest}" -nt "${marker}" ]]; then
        run_dir="${newest}"
    else
        run_dir="NONE"; [[ ${rc} -eq 0 ]] && rc=99
    fi
    rm -f "${marker}"
    printf '%s\t%s\t%s\t%s\n' "${ds}" "${tag}" "${rc}" "${run_dir}" >> "${RUNS_TSV}"
    echo "## [${ds}/${tag}] rc=${rc}  run_dir=${run_dir}"
}

echo "=========================================="
echo " test5 system ablation starting"
echo " Datasets   : ${DATASETS_TEST5[*]}"
echo " Workload   : MAX_GRAPH_STEPS=${GRAPH_STEPS_T5} (a fixed number of graph search steps; measures how long each configuration takes to complete it)"
echo " Safety cap : ${SAFETY_MAX}s (guards against runaway runs; normally ends at the graph-step count first)"
echo " GPUs       : WORLD_SIZE=${WORLD_SIZE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})   Out: ${OUT_DIR}"
echo "=========================================="

# Shared env: use "successful graph-side update steps" MAX_GRAPH_STEPS as the fair workload;
# epoch is set very large so it doesn't stop early; runtime is given a large safety cap to
# guard against runaway runs. This way all three configurations do the same number of graph
# updates, comparing wall-time alone.
common_env=(
    PHASE=search
    NUM_TEXT_WORKERS="${WORLD_SIZE}"
    SEARCH_EPOCHS="${EPOCH_CAP}"
    GRAPH_EPOCHS="${EPOCH_CAP}"
    TEXT_EPOCHS="${EPOCH_CAP}"
    MAX_GRAPH_STEPS="${GRAPH_STEPS_T5}"
    SEARCH_MAX_RUNTIME="${SAFETY_MAX}"
    MAX_RUNTIME="${SAFETY_MAX}"
)

for ds in "${DATASETS_TEST5[@]}"; do
    echo
    echo "==================== DATASET: ${ds} ===================="

    # FULL: online pruning ON + asynchronous Push
    run_cell "${ds}" FULL \
        env DATASET="${ds}" "${common_env[@]}" \
            USE_ADAPTIVE_PRUNING=true SYNC_PUSH=false bash run_sys.sh

    # NOPRUNE: w/o online pruning (pruning OFF), asynchronous Push
    run_cell "${ds}" NOPRUNE \
        env DATASET="${ds}" "${common_env[@]}" \
            USE_ADAPTIVE_PRUNING=false SYNC_PUSH=false bash run_sys.sh

    # NOSYNC: w/o asynchronous Push (degraded to synchronous lock-step), pruning ON
    run_cell "${ds}" NOSYNC \
        env DATASET="${ds}" "${common_env[@]}" \
            USE_ADAPTIVE_PRUNING=true SYNC_PUSH=true bash run_sys.sh
done

echo
echo "=========================================="
echo " summarizing -> ${OUT_DIR}/{results.csv,table.md}"
echo "=========================================="

OUT_DIR="${OUT_DIR}" RUNS_TSV="${RUNS_TSV}" python3 - <<'PY'
import csv, json, os

out_dir = os.environ["OUT_DIR"]
runs_tsv = os.environ["RUNS_TSV"]

CONFIGS = [
    ("FULL",    "TAGNAS (Full System)"),
    ("NOPRUNE", "w/o online pruning (no pruning)"),
    ("NOSYNC",  "w/o asynchronous Push (degraded to synchronous)"),
]
# AUC uses greedy (the discrete architecture used for deployment); best is preferred, falling back to final
AUC_ROWS = ["search_GASSO_greedy_best", "search_GASSO_greedy"]


def count_minus_header(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return None


def nodes_edges(ds):
    return (count_minus_header(f"data/{ds}/raw/node-data.csv"),
            count_minus_header(f"data/{ds}/raw/graph.csv"))


def read_auc(run_dir):
    p = os.path.join(run_dir, "link-prediction-eval-gasso.csv")
    if not os.path.isfile(p):
        return None
    table = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Strip the eval negative-sampling suffix (e.g. __neg_type_aware): for typed graphs
            # like amazon/twitter the row name is
            # search_GASSO_greedy_best__neg_type_aware, an exact match would miss it and cause a false FAIL
            table[r.get("gnn_model", "").split("__")[0]] = r
    for key in AUC_ROWS:
        if key in table:
            try:
                return float(table[key]["auc"])
            except (KeyError, ValueError):
                return None
    return None


def read_seconds(run_dir):
    p = os.path.join(run_dir, "search", "training_time.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8")).get("total_seconds")
    except (OSError, json.JSONDecodeError):
        return None


runs = {}
order = []
with open(runs_tsv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        runs[(r["dataset"], r["tag"])] = (r["rc"], r["run_dir"])
        if r["dataset"] not in order:
            order.append(r["dataset"])

cell = {}
for ds in order:
    for tag, _ in CONFIGS:
        rc, run_dir = runs.get((ds, tag), ("missing", "NONE"))
        auc = secs = None
        ok = False
        if rc == "0" and run_dir not in ("", "NONE") and os.path.isdir(run_dir):
            auc = read_auc(run_dir)
            secs = read_seconds(run_dir)
            ok = (auc is not None and secs is not None)
        cell[(ds, tag)] = dict(auc=auc, secs=secs, ok=ok)


def pct(ds, tag):
    base = cell.get((ds, "FULL"))
    c = cell[(ds, tag)]
    if not (base and base["ok"]) or not base["secs"]:
        return "N/A"
    if not c["ok"]:
        return "FAIL"
    return f"{(c['secs'] - base['secs']) / base['secs'] * 100:+.1f}%"


# results.csv
with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "nodes", "edges", "config",
                "test_auc", "search_hours", "time_increase_pct"])
    for ds in order:
        n, e = nodes_edges(ds)
        for tag, label in CONFIGS:
            c = cell[(ds, tag)]
            if not c["ok"]:
                w.writerow([ds, n, e, label, "FAIL", "FAIL", pct(ds, tag)])
            else:
                w.writerow([ds, n, e, label,
                            f"{c['auc']:.4f}", f"{c['secs']/3600:.3f}", pct(ds, tag)])

# table.md
with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as f:
    f.write("# test5 system ablation (speedup contribution of asynchronous bidirectional Push / online pruning)\n\n")
    f.write("| Dataset (Nodes/Edges) | Ablation config (Configuration) | Final searched Test AUC | "
            "Total search time (Hours) | Time-increase percentage |\n")
    f.write("|---|---|---|---|---|\n")
    for ds in order:
        n, e = nodes_edges(ds)
        ds_cell = f"{ds} ({n} / {e})"
        for i, (tag, label) in enumerate(CONFIGS):
            c = cell[(ds, tag)]
            first = ds_cell if i == 0 else ""
            auc = f"{c['auc']:.4f}" if c["ok"] else "FAIL"
            hrs = f"{c['secs']/3600:.3f}" if c["ok"] else "FAIL"
            f.write(f"| {first} | {label} | {auc} | {hrs} | {pct(ds, tag)} |\n")
        f.write("|  |  |  |  |  |\n")
    f.write("\n> Measures each configuration's duration under a fixed MAX_GRAPH_STEPS (same number of graph search steps = same workload). Time-increase percentage = (t - t_Full)/t_Full.\n")
    f.write("> FULL=full system; NOPRUNE=online pruning removed; NOSYNC=asynchronous Push removed (degraded to synchronous lock-step). AUC=the searched discrete architecture (greedy).\n")

print("[summary] wrote", os.path.join(out_dir, "results.csv"))
print("[summary] wrote", os.path.join(out_dir, "table.md"))
print()
print(open(os.path.join(out_dir, "table.md"), encoding="utf-8").read())
PY

echo
echo "[done] test5 complete:"
echo "  ${OUT_DIR}/results.csv"
echo "  ${OUT_DIR}/table.md"
echo "  ${OUT_DIR}/_runs.tsv"
