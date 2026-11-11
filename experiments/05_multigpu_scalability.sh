#!/usr/bin/env bash
# =========================================================================
# experiments/05_multigpu_scalability.sh -- multi-GPU scalability test (1/2/4 GPU throughput + speedup)
#
#   B2: DDP-Sync     -> run_ddp.sh        (WORLD_SIZE = number of GPUs)
#   B3: ZeRO-Offload -> run_deepspeed.sh  (NUM_GPUS   = number of GPUs)
#   TAGNAS (Ours)    -> run_sys.sh        (NUM_TEXT_WORKERS = number of GPUs; the graph worker runs on CPU)
#
# For each (system x GPU count), run a uniform wall-clock budget; throughput = nodes /
# total_seconds (a rate):
#   - B2/B3 use seed-nodes/s (throughput_nodes_per_sec)
#   - TAGNAS uses text-nodes/s (text_throughput_nodes_per_sec) -- the GPU is on the text side, so it scales with GPU count
#   Speedup = T_N / T_1 (an in-group ratio; different units across systems don't affect the speedup).
#
# If this machine doesn't have enough GPUs for a given count (e.g. only 2 GPUs but 4 are
# requested) -> that cell automatically becomes N/A; the code is ready for 4 GPUs.
#
# WARNING: only "dataset + GPU count + budget" are injected (GPU count is this experiment's
# controlled variable); all other hyperparameters live in the respective run_*.sh files.
#
# Usage:
#   bash experiments/05_multigpu_scalability.sh                                  # default goodreads-crime, GPU=1 2 4
#   DATASETS_TEST4=pubmed TIME_BUDGET=150 bash experiments/05_multigpu_scalability.sh   # validate with pubmed first
#   GPU_COUNTS="1 2" bash experiments/05_multigpu_scalability.sh
# =========================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# HuggingFace mirror + offline mode (models are already cached -> skip Hub network calls, avoiding the
# list_repo_templates tree API, which the mirror doesn't support and causes ConnectTimeout). Inherited by the child run_*.sh processes.
# Multi-GPU NCCL: on this machine, GPU groups that cross a PCIe bridge (e.g. 0,1,2 / 0,1,4) can
# deadlock on P2P/IB communication during the first ALLREDUCE (processes spin and hang at
# epoch 0 batch 0). We uniformly disable P2P/IB to take the safe path (no effect on single-GPU
# runs). Can be overridden via env: NCCL_P2P_DISABLE=0 bash testN.sh
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

read -ra DATASETS_TEST4 <<< "${DATASETS_TEST4:-goodreads-crime}"
GPU_COUNTS="${GPU_COUNTS:-4 2 1}"
TIME_BUDGET="${TIME_BUDGET:-500}"     # wall-clock cap per run (seconds, can be set manually). Throughput is a rate, so it's enough to get past warmup
OUT_DIR="${OUT_DIR:-results/test4}"

# TAGNAS's GPU only runs the text worker; goodreads-crime's text is very long, so
# TEXT_CHUNK_SIZE=128 would fill up an A6000's memory in distilgpt2's fp32 forward pass and
# prevent even the first text batch from completing.
# We give TAGNAS its own conservative config here, without affecting the comparable settings for B2/B3.
TAGNAS_TEXT_CHUNK_SIZE="${TAGNAS_TEXT_CHUNK_SIZE:-32}"
TAGNAS_GRAPH_BATCH_SIZE="${TAGNAS_GRAPH_BATCH_SIZE:-256}"
TAGNAS_GRAPH_K_HOP="${TAGNAS_GRAPH_K_HOP:-3}"
TAGNAS_NUM_LAYERS="${TAGNAS_NUM_LAYERS:-3}"
TAGNAS_VAL_EARLY_STOP="${TAGNAS_VAL_EARLY_STOP:-false}"
# TAGNAS throughput "warmup complete" starting signal: in metrics.jsonl, the first time this
# cumulative counter goes above 0 is the warm starting point, excluding cold-start (cache
# filling/slow first batch) from the throughput denominator. Default is graph_updates_total
# (the side with the longest cold start).
TAGNAS_WARM_SIGNAL="${TAGNAS_WARM_SIGNAL:-graph_updates_total}"
B5_MASTER_PORT="${B5_MASTER_PORT:-30530}"   # B5 pipeline rendezvous (keep < 32768)
# Strong scaling needs a FIXED problem size, but run_unitg.sh defaults MICROBATCHES to
# WORLD_SIZE, which makes the per-epoch contrastive-anchor count (the throughput numerator)
# grow with the GPU count -- pubmed measured 1024/2048/3267 seed nodes per epoch at 1/2/4 GPUs,
# so a 3.2x counting change would masquerade as speedup. Pin it instead; M=4 satisfies GPipe's
# M >= P for every column here.
B5_MICROBATCHES="${B5_MICROBATCHES:-4}"

# Restrict which systems this launch measures (the rest keep their rows in the ledger).
SYSTEMS_TEST4="${SYSTEMS_TEST4:-B2 B3 B5 TAGNAS}"
_want4() { [[ " ${SYSTEMS_TEST4} " == *" $1 "* ]]; }
# k-step distributed (local_avg periodic weighted averaging) = TAGNAS's default sync strategy;
# test4 explicitly aligns with it (K/warmup are tunable). Only takes effect with >=2 workers;
# with 1 worker it's automatically inert. Sweet spot is K=16/warmup=30. TAGNAS_LOCAL_AVG=0 falls back to per-step.
TAGNAS_LOCAL_AVG="${TAGNAS_LOCAL_AVG:-1}"           # 1=k-step (default); 0=fall back to the per-step all_reduce control
TAGNAS_SYNC_INTERVAL="${TAGNAS_SYNC_INTERVAL:-16}"  # K: number of batches between two weighted averages
TAGNAS_SYNC_WARMUP="${TAGNAS_SYNC_WARMUP:-30}"      # number of warmup syncs (every step averaged for the first N)

# Available GPU pool (can be set manually: CUDA_VISIBLE_DEVICES=2,3 bash experiments/05_multigpu_scalability.sh). Default 0,1.
# The scalability test needs to use 1/2/4 GPUs separately, so instead of exporting this
# globally we treat this pool as an "optional physical GPU list";
# each ngpu takes the first ngpu GPUs from the pool. AVAIL = pool size (determines which GPU
# counts can run and which are filled with N/A).
IFS=',' read -ra GPU_POOL <<< "${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
for i in "${!GPU_POOL[@]}"; do
    GPU_POOL[$i]="$(echo "${GPU_POOL[$i]}" | xargs)"
done
AVAIL="${#GPU_POOL[@]}"
[[ "${AVAIL}" -lt 1 ]] && AVAIL=0

for ds in "${DATASETS_TEST4[@]}"; do
    if [[ ! -d "data/${ds}/raw" ]]; then
        echo "[ERROR] data/${ds}/raw does not exist. Currently available datasets:"
        find data -mindepth 1 -maxdepth 1 -type d -printf '  - %f\n' | sort
        echo "       you might want: DATASETS_TEST4=goodreads-crime bash experiments/05_multigpu_scalability.sh"
        exit 2
    fi
done

mkdir -p "${OUT_DIR}"
RUNS_TSV="${OUT_DIR}/_runs.tsv"
# The ledger is append-only across launches and the summary is last-row-wins per
# (dataset, tag, ngpu), so a partial run (e.g. SYSTEMS_TEST4=B5) adds its cells without
# discarding the systems measured earlier. FRESH_LEDGER=1 forces a clean slate.
if [[ -n "${FRESH_LEDGER:-}" || ! -s "${RUNS_TSV}" ]]; then
    printf 'dataset\ttag\tngpu\trc\trun_dir\tthroughput\tthroughput_warm\n' > "${RUNS_TSV}"
fi

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

# run_cell <ds> <tag> <ngpu> <run_dir_glob> <json_subpath> <thr_key> [metrics_jsub] [warm_signal] -- <cmd...>
#   - metrics_jsub not passed (B2/B3): throughput_warm = throughput (no cold-start concept).
#   - metrics_jsub+warm_signal passed (TAGNAS): computes throughput starting "after warmup completes" from <run_dir>/<metrics_jsub>.
run_cell() {
    local ds="$1" tag="$2" ngpu="$3" glob="$4" jsub="$5" thrkey="$6"; shift 6
    # Optional trailing args (before the env... command): metrics subpath + warm signal. The
    # command itself never starts with these prefixes.
    local metrics_jsub="" warm_signal=""
    if [[ "${1:-}" == *metrics.jsonl ]]; then
        metrics_jsub="$1"; warm_signal="${2:-graph_updates_total}"; shift 2
    fi
    local marker rc newest run_dir thr thr_warm

    if [[ "${ngpu}" -gt "${AVAIL}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${tag}" "${ngpu}" 200 SKIP NA NA >> "${RUNS_TSV}"
        echo "## [${ds}/${tag}/${ngpu}gpu] SKIP -- needs ${ngpu} GPUs, this machine only has ${AVAIL} (recorded as N/A in the table)"
        return
    fi

    marker="$(mktemp)"
    echo
    echo "############################################################"
    echo "## [${ds}/${tag}/${ngpu}gpu] launching: $*"
    echo "############################################################"
    setsid "$@" &
    CURRENT_TRAIN_PID=$!
    wait "${CURRENT_TRAIN_PID}"; rc=$?
    CURRENT_TRAIN_PID=""

    newest="$(ls -dt ${glob} 2>/dev/null | head -1 || true)"
    newest="${newest%/}"
    if [[ -n "${newest}" && "${newest}" -nt "${marker}" ]]; then
        run_dir="${newest}"
    else
        run_dir="NONE"; [[ ${rc} -eq 0 ]] && rc=99
    fi
    rm -f "${marker}"

    thr="NA"; thr_warm="NA"
    if [[ "${run_dir}" != "NONE" ]]; then
        # Raw throughput including cold start (current logic): read thrkey from training_time.json.
        thr="$(python3 -c "import json,os;p=os.path.join('${run_dir}','${jsub}');d=json.load(open(p)) if os.path.isfile(p) else {};print(d.get('${thrkey}',''))" 2>/dev/null || true)"
        [[ -z "${thr}" ]] && thr="NA"
        # Warm throughput (starting after warmup completes): only computed when TAGNAS passes metrics_jsub; otherwise warm=cold.
        if [[ -n "${metrics_jsub}" ]]; then
            thr_warm="$(python3 bin/tagnas_warm_throughput.py "${run_dir}/${metrics_jsub}" --signal "${warm_signal}" --numerator text_pushes_total 2>/dev/null || echo NA)"
            [[ -z "${thr_warm}" ]] && thr_warm="NA"
        else
            thr_warm="${thr}"
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${tag}" "${ngpu}" "${rc}" "${run_dir}" "${thr}" "${thr_warm}" >> "${RUNS_TSV}"
    echo "## [${ds}/${tag}/${ngpu}gpu] rc=${rc}  run_dir=${run_dir}  throughput=${thr}  throughput_warm=${thr_warm}"
}

echo "=========================================="
echo " test4 multi-GPU scalability starting"
echo " Datasets : ${DATASETS_TEST4[*]}   GPU_COUNTS=${GPU_COUNTS}   available GPU pool=${GPU_POOL[*]} (${AVAIL} total; counts above this are filled with N/A)"
echo " Per-run budget: ${TIME_BUDGET}s   Out dir: ${OUT_DIR}"
echo " TAGNAS   : text_chunk=${TAGNAS_TEXT_CHUNK_SIZE} graph_batch=${TAGNAS_GRAPH_BATCH_SIZE} k_hop=${TAGNAS_GRAPH_K_HOP} num_layers=${TAGNAS_NUM_LAYERS}"
echo "=========================================="

for ds in "${DATASETS_TEST4[@]}"; do
    for ngpu in ${GPU_COUNTS}; do
        # Take the first ngpu GPUs from the available pool (e.g. pool=2,3 and ngpu=1 -> CVD=2; ngpu=2 -> CVD=2,3)
        CVD="$(IFS=,; echo "${GPU_POOL[*]:0:ngpu}")"

        _want4 B2 && run_cell "${ds}" B2 "${ngpu}" "model/runpytorch/gasso_${ds}_ddp_*" \
            "training_time.json" "throughput_nodes_per_sec" \
            env DATASETS="${ds}" CUDA_VISIBLE_DEVICES="${CVD}" WORLD_SIZE="${ngpu}" \
                MAX_RUNTIME="${TIME_BUDGET}" bash run_ddp.sh

        _want4 B3 && run_cell "${ds}" B3 "${ngpu}" "model/runpytorch/gasso_${ds}_b4_zero_*" \
            "training_time.json" "throughput_nodes_per_sec" \
            env DATASETS="${ds}" CUDA_VISIBLE_DEVICES="${CVD}" NUM_GPUS="${ngpu}" \
                MAX_RUNTIME="${TIME_BUDGET}" bash run_deepspeed.sh

        _want4 TAGNAS && run_cell "${ds}" TAGNAS "${ngpu}" "model/distributed/run-*" \
            "search/training_time.json" "text_throughput_nodes_per_sec" \
            "search/metrics.jsonl" "${TAGNAS_WARM_SIGNAL}" \
            env DATASET="${ds}" CUDA_VISIBLE_DEVICES="${CVD}" NUM_TEXT_WORKERS="${ngpu}" \
                TEXT_CHUNK_SIZE="${TAGNAS_TEXT_CHUNK_SIZE}" \
                GRAPH_BATCH_SIZE="${TAGNAS_GRAPH_BATCH_SIZE}" \
                GRAPH_K_HOP="${TAGNAS_GRAPH_K_HOP}" \
                NUM_LAYERS="${TAGNAS_NUM_LAYERS}" \
                VAL_EARLY_STOP="${TAGNAS_VAL_EARLY_STOP}" \
                TEXT_LOCAL_AVG="${TAGNAS_LOCAL_AVG}" \
                TEXT_SYNC_INTERVAL="${TAGNAS_SYNC_INTERVAL}" \
                TEXT_SYNC_WARMUP="${TAGNAS_SYNC_WARMUP}" \
                SEARCH_MAX_RUNTIME="${TIME_BUDGET}" MAX_RUNTIME="${TIME_BUDGET}" bash run_sys.sh

        # B5 scales differently from B2/B3 by construction: WORLD_SIZE is the number of PIPELINE
        # STAGES (1 GPU = one 112-layer stage, 4 GPUs = four 28-layer stages) and it also sets the
        # number of LM replicas and, through run_unitg.sh's default, the microbatch count. That is
        # what inter-layer flow parallelism means; the footnote below says so. No metrics.jsonl is
        # passed, so throughput_warm = throughput exactly as for B2/B3.
        _want4 B5 && run_cell "${ds}" B5 "${ngpu}" "model/runpytorch/unitg_${ds}_*" \
            "training_time.json" "throughput_nodes_per_sec" \
            env DATASETS="${ds}" CUDA_VISIBLE_DEVICES="${CVD}" WORLD_SIZE="${ngpu}" \
                MICROBATCHES="${B5_MICROBATCHES}" \
                MASTER_PORT="${B5_MASTER_PORT}" \
                MAX_RUNTIME="${TIME_BUDGET}" bash run_unitg.sh
    done
done

echo
echo "=========================================="
echo " summarizing -> ${OUT_DIR}/{results.csv,table.md}"
echo "=========================================="

OUT_DIR="${OUT_DIR}" RUNS_TSV="${RUNS_TSV}" python3 - <<'PY'
import csv, json, os

out_dir = os.environ["OUT_DIR"]
runs_tsv = os.environ["RUNS_TSV"]

BASELINES = [
    ("B2",     "B2: DDP-Sync"),
    ("B3",     "B3: ZeRO-Offload"),
    ("B5",     "B5: UniTG (pipeline)"),
    ("TAGNAS", "TAGNAS (Ours)"),
]

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

# (ds, tag, ngpu) -> (rc, throughput_cold, throughput_warm)
# cold = the raw value including cold start; warm = starting after TAGNAS's "first aligned
# update" (B2/B3 warm=cold).
data = {}
RUNDIR = {}
order = []
with open(runs_tsv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        data[(r["dataset"], r["tag"], int(r["ngpu"]))] = (
            r["rc"], r["throughput"], r.get("throughput_warm", r["throughput"]))
        RUNDIR[(r["dataset"], r["tag"], int(r["ngpu"]))] = r.get("run_dir", "")
        if r["dataset"] not in order:
            order.append(r["dataset"])

# GPU-count columns: every count some cell actually ran (plus 1 and 2), so a 3-card pool gives
# 1/2/3 columns and a 4-card pool 1/2/4 -- nothing is hard-wired to 4 GPUs.
COUNTS = sorted({n for (_, _, n), (rc, _, _) in data.items() if rc not in ("200", "missing")} | {1, 2})

def cell_val(ds, tag, n, which):
    """which='warm'|'cold'. Returns (value or None, status). Status: ok / N/A (not enough GPUs) / FAIL"""
    rc, cold, warm = data.get((ds, tag, n), ("missing", "NA", "NA"))
    if rc in ("200", "missing"):
        return None, "N/A"   # not enough GPUs (SKIP) or this GPU count wasn't run -> N/A (not a failure)
    if rc != "0":
        return None, "FAIL"
    v = to_float(warm if which == "warm" else cold)
    if v is not None and v > 0:
        return v, "ok"
    # The run itself finished; only the warm-basis value is missing (TAGNAS never reached the
    # warm marker -- e.g. graph_updates_total stayed at 0, so every graph batch was dropped).
    if which == "warm" and to_float(cold) is not None and rc == "0":
        return None, "no-warm"
    return None, "FAIL"

def cell_thr(ds, tag, n):
    """Primary throughput = warm (excludes cold start; B2/B3 warm=cold). Used for display and speedup."""
    return cell_val(ds, tag, n, "warm")

def fmt_thr(v, st):
    return f"{v:.1f}" if st == "ok" else st

def fmt_thr_cell(ds, tag, n):
    """Cell: primarily shows warm; TAGNAS additionally shows the raw value including cold start in parentheses."""
    vw, sw = cell_thr(ds, tag, n)
    base = fmt_thr(vw, sw)
    if sw == "no-warm":
        vc, sc = cell_val(ds, tag, n, "cold")
        return f"no warm window (cold {vc:.1f})" if sc == "ok" and vc is not None else "no warm window"
    if tag == "TAGNAS" and sw == "ok":
        vc, sc = cell_val(ds, tag, n, "cold")
        if sc == "ok" and vc is not None:
            base = f"{base} (incl. cold start {vc:.1f})"
    return base

def fmt_speedup(ds, tag, n):
    v1, s1 = cell_thr(ds, tag, 1)
    vn, sn = cell_thr(ds, tag, n)
    if s1 == "N/A" or sn == "N/A":
        return "N/A"
    if s1 == "no-warm" or sn == "no-warm":
        return "n/a (no warm window)"
    if s1 != "ok" or sn != "ok" or not v1:
        return "FAIL" if (s1 == "FAIL" or sn == "FAIL") else "N/A"
    return f"{vn / v1:.2f}x"

# ---- two lanes -------------------------------------------------------------------------------------
# Each system is reported on BOTH lanes, so a column always compares like with like:
#   text lane  = nodes/s through the LM (GPU):  TAGNAS text_throughput (warm, its ledger value) | B5 its LM's
#                lm_throughput_nodes_per_sec | B2/B3 their seed nodes/s (sync: every step runs LM and GNN on them)
#   graph lane = nodes/s through the GNN:       TAGNAS graph batches x batch size / time (CPU) | B5 its GNN's seed
#                nodes/s (its ledger value) | B2/B3 the same seed nodes/s
# The ledger holds one lane per system (TAGNAS: text, the others: graph); the other lane is read from the run's
# training_time.json. Speedups are computed within a lane.
PRIMARY = {"TAGNAS": "text"}          # everyone else's ledger value is the graph lane (B2/B3: both lanes)

def _json(ds, tag, n):
    rd = RUNDIR.get((ds, tag, n), "")
    for sub in ("search/training_time.json", "training_time.json"):
        p = os.path.join(rd, sub)
        if rd and os.path.isfile(p):
            return json.load(open(p))
    return None

def lane_val(ds, tag, n, lane):
    """(value or None, status) of this system on this lane."""
    if tag in ("B2", "B3") or PRIMARY.get(tag, "graph") == lane:
        return cell_thr(ds, tag, n)
    v, st = cell_thr(ds, tag, n)                      # status of the run itself (N/A / FAIL / ok)
    if st in ("N/A", "FAIL"):
        return None, st
    j = _json(ds, tag, n)
    key = "throughput_nodes_per_sec" if tag == "TAGNAS" else "lm_throughput_nodes_per_sec"
    x = to_float((j or {}).get(key))
    return (x, "ok") if x and x > 0 else (None, "FAIL")

def lane_cell(ds, tag, n, lane):
    if lane == "text" and tag == "TAGNAS":
        return fmt_thr_cell(ds, tag, n)                # keeps the "(incl. cold start ...)" annotation
    return fmt_thr(*lane_val(ds, tag, n, lane))

def lane_speedup(ds, tag, n, lane):
    v1, s1 = lane_val(ds, tag, 1, lane)
    vn, sn = lane_val(ds, tag, n, lane)
    if s1 == "N/A" or sn == "N/A":
        return "N/A"
    if s1 != "ok" or sn != "ok" or not v1:
        return "FAIL" if "FAIL" in (s1, sn) else "N/A"
    return f"{vn / v1:.2f}x"

LANES = [("text", "Text lane: nodes/s through the LM (GPU)"),
         ("graph", "Graph lane: nodes/s through the GNN (TAGNAS: CPU; others: GPU)")]

with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    hdr = ["dataset", "baseline", "lane"]
    for n in COUNTS:
        hdr += [f"gpu{n}_nodes_per_s"] + ([f"speedup_{n}gpu_ideal{n}"] if n > 1 else [])
    w.writerow(hdr)
    for lane, _ in LANES:
        for ds in order:
            for tag, label in BASELINES:
                row = [ds, label, lane]
                for n in COUNTS:
                    row += [fmt_thr(*lane_val(ds, tag, n, lane))] + ([lane_speedup(ds, tag, n, lane)] if n > 1 else [])
                w.writerow(row)

with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as f:
    f.write("# test4 multi-GPU scalability (throughput & speedup), per lane\n")
    for lane, title in LANES:
        f.write(f"\n## {title}\n\n")
        hdr = ["Dataset", "System architecture", "1 GPU"]
        for n in COUNTS[1:]:
            hdr += [f"{n} GPU", f"{n}-GPU speedup (ideal={n}.0)"]
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "---|" * len(hdr) + "\n")
        for ds in order:
            for i, (tag, label) in enumerate(BASELINES):
                cells = [ds if i == 0 else "", label, lane_cell(ds, tag, 1, lane)]
                for n in COUNTS[1:]:
                    cells += [lane_cell(ds, tag, n, lane), lane_speedup(ds, tag, n, lane)]
                f.write("| " + " | ".join(cells) + " |\n")
    f.write("\n> Two lanes, each compared like with like. Text lane = nodes/s through the LM on the GPU(s): TAGNAS's text "
            "workers; B5's LM (lm_throughput_nodes_per_sec); B2/B3's seed nodes (a synchronous step trains LM and GNN on "
            "the same nodes, so both lanes carry the same number). Graph lane = nodes/s through the GNN: TAGNAS's graph "
            "worker on CPU (graph batches x graph_batch_size / time, dropped batches included, exact-push extras not "
            "counted); B5's GNN seed nodes (contrastive anchors) on its pipeline stages. Speedup = T_N/T_1 within a lane.\n")
    f.write("> B5's text lane grows super-linearly with GPUs by construction: its LM trains only in the GNN pipeline's "
            "bubbles, and one GPU is one pipeline stage with almost no bubbles, so the 1-GPU LM rate is close to zero.\n")
    f.write("> TAGNAS's GPU count scales its TEXT workers; its graph worker is one CPU process in every column, so its "
            "graph lane is not expected to scale with GPUs.\n")
    f.write("> B5 scales by inter-layer flow parallelism, not data parallelism: N GPUs means N pipeline stages "
            "(1 GPU = one 112-layer stage, 4 GPUs = four 28-layer stages) plus N LM replicas and N microbatches per "
            "mini-batch. Its speedup therefore measures a different mechanism from B2/B3's data-parallel scaling.\n")
    f.write("> TAGNAS text-lane throughput starts counting after the \"first aligned update (cache warmup complete)\", "
            "excluding the drop-on-miss cold start; the value in parentheses is the raw figure including cold start.\n")
    f.write(f"> Datasets: {', '.join(order)}. N/A = this machine doesn't have enough GPUs for that column.\n")

print("[summary] wrote", os.path.join(out_dir, "results.csv"))
print("[summary] wrote", os.path.join(out_dir, "table.md"))
print()
print(open(os.path.join(out_dir, "table.md"), encoding="utf-8").read())
PY

echo
echo "[done] test4 complete:"
echo "  ${OUT_DIR}/results.csv"
echo "  ${OUT_DIR}/table.md"
echo "  ${OUT_DIR}/_runs.tsv"
