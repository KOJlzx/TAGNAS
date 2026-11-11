#!/usr/bin/env bash
# =========================================================================
# experiments/01_end_to_end.sh -- one-shot sequential run of all four systems + summary comparison table
#
# Four baselines (one group of 4 rows per dataset):
#   B1: DDP-Fixed (GAT+LM)   -> run_clip.sh      (fixed GNN)
#   B2: DDP-Sync (GNAS+LM)   -> run_ddp.sh       (synchronous GASSO)
#   B3: ZeRO-Offload (FP32)  -> run_deepspeed.sh (DeepSpeed ZeRO-Offload GASSO)
#   B5: UniTG (RevGNN+LM)    -> run_unitg.sh     (fixed deep RevGNN, flow parallelism + LM in bubbles)
#   TAGNAS (Ours)            -> run_sys.sh       (asynchronous distributed GASSO, PHASE=search)
#   (B4 is the ZeRO-3 variant used only by experiments/03_memory_limit.sh.)
#   SYSTEMS_TEST1="B1 B2 B3 B5 TAGNAS" (default) selects which systems run.
#
# Each system automatically runs link-prediction evaluation after training (built into each
# script). This script is only responsible for:
#   (1) Launching all four systems sequentially (overriding dataset/epochs/layers/batch via env)
#   (2) Recording the run-dir produced each time (manifest: results/test1/_runs.tsv)
#   (3) A final python summary step: extracting Test AUC / total duration / throughput,
#       computing Speedup, and emitting the table
#
# Metric definitions:
#   - Test AUC : greedy (discrete deployed architecture) @ best checkpoint
#   - Duration/throughput: each run's training_time.json (throughput_nodes_per_sec was added
#     in a previous task). All five systems report it with seed-node semantics -- the nodes that
#     actually enter the contrastive objective per step (B1/B2/B3: the batch's text rows, summed
#     over ranks; TAGNAS: batches x graph_batch_size; B5: the contrastive anchors of each
#     microbatch) -- not the nodes their towers compute over, which are far more numerous and
#     differ per system.
#   - Speedup  : B2 total duration / this row's total duration (B2 = 1.00x); if B2 fails, the
#     whole group is marked --
#
# Fault tolerance: a single cell's training failure (e.g. B3 OOM on citation) does not abort
#       the whole sweep -- that table cell is simply filled with FAIL.
#
# WARNING: experiments/01_end_to_end.sh only controls the "dataset" knob. All other training
#    hyperparameters (epoch / batch / num_layers / world_size / GNN type / max_runtime / PHASE /
#    CUDA device, etc.) should be changed directly in the corresponding run_clip.sh /
#    run_ddp.sh / run_deepspeed.sh / run_sys.sh file -- experiments/01_end_to_end.sh only
#    injects the dataset when launching each system and does not override any other env var.
#
# Usage:
#   bash experiments/01_end_to_end.sh                              # runs the pubmed twitter citation groups
#   DATASETS_TEST1="pubmed" bash experiments/01_end_to_end.sh      # runs pubmed only (validate the full pipeline)
# =========================================================================

# Note: set -e is intentionally NOT enabled -- we need to tolerate a single training failure and continue to the next cell
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate venv (if not already active)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Available GPU pool (can be set manually: CUDA_VISIBLE_DEVICES=2,3 bash experiments/01_end_to_end.sh). Default 0,1,4.
# WORLD_SIZE = number of GPUs each system runs in parallel with: B1/B2 = DDP world_size; B3 = NUM_GPUS;
#   TAGNAS = NUM_TEXT_WORKERS (the graph worker runs separately on CPU).
#   Default = number of GPUs listed in the pool (uses however many are listed); can be
#   explicitly overridden: WORLD_SIZE=2 bash experiments/01_end_to_end.sh.
IFS=',' read -ra GPU_POOL <<< "${CUDA_VISIBLE_DEVICES:-1, 5, 6}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_POOL[@]}}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > ${#GPU_POOL[@]} )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} is invalid or exceeds the number of visible GPUs ${#GPU_POOL[@]}" \
         "(CUDA_VISIBLE_DEVICES=${GPU_POOL[*]}). Please list enough GPUs in CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
# Take the first WORLD_SIZE GPUs from the pool as the ones actually used (visible GPU count = used GPU count, avoiding "N listed but only M used").
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_POOL[*]:0:WORLD_SIZE}")"

# HuggingFace mirror + offline mode (models are already cached -> skip Hub network calls, avoiding the
# list_repo_templates tree API, which the mirror doesn't support and causes ConnectTimeout). Inherited by the child run_*.sh processes.
# Multi-GPU NCCL: on this machine, GPU groups that cross a PCIe bridge (e.g. 0,1,2 / 0,1,4) can
# deadlock on P2P/IB communication during the first ALLREDUCE (processes spin and hang at epoch
# 0 batch 0). We uniformly disable P2P/IB to take the safe path (no effect on single-GPU runs).
# Can be overridden via env: NCCL_P2P_DISABLE=0 bash testN.sh
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
export LINK_NEG_SAMPLING="${LINK_NEG_SAMPLING:-auto}"   # default/type_aware/auto

# ──────────────────────────────────────────────────────────────────────
# Configurable settings
#   The "dataset" and "available GPUs" are set here; all other hyperparameters are changed
#   in the respective run_*.sh files.
# ──────────────────────────────────────────────────────────────────────
# Dataset (space-separated, overridable via env). All systems must run on the same dataset for a fair comparison.
read -ra DATASETS_TEST1 <<< "${DATASETS_TEST1:-twitter}"
# Systems to run (space-separated subset of "B1 B2 B3 B5 TAGNAS"); e.g. SYSTEMS_TEST1=B5 to add/verify one cell.
read -ra SYSTEMS_TEST1 <<< "${SYSTEMS_TEST1:-B1 B2 B3 B5 TAGNAS}"
_want() { local s; for s in "${SYSTEMS_TEST1[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

# Fixed number of training epochs (only affects B1/B2/B3 = run_clip/ddp/deepspeed; does not affect TAGNAS).
#   If EPOCHS=N is set -> runs the full N epochs (early stopping disabled via a large patience)
#     + the summary evaluation uses final_model;
#   If unset    -> falls back entirely to each run_*.sh's defaults (30 epochs + early-stop
#     patience=3 + uses best_model).
EPOCHS="${EPOCHS:-}"
EPOCH_ENV=()
if [[ -n "${EPOCHS}" ]]; then
    _BIG_PATIENCE=$((EPOCHS + 100000))   # far greater than the epoch count => early stopping never triggers => runs the full EPOCHS
    EPOCH_ENV=(NUM_EPOCHS="${EPOCHS}"
               EARLY_STOPPING_PATIENCE="${_BIG_PATIENCE}"   # variable name used by run_clip / run_ddp
               EARLY_STOP_PATIENCE="${_BIG_PATIENCE}")      # variable name used by run_deepspeed (different)
fi

# Each epoch uses only this fraction (0,1] of the text for alignment (a different random subset
# is redrawn each epoch, so accumulated over multiple epochs the full set is still covered).
# Empty = use all text (default). Only affects B1/B2/B3 (synchronous systems); TAGNAS streams
# text asynchronously and is unaffected by this parameter.
TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE:-0.5}"
# Convergence/early-stopping metric: auc = val link-prediction AUC stops improving (default);
# loss = val contrastive loss stops decreasing (old behavior).
# Only affects B1/B2/B3; the TAGNAS async system already uses val-AUC early stopping (Scheme B)
# and is unaffected by this parameter.
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-auc}"

OUT_DIR="${OUT_DIR:-results/test1}"
mkdir -p "${OUT_DIR}"
RUNS_TSV="${OUT_DIR}/_runs.tsv"
# Reset the manifest on every full run
printf 'dataset\ttag\trc\trun_dir\n' > "${RUNS_TSV}"

# DDP/DeepSpeed rendezvous port. Default avoids 29500, which is often held by leftover processes;
# if it still conflicts on a shared machine, override with B1_MASTER_PORT/B2_MASTER_PORT/B3_MASTER_PORT.
B1_MASTER_PORT="${B1_MASTER_PORT:-29610}"
B2_MASTER_PORT="${B2_MASTER_PORT:-29620}"
B3_MASTER_PORT="${B3_MASTER_PORT:-29630}"
B5_MASTER_PORT="${B5_MASTER_PORT:-30510}"

# ──────────────────────────────────────────────────────────────────────
# Common: run one baseline, capture the exit code + resolve this run's run-dir, write it to the manifest
#   run_baseline <dataset> <tag> <run_dir_glob> -- <command...>
# The command uses the env-prefix form (env VAR=.. bash run_xx.sh) so it is passed in as a whole argv.
# ──────────────────────────────────────────────────────────────────────
run_baseline() {
    local ds="$1" tag="$2" glob="$3"; shift 3
    local marker rc newest run_dir
    marker="$(mktemp)"
    echo
    echo "############################################################"
    echo "## [${ds}/${tag}] launching: $*"
    echo "############################################################"
    if "$@"; then rc=0; else rc=$?; fi

    # Sequential execution -> take the newest directory matching glob; it must be newer than marker (avoid picking up a stale artifact)
    newest="$(ls -dt ${glob} 2>/dev/null | head -1 || true)"
    newest="${newest%/}"
    if [[ -n "${newest}" && "${newest}" -nt "${marker}" ]]; then
        run_dir="${newest}"
    else
        run_dir="NONE"
        [[ ${rc} -eq 0 ]] && rc=99   # training reported success but produced no output directory -- also treated as an error
    fi
    rm -f "${marker}"

    printf '%s\t%s\t%s\t%s\n' "${ds}" "${tag}" "${rc}" "${run_dir}" >> "${RUNS_TSV}"
    echo "## [${ds}/${tag}] rc=${rc}  run_dir=${run_dir}"
}

# ──────────────────────────────────────────────────────────────────────
# Main sweep: run all four cells sequentially for each dataset
# ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " test1 sweep starting"
echo "=========================================="
echo " Datasets : ${DATASETS_TEST1[*]}"
echo " Systems  : ${SYSTEMS_TEST1[*]}"
echo " GPUs     : WORLD_SIZE=${WORLD_SIZE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo " Ports    : B1=${B1_MASTER_PORT} B2=${B2_MASTER_PORT} B3=${B3_MASTER_PORT} B5=${B5_MASTER_PORT}"
echo " Eval negs: LINK_NEG_SAMPLING=${LINK_NEG_SAMPLING}"
[[ -n "${EPOCHS}" ]] && echo " Epochs   : EPOCHS=${EPOCHS} (B1/B2/B3 run the full fixed epoch count + early stopping disabled + evaluation uses final_model)"
[[ -n "${TEXT_SUBSAMPLE}" ]] && echo " TextSub  : TEXT_SUBSAMPLE=${TEXT_SUBSAMPLE} (each epoch uses only this fraction of text for alignment; B1/B2/B3 only)"
echo " EarlyStop: EARLY_STOP_METRIC=${EARLY_STOP_METRIC} (B1/B2/B3 convergence criterion: auc=val AUC stops improving / loss=val loss stops decreasing)"
echo " (all other hyperparameters are decided by each run_*.sh itself; experiments/01_end_to_end.sh only injects the dataset + GPU count)"
echo " Out dir  : ${OUT_DIR}"
echo "=========================================="

for ds in "${DATASETS_TEST1[@]}"; do
    echo
    echo "==================== DATASET: ${ds} ===================="

    # B1: DDP-Fixed (GNN+LM) -- GNN type/epochs etc. are all set in run_clip.sh
    _want B1 && run_baseline "${ds}" B1 "model/runpytorch/${ds}_*_ddp_*" \
        env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" "${EPOCH_ENV[@]}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC="${EARLY_STOP_METRIC}" MASTER_PORT="${B1_MASTER_PORT}" bash run_clip.sh

    # B2: DDP-Sync (GNAS+LM)
    _want B2 && run_baseline "${ds}" B2 "model/runpytorch/gasso_${ds}_ddp_*" \
        env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" "${EPOCH_ENV[@]}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC="${EARLY_STOP_METRIC}" MASTER_PORT="${B2_MASTER_PORT}" bash run_ddp.sh

    # B3: ZeRO-Offload (FP32)
    _want B3 && run_baseline "${ds}" B3 "model/runpytorch/gasso_${ds}_b4_zero_*" \
        env DATASETS="${ds}" NUM_GPUS="${WORLD_SIZE}" "${EPOCH_ENV[@]}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC="${EARLY_STOP_METRIC}" MASTER_PORT="${B3_MASTER_PORT}" bash run_deepspeed.sh

    # B5: UniTG-style (fixed deep RevGNN, flow parallelism across WORLD_SIZE GPUs, LM fine-tuned in the
    # pipeline bubbles + periodic embedding sync). Other knobs live in run_unitg.sh.
    _want B5 && run_baseline "${ds}" B5 "model/runpytorch/unitg_${ds}_*" \
        env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" "${EPOCH_ENV[@]}" TEXT_SUBSAMPLE="${TEXT_SUBSAMPLE}" EARLY_STOP_METRIC="${EARLY_STOP_METRIC}" MASTER_PORT="${B5_MASTER_PORT}" bash run_unitg.sh

    # TAGNAS (Ours) -- asynchronous distributed (PHASE and other parameters are all set in run_sys.sh)
    _want TAGNAS && run_baseline "${ds}" TAGNAS "model/distributed/run-*" \
        env DATASET="${ds}" NUM_TEXT_WORKERS="${WORLD_SIZE}" bash run_sys.sh
done

# ──────────────────────────────────────────────────────────────────────
# Summary: read the manifest -> extract metrics -> emit the table (results.csv + table.md)
# ──────────────────────────────────────────────────────────────────────
echo
echo "=========================================="
echo " Summarizing results -> ${OUT_DIR}/{results.csv,table.md}"
echo "=========================================="

OUT_DIR="${OUT_DIR}" RUNS_TSV="${RUNS_TSV}" EPOCHS="${EPOCHS}" LINK_NEG_SAMPLING="${LINK_NEG_SAMPLING}" python3 - <<'PY'
import csv, json, os, fnmatch

out_dir   = os.environ["OUT_DIR"]
runs_tsv  = os.environ["RUNS_TSV"]
requested_neg_sampling = os.environ.get("LINK_NEG_SAMPLING", "auto")
TYPE_SUFFIX = "__neg_type_aware"

BASELINES = [
    ("B1",     "B1: DDP-Fixed (GAT+LM)"),
    ("B2",     "B2: DDP-Sync (GNAS+LM)"),
    ("B3",     "B3: ZeRO-Offload (FP32)"),
    ("B5",     "B5: UniTG (RevGNN-112+LM)"),
    ("TAGNAS", "TAGNAS (Ours)"),
]

# Metric source for each baseline: the eval csv filename, the gnn_model row to read (in
# priority order), and candidate json subpaths (in priority order; TAGNAS accommodates
# PHASE=search/retrain/both in run_sys.sh).
# B1 uses a fixed GNN (GNN_TYPES is set in run_clip.sh) which may not be GAT -- hence the
# wildcard match.
SPEC = {
    "B1":     dict(csv="link-prediction-eval-clip.csv",
                   rows=["ddp_clip_*@best_model", "ddp_clip_*@final_model"],
                   json=["training_time.json"]),
    "B2":     dict(csv="link-prediction-eval-gnn.csv",
                   rows=["ddp_GASSO_greedy@best_model", "ddp_GASSO_greedy@final_model"],
                   json=["training_time.json"]),
    "B3":     dict(csv="link-prediction-eval-gnn.csv",
                   rows=["ddp_GASSO_greedy@best_model", "ddp_GASSO_greedy@final_model"],
                   json=["training_time.json"]),
    "B5":     dict(csv="link-prediction-eval-unitg.csv",
                   rows=["unitg_*@best_model", "unitg_*@final_model"],
                   json=["training_time.json"]),
    "TAGNAS": dict(csv="link-prediction-eval-gasso.csv",
                   rows=["search_GASSO_greedy_best", "search_GASSO_greedy",
                         "retrain_GASSO_greedy_best", "retrain_GASSO_greedy"],
                   json=[os.path.join("search", "training_time.json"),
                         os.path.join("retrain", "training_time.json")]),
}

# Fixed-epoch mode (EPOCHS set by experiments/01_end_to_end.sh): evaluation uses final_model
# (instead of the default best_model), because running the full fixed epoch count calls for
# "the final model after training to epoch N". TAGNAS is unaffected.
if os.environ.get("EPOCHS", "").strip():
    SPEC["B1"]["rows"] = ["ddp_clip_*@final_model", "ddp_clip_*@best_model"]
    SPEC["B2"]["rows"] = ["ddp_GASSO_greedy@final_model", "ddp_GASSO_greedy@best_model"]
    SPEC["B3"]["rows"] = ["ddp_GASSO_greedy@final_model", "ddp_GASSO_greedy@best_model"]
    SPEC["B5"]["rows"] = ["unitg_*@final_model", "unitg_*@best_model"]


def count_lines_minus_header(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            n = sum(1 for _ in f)
        return max(n - 1, 0)
    except OSError:
        return None


def dataset_nodes_edges(ds):
    nodes = count_lines_minus_header(f"data/{ds}/raw/node-data.csv")
    edges = count_lines_minus_header(f"data/{ds}/raw/graph.csv")
    return nodes, edges


def _row_patterns(pat):
    if requested_neg_sampling == "default":
        return [pat]
    return [f"{pat}{TYPE_SUFFIX}", pat]


def read_auc(run_dir, spec):
    path = os.path.join(run_dir, spec["csv"])
    if not os.path.isfile(path):
        return None, None
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):           # the eval csv is comma-delimited
            table[r.get("gnn_model", "")] = r
    for pat in spec["rows"]:                  # take greedy@best -> @final in priority order (wildcards supported)
        for row_pat in _row_patterns(pat):
            for name in table:
                if fnmatch.fnmatch(name, row_pat):
                    try:
                        actual = "type_aware" if name.endswith(TYPE_SUFFIX) else "default"
                        return float(table[name]["auc"]), actual
                    except (KeyError, ValueError):
                        return None, None
    return None, None


def read_time_throughput(run_dir, spec):
    for rel in spec["json"]:                  # try multiple candidate json files in priority order
        path = os.path.join(run_dir, rel)
        if not os.path.isfile(path):
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return d.get("total_seconds"), d.get("throughput_nodes_per_sec")
    return None, None


# Read the manifest: {(ds, tag): (rc, run_dir)}
runs = {}
order = []   # preserve the order datasets first appear in
with open(runs_tsv, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        ds, tag, rc, run_dir = r["dataset"], r["tag"], r["rc"], r["run_dir"]
        runs[(ds, tag)] = (int(rc), run_dir)
        if ds not in order:
            order.append(ds)

# Compute per-cell metrics  cell[(ds,tag)] = dict(auc, t, thr, ok, ...)
cell = {}
for ds in order:
    for tag, _ in BASELINES:
        rc, run_dir = runs.get((ds, tag), (1, "NONE"))
        ok = (rc == 0 and run_dir not in ("", "NONE") and os.path.isdir(run_dir))
        auc = t = thr = neg_sampling = None
        if ok:
            spec = SPEC[tag]
            auc, neg_sampling = read_auc(run_dir, spec)
            t, thr = read_time_throughput(run_dir, spec)
            ok = (auc is not None and t is not None and thr is not None)
        cell[(ds, tag)] = dict(auc=auc, t=t, thr=thr, ok=ok, run_dir=run_dir, neg_sampling=neg_sampling)

# Speedup: each group is benchmarked against B2's total duration
for ds in order:
    base = cell.get((ds, "B2"))
    base_t = base["t"] if base and base["ok"] else None
    for tag, _ in BASELINES:
        c = cell[(ds, tag)]
        if c["ok"] and base_t and c["t"]:
            c["speedup"] = base_t / c["t"]
        else:
            c["speedup"] = None

# ---------- write results.csv ----------
csv_path = os.path.join(out_dir, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "nodes", "edges", "baseline",
                "test_auc", "neg_sampling", "total_time_s", "throughput_nodes_per_s", "speedup_x"])
    for ds in order:
        nodes, edges = dataset_nodes_edges(ds)
        for tag, label in BASELINES:
            c = cell[(ds, tag)]
            if not c["ok"]:
                w.writerow([ds, nodes, edges, label, "FAIL", "FAIL", "FAIL", "FAIL", ""])
            else:
                sp = "" if c["speedup"] is None else f"{c['speedup']:.2f}"
                w.writerow([ds, nodes, edges, label,
                            f"{c['auc']:.4f}", c["neg_sampling"] or "default",
                            f"{c['t']:.2f}", f"{c['thr']:.1f}", sp])

# ---------- write table.md ----------
def fmt(c, key, spec):
    if not c["ok"] or c[key] is None:
        return "FAIL"
    return spec % c[key]

md_path = os.path.join(out_dir, "table.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# test1 comparison results\n\n")
    f.write(f"> Eval negative sampling requested: `{requested_neg_sampling}`. "
            "`auto` prefers the `__neg_type_aware` row; plain graphs fall back to default.\n\n")
    f.write("| Dataset (Nodes/Edges) | System Config (Baselines) | Test AUC | Neg sampling | "
            "Total Training Time (s) | Throughput (Nodes/s) | Speedup (x) |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for ds in order:
        nodes, edges = dataset_nodes_edges(ds)
        ds_cell = f"{ds} ({nodes} / {edges})"
        for i, (tag, label) in enumerate(BASELINES):
            c = cell[(ds, tag)]
            first = ds_cell if i == 0 else ""
            auc = fmt(c, "auc", "%.4f")
            neg = c.get("neg_sampling") or ("FAIL" if not c["ok"] else "default")
            tt  = fmt(c, "t",   "%.2f")
            thr = fmt(c, "thr", "%.1f")
            if c["speedup"] is None:
                sp = "—" if c["ok"] else "FAIL"
            else:
                sp = f"{c['speedup']:.2f}×"
            f.write(f"| {first} | {label} | {auc} | {neg} | {tt} | {thr} | {sp} |\n")
        f.write("|  |  |  |  |  |  |  |\n")   # blank-row separator between groups

print(f"[summary] wrote {csv_path}")
print(f"[summary] wrote {md_path}")
print()
with open(md_path, encoding="utf-8") as f:
    print(f.read())
PY

echo
echo "[done] test1 fully complete. Results:"
echo "  ${OUT_DIR}/results.csv"
echo "  ${OUT_DIR}/table.md"
echo "  ${OUT_DIR}/_runs.tsv   (run-dir manifest)"
