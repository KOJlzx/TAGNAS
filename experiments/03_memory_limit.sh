#!/usr/bin/env bash
# =========================================================================
# experiments/03_memory_limit.sh -- Table 2: stress test at the limit + memory usage analysis (default 1000s/cell)
#
# Targets the largest-scale graph (goodreads-crime, 420K nodes) specifically, using data to
# "hammer" the systems:
#   the memory wall (OOM) of B1/B2, the PCIe thrashing of B3 -- to highlight how TAGNAS breaks
#   through the memory/communication wall.
#
# Four systems (B1 added):
#   B1: DDP-Fixed (GAT+LM)   -> run_clip.sh
#   B2: DDP-Sync  (GNAS+LM)  -> run_ddp.sh
#   B3: ZeRO-Offload (FP32)  -> run_deepspeed.sh
#   TAGNAS (Ours)            -> run_sys.sh
#
# 6 table columns:
#   - Max supported Batch Size (Max Nodes/Batch): try candidate batches from largest to
#     smallest; the first one that doesn't OOM is the limit
#   - Max throughput (Nodes/s)         : throughput_nodes_per_sec from running MEASURE_RUNTIME
#     seconds at the limit batch
#   - GPU pure-compute fraction (Compute vs I/O): gpu_active_seconds / total_seconds x 100
#     (CUDA-event basis)
#   - Bottleneck analysis at the limit  : fixed conclusion text (see the summary section BOTTLENECK)
#
# OOM handling: if every candidate batch OOMs for a system -> the max-batch column is filled
# with OOM, and throughput/fraction are filled with N/A.
#
# WARNING: only run one test at a time (the TAGNAS ports 29590/29591 are fixed and will
# collide); multi-GPU already has NCCL_P2P/IB_DISABLE built in.
#
# Usage:
#   DATASETS_TEST2=pubmed BATCH_CANDIDATES="64 32" MEASURE_RUNTIME=120 bash experiments/03_memory_limit.sh   # quick check on a small graph
#   NUM_LAYERS=1 GRAPH_K_HOP=1 DATASETS_TEST2=goodreads-crime CUDA_VISIBLE_DEVICES=0,3,5 bash experiments/03_memory_limit.sh  # the real stress test
# =========================================================================

set -uo pipefail   # -e is not enabled: a single cell's failure must not stop the run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Available GPU pool (can be set manually: CUDA_VISIBLE_DEVICES=2,3 bash experiments/03_memory_limit.sh). Default 0,1,2.
# WORLD_SIZE = number of GPUs each system runs in parallel with: B1/B2 = DDP world_size; B3 = NUM_GPUS;
#   TAGNAS = NUM_TEXT_WORKERS (the graph worker runs separately on CPU).
IFS=',' read -ra GPU_POOL <<< "${CUDA_VISIBLE_DEVICES:-1, 5, 6}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_POOL[@]}}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > ${#GPU_POOL[@]} )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} is invalid or exceeds the number of visible GPUs ${#GPU_POOL[@]}" \
         "(CUDA_VISIBLE_DEVICES=${GPU_POOL[*]}). Please list enough GPUs in CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_POOL[*]:0:WORLD_SIZE}")"

# Multi-GPU NCCL: on this machine, GPU groups that cross a PCIe bridge (e.g. 0,1,2 / 0,1,4) can
# deadlock on P2P/IB communication during the first ALLREDUCE (processes spin and hang at
# epoch 0 batch 0). We uniformly disable P2P/IB to take the safe path (no effect on single-GPU
# runs). Can be overridden via env: NCCL_P2P_DISABLE=0 bash testN.sh
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Memory defragmentation: the forward pass of ops like GAT allocates a large number of
# variously sized tensors, and the default caching allocator fragments memory (measured:
# B1@batch48's real working set is ~37.7GB but it OOMs early on a 47.5GB GPU due to 8.9GB of
# fragmentation).
# Enable expandable_segments so fragments can be coalesced and reclaimed, avoiding a false
# limit of "should fit but OOMs" (consistent with test4).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_OFFLINE="${HF_OFFLINE:-true}"
if [[ "${HF_OFFLINE}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# ──────────────────────────────────────────────────────────────────────
# Configurable settings
# ──────────────────────────────────────────────────────────────────────
# Stress-test dataset (default is the largest graph, goodreads-crime; can be changed to
# another large graph, e.g. citation).
read -ra DATASETS_TEST2 <<< "${DATASETS_TEST2:-citation}"
# Which systems to run (space-separated). Default runs all; to test a single one use
# SYSTEMS=B4 bash experiments/03_memory_limit.sh.
SYSTEMS="${SYSTEMS:-B1 B2 B3 B4 B5 B5g TAGNAS}"
_want() { [[ " ${SYSTEMS} " == *" $1 "* ]]; }
# Wall-clock cap (seconds) while measuring metrics at the limit; the table title says 1000s.
# Each candidate batch during probing also uses this cap.
MEASURE_RUNTIME="${MEASURE_RUNTIME:-500}"
# Candidate batches to probe from largest to smallest (nodes/batch). The first one that
# doesn't OOM is the "max supported batch size".
BATCH_CANDIDATES="${BATCH_CANDIDATES:-512 256 128 74 72 64 56 48 40 32 24 16 8 4 2 1}"
# TAGNAS: the GPU only runs the text worker, so here we probe "TEXT_CHUNK_SIZE = the largest
# text batch the GPU can hold" (the same concept as B1-B4's BATCH_SIZE and comparable to it);
# the graph worker is on CPU, so GRAPH_BATCH_SIZE is fixed here (it doesn't use GPU memory).
TAGNAS_GRAPH_BATCH_SIZE="${TAGNAS_GRAPH_BATCH_SIZE:-256}"
B5_MASTER_PORT="${B5_MASTER_PORT:-30540}"   # B5 pipeline rendezvous (keep < 32768)
# Candidate list for the B5g row: node counts per microbatch, not text batch sizes.
# The top entry is deliberately just above the largest target split rather than an arbitrary
# ceiling. At MICROBATCHES=1 the partition is B = ceil(N_train / knob), so every candidate above
# N_train collapses to the same B=1 run -- probing 400000/300000/200000/150000 on citation
# (N_train=112439) would repeat one configuration four times and then report 400000, a number the
# run never used. Each entry below yields a distinct B on citation: 1, 2, 3, 4, 6, 12, 23, 57, 113.
B5_NODE_CANDIDATES="${B5_NODE_CANDIDATES:-120000 60000 40000 30000 20000 10000 5000 2000 1000}"
# B5's text batch is probed above the shared BATCH_CANDIDATES ceiling: its GPU memory is
# dominated by the graph side, so 512 (the top of the shared list) is not a measured limit.
B5_TEXT_CANDIDATES="${B5_TEXT_CANDIDATES:-4096 2048 1024 512 256 128 64 32 16 8 4 2 1}"
# B5's text step read under TAGNAS's text measurement (default on). Without it B5 feeds a different
# and much cheaper text step: every node's rows joined and truncated to a fixed 128 tokens, in bf16,
# drawn from the whole train split. Measured on TAGNAS's own text tower at 74 rows, those three
# differences alone take the step from 31.8 GiB to 4.9 GiB (results/test2/verify_tagnas_text_memory.py),
# i.e. they, not the architecture, are most of the raw 1024-vs-74 row gap. Aligned, B5 reads the exact
# string TAGNAS reads for each node (verified 169343/169343 on citation), pads each batch to its
# longest row with the same 1024-token cap, runs the LM in fp32, and draws its contrastive anchors
# from the same fraction of train nodes as the TAGNAS probe (run_sys.sh's TRAIN_NODE_FRACTION
# default), because under pad-to-longest a larger pool meets longer outliers. B5's method is left
# alone -- early exit, pipeline, bubbles -- and so is use_cache=False, which changes no computation.
B5_TEXT_ALIGN="${B5_TEXT_ALIGN:-1}"
B5_ALIGN_FRACTION="${B5_ALIGN_FRACTION:-0.1}"
B5_TEXT_ENV=()
if [[ "${B5_TEXT_ALIGN}" == "1" ]]; then
    # LM_PROBE_WORST_CASE: B5's LM trains only in pipeline bubbles and takes tens of steps in the
    # probe window, where TAGNAS's text worker meets every node ~10 times; without it B5 would
    # usually never draw its longest text and would report a limit above its true worst case.
    # B5_ALIGN_MAX_LENGTH: the cap on B5's text. Both systems use the same 1024 tokenizer cap, but their
    # 10% pools are different random draws, and on citation B5's happens to contain two texts at the
    # cap while TAGNAS's longest is 945. Under pad-to-longest that alone costs B5 ~8% more memory at
    # the same row count, so the driver sets the cap to the longest text TAGNAS's probe can meet
    # (results/_tools/exp03_unified_text.sh computes it from TAGNAS's own data path).
    B5_TEXT_ENV=(TEXT_PER_NODE=tagnas LM_MAX_LENGTH="${B5_ALIGN_MAX_LENGTH:-1024}" LM_DYNAMIC_PADDING=1
                 NO_LM_AUTOCAST=1 LM_TRAIN_NODE_FRACTION="${B5_ALIGN_FRACTION}" LM_PROBE_WORST_CASE=1)
    # Same fine step as the TAGNAS re-probe in the region where both limits fall.
    B5_TEXT_CANDIDATES="${B5_TEXT_CANDIDATES_ALIGNED:-512 384 320 256 224 192 176 160 144 128 120 112 104 96 88 80 72 64 48 32 16}"
fi
TAGNAS_TEXT_CHUNK="${TAGNAS_TEXT_CHUNK:-128}"   # old default, no longer used by the TAGNAS probe call (TEXT_CHUNK is now the probed dimension)
# Extra timeout margin per candidate probe (seconds): the actual outer timeout = MEASURE_RUNTIME + PROBE_GRACE.
# Guards against "the first batch's forward pass fills memory and hangs", which would make the
# internal per-step cap ineffective and hang forever.
# The margin must be large enough for large-graph setup (goodreads' SVD + data loading takes
# about 10 minutes).
PROBE_GRACE="${PROBE_GRACE:-1200}"
OUT_DIR="${OUT_DIR:-results/test2}"
LOG_DIR="${OUT_DIR}/probe_logs"

# After an OOM, PyTorch/DDP/DeepSpeed sometimes doesn't exit immediately -- the parent process
# returns while spawned workers keep holding GPU memory. The two watchdogs below quickly mark
# this probe as failed and kill the whole process group.
OOM_POLL_INTERVAL="${OOM_POLL_INTERVAL:-5}"
# Threshold for judging a hang as "no log progress after training started". Large graphs
# (goodreads) have a long silent period during the first batch + multi-GPU sync, and 200s
# would be too short and cause false kills. Default is raised to 600s.
OOM_STALL_TIMEOUT="${OOM_STALL_TIMEOUT:-600}"
# Systems that print no tqdm progress bar (TAGNAS) never trip the "training started" marker, so
# the stall watchdog above would stay disarmed and a hang would burn the whole outer timeout.
# After this many seconds the watchdog arms anyway and a silent log counts as no progress.
PROBE_STALL_ARM_AFTER="${PROBE_STALL_ARM_AFTER:-180}"
# Criterion for "this tier can run": the probe log's tqdm progress has advanced to >= this many
# steps, with no OOM keyword -- that alone is taken as proof the batch fits in memory (= no
# OOM), even without producing training_time.json.
# This is the true signal test2 needs (finding the largest batch that doesn't OOM) -- progress
# bar advancing is what proves it fits.
PROBE_MIN_STEPS="${PROBE_MIN_STEPS:-20}"
# Sanity cap on the training step/s rate when estimating throughput from tqdm: exceeding it
# means we grabbed a non-training progress bar (e.g. HF "Loading weights" at thousands of
# it/s); discard it -> throughput filled with NA, avoiding absurd estimates like 6.25M nodes/s.
PROBE_MAX_SANE_RATE="${PROBE_MAX_SANE_RATE:-100}"
# Pre-probe gate: before probing each tier, first thoroughly reap leftover repo GPU processes
# from the previous tier (including escaped DeepSpeed workers), then poll until "this repo's
# usage on the target GPUs" is <= this threshold (MiB) or the timeout elapses. This avoids
# starting the next tier before the previous one has fully released memory, which would make
# every subsequent batch falsely OOM on GPUs already filled by our own leftovers.
PRE_PROBE_MEM_TOL_MIB="${PRE_PROBE_MEM_TOL_MIB:-500}"
PRE_PROBE_CLEAR_TIMEOUT="${PRE_PROBE_CLEAR_TIMEOUT:-120}"
CLEANUP_TERM_WAIT="${CLEANUP_TERM_WAIT:-8}"
CLEANUP_NEW_GPU_PROCS="${CLEANUP_NEW_GPU_PROCS:-true}"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"
RUNS_TSV="${OUT_DIR}/_runs.tsv"
# Append-only across launches (the summary keeps the last row per (dataset, tag)), so probing
# one system with SYSTEMS=... does not discard the others. FRESH_LEDGER=1 starts clean.
if [[ -n "${FRESH_LEDGER:-}" || ! -s "${RUNS_TSV}" ]]; then
    printf 'dataset\ttag\tmax_batch\trc\trun_dir\tthroughput\tgpu_active_s\ttotal_s\tnvml_util\n' > "${RUNS_TSV}"
fi

wait_cards_free_any_owner() {   # every visible card below 100 MiB, whoever held it
    local used
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null \
               | awk '$1 >= 100 {n++} END {print n + 0}')
        [[ "${used}" == "0" ]] && return
        sleep 3
    done
}

gpu_pid_snapshot() {
    local out_file="$1"
    : > "${out_file}"
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local gpu_ids="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu_ids}" \
        2>/dev/null | awk 'NF {print $1}' | sort -u > "${out_file}" || true
}

terminate_process_group() {
    local pg="$1" reason="$2"
    if kill -0 -- "-${pg}" 2>/dev/null; then
        echo "## [cleanup] ${reason}: kill -TERM process group -${pg}"
        kill -TERM -- "-${pg}" 2>/dev/null || true
        sleep "${CLEANUP_TERM_WAIT}"
    fi
    if kill -0 -- "-${pg}" 2>/dev/null; then
        echo "## [cleanup] ${reason}: kill -KILL process group -${pg}"
        kill -KILL -- "-${pg}" 2>/dev/null || true
        sleep 2
    fi
}

cleanup_new_repo_gpu_processes() {
    local before_file="$1" reason="$2"
    [[ "${CLEANUP_NEW_GPU_PROCS}" == "true" ]] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0

    local after_file
    after_file="$(mktemp)"
    gpu_pid_snapshot "${after_file}"

    local pids pid args kill_list=()
    pids="$(comm -13 "${before_file}" "${after_file}" 2>/dev/null || true)"
    rm -f "${after_file}"

    for pid in ${pids}; do
        args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
        [[ -n "${args}" ]] || continue
        if [[ "${args}" == *"${REPO_ROOT}"* || "${args}" == *"bin/congrat_train.py"* || \
              "${args}" == *"bin/ddp_train.py"* || "${args}" == *"bin/zero_offload_train.py"* || \
              "${args}" == *"async_coexec_train.py"* || "${args}" == *"unitg_train.py"* ]]; then
            kill_list+=("${pid}")
        fi
    done

    ((${#kill_list[@]} == 0)) && return 0
    echo "## [cleanup] ${reason}: kill leftover repo GPU pids: ${kill_list[*]}"
    kill -TERM "${kill_list[@]}" 2>/dev/null || true
    sleep "${CLEANUP_TERM_WAIT}"
    kill -KILL "${kill_list[@]}" 2>/dev/null || true
}

# Determine whether a GPU pid belongs to "this repo" (including DeepSpeed/mp.spawn escaped
# workers -- identified via the repo path/script name in args).
_is_repo_gpu_pid() {
    local args
    args="$(ps -p "$1" -o args= 2>/dev/null || true)"
    [[ -n "${args}" ]] || return 1
    [[ "${args}" == *"${REPO_ROOT}"* || "${args}" == *bin/congrat_train.py* || \
       "${args}" == *bin/ddp_train.py* || "${args}" == *bin/zero_offload_train.py* || \
       "${args}" == *async_coexec_train.py* ]]
}

# Total GPU memory (MiB) used by "this repo's" GPU processes on the target GPUs (the physical
# GPUs specified by CUDA_VISIBLE_DEVICES). Only counts our own processes; usage by other users
# is not counted (can't be cleaned up, only flagged separately in the gate).
repo_gpu_mem_on_targets() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    local gpu_ids="${CUDA_VISIBLE_DEVICES//[[:space:]]/}" sum=0 pid mem
    while IFS=', ' read -r pid mem; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        _is_repo_gpu_pid "${pid}" && sum=$(( sum + ${mem:-0} ))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "${gpu_ids}" 2>/dev/null)
    echo "${sum}"
}

# Kill by pid every "this repo" GPU process on the target GPUs directly (not by process group
# -- the rank/workers forked further by the DeepSpeed launcher often escape the setsid group,
# so a process-group kill can't reap them cleanly; this is a pid-based fallback).
reap_repo_gpu_procs_on_targets() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local gpu_ids="${CUDA_VISIBLE_DEVICES//[[:space:]]/}" pid kill_list=()
    while read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        _is_repo_gpu_pid "${pid}" && kill_list+=("${pid}")
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu_ids}" 2>/dev/null | awk 'NF{print $1}')
    ((${#kill_list[@]} == 0)) && return 0
    echo "## [pre-probe] reaping leftover repo GPU processes on target GPUs: ${kill_list[*]}"
    kill -TERM "${kill_list[@]}" 2>/dev/null || true
    sleep "${CLEANUP_TERM_WAIT}"
    kill -KILL "${kill_list[@]}" 2>/dev/null || true
}

# Pre-probe gate: first reap leftover repo workers, then poll until "this repo's usage on the
# target GPUs <= PRE_PROBE_MEM_TOL_MIB" or PRE_PROBE_CLEAR_TIMEOUT is exceeded. Usage from
# other users can't be cleared -- it's only flagged, not blocked on (waiting wouldn't help anyway).
wait_target_gpus_clear() {
    local ds="$1" tag="$2" waited=0 mine free_total
    reap_repo_gpu_procs_on_targets
    while :; do
        mine="$(repo_gpu_mem_on_targets)"
        (( mine <= PRE_PROBE_MEM_TOL_MIB )) && break
        if (( waited >= PRE_PROBE_CLEAR_TIMEOUT )); then
            echo "## [${ds}/${tag}] [pre-probe] after waiting ${waited}s the target GPUs still have ${mine} MiB of repo leftovers (reclaim is slow) -> continuing anyway"
            break
        fi
        echo "## [${ds}/${tag}] [pre-probe] target GPUs still have ${mine} MiB of unreleased repo memory -> reaping again and waiting ${OOM_POLL_INTERVAL}s"
        reap_repo_gpu_procs_on_targets
        sleep "${OOM_POLL_INTERVAL}"; waited=$(( waited + OOM_POLL_INTERVAL ))
    done
    free_total="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" 2>/dev/null | awk '{s+=$1} END{print s+0}')"
    echo "## [${ds}/${tag}] [pre-probe] target GPUs ready: repo leftovers approx ${mine:-0} MiB; total free on target GPUs ${free_total:-?} MiB (the rest is used by others and can't be cleaned up)"
}

# Extract the "highest completed step number" from tqdm progress in the probe log.
# The training progress bar looks like "Epoch 1/30: ... 500/16017 [07:48<...,  1.07it/s, ...]";
# take the largest N from "N/M [".
# Key: **only count within lines containing "Epoch N/M"** (training lines), excluding other
# tqdm bars such as HF's "Loading weights: 76/76 [" -- otherwise a run that OOMs right after
# the model finishes loading (76 weight shards) would have its "76/76" misread as "ran 76
# steps -> can run" (a false positive).
# If there is no training progress line at all, echoes 0 (= training never really started ->
# judged as failed).
probe_max_step() {
    local log="$1"
    [[ -f "${log}" ]] || { echo 0; return; }
    grep -aE 'Epoch [0-9]+/[0-9]+' "${log}" 2>/dev/null \
        | grep -oE '[0-9]+/[0-9]+ \[' | sed -E 's#/.*##' \
        | sort -n | tail -1 | grep -E '^[0-9]+$' || echo 0
}

# Estimate step/s from the last tqdm rate in the "training progress bar" (supports both
# "X.XXit/s" and "X.XXs/it").
# Estimated throughput nodes/s ~= step/s x batch x world_size (a fallback for when
# training_time.json isn't available).
# Key: only take the rate from lines containing "Epoch N/M" (training lines), **excluding
# other tqdm bars like "Loading weights"** -- otherwise, under multi-GPU, the HF weight-loading
# bar (thousands of it/s) could get grabbed by tail and inflate the throughput by thousands of
# times (this once produced the absurd 6.25M value).
# Also apply a sanity cap on step/s as a fallback: a training step rate can't reasonably exceed
# PROBE_MAX_SANE_RATE it/s; if it does, we assume we grabbed the wrong bar -> return empty.
probe_last_rate() {
    local log="$1" line rate
    [[ -f "${log}" ]] || { echo ""; return; }
    line="$(grep -aE 'Epoch [0-9]+/[0-9]+' "${log}" 2>/dev/null \
            | grep -oE '[0-9.]+(it/s|s/it)' | tail -1)"
    [[ -z "${line}" ]] && { echo ""; return; }
    if [[ "${line}" == *it/s ]]; then
        rate="${line%it/s}"
    else
        rate="$(python3 -c "print(1.0/${line%s/it})" 2>/dev/null || echo "")"
    fi
    [[ -z "${rate}" ]] && { echo ""; return; }
    # Sanity fallback: step/s exceeding the cap (default 100) almost certainly means we grabbed
    # the wrong non-training bar -> treat as invalid.
    python3 -c "import sys; r=float('${rate}'); sys.exit(0 if r<=${PROBE_MAX_SANE_RATE} else 1)" 2>/dev/null \
        && echo "${rate}" || echo ""
}

# ──────────────────────────────────────────────────────────────────────
# Combined probe+measure: try candidate batches from largest to smallest, the first one that
# doesn't OOM is the limit; read its metrics.
#   probe_and_measure <ds> <tag> <run_dir_glob> <json_subpath> <batch_env_name> -- <fixed env...> bash run_*.sh
#   - Internally, for each candidate b, runs: env <batch_env>=b <fixed env...> bash run_*.sh
#   - Judging "runs (no OOM)": no OOM keyword in the log, and (either <run_dir>/<json_subpath>
#     was produced or progress advanced >= PROBE_MIN_STEPS).
#     With json -> read the full throughput metrics; with only progress -> this batch still
#     counts as runnable, and throughput is estimated from the tqdm rate (marked ~).
# ──────────────────────────────────────────────────────────────────────
probe_and_measure() {
    local ds="$1" tag="$2" glob="$3" jsub="$4" batch_env="$5"; shift 5
    local b marker rc newest jpath probe_log before_gpu_pids
    local max_batch="OOM" thr="NA" gpu_act="NA" tot="NA" final_rc=1 final_run="NONE" nvml="NA" _util_log _util_pid
    local _probe_no_update=0

    echo
    echo "############################################################"
    # PROBE_CANDIDATES lets a caller override the sweep for a knob whose unit is not a text
    # batch size (B5g sweeps node counts, where 512..1 would be meaningless).
    local cands="${PROBE_CANDIDATES:-${BATCH_CANDIDATES}}"
    echo "## [${ds}/${tag}] probing the max batch  ${batch_env} in {${cands}}  (cap ${MEASURE_RUNTIME}s/attempt)"
    echo "############################################################"

    # Index loop instead of `for b in`: a candidate whose run lost a card to another user's job at
    # startup ("busy or unavailable") is retried as is once the cards are free, instead of being read
    # as "does not fit" and stepping down to a smaller batch.
    local _cl=(${cands}) _ci=0 _busy_retries=0
    while (( _ci < ${#_cl[@]} )); do
        b="${_cl[$_ci]}"
        marker="$(mktemp)"
        before_gpu_pids="$(mktemp)"
        probe_log="${LOG_DIR}/${ds}_${tag}_${batch_env}_${b}_$(date +%Y%m%d_%H%M%S).log"

        # Pre-probe gate: first reap leftovers from the previous tier (including escaped
        # DeepSpeed workers) + wait until this repo's usage on the target GPUs is approx 0
        # before starting, avoiding a false OOM on GPUs already filled by our own leftovers. The
        # "before" snapshot is taken right after this, so it represents the "clean" state.
        wait_target_gpus_clear "${ds}" "${tag}"
        gpu_pid_snapshot "${before_gpu_pids}"
        local _rank_logs="${LOG_DIR}/${ds}_${tag}_ranks"
        rm -rf "${_rank_logs}"

        echo "------ [${ds}/${tag}] trying ${batch_env}=${b} (outer timeout $(( MEASURE_RUNTIME + PROBE_GRACE ))s) ------"
        echo "------ [${ds}/${tag}] probe log: ${probe_log}"
        # setsid starts a new session (process group leader = $!); we poll our own timer, and on
        # timeout kill the entire [process group], reaping mp.spawn workers along with it. This
        # fixes the old bug where "timeout only killed the direct child while spawn workers
        # escaped as orphans holding memory" (which once let a hung candidate sit for 11 hours
        # filling 3 GPUs and contaminating every subsequent probe).
        rc=0
        # Device-level GPU utilisation, one sample per second per visible card (nvidia-smi / NVML):
        # the same ruler for every system, independent of how each one instruments its own code.
        _util_log="${probe_log%.log}.util"
        nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits \
            -i "${CUDA_VISIBLE_DEVICES:-0}" -l 1 > "${_util_log}" 2>/dev/null &
        _util_pid=$!
        setsid env "${batch_env}=${b}" "$@" > "${probe_log}" 2>&1 &
        local _pg=$! _to=$(( MEASURE_RUNTIME + PROBE_GRACE )) _w=0
        local _last_size=0 _last_change=0 _training_seen=0
        local _probe_had_oom=0
        local _oom_re='CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|DefaultCPUAllocator: can.t allocate|RuntimeError: CUDA error: out of memory|terminated with signal SIGKILL|process [0-9]+ terminated with signal SIGKILL|Killed'
        while kill -0 "${_pg}" 2>/dev/null; do
            sleep "${OOM_POLL_INTERVAL}"; _w=$((_w + OOM_POLL_INTERVAL))

            # Progress = growth of ANY of: the launcher's stdout, the per-rank logs, or files of this
            # attempt's run directory (TAGNAS writes metrics.jsonl every 10 s). The launcher's stdout
            # alone stays silent for many minutes on a large graph (TAGNAS: data store + worst-case
            # token scan ~5 min on amazon-baby), which used to be read as a hang and killed.
            local _size=0 _rd
            [[ -f "${probe_log}" ]] && _size="$(wc -c < "${probe_log}" 2>/dev/null || echo 0)"
            [[ -d "${_rank_logs}" ]] && _size=$(( _size + $(find "${_rank_logs}" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') ))
            _rd="$(ls -dt ${glob} 2>/dev/null | head -1)"
            if [[ -n "${_rd}" && "${_rd}" -nt "${marker}" ]]; then
                _size=$(( _size + $(find "${_rd}" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') ))
            fi
            if [[ "${_size}" != "${_last_size}" ]]; then
                _last_size="${_size}"
                _last_change="${_w}"
            fi

            if grep -Eiq "${_oom_re}" "${probe_log}" 2>/dev/null \
               || { [[ -d "${_rank_logs}" ]] && grep -Eiqr "${_oom_re}" "${_rank_logs}" 2>/dev/null; }; then
                echo "## [${ds}/${tag}] ${batch_env}=${b} OOM/Killed detected in log -> killing process group -${_pg} immediately"
                _probe_had_oom=1
                terminate_process_group "${_pg}" "OOM detected"
                rc=134; break
            fi

            if (( _training_seen == 0 )) && grep -Eq "Starting .*training|Epoch 1/" "${probe_log}" 2>/dev/null; then
                _training_seen="${_w}"
                _last_change="${_w}"
            fi
            if (( _training_seen == 0 && _w >= PROBE_STALL_ARM_AFTER )); then
                _training_seen="${_w}"      # no marker from this system; arm on elapsed time instead
            fi
            if (( _training_seen > 0 && OOM_STALL_TIMEOUT > 0 && _w - _last_change >= OOM_STALL_TIMEOUT )); then
                # Note: stall (no progress) != OOM. A true OOM goes through the _oom_re regex
                # path above. Here we just kill the process after the log has been silent too
                # long, to prevent a hang; whether it "can run" is decided afterward by the
                # criterion (steps advanced + no OOM keyword).
                echo "## [${ds}/${tag}] ${batch_env}=${b} no log progress for ${OOM_STALL_TIMEOUT}s after training started -> judged hang/no-progress (not OOM), killing process group -${_pg}"
                terminate_process_group "${_pg}" "training stalled"
                rc=124; break
            fi

            if (( _w >= _to )); then
                echo "## [${ds}/${tag}] ${batch_env}=${b} timed out after ${_to}s -> killing process group -${_pg} (including spawn workers)"
                terminate_process_group "${_pg}" "probe timeout"
                rc=137; break
            fi
        done
        (( rc == 137 )) || { wait "${_pg}" 2>/dev/null; rc=$?; }
        kill "${_util_pid}" 2>/dev/null; wait "${_util_pid}" 2>/dev/null

        # Whether or not the parent process has already exited, sweep once more for leftover
        # workers in the same PGID; on OOM it's common for the parent process to return while
        # its rank children are still alive, holding memory and blocking subsequent candidates.
        terminate_process_group "${_pg}" "post-probe reap"

        if grep -qs 'busy or unavailable' "${probe_log}" "${_rank_logs}"/*.log && (( _busy_retries < 50 )); then
            _busy_retries=$((_busy_retries + 1))
            echo "## [${ds}/${tag}] ${batch_env}=${b}: a card was taken by another job at startup (not OOM) -> waiting for free cards, retrying the same candidate (${_busy_retries})"
            rm -f "${marker}" "${before_gpu_pids}"
            wait_cards_free_any_owner
            continue
        fi

        newest="$(ls -dt ${glob} 2>/dev/null | head -1 || true)"; newest="${newest%/}"
        jpath=""
        [[ -n "${newest}" && "${newest}" -nt "${marker}" ]] && jpath="${newest}/${jsub}"
        rm -f "${marker}"

        # Judging "this tier can run (no OOM)": the core signal is "no OOM keyword in the log"
        # plus "training is actually progressing".
        #   _probe_had_oom: the watchdog's polling loop hit a true OOM regex (CUDA OOM/SIGKILL, etc.).
        #   has_json     : it ran long enough to produce training_time.json (full throughput metrics available).
        #   max_step     : the highest tqdm step reached; >= PROBE_MIN_STEPS proves it fits in memory.
        # For large graphs (goodreads) a single epoch far exceeds MEASURE_RUNTIME, so json is
        # almost never produced; in that case, as long as progress advanced enough we judge the
        # batch runnable (throughput estimated from the tqdm rate, marked with a ~ prefix), and
        # no longer misjudge it as OOM just because "no json" was produced.
        local has_json=0 max_step rate thrkey json_thr json_thr_pos=0
        [[ -n "${jpath}" && -f "${jpath}" ]] && has_json=1
        # Choose the throughput key by system: TAGNAS's GPU is on the text side -> read
        # text_throughput_nodes_per_sec; others read throughput_nodes_per_sec.
        thrkey="throughput_nodes_per_sec"; [[ "${tag}" == TAGNAS* ]] && thrkey="text_throughput_nodes_per_sec"
        # Throughput lower bound: having json isn't enough -- that key's throughput must also be
        # >0 to count as truly runnable, otherwise something like TAGNAS's text worker crashing
        # from CUDA OOM could still have the coordinator write out a degenerate json with
        # throughput=0, which would be misjudged as "runnable" (this once produced garbage data
        # with max_batch=1/throughput=0.03).
        if [[ "${has_json}" -eq 1 ]]; then
            json_thr="$(python3 -c "import json;print(json.load(open('${jpath}')).get('${thrkey}','') or '')" 2>/dev/null || true)"
            python3 -c "import sys;v='${json_thr}'.strip();sys.exit(0 if (v not in ('','None') and float(v)>0) else 1)" 2>/dev/null && json_thr_pos=1
        fi
        # Positive throughput is necessary but not sufficient for TAGNAS: under drop-on-miss a
        # worker can push embeddings, miss the cache on every batch and never apply a single
        # update, which still writes a json with a small positive nodes/s. Require that at least
        # one side actually stepped the optimiser, and flag that case separately from OOM -- it
        # is a convergence/cold-start failure, and stepping the batch size down does not fix it.
        if [[ "${tag}" == TAGNAS* && "${json_thr_pos}" -eq 1 ]]; then
            if ! python3 - "${jpath%/*}/metrics.jsonl" <<'PYCHK' 2>/dev/null
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError:
    sys.exit(0)          # no metrics file -> nothing to judge, keep the old behaviour
def top(k):
    return max((r.get(k) or 0) for r in rows) if rows else 0
sys.exit(0 if (top("graph_updates_total") > 0 or top("text_updates_total") > 0) else 1)
PYCHK
            then
                json_thr_pos=0
                _probe_no_update=$((_probe_no_update + 1))
                echo "## [${ds}/${tag}] ${batch_env}=${b} ran and wrote ${jsub}, but performed 0 graph and 0 text updates"
                echo "##        -> degenerate run (drop-on-miss never aligned), NOT an out-of-memory condition"
            fi
        fi
        max_step="$(probe_max_step "${probe_log}")"
        if [[ "${_probe_had_oom}" -eq 0 && ( "${json_thr_pos}" -eq 1 || "${max_step}" -ge "${PROBE_MIN_STEPS}" ) ]]; then
            # Runnable (no OOM): the first runnable one from largest to smallest is the "largest
            # non-OOM" = the max batch, record it and stop.
            max_batch="${b}"; final_rc=0; final_run="${newest}"
            # mean utilisation over the training phase (start-up / data loading excluded), all cards pooled
            nvml="$(awk -F', ' -v ncards="$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)" \
                    -f "${REPO_ROOT}/experiments/helpers/nvml_mean.awk" "${_util_log}" 2>/dev/null)"
            if [[ "${json_thr_pos}" -eq 1 ]]; then
                thr="${json_thr}"
                gpu_act="$(python3 -c "import json;print(json.load(open('${jpath}')).get('gpu_active_seconds',''))" 2>/dev/null || true)"
                tot="$(python3 -c "import json;print(json.load(open('${jpath}')).get('total_seconds',''))" 2>/dev/null || true)"
                echo "## [${ds}/${tag}] OK max batch=${b} (no OOM, rc=${rc}, ${jsub} produced, ${thrkey}=${thr}) -> stopping the probe"
            else
                # No json produced: estimate throughput nodes/s ~= step/s x batch x world_size
                # from the final tqdm rate, marked ~ to indicate an estimate.
                rate="$(probe_last_rate "${probe_log}")"
                if [[ -n "${rate}" ]]; then
                    thr="~$(python3 -c "print(round(${rate}*${b}*${WORLD_SIZE}, 1))" 2>/dev/null || echo NA)"
                else
                    thr="NA"
                fi
                gpu_act="NA"; tot="NA"
                # rc origin: 124=killed mid-run by the watchdog / 137=outer timeout / 143=SIGTERM'd / 0=finished within the window on its own.
                # Regardless of which, as long as progress advanced >= PROBE_MIN_STEPS with no OOM keyword, the batch is proven to fit in memory.
                _why="rc=${rc}"; (( rc == 124 )) && _why="rc=124(hung mid-run, reaped; not OOM)"
                echo "## [${ds}/${tag}] OK max batch=${b} (no OOM: reached step ${max_step} >= ${PROBE_MIN_STEPS}, no OOM keyword -> fits in memory; ${_why})"
                echo "##        did not produce ${jsub} (didn't finish within MEASURE_RUNTIME=${MEASURE_RUNTIME}s), throughput=${thr} is a tqdm estimate -> stopping the probe"
            fi
            rm -f "${before_gpu_pids}"
            break
        else
            cleanup_new_repo_gpu_processes "${before_gpu_pids}" "post-failure GPU reap"
            echo "## [${ds}/${tag}] ${batch_env}=${b} failure log tail:"
            tail -n 40 "${probe_log}" 2>/dev/null || true
            if [[ "${_probe_had_oom}" -eq 1 ]]; then
                echo "## [${ds}/${tag}] FAIL ${batch_env}=${b} confirmed OOM (log hit an OOM keyword, rc=${rc}) -> trying smaller"
            elif [[ "${has_json}" -eq 1 && "${json_thr_pos}" -eq 0 ]]; then
                echo "## [${ds}/${tag}] FAIL ${batch_env}=${b} produced json but ${thrkey}=${json_thr:-0}<=0 (worker may have crashed from CUDA OOM / failed to push) -> trying smaller"
            else
                echo "## [${ds}/${tag}] FAIL ${batch_env}=${b} did not meet the runnable criterion (max_step=${max_step}<${PROBE_MIN_STEPS}, no json, rc=${rc}) -> trying smaller"
            fi
            if (( _probe_no_update >= 2 )); then
                echo "## [${ds}/${tag}] ABORT: two consecutive candidates ran without a single update."
                echo "##        This is a cold-start/convergence failure, not a memory wall, so the sweep stops here"
                echo "##        instead of walking the rest of the candidate list. Recorded as NO-UPDATE."
                max_batch="NO-UPDATE"; final_rc=3; final_run="${newest}"
                rm -f "${before_gpu_pids}"
                break
            fi
        fi
        rm -f "${before_gpu_pids}"
        _ci=$((_ci + 1))
    done

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${ds}" "${tag}" "${max_batch}" "${final_rc}" "${final_run}" "${thr:-NA}" "${gpu_act:-NA}" "${tot:-NA}" "${nvml:-NA}" >> "${RUNS_TSV}"
    echo "## [${ds}/${tag}] max_batch=${max_batch}  throughput=${thr:-NA}  gpu_active=${gpu_act:-NA}s  total=${tot:-NA}s"
}

# ──────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " test2 memory stress test starting"
echo " Datasets : ${DATASETS_TEST2[*]}"
echo " GPUs     : WORLD_SIZE=${WORLD_SIZE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo " Budget   : ${MEASURE_RUNTIME}s/cell   candidate batches: ${BATCH_CANDIDATES}"
echo " Out dir  : ${OUT_DIR}"
echo "=========================================="

for ds in "${DATASETS_TEST2[@]}"; do
    echo
    echo "==================== DATASET: ${ds} ===================="

    # B1: DDP-Fixed -- probe BATCH_SIZE
    _want B1 && probe_and_measure "${ds}" B1 "model/runpytorch/${ds}_*_ddp_*" "training_time.json" BATCH_SIZE \
        env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" MAX_RUNTIME="${MEASURE_RUNTIME}" bash run_clip.sh

    # B2: DDP-Sync -- probe BATCH_SIZE
    _want B2 && probe_and_measure "${ds}" B2 "model/runpytorch/gasso_${ds}_ddp_*" "training_time.json" BATCH_SIZE \
        env DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" MAX_RUNTIME="${MEASURE_RUNTIME}" bash run_ddp.sh

    # B3: ZeRO-2 + optimizer offload -- probe BATCH_SIZE
    _want B3 && probe_and_measure "${ds}" B3 "model/runpytorch/gasso_${ds}_b4_zero_*" "training_time.json" BATCH_SIZE \
        env DATASETS="${ds}" NUM_GPUS="${WORLD_SIZE}" MAX_RUNTIME="${MEASURE_RUNTIME}" bash run_deepspeed.sh

    # B4: ZeRO-3 + full offload (param+optimizer) -- probe BATCH_SIZE (more aggressive
    # sharding, to see whether the activation wall can be broken through)
    # The glob uses *_b4_zero3_* to distinguish it from B3's *_b4_zero_*; EARLY_STOP_METRIC=loss
    # + NO_POST_EVAL=1 avoids the pitfalls of manual-forward val-AUC / sharded-checkpoint
    # evaluation under ZeRO-3 (we only need throughput, not those).
    _want B4 && probe_and_measure "${ds}" B4 "model/runpytorch/gasso_${ds}_b4_zero3_*" "training_time.json" BATCH_SIZE \
        env DATASETS="${ds}" NUM_GPUS="${WORLD_SIZE}" MAX_RUNTIME="${MEASURE_RUNTIME}" \
            DS_CONFIG="./configs/deepspeed/ds_config_zero3_offload.json" EARLY_STOP_METRIC=loss NO_POST_EVAL=1 bash run_deepspeed.sh

    # TAGNAS -- probes "TEXT_CHUNK_SIZE = the largest text batch the GPU can hold" (the graph
    # worker is on CPU, the GPU only runs the LM), on the same basis as B1-B4's BATCH_SIZE and
    # comparable to it; the graph side's GRAPH_BATCH_SIZE is fixed (doesn't use GPU memory).
    # Throughput reads text_throughput_nodes_per_sec (probe_and_measure already selects this
    # automatically by tag), and there's a throughput floor to reject degenerate runs.
    _want TAGNAS && probe_and_measure "${ds}" TAGNAS "model/distributed/run-*" "search/training_time.json" TEXT_CHUNK_SIZE \
        env DATASET="${ds}" NUM_TEXT_WORKERS="${WORLD_SIZE}" GRAPH_BATCH_SIZE="${TAGNAS_GRAPH_BATCH_SIZE}" \
            ASYNC_WORKER_LOG_DIR="${LOG_DIR}/${ds}_TAGNAS_ranks" \
            SEARCH_MAX_RUNTIME="${MEASURE_RUNTIME}" MAX_RUNTIME="${MEASURE_RUNTIME}" bash run_sys.sh

    # TAGNAS with length-bucketed, token-budget text chunks (TEXT_TOKEN_BUDGET): the probed knob is
    # the padded-token budget per chunk (rows x padded length); rows then vary with text length.
    _want TAGNASt && PROBE_CANDIDATES="${TAGNAS_TOKEN_CANDIDATES:-200000 160000 130000 110000 90000 75000 60000 45000 30000}" \
        probe_and_measure "${ds}" TAGNASt "model/distributed/run-*" "search/training_time.json" TEXT_TOKEN_BUDGET \
        env DATASET="${ds}" NUM_TEXT_WORKERS="${WORLD_SIZE}" GRAPH_BATCH_SIZE="${TAGNAS_GRAPH_BATCH_SIZE}" \
            ASYNC_WORKER_LOG_DIR="${LOG_DIR}/${ds}_TAGNASt_ranks" \
            SEARCH_MAX_RUNTIME="${MEASURE_RUNTIME}" MAX_RUNTIME="${MEASURE_RUNTIME}" bash run_sys.sh

    # B5 gets two rows because its memory profile is not shaped like the others'.
    #   B5  -- LM_BATCH_SIZE, the same "largest text batch the GPU can hold" as B1-B4/TAGNAS.
    #          Expect it not to OOM at all: B5's GPU memory is dominated by the graph side.
    #   B5g -- MAX_NODES_PER_MICROBATCH, the knob that actually drives B5's peak memory (a
    #          112-layer reversible stack's activations grow with the microbatch; measured 9.6 GiB
    #          on amazon-baby). Its unit is node counts, hence PROBE_CANDIDATES.
    _want B5 && PROBE_CANDIDATES="${B5_TEXT_CANDIDATES}" \
        probe_and_measure "${ds}" B5 "model/runpytorch/unitg_${ds}_*" "training_time.json" LM_BATCH_SIZE \
        env "${B5_TEXT_ENV[@]}" DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" MASTER_PORT="${B5_MASTER_PORT}" \
            MAX_RUNTIME="${MEASURE_RUNTIME}" NO_POST_EVAL=1 bash run_unitg.sh

    # MICROBATCHES=1 is what makes this probe meaningful: with the default M = WORLD_SIZE the
    # partition is B*M = 3 clusters whatever the knob says, pinning the microbatch at N/3 nodes,
    # so raising MAX_NODES_PER_MICROBATCH changes nothing. At M=1 the knob IS the microbatch node
    # count (B = ceil(N / knob)), so the largest candidate puts the whole training graph through
    # the 112-layer reversible stack in one shot.
    _want B5g && PROBE_CANDIDATES="${B5_NODE_CANDIDATES}" \
        probe_and_measure "${ds}" B5g "model/runpytorch/unitg_${ds}_*" "training_time.json" MAX_NODES_PER_MICROBATCH \
        env "${B5_TEXT_ENV[@]}" DATASETS="${ds}" WORLD_SIZE="${WORLD_SIZE}" MICROBATCHES=1 MASTER_PORT="${B5_MASTER_PORT}" \
            MAX_RUNTIME="${MEASURE_RUNTIME}" NO_POST_EVAL=1 bash run_unitg.sh
done

# ──────────────────────────────────────────────────────────────────────
# Summary: read the manifest -> compute max batch / throughput / GPU compute fraction -> emit
# the 6-column table
# ──────────────────────────────────────────────────────────────────────
echo
echo "=========================================="
echo " summarizing -> ${OUT_DIR}/{results.csv,table.md}"
echo "=========================================="

OUT_DIR="${OUT_DIR}" RUNS_TSV="${RUNS_TSV}" python3 - <<'PY'
import csv, json, os

out_dir = os.environ["OUT_DIR"]
runs_tsv = os.environ["RUNS_TSV"]

BASELINES = [
    ("B1",     "B1: DDP-Fixed (GAT+LM)"),
    ("B2",     "B2: DDP-Sync (GNAS+LM)"),
    ("B3",     "B3: ZeRO-2 Offload (FP32)"),
    ("B4",     "B4: ZeRO-3 Offload (param+optim)"),
    ("B5",     "B5: UniTG (text batch)"),
    ("B5g",    "B5: UniTG (graph microbatch nodes)"),
    ("TAGNAS", "TAGNAS (Ours)"),
    ("TAGNASt", "TAGNAS (Ours), token-budget chunks"),
]

# Bottleneck analysis at the limit: fixed conclusion text
BOTTLENECK = {
    "B1":     "Memory Wall: model parameters and graph structure compete for memory, causing very early OOM",
    "B2":     "Memory Wall: model parameters and graph structure compete for memory, causing very early OOM",
    "B3":     "PCIe Wall: the large offload volume triggers severe memory thrashing",
    "B4":     "PCIe Wall: ZeRO-3 also offloads parameters, so per-layer param-gather + CPU<->GPU round trips are the heaviest; it only saves parameter/optimizer memory, the activation wall remains",
    "B5":     "Text Wall under TAGNAS's text measurement: with its pipeline stage resident, 144 rows OOM and 128 hold on a 945-token worst-case batch; early exit (the LM stops after 2/3/5/6 layers by degree group) is what buys it 1.6x TAGNAS's rows",
    "B5g":    "No graph-side wall measured here: reversible layers keep activations flat in depth, so even one microbatch holding the entire 112,439-node training split fits; this knob bottoms out at the size of the data, not at memory",
    "TAGNAS": "Breaks through the memory and communication wall: CPU hosts an unbounded graph structure, GPU runs LM compute at full load",
    "TAGNASt": "Same system; text chunks are length-bucketed and bounded by padded tokens (rows x longest row) instead of rows, so padding no longer dominates the text step",
}
if os.environ.get("EXP03_FULL_GRAPH"):
    # the texts above quote the 10%-pool citation run's numbers; full-graph tables get neutral ones
    BOTTLENECK["B5"] = ("Text wall at the pool's longest text (worst case forced); early exit (the LM stops after "
                        "2/3/5/6 layers by degree group) is what buys it rows over a full-depth LM")
    BOTTLENECK["B5g"] = ("Graph-side knob: reversible layers keep activations flat in depth, so the limit, if any, "
                         "is the size of the training split rather than memory")


def count_minus_header(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return None


def nodes_edges(ds):
    return (count_minus_header(f"data/{ds}/raw/node-data.csv"),
            count_minus_header(f"data/{ds}/raw/graph.csv"))


def to_float(s):
    # throughput may be "~123.4" (estimated from the tqdm rate when there's no
    # training_time.json); strip the ~ prefix before converting.
    if isinstance(s, str):
        s = s.lstrip("~")
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# Read the manifest: {(ds, tag): row}
runs = {}
order = []
with open(runs_tsv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        runs[(r["dataset"], r["tag"])] = r
        if r["dataset"] not in order:
            order.append(r["dataset"])


def cell(ds, tag):
    """Returns (max_batch_str, throughput_str, compute_pct_str, nvml_util_str). OOM/missing -> OOM / N/A."""
    r = runs.get((ds, tag))
    if r is None:
        return "—", "N/A", "N/A", "N/A", "N/A", "N/A"
    mb = r.get("max_batch", "OOM")
    if mb == "NO-UPDATE":
        return "no update", "N/A", "N/A", "N/A", "N/A", "N/A"
    if mb == "OOM" or r.get("rc") != "0":
        return "OOM", "N/A", "N/A", "N/A", "N/A", "N/A"
    nv = to_float(r.get("nvml_util"))
    nv_s = "N/A" if nv is None else f"{nv:.1f}%"
    if tag == "TAGNASt":
        # the knob is a token budget; report it with the rows a chunk actually carried on average
        try:
            rows = [json.loads(l) for l in open(os.path.join(r["run_dir"], "search", "metrics.jsonl")) if l.strip()]
            last = [x for x in rows if "text_batches_total" in x][-1]
            mb = f"{int(mb):,} tokens (mean {last['text_pushes_total'] / max(last['text_batches_total'], 1):.0f} rows)"
        except Exception:  # noqa: BLE001
            mb = f"{mb} tokens"
    raw_thr = r.get("throughput") or ""
    estimated = isinstance(raw_thr, str) and raw_thr.lstrip().startswith("~")
    thr = to_float(raw_thr)
    gpu = to_float(r.get("gpu_active_s"))
    tot = to_float(r.get("total_s"))
    # Estimated values (no json, extrapolated from the final tqdm rate) are marked "~..(est)",
    # to distinguish them from measured throughput.
    thr_s = ("N/A" if thr is None else (f"~{thr:.1f} (est)" if estimated else f"{thr:.1f}"))
    if gpu is not None and tot and tot > 0:
        pct_s = f"{gpu / tot * 100:.1f}%"
    else:
        pct_s = "N/A"
    txt, gph = lanes(tag, r, thr)
    return mb, thr_s, pct_s, nv_s, txt, gph


def lanes(tag, r, thr):
    """(text-lane, graph-lane) nodes/s strings. The ledger holds one lane (TAGNAS: text; others: graph); the
    other comes from the run's training_time.json. B1-B4 are synchronous: one number serves both lanes."""
    def j(sub):
        p = os.path.join(r.get("run_dir", ""), sub)
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return {}
    def s(v):
        v = to_float(v)
        return "N/A" if v is None else f"{v:.1f}"
    if tag.startswith("TAGNAS"):
        jj = j("search/training_time.json")
        return s(jj.get("text_throughput_nodes_per_sec", thr)), s(jj.get("throughput_nodes_per_sec"))
    if tag.startswith("B5"):
        jj = j("training_time.json")
        return s(jj.get("lm_throughput_nodes_per_sec")), s(jj.get("throughput_nodes_per_sec", thr))
    return s(thr), s(thr)


# ---------- results.csv ----------
with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "nodes", "edges", "system",
                "max_batch_nodes", "text_lane_nodes_per_s", "graph_lane_nodes_per_s", "gpu_compute_pct",
                "gpu_util_nvml_pct", "bottleneck"])
    for ds in order:
        n, e = nodes_edges(ds)
        for tag, label in BASELINES:
            mb, thr_s, pct_s, nv_s, txt, gph = cell(ds, tag)
            w.writerow([ds, n, e, label, mb, txt, gph, pct_s, nv_s, BOTTLENECK[tag]])

# ---------- table.md ----------
with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as f:
    f.write("# Table 2: stress test at the limit + memory usage analysis\n\n")
    f.write("| Stress-test dataset (Nodes/Edges) | System architecture | Max supported Batch Size (Max Nodes/Batch) | "
            "Text lane (Nodes/s, LM on GPU) | Graph lane (Nodes/s, GNN) | GPU pure-compute fraction (Compute vs I/O) | "
            "GPU utilisation (nvidia-smi) | Bottleneck analysis |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for ds in order:
        n, e = nodes_edges(ds)
        ds_cell = f"{ds} ({n} / {e})"
        for i, (tag, label) in enumerate(BASELINES):
            mb, thr_s, pct_s, nv_s, txt, gph = cell(ds, tag)
            first = ds_cell if i == 0 else ""
            f.write(f"| {first} | {label} | {mb} | {txt} | {gph} | {pct_s} | {nv_s} | {BOTTLENECK[tag]} |\n")
        f.write("|  |  |  |  |  |  |  |  |\n")
    halos = set()
    for (_ds, _tag), _r in runs.items():
        if _tag.startswith("B5") and _r.get("rc") == "0":
            try:
                halos.add(int(json.load(open(os.path.join(_r["run_dir"], "manifest.json")))["config"].get("halo_hops", 0)))
            except Exception:  # noqa: BLE001
                pass
    if halos and halos != {0}:
        f.write("\n> B5 subgraph extraction = TAGNAS's: each microbatch is its METIS cluster (the seeds, the only loss anchors) "
                f"plus the seeds' k-hop neighbourhood, k = {sorted(halos)} (HALO_HOPS), as TAGNAS's graph worker extracts "
                "the 3-hop subgraph of its seed batch. UniTG's shipped microbatch is the induced cluster subgraph, which "
                "drops cross-cluster edges (31% of amazon-baby's, 6.9% of citation's).\n")
    f.write("\n> Throughput is given per lane so each column compares like with like. Text lane = nodes/s through the LM "
            "on the GPU(s) (TAGNAS: its text workers; B5: its LM, lm_throughput_nodes_per_sec). Graph lane = nodes/s through "
            "the GNN (TAGNAS: its graph worker on CPU, graph batches x graph_batch_size / time; B5: GNN seed nodes = "
            "contrastive anchors). B1-B4 are synchronous -- every step trains the LM and the GNN on the same nodes -- so "
            "one number fills both lanes.\n")
    f.write("> Max Batch Size = the first candidate batch, probed from largest to smallest, that does not OOM (OOM = every candidate ran out of memory).\n")
    f.write("> B1-B4 probe the text BATCH_SIZE; TAGNAS probes TEXT_CHUNK_SIZE (the text worker's GPU batch, the graph worker is on CPU and uses no GPU memory)"
            " -- both represent \"the largest text batch the GPU can hold\" and are comparable.\n")
    f.write("> B5 gets two rows because its memory is not shaped like the others': the text row probes LM_BATCH_SIZE, "
            "the graph row (B5g) probes MAX_NODES_PER_MICROBATCH at MICROBATCHES=1, where the partition is "
            "B = ceil(N_train / knob) and the knob therefore IS the microbatch node count. At the top of the sweep the "
            "cap exceeds N_train, so the effective microbatch there is min(knob, N_train) = the whole training graph in one shot.\n")
    if os.environ.get("EXP03_FULL_GRAPH"):
        f.write("> Full training graph for every system (TRAIN_NODE_FRACTION=1.0 for TAGNAS, LM_TRAIN_NODE_FRACTION=1.0 for B5; "
                "B5's GNN always holds the whole training split). B5 reads the exact per-node string TAGNAS reads, pads each batch to "
                "its longest row, runs its LM in fp32. Worst case forced on both: every TAGNAS text worker's first chunk and every B5 "
                "rank's first LM batch carry the longest text of the whole training pool (up to the shared 1024-token cap), so a "
                "short probe window cannot miss it. B5's remaining advantage in rows is its early exit, part of its method.\n")
    else:
        f.write("> B5 is measured under TAGNAS's text measurement. For every node it reads the exact string TAGNAS's text worker "
                "reads (built through TAGNAS's own data path; identical on all 169,343 citation nodes), pads each batch to its longest "
                "row, runs its LM in fp32, and draws its contrastive anchors from the same 10% fraction of train nodes as the TAGNAS "
                "probe. Both systems' worst-case batch is capped at the same 945 tokens -- the longest text in TAGNAS's probe pool -- "
                "and B5's first LM batch is forced to carry it, because B5's LM trains only in pipeline bubbles and would otherwise "
                "rarely meet its longest text inside the probe window, while TAGNAS's text worker meets every node about ten times. "
                "Under B5's shipped text settings (all of a node's rows joined and truncated to 128 tokens, bf16) the same row read "
                "1024: the gap to TAGNAS was ~14x in rows and is 1.6x once both read the same text the same way. What remains is "
                "B5's early exit, part of its method and kept.\n")
    f.write("> TAGNAS and B5 were probed on the same candidate grid below 128 (128 120 112 104 96 88 80 72 64 ...); TAGNAS's earlier "
            "74 came from a list that jumped from 128 straight to 74.\n")
    f.write("> B5 and (since 2026-09-24) TAGNAS run GPT-2 with use_cache=False; B1-B4 keep the default use_cache=True, an unused "
            "key/value cache (2.6 GiB at 80 rows x 945 tokens). It changes no computation.\n")
    f.write("> GPU pure-compute fraction = gpu_active_seconds / total_seconds, as each system instruments itself: B1-B4 and "
            "TAGNAS time forward+backward(+update) with CUDA events; B5 adds the host wall time of its GNN stage and of its LM "
            "training steps, which (a) leaves out its full-table LM encodes (GPU work, e.g. 273 s of a 503 s citation run) and "
            "(b) double-counts where its concurrent GNN and LM threads overlap. It is NOT comparable across systems. "
            "GPU utilisation (nvidia-smi) is: the driver's busy fraction of each visible card, sampled every second and averaged "
            "over cards from the first 10-second window averaging > 20% on (start-up, data loading and tokenisation "
            "excluded) -- one ruler for every system.\n")
    f.write("> The two B5 rows' throughput columns are not comparable with each other. The text row runs the shipped "
            "partition (M = WORLD_SIZE = 3, so 3x1024 contrastive anchors per epoch) with the probed LM batch; the graph row "
            "forces M=1 to make the knob bite, which leaves 1x1024 anchors per epoch and LM_BATCH_SIZE at its default 64, so "
            "their seed-node rates differ by how much work an epoch contains, not by how fast the big microbatch runs.\n")

print("[summary] wrote", os.path.join(out_dir, "results.csv"))
print("[summary] wrote", os.path.join(out_dir, "table.md"))
print()
print(open(os.path.join(out_dir, "table.md"), encoding="utf-8").read())
PY

echo
echo "[done] test2 memory stress test complete:"
echo "  ${OUT_DIR}/results.csv"
echo "  ${OUT_DIR}/table.md"
echo "  ${OUT_DIR}/_runs.tsv"
