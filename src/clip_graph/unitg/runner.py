"""Per-rank entry point: one process per GPU hosting a GNN stage thread and an LM thread."""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

import torch
import torch.distributed as dist

from .channel import SyncChannel
from .config import UniTGConfig
from .data import build_data
from .gnn_thread import GNNStageRunner
from .lm_thread import LMTrainer
from .pipeline import LMGate


def _logger(rank: int):
    def log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}][r{rank}] {msg}", flush=True)
    return log


def _warmup_groups(rank: int, world_size: int, device: torch.device, gnn_pg, lm_pg) -> None:
    """Initialise the LM's NCCL communicator and the pipeline's gloo pairs from the main
    thread, before any thread interleaving."""
    t = torch.ones(1, device=device)
    dist.all_reduce(t, group=lm_pg)
    if world_size > 1:
        for r in range(world_size - 1):
            for src, dst in ((r, r + 1), (r + 1, r)):
                buf = torch.ones(1)
                if rank == src:
                    dist.send(buf, dst=dst, group=gnn_pg)
                elif rank == dst:
                    dist.recv(buf, src=src, group=gnn_pg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def default_data_builder(config: UniTGConfig, rank: int, log):
    """Build the dataset; non-root ranks are silenced (the datamodule prints a lot)."""
    import contextlib
    import io

    quiet = contextlib.redirect_stdout(io.StringIO()) if rank != 0 else contextlib.nullcontext()
    with quiet:
        return build_data(config.dataset, config.config_path, lm_max_length=config.lm_max_length,
                          text_per_node=config.text_per_node,
                          tagnas_text_config=config.tagnas_text_config,
                          num_groups=config.lm_num_exits, log=log if rank == 0 else (lambda m: None))


def worker_main(rank: int, config: UniTGConfig, data_builder=None, lm_factory=None) -> None:
    """Entry point for ``torch.multiprocessing.spawn`` (one process per GPU / pipeline stage).

    ``data_builder(config, rank, log)`` and ``lm_factory()`` are injectable so
    the whole flow can be exercised on CPU with synthetic data in tests.
    """
    log = _logger(rank)
    # `kill -USR1 <pid>` dumps every thread's Python stack to <run_dir>/stack_r<rank>.txt
    # (hang diagnosis without ptrace).
    _stack_path = os.path.join(config.run_dir or ".", f"stack_r{rank}.txt")
    os.environ["UNITG_STACK_FILE"] = _stack_path            # also used by LMGate's stall dump
    _stack_file = open(_stack_path, "a")
    faulthandler.register(signal.SIGUSR1, file=_stack_file, all_threads=True, chain=False)
    world_size = config.world_size
    use_cuda = torch.cuda.is_available() and os.environ.get("UNITG_FORCE_CPU", "") == ""
    device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)   # claim the (Exclusive_Process) card before data loading

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    world_pg = dist.group.WORLD
    backend = "nccl" if use_cuda else "gloo"
    ranks = list(range(world_size))
    gnn_pg = dist.new_group(ranks, backend="gloo")      # pipeline P2P: gloo (see pipeline.py)
    lm_pg = dist.new_group(ranks, backend=backend)      # LM data parallelism: NCCL on GPUs
    _warmup_groups(rank, world_size, device, gnn_pg, lm_pg)
    log(f"process groups ready (gnn=gloo, lm={backend}, device={device})")

    graph_tel = text_tel = None    # bound before the try so `finally` can always close them
    try:
        builder = data_builder or default_data_builder
        # Rank 0 builds first so on-disk caches (graph pickle, SVD, tokens) are created once.
        if rank == 0:
            data = builder(config, rank, log)
            dist.barrier(group=world_pg)
        else:
            dist.barrier(group=world_pg)
            data = builder(config, rank, log)
        dist.barrier(group=world_pg)

        gate = LMGate(enabled=config.bubble_interleave)
        channel = SyncChannel()
        offline = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))

        # Per-iteration telemetry (experiments/04_time_breakdown.sh). One recorder per lane: the
        # recorder keeps unguarded per-instance state and fixes `worker` at construction, so two
        # threads cannot share one. Built here, in the main thread and before lm.start(), so the
        # file handles exist before either lane writes.
        #
        # Only rank 0 records. TAGNAS's text workers are interchangeable replicas, so it records
        # all of them and pools offline; B5's ranks are DIFFERENT pipeline stages (rank 0 is the
        # only one that runs features_fn and the only one that never receives activations), so
        # averaging across stages would mix structurally different phase vectors.
        if config.profile and rank == 0:
            from clip_graph.telemetry import TelemetryRecorder, derive_jsonl_path

            def _rec(lane: str):
                return TelemetryRecorder(
                    system="unitg", dataset=config.dataset, seed=config.seed, worker=lane,
                    device=torch.device("cpu"),      # external feed: the recorder syncs nothing itself
                    jsonl_path=derive_jsonl_path(config.profile_out, system="unitg",
                                                 dataset=config.dataset, seed=config.seed,
                                                 worker=f"{lane}_r{rank}"),
                    lit_module=None, rank=0, emit_barrier_sync_wait=False,
                    legacy_json_path=None,
                    warmup_iters=2 if lane == "text" else 1,
                )
            graph_tel, text_tel = _rec("graph"), _rec("text")

        runner = GNNStageRunner(config, data, rank, world_size, device, world_pg, gnn_pg, gate, channel,
                                log=log, telemetry=graph_tel)
        runner.setup()

        lm = LMTrainer(config, data, rank, world_size, device, lm_pg, gate, channel, log=log,
                       local_files_only=offline, telemetry=text_tel, lm_factory=lm_factory)
        lm.start()
        dist.barrier(group=world_pg)
        summary = runner.train()

        channel.request_stop()
        gate.resume()
        lm.join()
        if lm.error is not None:
            raise RuntimeError(f"LM thread on rank {rank} failed") from lm.error

        # Close after lm.join() so neither lane can write to a closed handle.
        for _tel in (graph_tel, text_tel):
            if _tel is not None:
                _tel.close(extra={"world_size": world_size, "num_layers": config.num_layers,
                                  "microbatches": config.microbatches, "objective": config.objective,
                                  "lm_objective": config.lm_objective})
        graph_tel = text_tel = None

        if rank == 0:
            _write_summary(config, data, summary, channel, lm, log)
        dist.barrier(group=world_pg)
    except BaseException:
        log("FAILED:\n" + traceback.format_exc())
        raise
    finally:
        for _tel in (graph_tel, text_tel):
            if _tel is not None:
                try:
                    _tel.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _write_summary(config: UniTGConfig, data, summary: dict, channel: SyncChannel, lm: LMTrainer, log) -> None:
    run_dir = config.run_dir
    lm_stats = dict(channel.lm_stats or lm.stats)
    total = float(summary["total_seconds"])
    gnn_nodes = int(summary["gnn_nodes_processed"])
    # Throughput uses seed-node semantics to align with the DDP/DeepSpeed scripts and TAGNAS: the
    # numerator is the nodes that actually enter the contrastive objective, not the nodes the towers
    # compute over (B1/B2/B3 run a full-graph GNN per step but credit only the batch's text rows;
    # TAGNAS runs a k-hop subgraph but credits only the seeds). The full-pass rate -- every training
    # node once per epoch, which is what this file used to report -- is kept separately below.
    gnn_seed_nodes = int(summary.get("gnn_seed_nodes_processed", gnn_nodes))
    # The LM counter is per rank; the replicas run in lock-step (gradients are all-reduced at every
    # step boundary and each step draws the same number of nodes), so scaling by world_size matches
    # the DDP baselines' all_reduce(SUM) over ranks.
    lm_nodes_rank = int(lm_stats.get("nodes_processed", 0))
    lm_nodes = lm_nodes_rank * int(config.world_size)
    guard = {
        "configured_min_lm_texts_before_stop": config.min_lm_texts_before_stop,
        "min_lm_texts_before_stop": int(summary["min_lm_texts_before_stop"]),
        "text_guard_met": lm_nodes >= int(summary["min_lm_texts_before_stop"]),
    }
    identity = {"paper_v3_attempt": config.paper_v3_attempt, "paper_v3_cell": config.paper_v3_cell}
    timing = {
        **identity,
        **guard,
        "total_seconds": total,
        "total_minutes": total / 60.0,
        "average_epoch_seconds": (sum(summary["epoch_seconds"]) / len(summary["epoch_seconds"])) if summary["epoch_seconds"] else 0.0,
        "epoch_seconds": summary["epoch_seconds"],
        "epochs_completed": summary["epochs_completed"],
        "world_size": config.world_size,
        "early_stopped": summary["stop_reason"] == "val_early_stop",
        "stop_reason": summary["stop_reason"],
        "total_seed_nodes_processed": gnn_seed_nodes,
        "throughput_nodes_per_sec": gnn_seed_nodes / total if total > 0 else 0.0,
        "total_nodes_processed_full_pass": gnn_nodes,
        "throughput_full_pass_nodes_per_sec": gnn_nodes / total if total > 0 else 0.0,
        "lm_nodes_processed": lm_nodes,
        "lm_nodes_processed_per_rank": lm_nodes_rank,
        "lm_throughput_nodes_per_sec": lm_nodes / total if total > 0 else 0.0,
        "lm_steps": lm_stats.get("steps", 0),
        "gpu_active_seconds": summary["gnn_compute_seconds"] + lm_stats.get("train_seconds", 0.0),
        "gnn_compute_seconds": summary["gnn_compute_seconds"],
        "gnn_comm_seconds": summary["gnn_comm_seconds"],
        "lm_train_seconds": lm_stats.get("train_seconds", 0.0),
        "lm_gate_wait_seconds": lm_stats.get("gate_wait_seconds", 0.0),
        "lm_encode_seconds": lm_stats.get("encode_seconds", 0.0),
        "num_syncs": summary["num_syncs"],
        "sync_seconds": summary["sync_seconds"],
        "best_epoch": summary["best_epoch"],
        "best_val_auc": summary["best_val_auc"],
        "best_elapsed_seconds": summary["best_elapsed_seconds"],
        "best_selection_seconds": summary["best_elapsed_seconds"],
    }
    with open(os.path.join(run_dir, "training_time.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    manifest = {
        **identity,
        **guard,
        "system": "unitg",
        "dataset": config.dataset,
        "lm_name": data.lm_name,
        "stop_reason": summary["stop_reason"],
        "total_training_seconds": total,
        "best_val_auc": summary["best_val_auc"],
        "best_epoch": summary["best_epoch"],
        "eval_split": summary.get("eval_split"),
        "num_syncs": summary["num_syncs"],
        "scheduler": summary.get("scheduler"),
        "lm_stats": lm_stats,
        "data": data.meta,
        "events": summary.get("events", []),
        "config": config.to_dict(),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    torch.save(lm.state_dict_cpu(), os.path.join(run_dir, "lm_final.pth"))
    log("=" * 60)
    log(f"[Time] Total wall time : {total:.2f}s ({total / 60:.2f} min)  stop={summary['stop_reason']}")
    log(f"[Time] Epochs run      : {summary['epochs_completed']}  syncs={summary['num_syncs']}  best_val_auc={summary['best_val_auc']}")
    log(f"[Throughput] gnn seed nodes/s={timing['throughput_nodes_per_sec']:.1f} "
        f"(full pass {timing['throughput_full_pass_nodes_per_sec']:.1f})  "
        f"lm nodes/s={timing['lm_throughput_nodes_per_sec']:.1f}  lm_steps={timing['lm_steps']}")
    log(f"Training completed! Artifacts in {run_dir}")
