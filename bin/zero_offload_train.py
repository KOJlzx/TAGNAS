"""ZeRO-Offload supernet baseline for joint graph/text architecture search.

Run with ``bash run_deepspeed.sh``. The DeepSpeed launcher initializes the
process group and wraps the supernet; JSON configuration files select
ZeRO stage, CPU offload, and precision.

Inspect arguments with ``python bin/zero_offload_train.py --help``.
"""

import argparse
import math
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

# B4 is still a GNAS workload (same model as bin/ddp_train.py), only the parallelism framework is swapped
from clip_graph.gassolit import LitClipGraph as LitGassoClipGraph
from clip_graph.telemetry import TelemetryRecorder, derive_jsonl_path
from clip_graph.data.gassodatamodule import (
    PubmedGraphTextDataModule as GassoPubmedGraphTextDataModule,
)

_DATA_ROOT = "data"
_REQUIRED_FILES: Tuple[str, ...] = ("graph.csv", "node-data.csv", "texts.csv")


def _discover_datasets(root: str = _DATA_ROOT) -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        raw_dir = os.path.join(root, name, "raw")
        if not os.path.isdir(raw_dir):
            continue
        if all(os.path.isfile(os.path.join(raw_dir, fn)) for fn in _REQUIRED_FILES):
            out.append(name)
    return out


@dataclass
class TrainingConfig:
    config_path: str
    deepspeed_config: str = "configs/deepspeed/ds_config_b4_offload.json"
    dataset: str = "pubmed"
    num_workers: int = 0
    num_epochs: int = 10
    learning_rate: float = 3e-5
    checkpoint_frequency: int = 5
    log_frequency: int = 10
    early_stopping_patience: int = 3
    output_root: str = "model/runpytorch"
    run_name: Optional[str] = None
    # Injected by the DeepSpeed CLI launcher
    local_rank: int = 0
    # Injected from main (rank 0 creates it, then it is synced to the other ranks via env var)
    run_dir: Optional[str] = None
    # None means fall back to model.init_args.gnn_params.num_layers from the YAML; if not None, the CLI overrides it.
    # I.e. the number of GNN layers / k_hop neighbor hops.
    num_layers: Optional[int] = None
    # None means fall back to data.init_args.batch_size from the YAML; if not None, the CLI overrides it.
    # Warning: GASSO uses square_contrastive_loss: with batch_size=1, CE([x],[0])=0 holds exactly,
    # so the whole training loss is identically 0, alpha gets no gradient, and the search is completely
    # ineffective. Must be >= 2.
    batch_size: Optional[int] = None
    # Training wall-clock cap (seconds). None means unlimited; if not None, training stops once the
    # cap is reached (used for the unified budget comparison in test3).
    max_runtime_seconds: Optional[float] = None
    # Each epoch only uses this fraction (0,1] of the text for alignment: each epoch only runs the
    # first ceil(total_batches * frac) batches.
    # Since BatchGraphTextDataset reshuffles every epoch, the truncated set is a different random
    # subset each time (accumulating over multiple epochs still covers the full set).
    # None = use all text (default, backward compatible).
    text_subsample_fraction: Optional[float] = None
    # Early-stopping / convergence monitoring metric: "auc" = val link-prediction AUC stops improving
    # (default, higher is better, uses the greedy deployed architecture);
    # "loss" = val contrastive loss stops decreasing (old behavior). In "auc" mode, once per epoch on
    # rank 0 a greedy(is_search=False) full-graph forward pass over the val subgraph computes AUC,
    # which is used to pick best_model and decide early stopping.
    early_stop_metric: str = "auc"
    # test6 time-breakdown: when enabled, each iteration is timed into 7 buckets (alongside _PhaseTimer),
    # and a profile_out JSON is written at the end of training.
    profile: bool = False
    profile_out: Optional[str] = None
    # Unified telemetry: per-iteration JSONL path (None = derived from profile_out's telemetry/ subdirectory).
    telemetry_jsonl: Optional[str] = None
    # For reproducibility, and to tag the telemetry JSONL with a seed label.
    seed: int = 42
    # Every N seconds, additionally compute val-AUC on rank0 and write it to run_dir/val_timeline.csv
    # (for dense time-to-accuracy sampling).
    # Independent of the per-epoch metrics.csv / early stopping; under ZeRO-3, _compute_val_auc returns
    # None, so nothing is written automatically.
    # 0 = disabled (default, behavior unchanged).
    val_eval_interval_seconds: float = 0.0
    # True = switch the periodic sampling to run link prediction on the test split instead (test1-2's
    # honest speedup comparison uses the test-set convention). Default False = val.
    val_eval_on_test: bool = False


# ---------------------------------------------------------------------------
# Files / directories
# ---------------------------------------------------------------------------


def _ds_zero_stage(deepspeed_config: str) -> int:
    """Read zero_optimization.stage from the deepspeed config JSON (defaults to 2 if unreadable)."""
    try:
        with open(deepspeed_config, "r", encoding="utf-8") as f:
            return int(json.load(f).get("zero_optimization", {}).get("stage", 2))
    except Exception:
        return 2


def _make_run_dir(config: TrainingConfig) -> str:
    root = os.path.abspath(os.path.expanduser(config.output_root))
    os.makedirs(root, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        # DeepSpeed starts independent interpreters: rank 0 cannot set an
        # environment variable for already-running sibling ranks. The launcher
        # supplies one shared ID, avoiding timestamp-boundary split directories.
        ts = (os.environ.get("PAPER_V3_ATTEMPT") or os.environ.get("TAGNAS_RUN_ID")
              or datetime.now().strftime("%Y%m%d_%H%M%S"))
        # ZeRO-3 (B4) and ZeRO-2 (B3) use different directory prefixes so that test2's glob can
        # tell them apart (`..._b4_zero_*` does not match `..._b4_zero3_*`).
        tag = "zero3" if _ds_zero_stage(config.deepspeed_config) >= 3 else "zero"
        name = f"gasso_{config.dataset}_b4_{tag}_{ts}"
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_training_artifacts(
    run_dir: str,
    config: TrainingConfig,
    model_config: OmegaConf,
) -> tuple[str, str, str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    train_log = os.path.join(run_dir, "training.log")
    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(dict(asdict(config), paper_v3_attempt=os.environ.get("PAPER_V3_ATTEMPT"),
                       paper_v3_cell=os.environ.get("PAPER_V3_CELL")), f, indent=2, ensure_ascii=False)
    # Also write the CLI/shell overrides back into the snapshot, so the eval side sees the correct actual structure.
    if config.num_layers is not None:
        try:
            model_config.model.init_args.gnn_params.num_layers = int(config.num_layers)
        except Exception:
            pass
    OmegaConf.save(model_config, os.path.join(run_dir, "model_config_snapshot.yaml"))
    # Also snapshot the ds_config to make paper reproduction easier
    if os.path.isfile(config.deepspeed_config):
        with open(config.deepspeed_config, "r", encoding="utf-8") as src:
            ds_snapshot = json.load(src)
        with open(os.path.join(run_dir, "ds_config_snapshot.json"), "w", encoding="utf-8") as dst:
            json.dump(ds_snapshot, dst, indent=2, ensure_ascii=False)
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch", "train_loss", "val_loss", "best_val_loss",
                "val_auc", "best_val_auc",
                "epoch_seconds",
                "lm_phase_ms", "gnn_phase_ms", "bw_step_phase_ms",
                "peak_gpu_mem_gb",
                "timestamp",
            ]
        )
    with open(train_log, "w", encoding="utf-8") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write(f"config_path={config.config_path}\n")
        f.write(f"ds_config={config.deepspeed_config}\n")
        f.write(f"dataset={config.dataset}\n\n")
    return metrics_csv, train_log, ckpt_dir


def _append_metric_row(
    metrics_csv: str,
    train_log: str,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
    epoch_seconds: float,
    phase_stats: Dict[str, float],
    peak_mem_gb: float,
    val_auc: Optional[float] = None,
    best_val_auc: Optional[float] = None,
) -> None:
    ts = datetime.now().isoformat()
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                epoch + 1,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                f"{best_val_loss:.6f}" if best_val_loss < float("inf") else "",
                "" if val_auc is None else f"{val_auc:.6f}",
                "" if not best_val_auc else f"{best_val_auc:.6f}",
                f"{epoch_seconds:.3f}",
                f"{phase_stats.get('lm', 0.0):.3f}",
                f"{phase_stats.get('gnn', 0.0):.3f}",
                f"{phase_stats.get('bw_step', 0.0):.3f}",
                f"{peak_mem_gb:.3f}",
                ts,
            ]
        )
    with open(train_log, "a", encoding="utf-8") as f:
        line = (
            f"epoch={epoch + 1} train_loss={train_loss:.6f}"
            f" lm_ms={phase_stats.get('lm', 0.0):.2f}"
            f" gnn_ms={phase_stats.get('gnn', 0.0):.2f}"
            f" bw_step_ms={phase_stats.get('bw_step', 0.0):.2f}"
            f" peak_mem_gb={peak_mem_gb:.2f}"
        )
        if val_loss is not None:
            line += f" val_loss={val_loss:.6f}"
        if best_val_loss < float("inf"):
            line += f" best_val_loss={best_val_loss:.6f}"
        if val_auc is not None:
            line += f" val_auc={val_auc:.6f}"
        if best_val_auc:
            line += f" best_val_auc={best_val_auc:.6f}"
        line += f" epoch_seconds={epoch_seconds:.3f}"
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Resume: continue from a DeepSpeed checkpoint in an existing run_dir
# ---------------------------------------------------------------------------


def _latest_epoch_tag(ckpt_dir: str) -> Optional[str]:
    """Find the epoch_N directory with the largest N under ckpt_dir. Prefer reading DeepSpeed's `latest` marker file."""
    if not os.path.isdir(ckpt_dir):
        return None
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.isfile(latest_file):
        try:
            with open(latest_file, encoding="utf-8") as f:
                tag = f.read().strip()
            if tag and os.path.isdir(os.path.join(ckpt_dir, tag)):
                return tag
        except OSError:
            pass
    best, best_n = None, -1
    epoch_re = re.compile(r"^epoch_(\d+)$")
    for name in os.listdir(ckpt_dir):
        m = epoch_re.match(name)
        if m and os.path.isdir(os.path.join(ckpt_dir, name)):
            n = int(m.group(1))
            if n > best_n:
                best, best_n = name, n
    return best


def _read_resume_state(metrics_csv: str, metric: str) -> Optional[Dict[str, Any]]:
    """Restore resume state from metrics.csv: number of completed epochs, best metric, early-stop patience, and historical per-epoch durations."""
    if not os.path.isfile(metrics_csv):
        return None
    rows: List[Dict[str, str]] = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None

    def _f(x: Optional[str]) -> Optional[float]:
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    last_epoch = int(rows[-1]["epoch"])          # 1-indexed: total number of completed epochs
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch0 = 0                               # 0-indexed best epoch
    for r in rows:
        e = int(r["epoch"])
        va = _f(r.get("val_auc"))
        vl = _f(r.get("val_loss"))
        if va is not None and va > best_val_auc:
            best_val_auc = va
            if metric == "auc":
                best_epoch0 = e - 1
        if vl is not None and vl < best_val_loss:
            best_val_loss = vl
            if metric != "auc":
                best_epoch0 = e - 1
    patience = max(last_epoch - (best_epoch0 + 1), 0)
    epoch_seconds = [(_f(r.get("epoch_seconds")) or 0.0) for r in rows]
    return dict(
        completed_epochs=last_epoch,
        best_val_auc=best_val_auc,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch0,
        patience_counter=patience,
        epoch_seconds=epoch_seconds,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _is_main(rank: int) -> bool:
    return rank == 0


def _all_reduce_mean(value: float, world_size: int, device: torch.device) -> float:
    """Average across ranks (does not affect gradients, pure metric aggregation)."""
    if world_size <= 1 or not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / float(world_size)


class _LightningDDPWrapper(nn.Module):
    """Lets DeepSpeed's forward correctly proxy to the LightningModule's step.

    Like DDP, DeepSpeed relies on forward actually passing through the wrapper to trigger
    backward-pass synchronization. Wrapping chain:
    DeepSpeedEngine -> _LightningDDPWrapper.forward -> lit.training_step.
    """

    def __init__(self, lit_module: nn.Module) -> None:
        super().__init__()
        self.lit_module = lit_module

    def forward(self, batch: Dict[str, Any], batch_idx: int, split: str) -> torch.Tensor:
        if split == "train":
            return self.lit_module.training_step(batch, batch_idx)
        if split == "val":
            return self.lit_module.validation_step(batch, batch_idx)
        raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# Phase Timer: quantifies time spent in the LM / GNN / Backward+Step phases using CUDA Events
# ---------------------------------------------------------------------------


class _PhaseTimer:
    """Captures the PCIe-blocking vs. compute time breakdown for the B4 paper.

    Instruments before/after the LM / GNN submodules via forward hooks, and explicitly records
    the backward+step segment.
    All events are recorded on the default stream; commit() does a full-device synchronize and
    then computes the elapsed-time deltas (in milliseconds).
    """

    # Candidate submodule names: covers the common naming conventions used by graphNAS / clip-graph / GASSO in src/clip_graph/
    LM_NAME_CANDIDATES = ("lm", "text_encoder", "text", "language_model", "lm_encoder", "text_tower")
    GNN_NAME_CANDIDATES = ("gnn", "graph_encoder", "graph", "gnn_encoder", "graph_tower", "gasso")

    def __init__(self, lit_module: nn.Module, *, verbose: bool = True) -> None:
        self._lm_starts: List[torch.cuda.Event] = []
        self._lm_ends: List[torch.cuda.Event] = []
        self._gnn_starts: List[torch.cuda.Event] = []
        self._gnn_ends: List[torch.cuda.Event] = []
        self._bw_starts: List[torch.cuda.Event] = []
        self._bw_ends: List[torch.cuda.Event] = []
        self.times: Dict[str, List[float]] = defaultdict(list)
        # Cumulative effective GPU compute time over the whole run (ms): sum of lm+gnn+bw per step,
        # not cleared by reset().
        # Dividing by 1000 at the end of training gives this rank's GPU-active seconds (used by test3).
        self.run_total_ms = 0.0
        self.lm_attached = False
        self.gnn_attached = False
        self._attach_hooks(lit_module, verbose=verbose)

    @staticmethod
    def _find(module: nn.Module, candidates) -> Optional[nn.Module]:
        for name in candidates:
            sub = getattr(module, name, None)
            if isinstance(sub, nn.Module):
                return sub
        return None

    def _attach_hooks(self, lit_module: nn.Module, *, verbose: bool) -> None:
        # Look on lit_module itself first; if not found, go one level deeper into lit_module.model
        lm_module = self._find(lit_module, self.LM_NAME_CANDIDATES)
        gnn_module = self._find(lit_module, self.GNN_NAME_CANDIDATES)
        inner = getattr(lit_module, "model", None)
        if isinstance(inner, nn.Module):
            if lm_module is None:
                lm_module = self._find(inner, self.LM_NAME_CANDIDATES)
            if gnn_module is None:
                gnn_module = self._find(inner, self.GNN_NAME_CANDIDATES)

        if lm_module is not None:
            lm_module.register_forward_pre_hook(self._lm_pre)
            lm_module.register_forward_hook(self._lm_post)
            self.lm_attached = True
            if verbose:
                print(f"[PhaseTimer] attached LM hook -> {type(lm_module).__name__}")
        elif verbose:
            print("[PhaseTimer] WARN: LM submodule not found; LM phase timing disabled")

        if gnn_module is not None:
            gnn_module.register_forward_pre_hook(self._gnn_pre)
            gnn_module.register_forward_hook(self._gnn_post)
            self.gnn_attached = True
            if verbose:
                print(f"[PhaseTimer] attached GNN hook -> {type(gnn_module).__name__}")
        elif verbose:
            print("[PhaseTimer] WARN: GNN submodule not found; GNN phase timing disabled")

    @staticmethod
    def _make_event() -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    # Submodule hooks
    def _lm_pre(self, module, inputs):  # noqa: ANN001
        self._lm_starts.append(self._make_event())

    def _lm_post(self, module, inputs, output):  # noqa: ANN001
        self._lm_ends.append(self._make_event())

    def _gnn_pre(self, module, inputs):  # noqa: ANN001
        self._gnn_starts.append(self._make_event())

    def _gnn_post(self, module, inputs, output):  # noqa: ANN001
        self._gnn_ends.append(self._make_event())

    # Explicit backward+step interval
    def mark_bw_start(self) -> None:
        self._bw_starts.append(self._make_event())

    def mark_bw_end(self) -> None:
        self._bw_ends.append(self._make_event())

    def commit(self) -> None:
        """Pair up this step's events, bucket their deltas, and clear the buffers."""
        torch.cuda.synchronize()

        def _accumulate(starts, ends, key):
            n = min(len(starts), len(ends))
            if n == 0:
                return 0.0
            total = 0.0
            for i in range(n):
                try:
                    total += starts[i].elapsed_time(ends[i])
                except Exception:
                    pass
            self.times[key].append(total)
            return total

        step_ms = (
            _accumulate(self._lm_starts, self._lm_ends, "lm")
            + _accumulate(self._gnn_starts, self._gnn_ends, "gnn")
            + _accumulate(self._bw_starts, self._bw_ends, "bw_step")
        )
        # Cumulative over the whole run (not cleared by reset) = this rank's effective GPU compute time
        self.run_total_ms += step_ms
        self._lm_starts.clear(); self._lm_ends.clear()
        self._gnn_starts.clear(); self._gnn_ends.clear()
        self._bw_starts.clear(); self._bw_ends.clear()

    def epoch_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, vals in self.times.items():
            out[key] = (sum(vals) / len(vals)) if vals else 0.0
        return out

    def reset(self) -> None:
        self.times.clear()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GraphModelTrainer:
    def __init__(self, args: argparse.Namespace, config: TrainingConfig) -> None:
        # DeepSpeed has already initialized the process group via the launcher + deepspeed.init_distributed
        self.rank = int(args.local_rank)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.config = config
        self.device = torch.device(f"cuda:{self.rank}")
        self.epoch = 0
        self.global_step = 0
        # Throughput accounting: total number of "seed nodes" processed by this rank (accumulated
        # across each step's batch of text nodes).
        # After training ends, all_reduce(SUM) across ranks gives the system-wide total -> Nodes/s = total_nodes / total_time.
        self.nodes_processed = 0
        # max-runtime cap: stop as soon as the deadline is reached (test3's unified 500s budget).
        self._deadline: Optional[float] = None
        self._time_up = False

        # Early stopping (rank 0 decides, broadcasts to the other ranks)
        self.best_val_loss = float("inf")
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = 0
        # Cache for val-AUC early stopping: the val subgraph doesn't change across epochs, so the
        # tensors + negative edges are only built/sampled once.
        self._val_auc_cache: Optional[tuple] = None

        if _is_main(self.rank):
            print(f"[rank{self.rank}] Initializing DeepSpeed B4 trainer on {self.device}")
            print(
                f"[rank{self.rank}] Model: gasso, Dataset: {config.dataset}, "
                f"world_size={self.world_size}, ds_config={config.deepspeed_config}"
            )
            print("_" * 100)

        torch.cuda.set_device(self.rank)

        # 1. Load the YAML
        self.model_config = OmegaConf.load(config.config_path)
        if config.num_layers is not None:
            # CLI/shell override: write it back into model.init_args.gnn_params.num_layers in the YAML,
            # so that when _setup_model later builds the GNN from model_args, it also sees this value.
            self.model_config.model.init_args.gnn_params.num_layers = int(config.num_layers)
            if _is_main(self.rank):
                print(f"[override] gnn_params.num_layers = {config.num_layers} (= k_hop)")
        try:
            self.k_hop = int(self.model_config.model.init_args.gnn_params.num_layers)
        except Exception:
            self.k_hop = 1

        # 2. Data
        self._setup_data()

        # 3. Model + DeepSpeed engine (also obtains the optimizer)
        self._setup_engine(args)

        # 4. Phase timer (the forward hook does not affect numerical results)
        self.phase_timer = _PhaseTimer(self._lit_module(), verbose=_is_main(self.rank))
        # test6 time-breakdown: unified telemetry (coexists with phase_timer, only enabled with --profile).
        # ZeRO's gradient sync happens inside engine.backward, and the optimizer step inside engine.step
        # (CPU offload); injecting a barrier would double-count and disturb the offload pipeline, hence
        # emit_barrier_sync_wait=False, with load imbalance attributed to Backward.
        self.profiler: Optional[TelemetryRecorder] = None
        if getattr(config, "profile", False):
            _legacy_out = config.profile_out or os.path.join(
                "results", "test6", f"breakdown_gnas_zero_{config.dataset}.json"
            )
            _jsonl = getattr(config, "telemetry_jsonl", None) or derive_jsonl_path(
                config.profile_out, system="gnas_zero", dataset=config.dataset,
                seed=getattr(config, "seed", 42), worker="single",
            )
            self.profiler = TelemetryRecorder(
                system="gnas_zero",
                dataset=config.dataset,
                seed=getattr(config, "seed", 42),
                worker="single",
                device=self.device,
                jsonl_path=_jsonl,
                lit_module=self._lit_module(),
                rank=self.rank,
                enabled=True,
                emit_barrier_sync_wait=False,
                legacy_json_path=_legacy_out,
                verbose=_is_main(self.rank),
            )

    # ---------- Submodule setup ----------

    def _setup_data(self) -> None:
        if _is_main(self.rank):
            print("Setting up data...")
        init_args = dict(self.model_config.data.init_args)
        data_dir = os.path.join(_DATA_ROOT, self.config.dataset)
        init_args["data_dir"] = data_dir
        if self.config.batch_size is not None:
            init_args["batch_size"] = int(self.config.batch_size)

        # GASSO uses square_contrastive_loss: CE([1x1 logit], target=[0]) is identically 0,
        # which zeroes out loss/grad/alpha entirely and makes the search completely ineffective.
        # This was the root cause of the all-zero loss in the previous gasso_pubmed_b4_zero_20260517
        # run. We hard-fail here to avoid hitting this pitfall again.
        effective_bs = int(init_args.get("batch_size", 0) or 0)
        if effective_bs <= 1:
            raise ValueError(
                f"GASSO contrastive training requires batch_size >= 2; the current batch_size={effective_bs} "
                f"would make square_contrastive_loss identically 0, zeroing out loss/grad/alpha and "
                f"disabling the search. "
                f"Please override via --batch-size N (N>=2) or BATCH_SIZE=N bash run_deepspeed.sh."
            )
        if _is_main(self.rank):
            print(f"[data] effective batch_size = {effective_bs}")

        raw_dir = os.path.join(data_dir, "raw")
        missing = [fn for fn in _REQUIRED_FILES if not os.path.isfile(os.path.join(raw_dir, fn))]
        if missing:
            available = _discover_datasets()
            raise ValueError(
                f"data/{self.config.dataset}/raw is missing file(s) {missing}. "
                f"Currently available datasets: {available}"
            )

        datamodule = GassoPubmedGraphTextDataModule(**init_args)

        # prepare_data is expected to be idempotent; no extra dist.barrier needed
        datamodule.prepare_data()
        # Consistent with bin/ddp_train.py: every rank gets the full graph, and load is split across
        # ranks via DistributedSampler
        datamodule.setup(node=0, world_size=1, stage="fit", k_hop=self.k_hop)

        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset
        # Keep a reference to val_dataset so val-AUC early stopping can fetch the split-induced graph_data on rank 0.
        self.val_dataset = val_dataset
        # Reference to the test split (when val_eval_on_test=True, the periodic sampling instead runs link prediction on test).
        self.test_dataset = getattr(datamodule, "test_dataset", None)

        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=self.train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.model_config.data.init_args.pin_memory,
            collate_fn=train_dataset.__collate__,
        )

        if val_dataset:
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                sampler=self.val_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.model_config.data.init_args.pin_memory,
                collate_fn=val_dataset.__collate__,
            )
        else:
            self.val_sampler = None
            self.val_dataloader = None
            if _is_main(self.rank):
                print("Warning: No validation dataset found. Early stopping disabled.")

    def _setup_engine(self, args: argparse.Namespace) -> None:
        if _is_main(self.rank):
            print("Setting up DeepSpeed engine (ZeRO-Offload)...")
        model_args = self.model_config.model.init_args.copy()
        model_args["lr"] = self.config.learning_rate
        lit_model = LitGassoClipGraph(**model_args)
        wrapped = _LightningDDPWrapper(lit_model)

        # Inject the CLI lr into ds_config's optimizer.params.lr, so users don't forget to keep them in sync
        with open(self.config.deepspeed_config, "r", encoding="utf-8") as f:
            ds_cfg = json.load(f)
        if "optimizer" in ds_cfg and "params" in ds_cfg["optimizer"]:
            ds_cfg["optimizer"]["params"]["lr"] = float(self.config.learning_rate)
        # Record the ZeRO stage: under stage3 (B4), parameters are sharded, so a few places that
        # "bypass model_engine and touch parameters directly" (the val-AUC manual forward pass, and
        # the best-snapshot state_dict) would get empty/sharded tensors and need to be gated by stage.
        self.zero_stage = int(ds_cfg.get("zero_optimization", {}).get("stage", 2))

        # DeepSpeed does not allow args.deepspeed_config and config= to both be set --
        # we want to go through config=ds_cfg (so the CLI lr can be injected), so clear args.deepspeed_config
        args.deepspeed_config = None

        # Classic DARTS approach: put alpha (alphas_normal/norm/act) in its own param group with no
        # weight_decay, to avoid AdamW's weight_decay=0.01 pushing the just-learned alpha back toward 0;
        # alpha is still an nn.ParameterList and still participates in backward together with the
        # weight params, taken over by DeepSpeed Stage 2.
        arch_param_names: set[str] = set()
        for n, _ in wrapped.named_parameters():
            if ".alphas_normal." in n or ".alphas_norm." in n or ".alphas_act." in n:
                arch_param_names.add(n)
        arch_params: list = []
        weight_params: list = []
        for n, p in wrapped.named_parameters():
            (arch_params if n in arch_param_names else weight_params).append(p)
        if _is_main(self.rank):
            print(f"[opt] arch params (α, no weight_decay): {len(arch_params)} tensors")
            print(f"[opt] weight params (with weight_decay): {len(weight_params)} tensors")
        param_groups = [
            {"params": weight_params},  # uses ds_cfg.optimizer's default weight_decay
            {"params": arch_params, "weight_decay": 0.0},
        ]

        # deepspeed.initialize takes over: process group + model wrapping + optimizer construction + ZeRO configuration
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=wrapped,
            model_parameters=param_groups,
            config=ds_cfg,
        )

        if _is_main(self.rank):
            print(
                f"[deepspeed] ZeRO stage={ds_cfg.get('zero_optimization', {}).get('stage', '?')}"
                f" offload_optimizer={ds_cfg.get('zero_optimization', {}).get('offload_optimizer', '?')}"
                f" fp16.enabled={ds_cfg.get('fp16', {}).get('enabled', False)}"
                f" bf16.enabled={ds_cfg.get('bf16', {}).get('enabled', False)}"
            )

    # ---------- helpers ----------

    def _lit_module(self) -> nn.Module:
        return self.model_engine.module.lit_module

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    # ---------- train / val ----------

    def train_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float], float]:
        self.model_engine.train()
        self.epoch = epoch
        # Time anchor for periodic dense val-AUC sampling: anchored as the "training start point" when the first epoch begins.
        if self.config.val_eval_interval_seconds and self.config.val_eval_interval_seconds > 0:
            if getattr(self, "_intra_val_start", None) is None:
                self._intra_val_start = time.time()
                self._intra_val_last = time.time()
                self._intra_val_csv = None
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        # Peak memory needed for the B4 paper: reset once per epoch to look only at this epoch's high-water mark
        torch.cuda.reset_peak_memory_stats(self.device)

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        self.phase_timer.reset()

        # Text subsampling: this epoch only runs the first _keep batches (reshuffled each epoch -> different random subset).
        _n_batches = len(self.train_dataloader)
        _keep = _n_batches
        if self.config.text_subsample_fraction is not None:
            _frac = float(self.config.text_subsample_fraction)
            if not (0.0 < _frac <= 1.0):
                raise ValueError(f"text_subsample_fraction must be in (0,1], got {_frac}")
            _keep = max(1, math.ceil(_n_batches * _frac))

        if _is_main(self.rank):
            iterator = tqdm(
                enumerate(self.train_dataloader),
                total=_keep,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                ncols=120,
            )
        else:
            iterator = enumerate(self.train_dataloader)

        if self.profiler is not None:
            self.profiler.epoch_start()
        for batch_idx, batch in iterator:
            if batch_idx >= _keep:
                break
            if self.profiler is not None:
                self.profiler.iter_begin()
                self.profiler.xfer_start()
            batch = self._move_batch_to_device(batch)
            if self.profiler is not None:
                self.profiler.xfer_end()
            # Throughput: number of seed nodes fed in this step (already deduplicated by text/node, at most batch_size)
            _n_nodes = int(batch["text_node_ids"].shape[0])
            self.nodes_processed += _n_nodes
            if self.profiler is not None:
                self.profiler.set_iter_meta(nodes_in_batch=_n_nodes)

            # forward triggers the LM/GNN submodule hooks (automatically captured inside PhaseTimer)
            loss = self.model_engine(batch, batch_idx, "train")

            # backward + step: this is where the bulk of ZeRO-Offload paging communication happens
            self.phase_timer.mark_bw_start()
            if self.profiler is not None:
                self.profiler.bwd_start()
            self.model_engine.backward(loss)
            if self.profiler is not None:
                self.profiler.bwd_end()
                self.profiler.sync_wait()   # timed barrier between backward and step = synchronization wait
                self.profiler.upd_start()
            self.model_engine.step()
            if self.profiler is not None:
                self.profiler.upd_end()
            self.phase_timer.mark_bw_end()

            self.phase_timer.commit()

            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            self.global_step += 1

            if _is_main(self.rank) and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "Loss": f"{loss_val:.4f}",
                        "Avg": f"{total_loss / num_batches:.4f}",
                    }
                )

            if self.profiler is not None:
                self.profiler.commit()

            # Periodic dense val-AUC sampling (for the time-to-accuracy curve): rank0 decides based
            # on wall-clock and broadcasts; only rank0 actually runs _compute_val_auc() (returns None
            # under ZeRO-3, so nothing is written); a final barrier resets state to avoid a collective deadlock.
            if self.config.val_eval_interval_seconds and self.config.val_eval_interval_seconds > 0:
                _do_eval = 0
                if _is_main(self.rank) and (
                    time.time() - self._intra_val_last
                ) >= self.config.val_eval_interval_seconds:
                    _do_eval = 1
                if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                    _flag = torch.tensor([_do_eval], device=self.device, dtype=torch.int64)
                    dist.broadcast(_flag, src=0)
                    _do_eval = int(_flag.item())
                if _do_eval:
                    _t_ev0 = time.time()      # training is paused from here until the barrier below
                    _va = self._compute_val_auc()
                    if _is_main(self.rank):
                        self._log_val_timeline(time.time() - self._intra_val_start, epoch, _va,
                                               eval_seconds=time.time() - _t_ev0)
                        self._intra_val_last = time.time()
                    self.model_engine.train()
                    if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                        dist.barrier()

            # max-runtime cap: decision is synchronized across ranks (otherwise ranks breaking at
            # different steps would deadlock).
            # Only done when a deadline is set (zero overhead for normal training).
            if self._deadline is not None:
                local_up = 1.0 if time.time() >= self._deadline else 0.0
                flag = torch.tensor([local_up], device=self.device)
                if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item() > 0:
                    self._time_up = True
                    break

        local_avg = total_loss / num_batches if num_batches > 0 else 0.0
        global_avg = _all_reduce_mean(local_avg, self.world_size, self.device)
        epoch_time = time.time() - epoch_start
        phase_stats = self.phase_timer.epoch_summary()
        peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        return global_avg, epoch_time, phase_stats, peak_mem_gb

    def validate(self) -> Optional[float]:
        if not self.val_dataloader:
            return None
        self.model_engine.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                loss = self.model_engine(batch, batch_idx, "val")
                total_loss += loss.item()
                num_batches += 1

        local_avg = total_loss / num_batches if num_batches > 0 else 0.0
        return _all_reduce_mean(local_avg, self.world_size, self.device)

    def _build_val_auc_cache(self) -> Optional[tuple]:
        """Build and cache the val subgraph tensors plus one-time sampled negative edges (val subgraph is unchanged across epochs).

        Returns (vx, vei, vnei, edge_attr) or None (no val / subgraph has no edges). Only called on rank 0.
        """
        from clip_graph.eval_neg_sampling import sample_negative_edges

        eval_dataset = (getattr(self, "test_dataset", None) if getattr(self.config, "val_eval_on_test", False)
                        else getattr(self, "val_dataset", None))
        if eval_dataset is None:
            return None
        graph_data = eval_dataset.dataset.graph_data
        vx = graph_data.x.float().to(self.device)
        vei = graph_data.edge_index.to(self.device)
        if vei.size(1) == 0:
            if _is_main(self.rank):
                print("[val-auc] val subgraph has no edges, cannot compute AUC, falling back to loss-based early stopping")
            return None
        edge_attr = getattr(graph_data, "edge_attr", None)
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros(vei.size(1), 1, dtype=vx.dtype)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = edge_attr.float().to(self.device)

        _NEG_SEED = 2969591811
        torch.manual_seed(_NEG_SEED)
        vnei, neg_info = sample_negative_edges(
            pos_edge_index=vei,
            num_nodes=vx.size(0),
            graph_node_ids=getattr(graph_data, "node_ids", None),
            dataset_name=self.config.dataset,
            data_dir=os.path.join(_DATA_ROOT, self.config.dataset),
            mode="auto",
            seed=_NEG_SEED,
            device=self.device,
        )
        if _is_main(self.rank):
            print(
                f"[val-auc] cache built: nodes={vx.size(0)} pos={vei.size(1)} "
                f"neg={vnei.size(1)} neg_mode={neg_info.actual_mode}"
            )
        return vx, vei, vnei, edge_attr

    def _compute_val_auc(self, collective: bool = True) -> Optional[float]:
        """On rank 0, run one greedy(is_search=False) full-graph forward pass over the val subgraph to compute link-prediction AUC.

        Non-rank-0 processes return None immediately (the other ranks rejoin at the main loop's
        dist.broadcast and never enter this section, avoiding a deadlock with ZeRO communication).
        Returns None on failure.
        """
        # Under ZeRO-3 (stage>=3) the parameters are sharded and every submodule carries DeepSpeed's
        # fetch/release hooks, which are collectives: a forward run on rank 0 alone desynchronises the
        # ranks' parameter coordinators and the next training step deadlocks (observed). So under ZeRO-3
        # EVERY rank runs the same evaluation forward (the main loop brings all ranks here together; the
        # eval graph and its negatives are built identically on each rank), the hooks gather the weights
        # as in training, and only rank 0 computes the AUC. ZeRO-1/2 keep full weights: rank 0 alone.
        if getattr(self, "zero_stage", 2) >= 3:
            if not collective:
                # Called on rank 0 only (epoch-end early-stop path): the other ranks are not here. Keep
                # the old behaviour -- no AUC, the caller falls back to loss.
                return None
            return self._compute_val_auc_rank0(all_ranks=True)
        if not _is_main(self.rank):
            return None
        return self._compute_val_auc_rank0()

    def _compute_val_auc_rank0(self, all_ranks: bool = False) -> Optional[float]:
        """Evaluation body: the forward runs on this rank; the AUC is computed on rank 0 only.
        all_ranks=True (ZeRO-3): every rank calls this so the parameter hooks' collectives line up."""
        import torch.nn.functional as F
        import sklearn.metrics as mt
        import torch_geometric as pyg

        if self._val_auc_cache is None:
            self._val_auc_cache = self._build_val_auc_cache()
        if self._val_auc_cache is None:
            return None
        vx, vei, vnei, edge_attr = self._val_auc_cache

        lit = self._lit_module()
        gnn = lit.model.gnn
        gnn_proj = lit.model.gnn_proj
        self.model_engine.eval()
        try:
            layers = gnn.num_layers
            gnn.adjs = (
                vei.unsqueeze(0).repeat(layers, 1, 1),
                edge_attr.unsqueeze(0).repeat(layers, 1, 1),
            )
            gnn.step = 1
            with torch.no_grad():
                out = gnn(vx, vei, edge_attr, is_search=False)  # greedy deployed architecture
                h = out["output"] if out["output"] is not None else out["last_hidden_state"]
                z = F.normalize(gnn_proj(h), p=2, dim=1)
                decoder = pyg.nn.models.autoencoder.InnerProductDecoder()
                if all_ranks and not _is_main(self.rank):
                    return None
                pos_dec = decoder(z, vei, sigmoid=True)
                neg_dec = decoder(z, vnei, sigmoid=True)
                pred = torch.cat([pos_dec, neg_dec], dim=0).detach().cpu().numpy()
                y = torch.cat(
                    [z.new_ones(vei.size(1)), z.new_zeros(vnei.size(1))], dim=0
                ).long().detach().cpu().numpy()
            return float(mt.roc_auc_score(y, pred))
        except Exception as exc:  # noqa: BLE001
            print(f"[val-auc] computation failed, falling back to loss-based early stopping: {exc}")
            return None

    def _log_val_timeline(self, elapsed_seconds: float, epoch: int,
                          val_auc: Optional[float], eval_seconds: Optional[float] = None) -> None:
        """Append one (elapsed, epoch, step, val_auc) row to run_dir/val_timeline.csv (rank0 only).

        This is the dense sampling data source for the time-to-accuracy curve; it is independent
        of metrics.csv (per-epoch) and does not participate in early stopping.
        Nothing is written when val_auc is None (ZeRO-3 / no val edges, etc.).
        """
        if not _is_main(self.rank) or self.config.run_dir is None or val_auc is None:
            return
        if getattr(self, "_intra_val_csv", None) is None:
            self._intra_val_csv = os.path.join(self.config.run_dir, "val_timeline.csv")
            with open(self._intra_val_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["elapsed_seconds", "epoch", "global_step", "val_auc", "timestamp", "eval_seconds"]
                )
        with open(self._intra_val_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [f"{elapsed_seconds:.3f}", epoch + 1, self.global_step,
                 f"{val_auc:.6f}", datetime.now().isoformat(),
                 "" if eval_seconds is None else f"{eval_seconds:.3f}"]
            )

    # ---------- checkpoint ----------

    def save_checkpoint(self, ckpt_dir: str, tag: str) -> None:
        # DeepSpeed handles rank synchronization + ZeRO parameter gathering internally; no manual if rank == 0 needed
        self.model_engine.save_checkpoint(ckpt_dir, tag=tag)
        if _is_main(self.rank):
            print(f"DeepSpeed checkpoint saved -> {os.path.join(ckpt_dir, tag)}")

    def snapshot_best_state(self) -> None:
        """On rank 0, copy the current lit module state to CPU as the best candidate.

        Under Stage 2, lit_module.state_dict() is complete; under Stage 3 it is only sharded
        (an incomplete state is obtained).
        ZeRO-3 (stage>=3) skips the snapshot entirely: the best_model.pth save block already has
        an `is not None` guard that will skip it automatically; test2 only reads training_time.json
        for throughput and doesn't need a best ckpt (getting the full weights requires zero_to_fp32.py separately).
        """
        if getattr(self, "zero_stage", 2) >= 3:
            return
        if not _is_main(self.rank):
            return
        self.best_model_state = {
            k: v.detach().cpu().clone() for k, v in self._lit_module().state_dict().items()
        }


# ---------------------------------------------------------------------------
# Single-dataset training entry point
# ---------------------------------------------------------------------------


def run_one_dataset(args: argparse.Namespace, config: TrainingConfig) -> None:
    rank = int(args.local_rank)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Resume: reuse the given run_dir (don't create a new timestamped directory), continuing from its checkpoints/.
    resume_run_dir = getattr(args, "resume_run_dir", None)
    resuming = bool(resume_run_dir)
    if resuming:
        config.run_dir = os.path.abspath(os.path.expanduser(resume_run_dir))
        os.environ["TAGNAS_RUN_DIR"] = config.run_dir

    # A launcher-provided ID lets every rank independently resolve the same path.
    if config.run_dir is None:
        if _is_main(rank):
            run_dir = _make_run_dir(config)
            os.environ["TAGNAS_RUN_DIR"] = run_dir
            config.run_dir = run_dir
        else:
            config.run_dir = os.environ.get("TAGNAS_RUN_DIR")
            if config.run_dir is None:
                # The inherited ID is shared; mutations of rank 0's environment are not.
                config.run_dir = _make_run_dir(config)

    run_dir = config.run_dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    train_log = os.path.join(run_dir, "training.log")

    if _is_main(rank):
        if resuming:
            # Resuming does not rewrite artifacts -- otherwise metrics.csv would be truncated by
            # "w" mode and lose the already-trained epochs.
            # Only append a resume separator marker to training.log.
            with open(train_log, "a", encoding="utf-8") as f:
                f.write(f"\n# ===== RESUMED at {datetime.now().isoformat()} =====\n")
            print(f"[resume] Reusing run_dir (not rewriting artifacts): {run_dir}")
        else:
            _write_training_artifacts(run_dir, config, OmegaConf.load(config.config_path))
            print(f"Run directory: {run_dir}")

    trainer = GraphModelTrainer(args, config)

    train_total_start = time.time()
    if config.max_runtime_seconds is not None:
        trainer._deadline = train_total_start + float(config.max_runtime_seconds)
    epoch_times: list[float] = []
    epochs_completed = 0
    early_stopped = False
    best_checkpoint_seconds = None
    best_selection_seconds = None

    # Resume: once the engine is built, restore model / optimizer state from the checkpoint, and
    # restore the best-metric / early-stop counters.
    start_epoch = 0
    if resuming:
        resume_tag = getattr(args, "resume_tag", None) or _latest_epoch_tag(ckpt_dir)
        if resume_tag is None:
            raise SystemExit(f"[resume] No epoch_* checkpoint found under {ckpt_dir}")
        load_path, _client = trainer.model_engine.load_checkpoint(
            ckpt_dir, tag=resume_tag,
            load_optimizer_states=True, load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise SystemExit(f"[resume] load_checkpoint failed: {ckpt_dir}/{resume_tag}")

        m = re.match(r"^epoch_(\d+)$", resume_tag)
        st = _read_resume_state(metrics_csv, config.early_stop_metric) or {}
        # tag=epoch_N means N epochs have already been completed -> continue from the 0-indexed Nth epoch.
        start_epoch = int(m.group(1)) if m else int(st.get("completed_epochs", 0))
        trainer.best_val_auc = st.get("best_val_auc", 0.0)
        trainer.best_val_loss = st.get("best_val_loss", float("inf"))
        trainer.best_epoch = st.get("best_epoch", 0)
        trainer.patience_counter = st.get("patience_counter", 0)
        # Push the historical per-epoch durations into epoch_times, so the final avg/total stats reflect the whole run.
        epoch_times.extend(st.get("epoch_seconds", [])[:start_epoch])
        # If the resume point happens to be exactly the best epoch (e.g. resuming from latest when
        # latest is also best), use the currently loaded weights as the best snapshot, so
        # best_model.pth can still be written at the end.
        if start_epoch == trainer.best_epoch + 1:
            trainer.snapshot_best_state()
        if _is_main(rank):
            print(
                f"[resume] Loaded {resume_tag} -> start_epoch={start_epoch} "
                f"best_val_auc={trainer.best_val_auc:.4f} best_epoch={trainer.best_epoch + 1} "
                f"patience={trainer.patience_counter}/{config.early_stopping_patience}"
            )

    has_val = trainer.val_dataloader is not None

    if not has_val and _is_main(rank):
        print(
            f"Warning: No validation dataset. Training without early stopping. "
            f"DeepSpeed B4 launching for {config.num_epochs} epochs on {trainer.world_size} GPUs..."
        )
    elif _is_main(rank):
        print(
            f"DeepSpeed B4 starting for {config.num_epochs} epochs on {trainer.world_size} GPUs "
            f"(early stopping patience={config.early_stopping_patience})..."
        )

    for epoch in range(start_epoch, config.num_epochs):
        train_loss, epoch_time, phase_stats, peak_mem_gb = trainer.train_epoch(epoch)
        epoch_times.append(epoch_time)
        epochs_completed = epoch + 1

        if _is_main(rank):
            print(
                f"Epoch {epoch + 1} train_loss={train_loss:.4f} time={epoch_time:.2f}s "
                f"| LM={phase_stats.get('lm', 0.0):.2f}ms "
                f"GNN={phase_stats.get('gnn', 0.0):.2f}ms "
                f"BW+Step={phase_stats.get('bw_step', 0.0):.2f}ms "
                f"| Peak GPU Mem={peak_mem_gb:.2f} GB"
            )

        # max-runtime: stop immediately once reached (before val/ckpt), so total_seconds stays close to the budget.
        # _time_up is already synchronized across ranks inside train_epoch, so all ranks break together and don't block val's all_reduce.
        if trainer._time_up:
            if _is_main(rank):
                print(f"[max-runtime] Reached the {config.max_runtime_seconds}s budget, stopping training")
            break

        ds = trainer.train_dataloader.dataset
        if hasattr(ds, "next_epoch"):
            ds.next_epoch()

        val_loss: Optional[float] = None
        val_auc: Optional[float] = None
        if has_val:
            val_loss = trainer.validate()
            if _is_main(rank):
                print(f"Epoch {epoch + 1} val_loss={val_loss:.4f}")
                # Choice of monitoring metric: auc = val link-prediction AUC (greedy, higher is better);
                # loss = val contrastive loss (lower is better).
                # In auc mode, only rank 0 computes the full-graph forward pass; if it can't be
                # computed, this epoch falls back to using loss.
                metric = config.early_stop_metric
                if metric == "auc":
                    val_auc = trainer._compute_val_auc(collective=False)   # rank 0 only here
                use_auc = metric == "auc" and val_auc is not None
                if metric == "auc" and val_auc is None:
                    print("[val-auc] No AUC for this epoch, falling back to val_loss for early stopping")

                # Add an epsilon to the improvement check to avoid spurious patience++ when metrics are exactly equal
                _IMPROVE_EPS = 1e-6
                if use_auc:
                    improved = val_auc > trainer.best_val_auc + _IMPROVE_EPS
                    print(f"Epoch {epoch + 1} val_auc={val_auc:.4f}")
                else:
                    improved = val_loss < trainer.best_val_loss - _IMPROVE_EPS

                if improved:
                    best_selection_seconds = time.time() - train_total_start
                    if use_auc:
                        print(f"✓ Validation AUC improved {trainer.best_val_auc:.4f} -> {val_auc:.4f}")
                        trainer.best_val_auc = val_auc
                    else:
                        print(f"✓ Validation improved {trainer.best_val_loss:.4f} -> {val_loss:.4f}")
                        trainer.best_val_loss = val_loss
                    trainer.best_epoch = epoch
                    trainer.patience_counter = 0
                    trainer.snapshot_best_state()
                    if trainer.best_model_state is not None:
                        best_checkpoint_seconds = time.time() - train_total_start
                else:
                    trainer.patience_counter += 1
                    _what = "AUC" if use_auc else "loss"
                    print(
                        f"✗ No {_what} improvement. Patience: "
                        f"{trainer.patience_counter}/{config.early_stopping_patience}"
                    )

        if _is_main(rank):
            _append_metric_row(
                metrics_csv, train_log, epoch, train_loss, val_loss,
                trainer.best_val_loss, epoch_time, phase_stats, peak_mem_gb,
                val_auc=val_auc, best_val_auc=trainer.best_val_auc,
            )

        # Early-stop broadcast: rank 0 decides, other ranks exit in sync (not a barrier, a control-flow broadcast)
        if has_val:
            stop_signal = torch.tensor(
                [
                    1
                    if _is_main(rank) and trainer.patience_counter >= config.early_stopping_patience
                    else 0
                ],
                device=trainer.device,
                dtype=torch.int64,
            )
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                if _is_main(rank):
                    print(f"\nEarly stopping at epoch {epoch + 1}.")
                early_stopped = True
                break

        if (epoch + 1) % config.checkpoint_frequency == 0:
            trainer.save_checkpoint(ckpt_dir, tag=f"epoch_{epoch + 1}")

    trainer.save_checkpoint(ckpt_dir, tag="final")
    if getattr(trainer, "zero_stage", 2) < 3 and _is_main(rank):
        # ZeRO-2 keeps complete parameters on each rank. Export the same format
        # as the val-best state so the final checkpoint is independently evaluable.
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu().clone() for k, v in trainer._lit_module().state_dict().items()
                },
                "epoch": epochs_completed - 1,
                "global_step": trainer.global_step,
                "best_val_loss": trainer.best_val_loss,
            },
            os.path.join(run_dir, "final_model.pth"),
        )
    if has_val and trainer.best_model_state is not None and _is_main(rank):
        # rank 0 writes the cached best state out as its own .pth; under Stage 3 this is only a single-rank shard, for reference only
        best_pth = os.path.join(run_dir, "best_model.pth")
        torch.save(
            {
                "model_state_dict": trainer.best_model_state,
                "epoch": trainer.best_epoch,
                "global_step": trainer.global_step,
                "best_val_loss": trainer.best_val_loss,
            },
            best_pth,
        )
        print(f"Best model (rank 0 snapshot) saved to {best_pth}")

    total_time = time.time() - train_total_start

    # Throughput: sum across ranks gives the system-wide total of seed nodes processed (every rank
    # must participate in this collective, and it must happen before the DeepSpeed process group is
    # destroyed). Nodes/s = total_nodes / total_time.
    nodes_t = torch.tensor(
        [float(trainer.nodes_processed)], dtype=torch.float64, device=trainer.device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(nodes_t, op=dist.ReduceOp.SUM)
    total_nodes = int(nodes_t.item())
    throughput = total_nodes / total_time if total_time > 0 else 0.0

    # Effective GPU compute time: PhaseTimer's cumulative total (lm+gnn+bw), summed across ranks /world_size = per-GPU average (seconds).
    gpu_t = torch.tensor(
        [float(trainer.phase_timer.run_total_ms)], dtype=torch.float64, device=trainer.device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(gpu_t, op=dist.ReduceOp.SUM)
    gpu_active_seconds = (gpu_t.item() / 1000.0) / max(trainer.world_size, 1)

    # test6 time-breakdown: per-iteration JSONL has already been written during training; here rank0 finalizes and also writes the legacy averaged format.
    if trainer.profiler is not None:
        trainer.profiler.close(extra={
            "early_stopped": early_stopped, "epochs_completed": epochs_completed,
        })

    if _is_main(rank):
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        print("=" * 60)
        print(f"[Time] Total wall time : {total_time:.2f}s ({total_time / 60:.2f} min)")
        print(f"[Time] Avg epoch time  : {avg_epoch_time:.2f}s")
        print(f"[Time] Epochs run      : {epochs_completed}")
        print(f"[Time] World size      : {trainer.world_size} GPUs")
        print(f"[Time] Early stopped   : {early_stopped}")
        print(
            f"[Throughput] nodes/s={throughput:.1f}  "
            f"(total_nodes={total_nodes}, total_time={total_time:.2f}s)"
        )
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(f"\nfinished={datetime.now().isoformat()}\n")
            f.write(f"total_training_seconds={total_time:.2f}\n")
            f.write(f"avg_epoch_seconds={avg_epoch_time:.2f}\n")
            f.write(f"epochs_completed={epochs_completed}\n")
            f.write(f"world_size={trainer.world_size}\n")
            f.write(f"early_stopped={early_stopped}\n")
            f.write(f"total_nodes_processed={total_nodes}\n")
            f.write(f"throughput_nodes_per_sec={throughput:.4f}\n")
            f.write(f"gpu_active_seconds={gpu_active_seconds:.4f}\n")
            if trainer.best_val_loss < float("inf"):
                f.write(
                    f"best_epoch={trainer.best_epoch + 1} "
                    f"best_val_loss={trainer.best_val_loss:.6f}\n"
                )
        with open(os.path.join(run_dir, "training_time.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_seconds": total_time,
                    "average_epoch_seconds": avg_epoch_time,
                    "epoch_seconds": epoch_times,
                    "epochs_completed": epochs_completed,
                    "world_size": trainer.world_size,
                    "early_stopped": early_stopped,
                    "stop_reason": "val_early_stop" if early_stopped else (
                        "max_runtime_seconds" if trainer._time_up else "epochs_completed"),
                    "train_ok": True,
                    "paper_v3_attempt": os.environ.get("PAPER_V3_ATTEMPT"),
                    "paper_v3_cell": os.environ.get("PAPER_V3_CELL"),
                    "total_nodes_processed": total_nodes,
                    "throughput_nodes_per_sec": throughput,
                    "gpu_active_seconds": gpu_active_seconds,
                    "best_epoch": trainer.best_epoch + 1
                    if trainer.best_model_state is not None
                    else None,
                    "best_checkpoint_seconds": best_checkpoint_seconds,
                    "best_selection_seconds": best_selection_seconds,
                    "best_val_auc": trainer.best_val_auc if trainer.best_model_state is not None else None,
                    "best_val_loss": trainer.best_val_loss
                    if trainer.best_val_loss < float("inf")
                    else None,
                },
                f,
                indent=2,
            )
        print(f"Training completed! Artifacts in {run_dir}")


# ---------------------------------------------------------------------------
# Post-train link-prediction evaluation (only rank 0 triggers the subprocess; the other ranks do
# not repeat it; the evaluation runs once at the end of main(), after all datasets have finished
# training, to avoid entangling it with the DeepSpeed process group lifecycle)
# ---------------------------------------------------------------------------


def _run_link_prediction_after_train(
    run_dir: str,
    dataset_name: str,
    eval_script_rel: str = "bin/eval_link_pred_ddp.py",
    split: str = "test",
    device: str = "cuda",
    negative_sampling: str = "auto",
) -> None:
    import re
    import subprocess
    import sys as _sys

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"[post-eval] run_dir does not exist: {run_dir}")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    specs: List[str] = []
    for tag, fname in [("best", "best_model.pth"), ("final", "final_model.pth")]:
        if (os.path.isfile(os.path.join(run_dir, fname))
                or os.path.isfile(os.path.join(ckpt_dir, fname))):
            specs.append(tag)
    if os.path.isdir(ckpt_dir):
        epoch_re = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
        epoch_nums = sorted(
            int(m.group(1))
            for f in os.listdir(ckpt_dir)
            for m in [epoch_re.match(f)] if m
        )
        for n in epoch_nums:
            specs.append(f"epoch_{n}")

    if not specs:
        raise FileNotFoundError(f"[post-eval] No checkpoints found under {run_dir}")

    repo_root = str(REPO_ROOT)
    eval_script_path = os.path.join(repo_root, eval_script_rel)
    if not os.path.isfile(eval_script_path):
        raise FileNotFoundError(f"[post-eval] eval script does not exist: {eval_script_path}")

    print()
    print("=" * 60)
    print(f"[post-eval] Link-prediction evaluation (dataset={dataset_name})")
    print(f"  run_dir : {run_dir}")
    print(f"  ckpts   : {specs}")
    print(f"  negative_sampling: {negative_sampling}")
    print("=" * 60)

    for spec in specs:
        print(f"\n[post-eval] checkpoint={spec}")
        cmd = [
            _sys.executable, eval_script_path,
            "--run-dir", run_dir,
            "--split", split,
            "--device", device,
            "--checkpoint", spec,
            "--negative-sampling", negative_sampling,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAGNAS B4: ZeRO-Offload GNAS trainer (DeepSpeed)")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (e.g. pubmed,goodreads-crime). "
             "If omitted, uses --datasets-all or the hardcoded default ['pubmed']",
    )
    parser.add_argument(
        "--datasets-all",
        action="store_true",
        help="Run all available datasets under data/ (in lexicographic order)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/clip-graph/inductive-causal/pubmed/gassobase.yaml",
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stop-metric", choices=["loss", "auc"], default="auc",
                        help="Convergence/early-stopping monitoring metric: auc = val link-prediction AUC (greedy) stops improving (default); "
                             "loss = val contrastive loss stops decreasing (old behavior).")
    parser.add_argument("--checkpoint-frequency", type=int, default=10)
    parser.add_argument("--output-root", type=str, default="model/runpytorch")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override gnn_params.num_layers from the YAML (= GNN k_hop / number of layers). If omitted, uses the YAML config.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override data.init_args.batch_size from the YAML. GASSO contrastive requires >= 2; "
                             "with 1, cross-entropy is identically 0 and the whole training is ineffective. If omitted, uses the YAML config.")
    parser.add_argument("--max-runtime-seconds", type=float, default=None,
                        help="Training wall-clock cap (seconds). If omitted, runs the full epoch count; used for the unified budget comparison in test3.")
    parser.add_argument("--text-subsample-fraction", type=float, default=None,
                        help="Each epoch only uses this fraction (0,1] of the text for alignment (re-randomized each epoch). If omitted, uses all text.")
    parser.add_argument("--no-post-eval", action="store_true",
                        help="Do not automatically run link-prediction evaluation after training (runs by default, only triggered on rank 0)")
    parser.add_argument("--post-eval-split", choices=["train", "val", "test"], default="test",
                        help="Split used for post-train link prediction, default test")
    parser.add_argument("--resume-run-dir", type=str, default=None,
                        help="Resume: reuse this existing run_dir (don't create a new timestamped directory), continuing from its checkpoints/. "
                             "metrics.csv / training.log are appended to rather than overwritten. Only for single-dataset resume.")
    parser.add_argument("--resume-tag", type=str, default=None,
                        help="Checkpoint tag to resume from (e.g. epoch_23). If omitted, automatically uses latest / the largest epoch_N.")
    parser.add_argument("--profile", action="store_true",
                        help="test6: enable per-iteration 7-bucket time breakdown timing, writes JSON at the end of training")
    parser.add_argument("--profile-out", type=str, default=None,
                        help="Time-breakdown JSON output path (default results/test6/breakdown_gnas_zero_<ds>.json)")
    parser.add_argument("--telemetry-jsonl", type=str, default=None,
                        help="Unified telemetry per-iteration JSONL path (default derived from --profile-out's telemetry/ subdirectory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for reproducibility, and to tag the telemetry JSONL)")
    parser.add_argument("--val-eval-interval-seconds", type=float, default=0.0,
                        help="Every N seconds, additionally compute val-AUC and write it to val_timeline.csv (dense time-to-accuracy sampling); 0=disabled")
    parser.add_argument("--val-eval-on-test", action="store_true",
                        help="Switch the periodic sampling to run link prediction on the test split instead (test1-2's honest speedup comparison uses the test-set convention)")
    # Required by DeepSpeed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Automatically injected by the DeepSpeed launcher, do not set manually",
    )
    # DeepSpeed itself registers flags like --deepspeed / --deepspeed_config / --deepscale;
    # do not manually add_argument("--deepspeed_config", ...) again, or argparse will report a conflicting option
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def _build_config(args: argparse.Namespace, dataset: str) -> TrainingConfig:
    return TrainingConfig(
        config_path=args.config_path,
        deepspeed_config=args.deepspeed_config,
        dataset=dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stop_metric=args.early_stop_metric,
        checkpoint_frequency=args.checkpoint_frequency,
        output_root=args.output_root,
        local_rank=args.local_rank,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_runtime_seconds=args.max_runtime_seconds,
        text_subsample_fraction=args.text_subsample_fraction,
        profile=getattr(args, "profile", False),
        profile_out=getattr(args, "profile_out", None),
        telemetry_jsonl=getattr(args, "telemetry_jsonl", None),
        seed=getattr(args, "seed", 42),
        val_eval_interval_seconds=getattr(args, "val_eval_interval_seconds", 0.0),
        val_eval_on_test=getattr(args, "val_eval_on_test", False),
    )


def main() -> None:
    args = parse_cli()
    if args.datasets_all:
        ds_list = _discover_datasets()
        if not ds_list:
            raise SystemExit(
                f"No available datasets found under data/ (requires raw/{{ {','.join(_REQUIRED_FILES)} }})"
            )
    elif args.datasets:
        ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        ds_list = ["pubmed"]

    if int(args.local_rank) == 0:
        print(f"[launcher] datasets to train (sequential): {ds_list}")

    trained_runs: List[Tuple[str, str]] = []  # [(run_dir, dataset_name), ...]

    for dataset_name in ds_list:
        config = _build_config(args, dataset_name)
        try:
            run_one_dataset(args, config)
            torch.cuda.empty_cache()
            if config.run_dir:
                trained_runs.append((config.run_dir, dataset_name))
        except KeyboardInterrupt:
            print("Training interrupted by user")
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    # After all datasets have finished training, rank 0 uniformly runs link-prediction evaluation
    # for each run_dir.
    # No dist.barrier is entered here -- the DeepSpeed process group is torn down/rebuilt multiple
    # times across the loop, so rank 0 only needs to evaluate serially here; the other ranks exit directly.
    if not args.no_post_eval and int(args.local_rank) == 0 and trained_runs:
        print(f"\n[post-eval] All training finished, starting link-prediction evaluation ({len(trained_runs)} run_dir(s))")
        for run_dir, ds_name in trained_runs:
            try:
                _run_link_prediction_after_train(
                    run_dir=run_dir,
                    dataset_name=ds_name,
                    eval_script_rel="bin/eval_link_pred_ddp.py",
                    split=args.post_eval_split,
                    # The live rank-0 process retains its CUDA context after training.
                    # A subprocess cannot claim that Exclusive_Process GPU. Offline
                    # evaluation uses the same saved weights and inputs on the CPU.
                    device="cpu",
                    negative_sampling=os.environ.get("LINK_NEG_SAMPLING", "auto"),
                )
            except Exception:
                import traceback
                print(f"[post-eval] Link-prediction error (dataset={ds_name}):")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
