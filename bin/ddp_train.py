"""DDP supernet baseline for joint graph/text architecture search.

Run with ``bash run_ddp.sh`` or inspect ``python bin/ddp_train.py --help``.
The launcher uses mp.spawn; each rank loads the full graph and a
DistributedSampler partitions text batches across ranks.
"""

import argparse
import math
import csv
import json
import os
import time
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

# Standalone script: only runs gasso
from clip_graph.gassolit import LitClipGraph as LitGassoClipGraph
from clip_graph.data.gassodatamodule import (
    PubmedGraphTextDataModule as GassoPubmedGraphTextDataModule,
)
from clip_graph.telemetry import TelemetryRecorder, derive_jsonl_path

# Dataset discovery: dynamically scans data/<name>/raw/; any directory containing the
# three required CSV files is treated as an available dataset.
# This way, adding a new dataset under data/ later doesn't require touching this code.
_DATA_ROOT = "data"
_REQUIRED_FILES: Tuple[str, ...] = ("graph.csv", "node-data.csv", "texts.csv")


def _discover_datasets(root: str = _DATA_ROOT) -> List[str]:
    """Scan data/<name>/raw/ and return dataset names that have the complete set of three CSV files (in lexicographic order)."""
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
    dataset: str = "pubmed"
    num_workers: int = 0
    num_epochs: int = 10
    learning_rate: float = 3e-5
    checkpoint_frequency: int = 5
    log_frequency: int = 10
    early_stopping_patience: int = 3
    # DDP defaults to two GPUs; can be changed to 1 to fall back to a single GPU
    world_size: int = 2
    gradient_clip_norm: float = 1.0
    output_root: str = "model/runpytorch"
    run_name: Optional[str] = None
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    # Computed by the launcher right before spawn and injected here; all ranks share the same directory
    run_dir: Optional[str] = None
    # find_unused_parameters: GASSO has an architecture alpha branch, and after softmax some
    # branches may not always participate in the gradient
    ddp_find_unused_parameters: bool = True
    # None means fall back to data.init_args.batch_size in the YAML; when not None, the CLI overrides it
    batch_size: Optional[int] = None
    # None means fall back to model.init_args.gnn_params.num_layers in the YAML; when not None,
    # the CLI overrides it. I.e. the number of GNN message-passing layers / k_hop neighbor hops.
    num_layers: Optional[int] = None
    # Training wall-clock cap (seconds). None = unlimited; when not None, training stops once this
    # duration is reached (used for the test3 unified budget comparison).
    max_runtime_seconds: Optional[float] = None
    # Each epoch only uses this fraction (0,1] of the text for alignment: each epoch only runs the
    # first ceil(total_batches * frac) batches.
    # Because BatchGraphTextDataset reshuffles every epoch, the truncated subset is a different
    # random subset each time (accumulating over multiple epochs still covers the full data).
    # None = use all text (default, backward compatible).
    text_subsample_fraction: Optional[float] = None
    # Early-stopping / convergence monitoring metric: "auc" = val link-prediction AUC stops
    # improving (default, higher is better, uses the greedy deployed architecture);
    # "loss" = val contrastive loss stops decreasing (old behavior). In auc mode, once per epoch
    # on rank 0 a greedy(is_search=False) full-graph forward pass is run on the val subgraph to
    # compute AUC, which is used to select best_model and decide early stopping.
    early_stop_metric: str = "auc"
    # Adam eps override: None = fall back to YAML (1e-4). Matching DeepSpeed's 1e-8 fixes a bug
    # where the search gets stuck on a plateau.
    adam_eps: Optional[float] = None
    # weight_decay override: None = fall back to YAML (0.0). Matching DeepSpeed's 0.01 suppresses
    # contrastive embedding collapse (applies to the weights group; alpha stays fixed at 0).
    weight_decay: Optional[float] = None
    # test6 time-breakdown: once enabled, each iteration is timed into 7 buckets, and a profile_out
    # JSON is written at the end of training.
    profile: bool = False
    profile_out: Optional[str] = None
    # Unified telemetry: per-iteration JSONL path (None = derived from profile_out's telemetry/ subdir).
    telemetry_jsonl: Optional[str] = None
    # Reproducibility + used to tag the telemetry JSONL with a seed label.
    seed: int = 42
    # Every N seconds, additionally compute val-AUC on rank0 and append to run_dir/val_timeline.csv
    # (for dense time-to-accuracy sampling).
    # Fully independent of the per-epoch metrics.csv / early stopping; just an extra side-channel
    # record. 0 = disabled (default, no behavior change).
    val_eval_interval_seconds: float = 0.0
    # True = periodic sampling instead performs link prediction on the test split (test1-2 honest
    # speedup comparison uses the test-set convention). Default False = val.
    val_eval_on_test: bool = False


# ---------------------------------------------------------------------------
# File / directory
# ---------------------------------------------------------------------------


def _make_run_dir(config: TrainingConfig) -> str:
    root = os.path.abspath(os.path.expanduser(config.output_root))
    os.makedirs(root, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"gasso_{config.dataset}_ddp_{ts}"
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
    # Also write CLI/shell overrides back into the snapshot so the eval side sees the correct actual structure.
    if config.num_layers is not None:
        try:
            model_config.model.init_args.gnn_params.num_layers = int(config.num_layers)
        except Exception:
            pass  # Don't block snapshot writing if the YAML structure is unexpected
    OmegaConf.save(model_config, os.path.join(run_dir, "model_config_snapshot.yaml"))
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["epoch", "train_loss", "val_loss", "best_val_loss",
             "val_auc", "best_val_auc", "epoch_seconds", "timestamp"]
        )
    with open(train_log, "w", encoding="utf-8") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write(f"config_path={config.config_path}\n")
        f.write(f"dataset={config.dataset}\n")
        f.write(f"world_size={config.world_size}\n\n")
    return metrics_csv, train_log, ckpt_dir


def _append_metric_row(
    metrics_csv: str,
    train_log: str,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
    epoch_seconds: float,
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
                ts,
            ]
        )
    with open(train_log, "a", encoding="utf-8") as f:
        line = f"epoch={epoch + 1} train_loss={train_loss:.6f}"
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
# DDP utilities
# ---------------------------------------------------------------------------


def _ddp_timeout() -> timedelta:
    return timedelta(seconds=float(os.environ.get("DDP_TIMEOUT_SECONDS", "1800")))


def _setup_distributed(rank: int, world_size: int, config: TrainingConfig) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    timeout = _ddp_timeout()
    print(
        f"[rank{rank}] init_process_group start backend={config.backend} "
        f"port={config.master_port} timeout={timeout.total_seconds():.0f}s",
        flush=True,
    )
    if config.backend == "nccl":
        torch.cuda.set_device(rank)
        try:
            dist.init_process_group(
                backend=config.backend,
                rank=rank,
                world_size=world_size,
                device_id=torch.device(f"cuda:{rank}"),
                timeout=timeout,
            )
        except TypeError:
            dist.init_process_group(
                backend=config.backend, rank=rank, world_size=world_size, timeout=timeout,
            )
    else:
        dist.init_process_group(
            backend=config.backend, rank=rank, world_size=world_size, timeout=timeout,
        )
    print(f"[rank{rank}] init_process_group done", flush=True)


def _barrier() -> None:
    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _rank_log(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)


def _wait_for_file(path: str, rank: int, description: str) -> None:
    timeout_seconds = float(os.environ.get("DDP_SETUP_TIMEOUT_SECONDS", "600"))
    deadline = time.time() + timeout_seconds
    while not os.path.isfile(path):
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {description}: {path}")
        time.sleep(0.25)
    _rank_log(rank, f"{description} ready: {path}")


def _all_reduce_mean(value: float, world_size: int, device: torch.device) -> float:
    """Average across ranks (does not affect gradients, pure metric aggregation)."""
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / float(world_size))


class _LightningDDPWrapper(nn.Module):
    """Allows DDP's forward to correctly proxy to the LightningModule's step.

    DDP's backward-sync relies on forward actually passing through the wrapper (so it can
    register hooks like `_DDPSink`). Calling `ddp_model.module.training_step(...)` directly
    skips these hooks on some PyTorch versions, so we wrap it here:
    DDP -> _LightningDDPWrapper.forward -> lit.training_step.
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
# Trainer
# ---------------------------------------------------------------------------


class GraphModelTrainer:
    def __init__(self, rank: int, world_size: int, config: TrainingConfig):
        _rank_log(rank, "GraphModelTrainer.__init__ start")
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f"cuda:{rank}")
        self.epoch = 0
        self.global_step = 0
        # Throughput stats: total number of "seed nodes" processed by this rank (accumulated
        # from the batch text node count of each step).
        # At the end of training, all_reduce(SUM) across ranks gives the system-wide total ->
        # Nodes/s = total_nodes / total_time.
        self.nodes_processed = 0
        # GPU active compute time (ms, this rank): accumulated over the forward + backward+step
        # spans measured via CUDA events.
        # At the end of training, summed across ranks / world_size gives the per-GPU average.
        self.gpu_active_ms = 0.0
        # max-runtime cap: stop training as soon as the deadline is reached (used for the test3
        # unified 500s budget).
        self._deadline: Optional[float] = None
        self._time_up = False

        # AMP/fp16 is no longer used: GASSO's multi-branch mixed architecture + large graphs
        # (e.g. goodreads-crime) produce neighbor-aggregation value ranges far beyond fp16's
        # max=65504, causing NaNs and training crashes.
        # Training runs entirely in fp32; on a P100, fp16 offered limited speedup for the
        # multi-branch architecture anyway.

        # Early-stopping related variables (only truly used on rank 0; other ranks sync the
        # decision via broadcast)
        self.best_val_loss = float("inf")
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = 0
        # Cache for val-AUC early stopping: the val subgraph is unchanged across epochs, so
        # tensors + negative edges are only built/sampled once.
        self._val_auc_cache: Optional[tuple] = None
        # test6 time-breakdown timer (injected by _ddp_main after the trainer is built; None = no timing).
        self.profiler: Optional[TelemetryRecorder] = None

        if _is_main(rank):
            print(f"[rank{rank}] Initializing DDP Trainer on device: {self.device}")
            print(f"[rank{rank}] Model: gasso, Dataset: {self.config.dataset}, world_size={world_size}")
            print("_" * 100)

        # 1. Load config
        _rank_log(rank, "OmegaConf.load start")
        self.model_config = OmegaConf.load(config.config_path)
        _rank_log(rank, "OmegaConf.load done")
        if config.num_layers is not None:
            # CLI/shell override: write back into model.init_args.gnn_params.num_layers in the
            # YAML, so that when _setup_model later builds the GNN from model_args, it also sees
            # this value.
            self.model_config.model.init_args.gnn_params.num_layers = int(config.num_layers)
            if _is_main(self.rank):
                print(f"[override] gnn_params.num_layers = {config.num_layers} (= k_hop)")
        try:
            self.k_hop = int(self.model_config.model.init_args.gnn_params.num_layers)
        except Exception:
            self.k_hop = 1

        # 2. Data
        self._setup_data()

        # 3. Model + DDP wrapping
        self._setup_model()

        # 4. Optimizer
        self._setup_optimizer()

    # ---------- Submodule setup ----------

    def _setup_data(self) -> None:
        if _is_main(self.rank):
            print("Setting up data...")
        init_args = dict(self.model_config.data.init_args)
        data_dir = os.path.join(_DATA_ROOT, self.config.dataset)
        init_args["data_dir"] = data_dir
        if self.config.batch_size is not None:
            init_args["batch_size"] = int(self.config.batch_size)
        sim_smoothing = float(self.model_config.model.init_args.get("sim_smoothing", 0.0) or 0.0)
        if sim_smoothing <= 0.0:
            init_args["compute_sim_mutual"] = False
            if _is_main(self.rank):
                print("[data] sim_smoothing=0, skip sim_mutual precompute", flush=True)

        # Validated based on file existence: data/<name>/raw/{graph,node-data,texts}.csv just
        # needs to be complete.
        # More flexible than a hardcoded whitelist; new datasets can be dropped in without
        # touching this code.
        raw_dir = os.path.join(data_dir, "raw")
        missing = [fn for fn in _REQUIRED_FILES if not os.path.isfile(os.path.join(raw_dir, fn))]
        if missing:
            available = _discover_datasets()
            raise ValueError(
                f"data/{self.config.dataset}/raw is missing files {missing}. "
                f"Currently available datasets: {available}"
            )

        datamodule = GassoPubmedGraphTextDataModule(**init_args)

        # rank 0 prepares/caches the data first; other ranks wait for it to finish writing
        # before reading, to avoid cache contention.
        if _is_main(self.rank):
            _rank_log(self.rank, "prepare_data start")
            datamodule.prepare_data()
            _rank_log(self.rank, "prepare_data done")
        else:
            cache_path = getattr(datamodule, "_graph_data_cache_path", None)
            if cache_path:
                _wait_for_file(cache_path, self.rank, "graph cache")
            _rank_log(self.rank, "prepare_data cache check start")
            datamodule.prepare_data()  # Other ranks just hit the cache
            _rank_log(self.rank, "prepare_data cache check done")

        # Consistent with single-machine training: world_size=1, node=0, every rank gets the
        # full graph.
        # Load partitioning across ranks is handled below by the DistributedSampler.
        _rank_log(self.rank, "datamodule.setup start")
        datamodule.setup(node=0, world_size=1, stage="fit", k_hop=self.k_hop)
        _rank_log(self.rank, "datamodule.setup done")

        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset
        # Keep a reference to val_dataset so val-AUC early stopping can fetch the split-induced
        # graph_data on rank 0.
        self.val_dataset = val_dataset
        # Reference to the test split (when val_eval_on_test=True, periodic sampling performs
        # link prediction on test instead).
        self.test_dataset = getattr(datamodule, "test_dataset", None)

        # BatchGraphTextDataset has its own within-epoch shuffle (`next_epoch()`),
        # so DistributedSampler uses shuffle=False and is only responsible for evenly splitting
        # batch indices [0..len) across ranks.
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
                print("Warning: No validation dataset found. Early stopping will be disabled.")

    def _setup_model(self) -> None:
        if _is_main(self.rank):
            print("Setting up gasso model...")
        model_args = self.model_config.model.init_args.copy()
        model_args["lr"] = self.config.learning_rate
        _rank_log(self.rank, "LitGassoClipGraph init start")
        lit_model = LitGassoClipGraph(**model_args).to(self.device)
        _rank_log(self.rank, "LitGassoClipGraph init done")
        wrapped = _LightningDDPWrapper(lit_model).to(self.device)
        _rank_log(self.rank, "DDP wrap start")
        self.model = DDP(
            wrapped,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
        )
        _rank_log(self.rank, "DDP wrap done")

    def _setup_optimizer(self) -> None:
        ma = self.model_config.model.init_args
        betas = ma.get("betas", (0.9, 0.999))
        betas = tuple(betas) if not isinstance(betas, tuple) else betas
        eps = float(ma.get("eps", 1e-8))
        # CLI/env override for Adam eps: the YAML default of 1e-4 causes the Adam update for 95%
        # of parameters (where sqrt(v) < eps) to be dominated by eps, degrading to a slow
        # ~proportional-to-grad regime that stalls search/training on a plateau; matching
        # DeepSpeed's 1e-8 gives proper full Adam normalization.
        if getattr(self.config, "adam_eps", None) is not None:
            if _is_main(self.rank):
                print(f"[override] AdamW eps = {float(self.config.adam_eps)} (overriding YAML's {eps})")
            eps = float(self.config.adam_eps)
        # weight_decay: the YAML default (0.0) can be overridden via CLI/env. Matching
        # DeepSpeed's 0.01 suppresses contrastive collapse.
        wd = float(ma.weight_decay)
        if getattr(self.config, "weight_decay", None) is not None:
            wd = float(self.config.weight_decay)
        # Two DARTS groups: alpha (architecture params) form their own group with no
        # weight_decay (to avoid wd pushing a newly learned alpha back toward 0);
        # weights get wd. Fully aligned with bin/zero_offload_train.py (previously DDP used a
        # single group with no wd on weights).
        arch_params, weight_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ".alphas_normal." in n or ".alphas_norm." in n or ".alphas_act." in n:
                arch_params.append(p)
            else:
                weight_params.append(p)
        if _is_main(self.rank):
            print(f"[opt] weights(wd={wd}): {len(weight_params)} tensors | "
                  f"arch α(wd=0): {len(arch_params)} tensors | eps={eps}")
        self.optimizer = optim.AdamW(
            [
                {"params": weight_params, "weight_decay": wd},
                {"params": arch_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            eps=eps,
            betas=betas,
        )

    # ---------- Helpers ----------

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    def _unwrapped_state_dict(self) -> Dict[str, torch.Tensor]:
        # DDP -> _LightningDDPWrapper -> LitGassoClipGraph
        return self.model.module.lit_module.state_dict()

    # ---------- train / val ----------

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.epoch = epoch
        # Time anchor for periodic dense val-AUC sampling: anchored as the "training start" when
        # the first epoch begins.
        if self.config.val_eval_interval_seconds and self.config.val_eval_interval_seconds > 0:
            if getattr(self, "_intra_val_start", None) is None:
                self._intra_val_start = time.time()
                self._intra_val_last = time.time()
                self._intra_val_csv = None
        if self.train_sampler is not None:
            # The sampler is currently set to shuffle=False, so set_epoch doesn't actually
            # change the order, but it's kept in case shuffle=True is enabled later
            self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        # Text subsampling: this epoch only runs the first _keep batches (reshuffled each epoch
        # -> a different random subset each time).
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
                ncols=100,
            )
        else:
            iterator = enumerate(self.train_dataloader)

        if self.profiler is not None:
            self.profiler.epoch_start()
        use_cuda = self.device.type == "cuda"
        for batch_idx, batch in iterator:
            if batch_idx >= _keep:
                break
            if self.profiler is not None:
                self.profiler.iter_begin()
                self.profiler.xfer_start()
            batch = self._move_batch_to_device(batch)
            if self.profiler is not None:
                self.profiler.xfer_end()
            # Throughput: number of seed nodes fed in this step (already deduplicated by
            # text/node, at most batch_size)
            _n_nodes = int(batch["text_node_ids"].shape[0])
            self.nodes_processed += _n_nodes
            if self.profiler is not None:
                self.profiler.set_iter_meta(nodes_in_batch=_n_nodes)

            # GPU active compute time: wrap the forward and backward+step spans with CUDA
            # events, accumulated after synchronize (CPU gaps between steps are not counted,
            # i.e. treated as Idle/Wait).
            if use_cuda:
                ev_fwd0 = torch.cuda.Event(enable_timing=True)
                ev_fwd1 = torch.cuda.Event(enable_timing=True)
                ev_bwd1 = torch.cuda.Event(enable_timing=True)
                ev_fwd0.record()

            # Goes through DDP's forward, ensuring grad sync fires correctly on backward
            loss = self.model(batch, batch_idx, "train")
            if use_cuda:
                ev_fwd1.record()

            if self.profiler is not None:
                self.profiler.bwd_start()
            loss.backward()
            if self.profiler is not None:
                self.profiler.bwd_end()
                self.profiler.sync_wait()   # Timed barrier between backward and step = sync wait
                self.profiler.upd_start()
            if self.config.gradient_clip_norm > 0:
                nn_utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.optimizer.step()
            if self.profiler is not None:
                self.profiler.upd_end()
            if use_cuda:
                ev_bwd1.record()
                torch.cuda.synchronize(self.device)
                self.gpu_active_ms += ev_fwd0.elapsed_time(ev_fwd1) + ev_fwd1.elapsed_time(ev_bwd1)
            self.optimizer.zero_grad()

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

            # Periodic dense val-AUC sampling (for the time-to-accuracy curve): rank0 decides
            # based on wall-clock and broadcasts; only rank0 actually calls _compute_val_auc(),
            # and a final barrier resets state -- this avoids deadlocking the next collective if
            # only rank0 ran it.
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
                    self.model.train()
                    if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                        dist.barrier()

            # max-runtime cap: stop as soon as it's reached (per-step granularity, keeping total
            # duration close to the budget).
            # The decision must be synced across ranks -- otherwise ranks would break on
            # different steps and deadlock at the next all_reduce/barrier. A single 1-element
            # all_reduce(MAX) is used to reach a consistent decision.
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
        # Aggregate across ranks to get the global train loss
        global_avg = _all_reduce_mean(local_avg, self.world_size, self.device)
        epoch_time = time.time() - epoch_start
        return global_avg, epoch_time

    def validate(self) -> Optional[float]:
        if not self.val_dataloader:
            return None
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                loss = self.model(batch, batch_idx, "val")
                total_loss += loss.item()
                num_batches += 1

        local_avg = total_loss / num_batches if num_batches > 0 else 0.0
        return _all_reduce_mean(local_avg, self.world_size, self.device)

    def _build_val_auc_cache(self) -> Optional[tuple]:
        """Build and cache the val subgraph tensors + one-time sampled negative edges (the val
        subgraph is unchanged across epochs).

        Returns (vx, vei, vnei, edge_attr) or None (no val / subgraph has no edges). Only
        called on rank 0.
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
        # GASSO encode requires edge_attr; if missing, pad with zeros and unsqueeze (same as eval_link_pred_ddp).
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

    def _compute_val_auc(self) -> Optional[float]:
        """Run a single greedy(is_search=False) full-graph forward pass on the val subgraph on
        rank 0 to compute link-prediction AUC.

        Returns None directly on non-rank-0 (to avoid DDP collective deadlock). Returns None on failure.
        """
        if not _is_main(self.rank):
            return None
        import torch.nn.functional as F
        import sklearn.metrics as mt
        import torch_geometric as pyg

        if self._val_auc_cache is None:
            self._val_auc_cache = self._build_val_auc_cache()
        if self._val_auc_cache is None:
            return None
        vx, vei, vnei, edge_attr = self._val_auc_cache

        gnn = self.model.module.lit_module.model.gnn
        gnn_proj = self.model.module.lit_module.model.gnn_proj
        self.model.eval()
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
        """Append a (elapsed, epoch, step, val_auc) row to run_dir/val_timeline.csv (rank0 only).

        This is the dense sampling data source for the time-to-accuracy curve; independent of
        metrics.csv (per epoch), and not used for early stopping.
        Not written to disk when val_auc is None (no val edges, etc.).
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

    def save_checkpoint(self, path: str) -> None:
        if not _is_main(self.rank):
            return
        state = {
            "model_state_dict": self._unwrapped_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def save_best_model(self, path: str) -> None:
        if not _is_main(self.rank):
            return
        if self.best_model_state is not None:
            state = {
                "model_state_dict": self.best_model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.best_epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            }
            torch.save(state, path)
            print(
                f"Best model saved to {path} (Epoch {self.best_epoch + 1}, Val Loss: {self.best_val_loss:.4f})"
            )
        else:
            print("Warning: No best model state to save.")


# ---------------------------------------------------------------------------
# Single-rank training loop (mp.spawn entry point)
# ---------------------------------------------------------------------------


def _ddp_main(rank: int, world_size: int, config: TrainingConfig) -> None:
    _setup_distributed(rank, world_size, config)
    try:
        _rank_log(rank, "_ddp_main entered")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        run_dir = config.run_dir
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        train_log = os.path.join(run_dir, "training.log")

        if _is_main(rank):
            _write_training_artifacts(run_dir, config, OmegaConf.load(config.config_path))
            print(f"Run directory: {run_dir}")

        _rank_log(rank, "run_dir artifact setup complete")

        trainer = GraphModelTrainer(rank, world_size, config)
        _rank_log(rank, "GraphModelTrainer.__init__ done")

        # test6 time-breakdown: attaches a forward hook on lit_module, timing each iteration
        # into 7 buckets.
        # Unified telemetry: DDP's real gradient sync overlaps with backward, so a timed
        # barrier is inserted after backward to measure load imbalance = SyncWait.
        if config.profile:
            _legacy_out = config.profile_out or os.path.join(
                "results", "test6", f"breakdown_gnas_ddp_{config.dataset}.json"
            )
            _jsonl = config.telemetry_jsonl or derive_jsonl_path(
                config.profile_out, system="gnas_ddp", dataset=config.dataset,
                seed=config.seed, worker="single",
            )
            trainer.profiler = TelemetryRecorder(
                system="gnas_ddp",
                dataset=config.dataset,
                seed=config.seed,
                worker="single",
                device=trainer.device,
                jsonl_path=_jsonl,
                lit_module=trainer.model.module.lit_module,
                rank=rank,
                enabled=True,
                emit_barrier_sync_wait=True,
                legacy_json_path=_legacy_out,
                verbose=_is_main(rank),
            )

        best_model_name = os.path.join(run_dir, "best_model.pth")
        final_model_name = os.path.join(run_dir, "final_model.pth")

        train_total_start = time.time()
        if config.max_runtime_seconds is not None:
            trainer._deadline = train_total_start + float(config.max_runtime_seconds)
        epoch_times: list[float] = []
        epochs_completed = 0
        early_stopped = False
        best_checkpoint_seconds = None
        best_selection_seconds = None

        if trainer.val_dataloader is None:
            if _is_main(rank):
                print("Warning: No validation dataset. Training without early stopping.")
                print(f"Starting DDP training for {config.num_epochs} epochs on {world_size} GPUs...")

            for epoch in range(config.num_epochs):
                train_loss, epoch_time = trainer.train_epoch(epoch)
                epoch_times.append(epoch_time)
                epochs_completed = epoch + 1
                if _is_main(rank):
                    print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}  Time: {epoch_time:.2f}s")
                    _append_metric_row(
                        metrics_csv, train_log, epoch, train_loss, None,
                        trainer.best_val_loss, epoch_time,
                    )

                # max-runtime: train_epoch already reached the deadline (break synced across
                # ranks) -> stop immediately here,
                # skipping any further checkpointing so total_seconds stays close to the budget
                # (needed for Active/Idle to be comparable).
                if trainer._time_up:
                    if _is_main(rank):
                        print(f"[max-runtime] reached {config.max_runtime_seconds}s budget, stopping training")
                    break

                ds = trainer.train_dataloader.dataset
                if hasattr(ds, "next_epoch"):
                    ds.next_epoch()
                _barrier()

                if (epoch + 1) % config.checkpoint_frequency == 0:
                    save_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                    trainer.save_checkpoint(save_path)
                    _barrier()

            trainer.save_checkpoint(final_model_name)
            _barrier()
        else:
            if _is_main(rank):
                print(
                    f"Starting DDP training for {config.num_epochs} epochs on {world_size} GPUs "
                    f"(early stopping patience={config.early_stopping_patience})..."
                )

            for epoch in range(config.num_epochs):
                train_loss, epoch_time = trainer.train_epoch(epoch)
                epoch_times.append(epoch_time)
                epochs_completed = epoch + 1
                if _is_main(rank):
                    print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}  Time: {epoch_time:.2f}s")

                # max-runtime: stop immediately once reached (before val/ckpt), so total_seconds
                # stays close to the budget.
                # _time_up is already synced across ranks within train_epoch, so all ranks break
                # together and won't stall val's all_reduce.
                if trainer._time_up:
                    if _is_main(rank):
                        print(f"[max-runtime] reached {config.max_runtime_seconds}s budget, stopping training")
                    break

                ds = trainer.train_dataloader.dataset
                if hasattr(ds, "next_epoch"):
                    ds.next_epoch()
                _barrier()

                val_loss = trainer.validate()
                val_auc = None
                if _is_main(rank):
                    print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}")
                    # Choice of monitoring metric: auc = val link-prediction AUC (greedy, higher
                    # is better); loss = val contrastive loss (lower is better).
                    # In auc mode only rank 0 computes the full-graph forward; if it can't be
                    # computed, this epoch falls back to using loss.
                    metric = config.early_stop_metric
                    if metric == "auc":
                        val_auc = trainer._compute_val_auc()
                    use_auc = metric == "auc" and val_auc is not None
                    if metric == "auc" and val_auc is None:
                        print("[val-auc] no AUC this epoch, falling back to val_loss for early-stopping decision")

                    _EPS = 1e-6
                    if use_auc:
                        improved = val_auc > trainer.best_val_auc + _EPS
                        print(f"Epoch {epoch + 1} Val AUC: {val_auc:.4f}")
                    else:
                        improved = val_loss < trainer.best_val_loss - _EPS

                    if improved:
                        best_selection_seconds = time.time() - train_total_start
                        if use_auc:
                            print(
                                f"✓ Validation AUC improved from {trainer.best_val_auc:.4f} to {val_auc:.4f}"
                            )
                            trainer.best_val_auc = val_auc
                        else:
                            print(
                                f"✓ Validation loss improved from {trainer.best_val_loss:.4f} to {val_loss:.4f}"
                            )
                            trainer.best_val_loss = val_loss
                        trainer.best_epoch = epoch
                        trainer.patience_counter = 0
                        # DDP ranks all have identical weights, so it's enough to copy the best out from rank 0
                        trainer.best_model_state = {
                            k: v.cpu().clone() for k, v in trainer._unwrapped_state_dict().items()
                        }
                        best_checkpoint_seconds = time.time() - train_total_start
                    else:
                        trainer.patience_counter += 1
                        _what = "AUC" if use_auc else "loss"
                        print(
                            f"✗ Validation {_what} did not improve. "
                            f"Patience: {trainer.patience_counter}/{config.early_stopping_patience}"
                        )
                    _append_metric_row(
                        metrics_csv, train_log, epoch, train_loss, val_loss,
                        trainer.best_val_loss, epoch_time,
                        val_auc=val_auc, best_val_auc=trainer.best_val_auc,
                    )

                # Early-stopping decision: rank 0 decides and broadcasts to other ranks, to
                # avoid ranks inconsistently exiting
                stop_signal = torch.tensor(
                    [
                        1
                        if _is_main(rank)
                        and trainer.patience_counter >= config.early_stopping_patience
                        else 0
                    ],
                    device=trainer.device,
                    dtype=torch.int64,
                )
                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1:
                    if _is_main(rank):
                        print(
                            f"\nEarly stopping triggered! No improvement for {config.early_stopping_patience} epochs."
                        )
                        if config.early_stop_metric == "auc":
                            print(
                                f"Best validation AUC: {trainer.best_val_auc:.4f} at epoch {trainer.best_epoch + 1}"
                            )
                        else:
                            print(
                                f"Best validation loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch + 1}"
                            )
                    early_stopped = True
                    break

                if (epoch + 1) % config.checkpoint_frequency == 0:
                    save_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                    trainer.save_checkpoint(save_path)
                    _barrier()

            trainer.save_best_model(best_model_name)
            trainer.save_checkpoint(final_model_name)
            _barrier()

        total_time = time.time() - train_total_start

        # Throughput: sum across ranks to get the system-wide total number of processed seed
        # nodes (must happen before destroy_process_group, and all ranks must participate in
        # this collective). Nodes/s = total_nodes / total_time.
        nodes_t = torch.tensor(
            [float(trainer.nodes_processed)], dtype=torch.float64, device=trainer.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nodes_t, op=dist.ReduceOp.SUM)
        total_nodes = int(nodes_t.item())
        throughput = total_nodes / total_time if total_time > 0 else 0.0

        # GPU active compute time: sum across ranks, then /world_size = per-GPU average (seconds).
        gpu_t = torch.tensor(
            [float(trainer.gpu_active_ms)], dtype=torch.float64, device=trainer.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gpu_t, op=dist.ReduceOp.SUM)
        gpu_active_seconds = (gpu_t.item() / 1000.0) / max(world_size, 1)

        # test6 time-breakdown: the per-iteration JSONL was already written during training;
        # here rank0 finalizes it and also writes the legacy averaged format.
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
            print(f"[Time] World size      : {world_size} GPUs")
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
                f.write(f"world_size={world_size}\n")
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
                        "world_size": world_size,
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
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Post-train link-prediction evaluation
# ---------------------------------------------------------------------------


def _run_link_prediction_after_train(
    run_dir: str,
    dataset_name: str,
    eval_script_rel: str = "bin/eval_link_pred_ddp.py",
    split: str = "test",
    device: str = "cuda",
    extra_args: Optional[List[str]] = None,
) -> None:
    """After training, run link prediction on all checkpoints under this run_dir (mirroring the
    post-training automatic evaluation in run_sys.sh).

    Coverage: ``best_model.pth`` / ``final_model.pth`` as well as
    ``checkpoints/checkpoint_epoch_*.pth``. Each checkpoint runs ``eval_script_rel`` in its own
    subprocess; a failed checkpoint raises, with final results appended to
    ``<run-dir>/link-prediction-eval-*.csv``.
    """
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
        raise FileNotFoundError(f"[post-eval] No checkpoints found in {run_dir}")

    repo_root = str(REPO_ROOT)
    eval_script_path = os.path.join(repo_root, eval_script_rel)
    if not os.path.isfile(eval_script_path):
        raise FileNotFoundError(f"[post-eval] eval script does not exist: {eval_script_path}")

    print()
    print("=" * 60)
    print(f"[post-eval] Link-prediction evaluation (dataset={dataset_name})")
    print(f"  run_dir : {run_dir}")
    print(f"  script  : {eval_script_rel}")
    print(f"  split   : {split}")
    print(f"  ckpts   : {specs}")
    print("=" * 60)

    for spec in specs:
        print(f"\n[post-eval] checkpoint={spec}")
        cmd = [
            _sys.executable, eval_script_path,
            "--run-dir", run_dir,
            "--split", split,
            "--device", device,
            "--checkpoint", spec,
        ]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    csv_candidates = [
        os.path.join(run_dir, "link-prediction-eval-gnn.csv"),
        os.path.join(run_dir, "link-prediction-eval-clip.csv"),
    ]
    for p in csv_candidates:
        if os.path.isfile(p):
            print(f"\n[post-eval] results summary -> {p}")


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def train_ddp(config: TrainingConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA but no CUDA device is available.")
    available = torch.cuda.device_count()
    if config.world_size > available:
        raise RuntimeError(
            f"world_size={config.world_size} exceeds available GPUs={available}"
        )

    config.run_dir = _make_run_dir(config)
    print(f"[launcher] Spawning {config.world_size} DDP workers, run_dir={config.run_dir}")

    mp.spawn(
        _ddp_main,
        args=(config.world_size, config),
        nprocs=config.world_size,
        join=True,
    )


def parse_cli() -> Tuple[List[str], Dict[str, Any]]:
    """Parse command-line arguments and return (list of datasets to train, dict used to override
    TrainingConfig).

    Dataset source priority: --datasets-all > --datasets > hardcoded default ["goodreads-crime"].
    All other parameters have defaults matching the original hardcoded values, so behavior is
    fully equivalent when no CLI args are passed.
    """
    parser = argparse.ArgumentParser(description="GraphNAS DDP standalone trainer (gasso)")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (e.g. goodreads-crime,amazon-baby). "
             "If omitted, falls back to --datasets-all or the hardcoded default ['goodreads-crime']",
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
    parser.add_argument("--output-root", type=str, default="model/runpytorch")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=str, default="29500",
                        help="DDP rendezvous port; change it if there's a port conflict.")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stop-metric", choices=["loss", "auc"], default="auc",
                        help="Convergence/early-stopping monitoring metric: auc = val "
                             "link-prediction AUC (greedy) stops improving (default); "
                             "loss = val contrastive loss stops decreasing (old behavior).")
    parser.add_argument("--checkpoint-frequency", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override data.init_args.batch_size in the YAML. If omitted, uses the YAML config.")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override gnn_params.num_layers in the YAML (= GNN k_hop / number of layers). If omitted, uses the YAML config.")
    parser.add_argument("--max-runtime-seconds", type=float, default=None,
                        help="Training wall-clock cap (seconds). If omitted, runs the full epoch count; used for the test3 unified budget comparison.")
    parser.add_argument("--text-subsample-fraction", type=float, default=None,
                        help="Each epoch only uses this fraction (0,1] of the text for alignment (re-sampled randomly each epoch). If omitted = use all text.")
    parser.add_argument("--no-post-eval", action="store_true",
                        help="Don't automatically run link-prediction evaluation after training completes (runs by default)")
    parser.add_argument("--post-eval-split", choices=["train", "val", "test"], default="test",
                        help="Split used for post-train link prediction, default test")
    parser.add_argument("--profile", action="store_true",
                        help="test6: enable per-iteration 7-bucket time-breakdown timing, write JSON at the end of training")
    parser.add_argument("--profile-out", type=str, default=None,
                        help="Time-breakdown JSON output path (default results/test6/breakdown_gnas_ddp_<ds>.json)")
    parser.add_argument("--telemetry-jsonl", type=str, default=None,
                        help="Unified telemetry per-iteration JSONL path (default derived from --profile-out's telemetry/ subdirectory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for reproducibility + tagging the telemetry JSONL)")
    parser.add_argument("--adam-eps", type=float, default=None,
                        help="Override AdamW eps (YAML default of 1e-4 stalls search; matching DeepSpeed uses 1e-8)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight_decay (YAML default is 0.0; matching DeepSpeed uses 0.01 to suppress contrastive collapse; only applies to the weights group)")
    parser.add_argument("--val-eval-interval-seconds", type=float, default=0.0,
                        help="Every N seconds, additionally compute val-AUC and write to val_timeline.csv (dense sampling of the time-to-accuracy curve); 0 = disabled")
    parser.add_argument("--val-eval-on-test", action="store_true",
                        help="Perform periodic sampled link prediction on the test split instead (test1-2 honest speedup comparison uses the test-set convention)")
    args = parser.parse_args()

    if args.datasets_all:
        ds_list = _discover_datasets()
        if not ds_list:
            raise SystemExit(f"No available datasets found under data/ (requires raw/{{ {','.join(_REQUIRED_FILES)} }})")
    elif args.datasets:
        ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        ds_list = ["goodreads-crime"]

    overrides: Dict[str, Any] = {
        "config_path": args.config_path,
        "output_root": args.output_root,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "world_size": args.world_size,
        "master_port": str(args.master_port),
        "num_workers": args.num_workers,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stop_metric": args.early_stop_metric,
        "checkpoint_frequency": args.checkpoint_frequency,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "max_runtime_seconds": args.max_runtime_seconds,
        "text_subsample_fraction": args.text_subsample_fraction,
        "adam_eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "profile": args.profile,
        "profile_out": args.profile_out,
        "telemetry_jsonl": args.telemetry_jsonl,
        "seed": args.seed,
        "val_eval_interval_seconds": args.val_eval_interval_seconds,
        "val_eval_on_test": args.val_eval_on_test,
    }
    # post-eval related params go through a side channel in overrides that TrainingConfig won't consume
    overrides["__post_eval__"] = {
        "enabled": not args.no_post_eval,
        "split": args.post_eval_split,
        "negative_sampling": os.environ.get("LINK_NEG_SAMPLING", "auto"),
    }
    return ds_list, overrides


if __name__ == "__main__":
    ds_list, overrides = parse_cli()
    post_eval_cfg = overrides.pop("__post_eval__", {"enabled": True, "split": "test"})
    print(f"[launcher] datasets to train: {ds_list}")
    for dataset_name in ds_list:
        config = TrainingConfig(
            dataset=dataset_name,
            log_frequency=10,
            **overrides,
        )

        try:
            train_ddp(config)
            torch.cuda.empty_cache()
            if post_eval_cfg.get("enabled", True) and config.run_dir:
                try:
                    _run_link_prediction_after_train(
                        run_dir=config.run_dir,
                        dataset_name=dataset_name,
                        eval_script_rel="bin/eval_link_pred_ddp.py",
                        split=post_eval_cfg.get("split", "test"),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        extra_args=["--negative-sampling", post_eval_cfg.get("negative_sampling", "auto")],
                    )
                except Exception:
                    import traceback
                    print("[post-eval] Link-prediction evaluation failed:")
                    traceback.print_exc()
                    raise
        except KeyboardInterrupt:
            print("Training interrupted by user")
            raise
        except Exception:
            import traceback

            traceback.print_exc()
            raise
