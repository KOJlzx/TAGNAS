"""
Asynchronous co-execution training runtime for GraphNAS + LM distributed training.

This module implements the training services for three process roles:

  ┌─────────────────────────────────────────────────────────┐
  │                 CoordinatorTrainingService               │
  │     (rank 0: global cache routing, metrics, checkpoints) │
  └────────────┬──────────────────────────────┬──────────────┘
               │  RPC push / lookup           │  RPC push / lookup
  ┌────────────▼─────────┐      ┌─────────────▼─────────────┐
  │ GraphWorkerTraining  │      │ TextWorkerTrainingService  │
  │  Service (rank 1)    │      │  (rank 2, 3, ...)          │
  │  GNN encoding on CPU │      │  LM encoding on GPU        │
  └──────────────────────┘      └───────────────────────────┘

Data flow:
  1. Each worker draws a node-id batch from StreamingNodeScheduler.
  2. Forward pass produces embeddings, wraps them as EmbeddingRecord objects, and
     pushes them to the coordinator.
  3. The coordinator writes records into the global cache and broadcasts fresh
     embeddings to the opposite modality workers.
  4. Backward computation needs peer embeddings, so workers first check the local
     cache and then fall back to coordinator RPC lookup.
  5. If records are still missing, drop-on-miss skips the gradient update while
     still pushing newly computed embeddings.

Shutdown model:
  - After a worker reaches its epoch limit, it calls _signal_finished() but keeps
    serving forward+push work until the coordinator stops the job.
  - The coordinator broadcasts shutdown_event after every worker reports finished.
  - Workers exit when they observe shutdown_event at a loop boundary.

RPC safety:
  - Worker-to-coordinator RPC calls are protected so teardown races do not crash
    workers after the coordinator has already shut down RPC.
  - New broadcasts are avoided during shutdown.
"""

from __future__ import annotations

from collections import Counter
import contextlib
import threading
import json
import os
from pathlib import Path
import time
from threading import Condition, Event, Lock
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.utils.checkpoint as torch_checkpoint
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
import sklearn.metrics as sklearn_metrics
import torch_geometric.utils as pyg_utils
import torch_geometric.nn.models.autoencoder as pyg_ae

from .. import losses as ls
from .cache import AsyncEmbeddingCache, clone_embedding_records
from .config import AsyncTrainConfig
from .pubmed_runtime import PubmedRuntimeStore
from .scheduler import CacheAwareNodeScheduler, StreamingNodeScheduler, TokenBudgetNodeScheduler, shard_node_ids
from .training_models import (
    GraphGenotype,
    GraphTower,
    TextTower,
    apply_graph_genotype,
    clone_state_to_cpu,
    export_graph_genotype,
)
from .types import CacheKey, EmbeddingRecord
from ..telemetry import TelemetryRecorder, derive_jsonl_path
from .lag import DelayBuffer, LagTracker, StalenessStats, delayed_modality, used_lag_payload


def _parameter_buckets(
    parameters: Iterable[torch.Tensor], max_bytes: int = 64 * 1024 * 1024,
) -> list[list[tuple[int, int, int]]]:
    """Deterministic, bounded dense slices; large parameters may span buckets."""
    if max_bytes <= 0:
        raise ValueError("local averaging bucket size must be positive")
    buckets, current, used, kind = [], [], 0, None
    for index, parameter in enumerate(parameters):
        parameter_kind = (parameter.device, parameter.dtype)
        if current and parameter_kind != kind:
            buckets.append(current)
            current, used = [], 0
        kind = parameter_kind
        capacity = max(1, max_bytes // parameter.element_size())
        offset = 0
        while offset < parameter.numel():
            count = min(parameter.numel() - offset, capacity - used)
            current.append((index, offset, count))
            offset, used = offset + count, used + count
            if used == capacity:
                buckets.append(current)
                current, used = [], 0
    if current:
        buckets.append(current)
    return buckets


@torch.no_grad()
def _bucketed_weighted_average(
    parameters: Iterable[torch.Tensor], optimizer: torch.optim.Optimizer, weight: float, group: Any,
    *, measure_divergence: bool = False, max_bytes: int = 64 * 1024 * 1024,
) -> float | None:
    """Average the same parameters and existing Adam moments with fewer collectives.

    Missing moment state contributes zeros but remains absent, as in the retained
    algorithm. Each row owns its original storage; flat communication buffers are
    temporary. Floating-point reduction grouping may differ from scalar tensors.
    """
    parameters = list(parameters)
    if not parameters:
        return 0.0 if measure_divergence else None
    buckets = _parameter_buckets(parameters, max_bytes)
    norm_before = norm_after = None
    divergence = None
    if measure_divergence:
        norm_before = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
        norm_after = torch.zeros_like(norm_before)
    for field in (None, "exp_avg", "exp_avg_sq"):
        for slices in buckets:
            first = parameters[slices[0][0]]
            flat = torch.zeros(sum(count for _, _, count in slices), device=first.device, dtype=first.dtype)
            views, sources, destinations = [], [], []
            cursor = 0
            for index, offset, count in slices:
                parameter = parameters[index]
                source = parameter if field is None else optimizer.state.get(parameter, {}).get(field)
                view = flat.narrow(0, cursor, count)
                cursor += count
                if source is None:
                    continue
                if source.device != first.device or source.dtype != first.dtype:
                    raise ValueError("local averaging requires Adam moments to retain their parameter dtype/device")
                if not source.is_contiguous():
                    raise ValueError("local averaging requires contiguous dense parameters and moment tensors")
                destination = source.view(-1).narrow(0, offset, count)
                views.append(view)
                sources.append(destination)
                destinations.append(destination)
            if views:
                torch._foreach_copy_(views, sources)
            if field is None and measure_divergence:
                norm_before.add_(flat.double().square().sum(), alpha=weight)
            flat.mul_(weight)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            if field is None and measure_divergence:
                norm_after.add_(flat.double().square().sum())
            if destinations:
                torch._foreach_copy_(destinations, views)
        if field is None and measure_divergence:
            dist.all_reduce(norm_before, op=dist.ReduceOp.SUM, group=group)
            divergence = float(torch.sqrt(torch.clamp(norm_before - norm_after, min=0)).item())
    return divergence


def _mean_staleness(records, peer_seen: dict) -> float:
    """Worker-side estimate of the peer-embedding lag, in the PRODUCER's successful updates, for the
    test6 telemetry: (latest producer_step this worker has seen from that producer) - producer_step.

    The old formula subtracted the consumer's own update count from the producer's, two counters of
    different workers that advance at different rates. The exact lag (against the producer's true
    current step) is computed by the coordinator from the ``lag_used`` payload (see lag.py); this
    estimate can only lag behind it by the broadcast latency. Records without a producer_step are ignored.
    """
    diffs = [max(0, int(peer_seen.get(str(r.producer), r.producer_step)) - int(r.producer_step))
             for r in records if getattr(r, "producer_step", -1) >= 0]
    return float(sum(diffs) / len(diffs)) if diffs else 0.0


def _note_peer_steps(peer_seen: dict, records) -> None:
    """Track, per producer, the highest producer_step seen in received peer embeddings."""
    for r in records:
        p = str(r.producer)
        if getattr(r, "producer_step", -1) > peer_seen.get(p, -1):
            peer_seen[p] = int(r.producer_step)


# ──────────────────────────────────────────────────────────────────────
# Part 1: global service registration and RPC handlers
# ──────────────────────────────────────────────────────────────────────
#
# PyTorch RPC requires remotely invoked functions to be module-level free
# functions rather than bound methods. Each process registers its service object
# in the module-level _TRAIN_SERVICE singleton, and RPC handlers resolve it via
# get_service().

_TRAIN_SERVICE: "BaseTrainingService | None" = None


def install_service(service: "BaseTrainingService") -> None:
    """Register the current process training service as a global singleton.

    Each process calls this once. All later RPC handlers retrieve the service via
    get_service().
    """
    global _TRAIN_SERVICE
    _TRAIN_SERVICE = service


def get_service() -> "BaseTrainingService":
    """Return the current process training service singleton."""
    if _TRAIN_SERVICE is None:
        raise RuntimeError("training runtime service has not been installed for this process")
    return _TRAIN_SERVICE


# --- RPC handlers executed on the coordinator side ---

def _rpc_receive_push(
    modality: str,
    records: list[EmbeddingRecord],
    source_name: str,
    telemetry: dict[str, Any] | None = None,
) -> bool:
    """Push fresh worker embeddings into the coordinator global cache.

    The coordinator writes records to the modality cache, broadcasts fresh records
    to peer-modality workers, and records telemetry such as batch counts and loss.
    """
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    return service.receive_push(modality, records, source_name, telemetry=telemetry)


def _rpc_lookup_cache(modality: str, node_ids: list[int]) -> list[EmbeddingRecord]:
    """Lookup embeddings for node_ids in the coordinator global cache.

    This is the remote fallback path for local-cache misses. The returned list may
    be shorter than the request when some records are absent.
    """
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    return service.lookup_embeddings(modality, node_ids)


def _rpc_ingest_remote_push(modality: str, records: list[EmbeddingRecord], source_name: str) -> bool:
    """Broadcast fresh embeddings from the coordinator to peer workers.

    For example, graph embeddings pushed by graph_worker are broadcast to all text
    workers. Recipients write them into local caches for future hits.
    """
    service = get_service()
    service.ingest_remote_push(modality, records, source_name)
    return True


def _rpc_request_start() -> bool:
    """Signal that a worker can start its training loop.

    Workers block in wait_until_start() after initialization and begin drawing
    batches only after receiving this signal.
    """
    service = get_service()
    service.allow_start()
    return True


def _rpc_request_shutdown() -> bool:
    """Ask a worker to leave its training loop gracefully.

    Workers check shutdown_event at batch-loop boundaries and exit after the
    current batch when requested.
    """
    service = get_service()
    service.request_shutdown()
    return True


def _rpc_mark_worker_ready(role_name: str) -> bool:
    """Report that a worker has completed initialization and is ready.

    The coordinator broadcasts start only after every worker is ready, ensuring all
    models, caches, and schedulers have been constructed.
    """
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    service.mark_worker_ready(role_name)
    return True


def _rpc_mark_worker_finished(role_name: str) -> bool:
    """Report that a worker has completed all assigned epochs.

    The coordinator broadcasts shutdown after all workers are finished. Workers do
    not exit immediately after reporting completion; they keep forward+push service
    alive until the shutdown signal arrives.
    """
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    service.mark_worker_finished(role_name)
    return True


# --- RPC handlers for sync_push lock-step ablation ---

def _rpc_sync_barrier(role_name: str) -> bool:
    """Lock-step barrier used by sync_push mode.

    Returns True when both graph and text sides may proceed. False indicates
    shutdown and tells workers to leave their loop.
    """
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    return service.sync_barrier_arrive(role_name)


def _rpc_report_telemetry(modality: str, telemetry: dict[str, Any]) -> bool:
    """Report telemetry after the update in sync mode."""
    service = get_service()
    assert isinstance(service, CoordinatorTrainingService)
    service.report_telemetry(modality, telemetry)
    return True


# --- RPC handlers executed on text workers ---

def _rpc_set_temperature(raw_tau: float) -> bool:
    """Synchronize the learnable contrastive temperature to text workers.

    graph_worker owns the learnable log-space tau parameter and broadcasts it after
    successful updates so all text workers use the same contrastive temperature.
    """
    service = get_service()
    assert isinstance(service, TextWorkerTrainingService)
    service.set_temperature(raw_tau)
    return True


# --- Generic RPC handlers callable on multiple roles ---

def _rpc_save_checkpoint(path: str) -> bool:
    """Ask a worker to save its model checkpoint.

    The coordinator invokes this periodically and during shutdown because model
    weights live inside worker processes.
    """
    service = get_service()
    service.save_checkpoint(path)
    return True


def _rpc_export_genotype() -> dict[str, Any]:
    """Export the current searched GNN genotype from the graph worker.

    Used during search. The coordinator serializes the returned dictionary into
    genotype.json.
    """
    service = get_service()
    assert isinstance(service, GraphWorkerTrainingService)
    return service.export_genotype().to_dict()


def _rpc_run_val_eval() -> dict[str, Any]:
    """Run link-prediction validation on the graph worker.

    This reads graph_tower parameters directly from the RPC thread without copying
    or locking.
    """
    service = get_service()
    assert isinstance(service, GraphWorkerTrainingService)
    return service.run_validation_eval()


def _rpc_export_pruning_summary() -> dict[str, Any]:
    """Export the adaptive-pruning summary from the graph worker.

    This is meaningful during search when adaptive pruning is enabled. With pruning
    disabled, or before pruning fires, every list is empty.
    """
    service = get_service()
    assert isinstance(service, GraphWorkerTrainingService)
    return service.graph_tower.gnn.pruning_summary()


# ──────────────────────────────────────────────────────────────────────
# Part 2: BaseTrainingService, shared by all runtime roles
# ──────────────────────────────────────────────────────────────────────

class BaseTrainingService:
    """Base class for coordinator, graph worker, and text workers.

    Provides process-local start/shutdown signals, the shared push_embeddings()
    helper used by workers, and role-specific virtual methods that subclasses
    override.
    """

    def __init__(
        self,
        *,
        config: AsyncTrainConfig,
        role_name: str,
        data_store: PubmedRuntimeStore,
        model_args: dict[str, Any],
    ) -> None:
        self.config = config            # Global runtime configuration.
        self.role_name = role_name      # Current process role, e.g. graph_worker.
        self.data_store = data_store    # Data access layer for graph/text batches.
        self.model_args = model_args    # Model hyperparameters loaded from YAML.
        training_nodes = (data_store.scheduler_train_nodes() if hasattr(data_store, "scheduler_train_nodes")
                          else data_store.split_nodes("train"))
        self.training_node_count = len(set(map(int, training_nodes)))
        config.validate_training_cache_capacity(self.training_node_count)
        self.shutdown_event = Event()   # Set when the coordinator broadcasts shutdown.
        self.start_event = Event()      # Set when the coordinator broadcasts start.

    def allow_start(self) -> None:
        """Set the start signal and unblock wait_until_start()."""
        self.start_event.set()

    def wait_until_start(self) -> None:
        """Block until the coordinator confirms that all workers are ready."""
        self.start_event.wait()

    def request_shutdown(self) -> None:
        """Set the shutdown signal; loops exit after the current batch boundary."""
        self.shutdown_event.set()

    def wait_until_shutdown(self) -> None:
        """Block until the shutdown signal is set."""
        self.shutdown_event.wait()

    def push_embeddings(
        self,
        modality: str,
        records: list[EmbeddingRecord],
        *,
        telemetry: dict[str, Any],
    ) -> bool:
        """Push embeddings from this forward pass to the coordinator global cache.

        Args:
            modality: "graph" or "text".
            records: EmbeddingRecord objects produced by this batch.
            telemetry: Optional batch telemetry such as loss, epoch, and hit rate.

        Returns:
            True when the coordinator receives the push; False during shutdown or
            a teardown-time RPC failure. A live push failure raises rather than
            silently losing a scheduled seed batch.

        Safety: teardown-time RPC failures are intentionally swallowed so worker
        cleanup can continue.
        """
        if self.shutdown_event.is_set():
            return False
        try:
            accepted = bool(
                rpc.rpc_sync(
                    self.config.coordinator_name,
                    _rpc_receive_push,
                    args=(modality, records, self.role_name, telemetry),
                )
            )
        except Exception as exc:
            if self.shutdown_event.is_set():
                return False
            raise RuntimeError(f"{self.role_name} {modality} embedding push failed during training") from exc
        if not accepted and not self.shutdown_event.is_set():
            raise RuntimeError(f"{self.role_name} {modality} embedding push was rejected during training")
        return accepted

    def _sync_barrier(self) -> bool:
        """Wait at the coordinator lock-step barrier; False means shutdown."""
        if self.shutdown_event.is_set():
            return False
        try:
            return bool(rpc.rpc_sync(self.config.coordinator_name, _rpc_sync_barrier, args=(self.role_name,)))
        except Exception:
            return False

    def _report_telemetry(self, modality: str, telemetry: dict[str, Any]) -> None:
        """Report real telemetry after a telemetry-free sync_push embedding push."""
        if self.shutdown_event.is_set():
            return
        with contextlib.suppress(Exception):
            rpc.rpc_sync(self.config.coordinator_name, _rpc_report_telemetry, args=(modality, telemetry))

    def run_training_loop(self) -> None:
        """Role-specific training loop implemented by worker subclasses."""
        raise RuntimeError(f"Role {self.role_name!r} does not own a local training loop")

    @property
    def _peer_seen(self) -> dict:
        """producer -> highest producer_step received so far (worker-side lag estimate)."""
        d = self.__dict__.get("_peer_seen_dict")
        if d is None:
            d = self.__dict__["_peer_seen_dict"] = {}
        return d

    def _stale_note(self, records) -> None:
        """staleness_stats: remember which training nodes the local peer cache now holds."""
        if not getattr(self.config, "staleness_stats", False):
            return
        st = self.__dict__.get("_stale_state")
        if st is None:
            st = self.__dict__["_stale_state"] = {
                "train": {int(n) for n in self.data_store.scheduler_train_nodes()}, "cov": set()}
        train, cov = st["train"], st["cov"]
        for r in records:
            nid = int(r.key.node_id)
            if nid in train:
                cov.add(nid)

    def _stale_coverage(self) -> float:
        """Share of training nodes present in the local peer cache (staleness_stats)."""
        st = self.__dict__.get("_stale_state")
        return len(st["cov"]) / max(1, len(st["train"])) if st else 0.0

    def ingest_remote_push(self, modality: str, records: list[EmbeddingRecord], source_name: str) -> None:
        """Ingest peer embeddings broadcast by the coordinator into the local cache."""
        raise RuntimeError(f"Role {self.role_name!r} cannot ingest {modality!r} pushes")

    def save_checkpoint(self, path: str) -> None:
        """Save this worker model checkpoint to path."""
        raise RuntimeError(f"Role {self.role_name!r} cannot save checkpoints")


# ──────────────────────────────────────────────────────────────────────
# Part 3: CoordinatorTrainingService, global cache routing and training control
# ──────────────────────────────────────────────────────────────────────

class CoordinatorTrainingService(BaseTrainingService):
    """Coordinator role, usually rank 0.

    Responsibilities:
      1. Maintain graph/text global caches, receive worker pushes, and broadcast.
      2. Serve lookup fallback requests for worker local-cache misses.
      3. Aggregate worker telemetry such as batch counts, loss, and cache hits.
      4. Periodically append metrics to metrics.jsonl.
      5. Periodically trigger worker checkpoint saves through RPC.
      6. Collect worker-finished signals and broadcast shutdown.

    The coordinator does not run forward/backward computation.
    """

    def __init__(
        self,
        *,
        config: AsyncTrainConfig,
        data_store: PubmedRuntimeStore,
        model_args: dict[str, Any],
    ) -> None:
        super().__init__(config=config, role_name=config.coordinator_name, data_store=data_store, model_args=model_args)
        # Embedding lag: exact measurement (LagTracker) and optional injection (DelayBuffer), see lag.py.
        self._lag = LagTracker()
        self._delay = DelayBuffer(steps=int(getattr(config, "lag_inject_steps", 0) or 0),
                                  seconds=float(getattr(config, "lag_inject_seconds", 0.0) or 0.0))
        self._lag_inject = self._delay.steps > 0 or self._delay.seconds > 0
        self._stale = StalenessStats() if getattr(config, "staleness_stats", False) else None
        if self._lag_inject:
            delayed_modality("graph", config.lag_inject_direction)       # validates the direction
            threading.Thread(target=self._lag_release_loop, name="lag-release", daemon=True).start()

        # --- Global caches ---
        # graph_cache stores graph embeddings pushed by graph_worker.
        # text_cache stores text embeddings pushed by text workers.
        # Both caches live on CPU and use LRU eviction with config-defined capacity.
        self.graph_cache = AsyncEmbeddingCache(
            device="cpu",
            capacity=config.graph_cache_capacity,
            dtype=config.cache_dtype,
            name="coordinator-graph-cache",
        )
        self.text_cache = AsyncEmbeddingCache(
            device="cpu",
            capacity=config.text_cache_capacity,
            dtype=config.cache_dtype,
            name="coordinator-text-cache",
        )

        # --- Thread-safe counters ---
        self.stats = Counter()          # Accumulator for all telemetry metrics.
        self._stats_lock = Lock()       # Protects stats and _finished_workers.

        # --- Synchronization and scheduling state ---
        self._ready_workers: set[str] = set()   # Workers that reported readiness.
        self._ready_event = Event()             # Set when every worker is ready.
        self._start_time = 0.0                  # Monotonic timestamp at training start.
        self._last_metrics_flush = 0.0          # Last metrics.jsonl append time.
        self._last_checkpoint_save = 0.0        # Last checkpoint save time.
        self._last_checkpoint_updates = 0       # Update count at the last checkpoint save.
        self._stop_reason = "running"           # Stop reason written to manifest.json.

        # --- sync_push lock-step barrier (2 parties: graph_worker + text_worker_0) ---
        # Generation-based reusable barrier. Both parties must arrive before either
        # proceeds; shutdown releases waiters immediately with False.
        self._sync_barrier_cond = Condition()
        self._sync_barrier_count = 0
        self._sync_barrier_gen = 0
        self._sync_barrier_parties = 2

        # --- Epoch-completion tracking ---
        # Workers call mark_worker_finished() after local epochs complete. The
        # coordinator broadcasts shutdown once all workers are finished.
        self._finished_workers: set[str] = set()

        # --- Validation early stopping (Scheme B) ---
        self._last_val_eval_time = 0.0          # Last validation monotonic time.
        self._best_val_auc = 0.0                # Best validation AUC seen so far.
        self._best_checkpoint_seconds: float | None = None
        self._best_selection_seconds: float | None = None
        self._val_no_improve_count = 0          # Consecutive non-improving evaluations.
        self._val_patience_ref = 0.0            # AUC the patience is measured against (moves only by > val_min_delta).
        self._logged_early_stop_defer = False   # One-shot log when an early stop is deferred (see _early_stop_blocked_by_text).
        self._val_history: list[dict[str, Any]] = []  # Full validation history.

    def run_training(self) -> dict[str, Any]:
        """Coordinator loop: wait for readiness, start workers, monitor, shut down.

        Returns the manifest dictionary containing stop_reason and final_metrics.
        """
        # 1. Wait for all workers to finish initialization.
        self.wait_for_workers_ready()

        # 2. Create output directory and reset the metrics file.
        self.config.train_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.metrics_path.write_text("", encoding="utf-8")

        # 3. Broadcast the start signal so workers begin drawing batches.
        self.allow_start()
        self._start_time = time.monotonic()
        self._last_metrics_flush = self._start_time
        self._last_checkpoint_save = self._start_time
        self._broadcast_start_signal()

        try:
            # 4. Main polling loop.
            while not self.shutdown_event.is_set():
                now = time.monotonic()

                # Stop normally when every worker has completed its epochs.
                if self._all_workers_finished():
                    self._stop_reason = "all_epochs_completed"
                    self.request_shutdown()
                    break

                # Enforce hard limits such as runtime or update-step ceilings.
                stop_reason = self._limit_stop_reason(now)
                if stop_reason is not None:
                    self._stop_reason = stop_reason
                    self.request_shutdown()
                    break

                # Validation via graph-worker link-prediction eval: it drives early stopping, and with
                # val_eval_always it also runs (for the AUC-over-time curve) when early stopping is off.
                if (self.config.val_early_stop or getattr(self.config, "val_eval_always", False)) \
                        and self._should_run_val_eval(now):
                    val_result = self._run_val_eval(now)
                    if self.config.val_early_stop and val_result is not None \
                            and self._val_no_improve_count >= self.config.val_patience:
                        if self._early_stop_blocked_by_text():
                            # Patience is exhausted but the text tower has never been updated, so
                            # the plateau says nothing about convergence. Keep running; the wall
                            # clock still bounds the run.
                            if not self._logged_early_stop_defer:
                                self._logged_early_stop_defer = True
                                with self._stats_lock:
                                    _gc = len(self.graph_cache)
                                    _text_updates = int(self.stats["text_updates_total"])
                                print(f"[coordinator] val patience exhausted at "
                                      f"{self._val_no_improve_count}/{self.config.val_patience} but the text side has "
                                      f"{_text_updates} aligned updates (graph cache covers {_gc} nodes) -- deferring early stop "
                                      f"until it has at least {self.config.val_early_stop_min_text_updates}",
                                      flush=True)
                        else:
                            with self._stats_lock:
                                self._text_guard_at_stop = int(self.stats["text_updates_total"])
                            self._stop_reason = "val_early_stop"
                            self.request_shutdown()
                            break

                # Periodically append metrics to metrics.jsonl.
                if self._should_flush_metrics(now):
                    self._append_metrics(self.metrics_snapshot(now))
                    self._last_metrics_flush = now

                # Periodically trigger checkpoint saves.
                if self._should_save_checkpoint(now):
                    self._save_latest_artifacts()
                    self._last_checkpoint_save = now
                    self._last_checkpoint_updates = self.successful_updates_total()

                time.sleep(0.1)
        finally:
            # 5. Always clean up after normal exit or exceptions.
            final_timestamp = time.monotonic()
            if self._stop_reason == "running":
                self._stop_reason = "shutdown_requested"

            # Broadcast shutdown to every worker.
            self._broadcast_shutdown_signal()

            # Save final checkpoints and genotype.
            self._save_latest_artifacts()

            # Write final metrics and manifest.
            final_metrics = self.metrics_snapshot(final_timestamp, final=True)
            self._append_metrics(final_metrics)

            # Total wall-clock training time in seconds, also surfaced in manifest.
            total_training_seconds = float(final_timestamp - self._start_time)

            # Throughput uses seed-node semantics to align with DDP/DeepSpeed scripts.
            # Count real seeds, including each partial end-of-pass batch. Extra
            # neighbor pushes are separate work and do not inflate this count.
            total_nodes_processed = int(final_metrics["graph_seed_nodes_total"])
            throughput_nodes_per_sec = (
                total_nodes_processed / total_training_seconds
                if total_training_seconds > 0 else 0.0
            )

            # Text-side throughput for test4 multi-GPU scaling: pushed text records
            # correspond to processed text nodes and should scale with text workers.
            # Graph throughput is capped by the single CPU graph worker, so text
            # throughput is the relevant scalability signal.
            text_nodes_processed = int(final_metrics["text_pushes_total"])
            text_throughput_nodes_per_sec = (
                text_nodes_processed / total_training_seconds
                if total_training_seconds > 0 else 0.0
            )

            # Legacy diagnostic event spans, never an SM/kernel-active measurement.
            # Keep the numeric field for existing consumers and label disabled timing.
            text_gpu_active_ms = float(final_metrics["text_gpu_active_ms"])
            gpu_active_seconds = (
                (text_gpu_active_ms / 1000.0) / max(self.config.num_text_workers, 1)
            )

            manifest = {
                "stage": self.config.stage,
                "graph_scheduler_policy": self.config.graph_scheduler_policy,
                "training_node_count": self.training_node_count,
                "paper_v3_attempt": os.environ.get("PAPER_V3_ATTEMPT"),
                "paper_v3_cell": os.environ.get("PAPER_V3_CELL"),
                "stop_reason": self._stop_reason,
                "total_training_seconds": total_training_seconds,
                "final_metrics": final_metrics,
                "val_history": self._val_history,
                "text_updates_at_stop_decision": getattr(self, "_text_guard_at_stop", None),
                "text_guard_min": self.config.val_early_stop_min_text_updates,
                "best_val_auc": self._best_val_auc,
                "best_checkpoint_seconds": self._best_checkpoint_seconds,
                "best_selection_seconds": self._best_selection_seconds,
                "train_ok": not (getattr(self, "_latest_artifact_save_errors", [])
                                 or getattr(self, "_best_artifact_save_errors", [])),
                "artifact_save_errors": {
                    "final": getattr(self, "_latest_artifact_save_errors", []),
                    "best": getattr(self, "_best_artifact_save_errors", []),
                },
                "best_ckpt": {
                    "graph": str(self.config.graph_best_ckpt_path),
                    "text": str(self.config.text_best_ckpt_path),
                    "genotype": (
                        str(self.config.genotype_best_file_path)
                        if self.config.stage == "search" else None
                    ),
                },
            }
            # Adaptive pruning: export the per-layer/group pruned-candidate summary
            # from graph_worker. RPC failure must not block manifest writing.
            if self.config.use_adaptive_pruning and self.config.stage == "search":
                with contextlib.suppress(Exception):
                    manifest["pruned_candidates"] = rpc.rpc_sync(
                        self.config.graph_worker_name,
                        _rpc_export_pruning_summary,
                    )
            # Record retrain warm-start sources for reproducibility and diagnosis.
            if self.config.stage == "retrain":
                manifest["warm_start"] = {
                    "enabled": bool(self.config.retrain_warm_start),
                    "graph_source": self.config.warm_start_graph_ckpt,
                    "text_source": self.config.warm_start_text_ckpt,
                }
            with self.config.manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)

            # Write a standalone training_time.json to match bin/ddp_train.py artifacts.
            time_path = self.config.train_output_dir / "training_time.json"
            with time_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": self.config.stage,
                        "graph_scheduler_policy": self.config.graph_scheduler_policy,
                        "training_node_count": self.training_node_count,
                        "paper_v3_attempt": os.environ.get("PAPER_V3_ATTEMPT"),
                        "paper_v3_cell": os.environ.get("PAPER_V3_CELL"),
                        "stop_reason": self._stop_reason,
                        "total_seconds": total_training_seconds,
                        "best_checkpoint_seconds": self._best_checkpoint_seconds,
                        "best_selection_seconds": self._best_selection_seconds,
                        "train_ok": manifest["train_ok"],
                        "total_minutes": total_training_seconds / 60.0,
                        "gpu_active_seconds": gpu_active_seconds,
                        "gpu_active_seconds_measured": final_metrics["text_cuda_event_timing_measured"],
                        "gpu_active_seconds_semantics": final_metrics["text_gpu_active_ms_semantics"],
                        "total_nodes_processed": total_nodes_processed,
                        "throughput_nodes_per_sec": throughput_nodes_per_sec,
                        "text_nodes_processed": text_nodes_processed,
                        "text_throughput_nodes_per_sec": text_throughput_nodes_per_sec,
                        "num_text_workers": self.config.num_text_workers,
                    },
                    handle,
                    indent=2,
                )

            # test6 time breakdown is emitted as per-iteration JSONL directly by
            # graph/text worker processes. The coordinator no longer merges it into
            # the old single-row breakdown format.

            print(
                f"[Coordinator:{self.config.stage}] total_training_seconds="
                f"{total_training_seconds:.2f}s ({total_training_seconds / 60.0:.2f} min)"
            )
            print(
                f"[Coordinator:{self.config.stage}] throughput_nodes_per_sec="
                f"{throughput_nodes_per_sec:.1f}  (total_nodes={total_nodes_processed}, "
                f"total_time={total_training_seconds:.2f}s)"
            )
            print(f"[Coordinator:{self.config.stage}] manifest={self.config.manifest_path}")
            print(f"[Coordinator:{self.config.stage}] training_time={time_path}")
        return manifest

    # --- Cache operations ---

    def lookup_embeddings(self, modality: str, node_ids: list[int]) -> list[EmbeddingRecord]:
        """Lookup embeddings for node_ids from the global cache.

        Returns only records that exist in cache, so the result may be shorter than
        the request. Workers call this through RPC after local-cache misses.
        """
        cache = self._cache_for_modality(modality)
        keys = [CacheKey(modality, int(node_id)) for node_id in node_ids]
        hits = cache.get_many(keys)
        return clone_embedding_records(
            (hits[key] for key in keys if key in hits), "cpu", dtype=self.config.cache_dtype,
        )

    def receive_push(
        self,
        modality: str,
        records: list[EmbeddingRecord],
        source_name: str,
        telemetry: dict[str, Any] | None = None,
    ) -> bool:
        """Receive worker embeddings, update cache, and broadcast to peer workers.

        Flow:
          1. Record telemetry such as batch count, loss, and hit rate.
          2. Insert records into the modality cache; stale versions are rejected.
          3. Broadcast only newly inserted fresh records to peer workers.

        Returns True when processing completes normally.
        """
        # Record telemetry under the stats lock.
        normalized_telemetry = dict(telemetry or {})
        normalized_telemetry.setdefault("worker_name", source_name)
        # Persist pruning and LM local-averaging events individually for auditability.
        pruning_events = normalized_telemetry.pop("pruning_events", None)
        sync_events = normalized_telemetry.pop("sync_events", None)
        # Lag: the pusher reports the peer embeddings its update used (a graph push describes text
        # embeddings the graph worker trained on, and vice versa); the producers' current step/batch is
        # refreshed from this push's own records BEFORE any injected delay, so it is always exact.
        lag_used = normalized_telemetry.pop("lag_used", None)
        stale_g = normalized_telemetry.pop("stale_G", None)
        stale_t = normalized_telemetry.pop("stale_T", None)
        stale_cov = normalized_telemetry.pop("stale_coverage", None)
        if self._stale is not None:
            if stale_g:
                self._stale.add("lag_G", stale_g)
            if stale_t:
                for producer, values in stale_t.items():
                    self._stale.add("lag_T", values)
                    self._stale.add(f"lag_T/{producer}", values)
            if stale_cov is not None:
                self._stale.coverage[source_name] = float(stale_cov)
                gate = float(getattr(self.config, "staleness_cover_gate", 0.98))
                if modality == "text" and stale_cov >= gate and source_name not in self._stale.gate_open_at:
                    self._stale.gate_open_at[source_name] = float(time.monotonic() - self._start_time)
        self._lag.update_from_records(records)
        self._lag.record_usage("graph_uses_text" if modality == "graph" else "text_uses_graph", lag_used)
        self._record_telemetry(modality, normalized_telemetry)
        if pruning_events:
            for event in pruning_events:
                self._append_metrics({"event": "pruning", **event})
        if sync_events:
            _sync_now = time.monotonic()
            for event in sync_events:
                self._append_metrics({
                    "event": "text_sync",
                    "uptime_seconds": float(_sync_now - self._start_time),
                    **event,
                })

        cpu_records = clone_embedding_records(records, "cpu", dtype=self.config.cache_dtype)
        # Lag injection: hold the push back from the global cache and the broadcast (so both the local
        # caches and the RPC fallback only ever see it late) until its producer has processed
        # lag_inject_steps more batches and lag_inject_seconds have passed (released by _lag_release_loop).
        if cpu_records and self._lag_inject and delayed_modality(modality, self.config.lag_inject_direction):
            first = cpu_records[0]
            self._delay.add(first.producer, first.producer_batch, first.created_at,
                            (modality, cpu_records, source_name))
            return True
        return self._commit_records(modality, cpu_records, source_name)

    def _commit_records(self, modality: str, cpu_records: list[EmbeddingRecord], source_name: str) -> bool:
        """Insert into the global cache (stale versions rejected) and broadcast the fresh records."""
        cache = self._cache_for_modality(modality)
        inserted, _ = cache.put_many(cpu_records)
        if not inserted:
            return True

        # Broadcast only records that were actually inserted.
        inserted_keys = set(inserted)
        fresh_records = [record for record in cpu_records if record.key in inserted_keys]
        if not fresh_records:
            return True

        # Broadcast to peer-modality workers. Avoid new RPC calls during shutdown.
        if not self.shutdown_event.is_set():
            for target in self._broadcast_targets_for_modality(modality):
                with contextlib.suppress(Exception):
                    rpc.rpc_async(target, _rpc_ingest_remote_push, args=(modality, fresh_records, source_name))
                with self._stats_lock:
                    self.stats["push_broadcast_count"] += 1
        return True

    def _lag_release_loop(self) -> None:
        """Commit held pushes once due (lag injection only); polls every 50 ms."""
        while not self.shutdown_event.is_set():
            for modality, cpu_records, source_name in self._delay.pop_due(self._lag.latest_batch):
                with contextlib.suppress(Exception):
                    self._commit_records(modality, cpu_records, source_name)
            self.shutdown_event.wait(0.05)

    # --- Worker lifecycle management ---

    def mark_worker_ready(self, role_name: str) -> None:
        """Record that a worker finished initialization."""
        with self._stats_lock:
            self._ready_workers.add(role_name)
            # world_size - 1 excludes the coordinator itself.
            if len(self._ready_workers) >= self.config.world_size - 1:
                self._ready_event.set()

    def mark_worker_finished(self, role_name: str) -> None:
        """Record that a worker has completed all assigned epochs.

        Once all workers are finished, _all_workers_finished() returns True and the
        coordinator broadcasts shutdown.
        """
        with self._stats_lock:
            self._finished_workers.add(role_name)

    def wait_for_workers_ready(self) -> None:
        """Block until every worker reports ready, or raise on timeout."""
        if self.config.world_size <= 1:
            return
        if not self._ready_event.wait(timeout=self.config.rpc_timeout_seconds):
            raise TimeoutError(
                f"Timed out waiting for workers to become ready; got {sorted(self._ready_workers)}"
            )

    # --- Metrics and checkpointing ---

    def successful_updates_total(self) -> int:
        """Return the cumulative successful update count across graph and text."""
        with self._stats_lock:
            return int(self.stats["graph_updates_total"] + self.stats["text_updates_total"])

    def metrics_snapshot(self, now: float, *, final: bool = False) -> dict[str, Any]:
        """Build a metrics snapshot for metrics.jsonl.

        Includes stage metadata, uptime, epoch configuration, and per-modality
        batch/update/drop/loss/cache-hit statistics.
        """
        with self._stats_lock:
            graph_loss_count = float(self.stats["graph_loss_count"])
            text_loss_count = float(self.stats["text_loss_count"])
            graph_cache_size, text_cache_size = len(self.graph_cache), len(self.text_cache)
            snapshot = {
                "stage": self.config.stage,
                "final": final,
                "stop_reason": self._stop_reason if final else None,
                "uptime_seconds": float(now - self._start_time),
                # Epoch configuration; graph and text can differ.
                "graph_epochs": self.config.graph_num_epochs,
                "text_epochs": self.config.text_num_epochs,
                "finished_workers": sorted(self._finished_workers),
                # Graph-side statistics.
                "graph_batches_total": float(self.stats["graph_batches_total"]),
                "graph_seed_nodes_total": float(self.stats["graph_seed_nodes_total"]),
                "graph_scheduler_policy": self.config.graph_scheduler_policy,
                "graph_completed_cycles": int(self.stats["graph_completed_cycles"]),
                "training_node_count": self.training_node_count,
                "graph_updates_total": float(self.stats["graph_updates_total"]),
                "graph_drops_total": float(self.stats["graph_drops_total"]),
                "graph_loss_mean": float(self.stats["graph_loss_sum"] / graph_loss_count) if graph_loss_count > 0 else 0.0,
                "graph_local_hit_nodes": float(self.stats["graph_local_hit_nodes"]),
                "graph_remote_hit_nodes": float(self.stats["graph_remote_hit_nodes"]),
                "graph_miss_nodes": float(self.stats["graph_miss_nodes"]),
                "graph_pushes_total": float(self.stats["graph_pushes_total"]),
                # Batches seeded from already-cached nodes; equals graph_batches_total
                # once the cold start is over (0 when graph_cache_sampling is off).
                "graph_cache_sampled_batches": float(self.stats["graph_cache_sampled_batches"]),
                "graph_weight_updates_total": float(self.stats["graph_weight_updates_total"]),
                "graph_arch_updates_total": float(self.stats["graph_arch_updates_total"]),
                # Text-side statistics.
                "text_batches_total": float(self.stats["text_batches_total"]),
                "text_updates_total": float(self.stats["text_updates_total"]),
                "text_drops_total": float(self.stats["text_drops_total"]),
                "text_loss_mean": float(self.stats["text_loss_sum"] / text_loss_count) if text_loss_count > 0 else 0.0,
                "text_local_hit_nodes": float(self.stats["text_local_hit_nodes"]),
                "text_remote_hit_nodes": float(self.stats["text_remote_hit_nodes"]),
                "text_miss_nodes": float(self.stats["text_miss_nodes"]),
                "text_pushes_total": float(self.stats["text_pushes_total"]),
                "text_gpu_active_ms": float(self.stats["text_gpu_active_ms"]),
                "text_cuda_event_timing_measured": bool(
                    self.stats["text_batches_total"] > 0
                    and self.stats["text_cuda_event_timing_measured_batches"] == self.stats["text_batches_total"]
                ),
                "text_gpu_active_ms_semantics": "legacy CUDA event spans; not kernel-active utilization; zero when disabled",
                # Broadcast and cache statistics.
                "push_broadcast_count": float(self.stats["push_broadcast_count"]),
                "coordinator_graph_cache_size": float(graph_cache_size),
                "coordinator_text_cache_size": float(text_cache_size),
                "coordinator_graph_cache_coverage": graph_cache_size / max(1, self.training_node_count),
                "coordinator_text_cache_coverage": text_cache_size / max(1, self.training_node_count),
                # Lag of the peer embeddings actually used in updates, per direction (lag.py): age (s),
                # producer successful-update steps, producer batches -- mean / p50 / p90 / max.
                "lag": self._lag.summary(),
                "lag_held_pushes": float(len(self._delay)),
                # Per producer: (successful updates, batches) as of its latest push -- per-replica step counts.
                "producer_progress": {p: {"updates": sb[0], "batches": sb[1]} for p, sb in sorted(self._lag.latest.items())},
            }
        if self._stale is not None:
            # lag_G (graph updates; after the C_G coverage gate) / lag_T (text-replica updates), exact
            # histograms summarised; coverage = each consumer's peer-cache share of training nodes.
            snapshot["staleness"] = {
                **self._stale.summary(),
                "coverage": dict(self._stale.coverage),
                "cg_gate_open_at": dict(self._stale.gate_open_at),
                "n_train": len(self.data_store.scheduler_train_nodes()),
            }
            if final:
                with contextlib.suppress(Exception):
                    with (self.config.train_output_dir / "staleness_hist.json").open("w", encoding="utf-8") as fh:
                        json.dump({k: {str(a): b for a, b in h.items()} for k, h in self._stale.histograms().items()}, fh)
        return snapshot

    def _append_metrics(self, record: dict[str, Any]) -> None:
        """Append one metrics record to metrics.jsonl as a JSON line."""
        with self.config.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _record_telemetry(self, modality: str, telemetry: dict[str, Any]) -> None:
        """Accumulate worker telemetry into stats counters.

        Counters are prefixed by modality ("graph_" or "text_") and track total
        batches, successful updates, drop-on-miss count, loss sums/counts, cache
        hit/miss nodes, and pushed embedding records.

        Graph-side telemetry also separates weight and architecture updates.
        """
        prefix = str(modality)
        with self._stats_lock:
            self.stats[f"{prefix}_batches_total"] += float(telemetry.get("batch_seen", 0.0))
            self.stats[f"{prefix}_updates_total"] += float(telemetry.get("update_applied", 0.0))
            self.stats[f"{prefix}_drops_total"] += float(telemetry.get("drop_on_miss", 0.0))
            self.stats[f"{prefix}_local_hit_nodes"] += float(telemetry.get("local_hit_nodes", 0.0))
            self.stats[f"{prefix}_remote_hit_nodes"] += float(telemetry.get("remote_hit_nodes", 0.0))
            self.stats[f"{prefix}_miss_nodes"] += float(telemetry.get("miss_nodes", 0.0))
            self.stats[f"{prefix}_pushes_total"] += float(telemetry.get("pushed_records", 0.0))
            if prefix == "graph":
                self.stats["graph_seed_nodes_total"] += float(telemetry.get("seed_nodes", 0.0))
                self.stats["graph_completed_cycles"] = max(
                    self.stats["graph_completed_cycles"], float(telemetry.get("completed_cycles", 0.0))
                )
            # Batches whose seeds came from the already-cached node pool; the gap to
            # {prefix}_batches_total is the cold-start window (see CacheAwareNodeScheduler).
            self.stats[f"{prefix}_cache_sampled_batches"] += float(telemetry.get("batch_from_cache", 0.0))
            # Legacy diagnostic event spans; the accompanying flag distinguishes
            # disabled timing from a measured zero. These are not kernel activity.
            self.stats[f"{prefix}_gpu_active_ms"] += float(telemetry.get("gpu_active_ms", 0.0))
            self.stats[f"{prefix}_cuda_event_timing_measured_batches"] += float(
                bool(telemetry.get("cuda_event_timing_measured", False))
            )
            # test6 phase timings are written directly by workers as per-iteration JSONL.

            # Only successful updates contribute to mean loss.
            if "loss" in telemetry and float(telemetry.get("update_applied", 0.0)) > 0:
                self.stats[f"{prefix}_loss_sum"] += float(telemetry["loss"])
                self.stats[f"{prefix}_loss_count"] += 1.0

            # Graph-side search separates weight-phase and arch-phase updates.
            phase = telemetry.get("phase")
            if prefix == "graph" and phase == "weight":
                self.stats["graph_weight_updates_total"] += float(telemetry.get("update_applied", 0.0))
            if prefix == "graph" and phase == "arch":
                self.stats["graph_arch_updates_total"] += float(telemetry.get("update_applied", 0.0))

    def report_telemetry(self, modality: str, telemetry: dict[str, Any]) -> None:
        """Report telemetry separately in sync_push mode.

        sync_push pushes embeddings first with telemetry={}, then reports real
        update telemetry after loss/backward/step. This mirrors receive_push
        accounting, including persisted pruning events.
        """
        normalized = dict(telemetry or {})
        pruning_events = normalized.pop("pruning_events", None)
        self._record_telemetry(modality, normalized)
        if pruning_events:
            for event in pruning_events:
                self._append_metrics({"event": "pruning", **event})

    def sync_barrier_arrive(self, role_name: str) -> bool:
        """Two-party lock-step barrier for sync_push mode.

        Returns True when both graph and text sides arrive. During shutdown it
        returns False immediately so callers can avoid deadlock.
        """
        del role_name  # Two-party barrier today; keep the argument for future diagnostics.
        with self._sync_barrier_cond:
            if self.shutdown_event.is_set():
                return False
            gen = self._sync_barrier_gen
            self._sync_barrier_count += 1
            if self._sync_barrier_count >= self._sync_barrier_parties:
                # Both parties arrived; advance the generation and wake the peer.
                self._sync_barrier_count = 0
                self._sync_barrier_gen += 1
                self._sync_barrier_cond.notify_all()
                return True
            # Wait for the peer with timeout polling so shutdown can wake promptly.
            while self._sync_barrier_gen == gen and not self.shutdown_event.is_set():
                self._sync_barrier_cond.wait(timeout=0.5)
            return self._sync_barrier_gen != gen  # Generation advanced means proceed; otherwise shutdown.

    # --- Internal helpers ---

    def _all_workers_finished(self) -> bool:
        """Return whether every worker has reported finished.

        The required set includes graph_worker and all text workers, excluding the
        coordinator.
        """
        required = {self.config.role_for_rank(rank) for rank in range(1, self.config.world_size)}
        return required.issubset(self._finished_workers)

    def _broadcast_start_signal(self) -> None:
        """Send the start signal to all workers."""
        for target_rank in range(1, self.config.world_size):
            rpc.rpc_sync(self.config.role_for_rank(target_rank), _rpc_request_start)

    def _broadcast_shutdown_signal(self) -> None:
        """Send the shutdown signal to all workers.

        RPC failures from already-exited workers are suppressed so other workers
        still receive the notification.
        """
        # Wake workers that may be blocked in the sync barrier.
        with self._sync_barrier_cond:
            self._sync_barrier_cond.notify_all()
        for target_rank in range(1, self.config.world_size):
            with contextlib.suppress(Exception):
                rpc.rpc_sync(self.config.role_for_rank(target_rank), _rpc_request_shutdown)

    @contextlib.contextmanager
    def _record_artifact_error(self, target: str, part: str):
        """Preserve teardown resilience while making a failed save unacceptably explicit."""
        try:
            yield
        except Exception as exc:
            getattr(self, f"_{target}_artifact_save_errors").append(
                {"part": part, "error": type(exc).__name__},
            )

    def _save_latest_artifacts(self) -> None:
        """Trigger worker checkpoint saves and genotype export.

        Model weights live in worker processes, so checkpoints must be saved
        through RPC. The coordinator pulls genotype metadata and writes it locally.
        """
        self._latest_artifact_save_errors = []
        # Save graph checkpoint in the graph_worker process.
        with self._record_artifact_error("latest", "graph"):
            rpc.rpc_sync(self.config.graph_worker_name, _rpc_save_checkpoint, args=(str(self.config.graph_ckpt_path),))
        # Save one text checkpoint from text_worker_0.
        with self._record_artifact_error("latest", "text"):
            rpc.rpc_sync(self.config.text_worker_name(0), _rpc_save_checkpoint, args=(str(self.config.text_ckpt_path),))
        # The search stage additionally exports the GNN genotype.
        if self.config.stage == "search":
            with self._record_artifact_error("latest", "genotype"):
                genotype = rpc.rpc_sync(self.config.graph_worker_name, _rpc_export_genotype)
                self.config.genotype_file_path.parent.mkdir(parents=True, exist_ok=True)
                with self.config.genotype_file_path.open("w", encoding="utf-8") as handle:
                    json.dump(genotype, handle, indent=2)
        # The retrain stage copies its genotype into the stage directory so eval
        # scripts can find {phase}/genotype.json.
        elif self.config.stage == "retrain" and self.config.genotype_path:
            with self._record_artifact_error("latest", "genotype"):
                src = Path(self.config.genotype_path).expanduser()
                dst = self.config.train_output_dir / "genotype.json"
                if src.exists() and src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def _save_best_artifacts(self) -> None:
        """Save best-validation-AUC checkpoints to *_best paths.

        These paths are separate from latest/final artifacts so final shutdown saves
        cannot overwrite the best checkpoint.
        """
        self._best_artifact_save_errors = []
        with self._record_artifact_error("best", "graph"):
            rpc.rpc_sync(
                self.config.graph_worker_name,
                _rpc_save_checkpoint,
                args=(str(self.config.graph_best_ckpt_path),),
            )
        with self._record_artifact_error("best", "text"):
            rpc.rpc_sync(
                self.config.text_worker_name(0),
                _rpc_save_checkpoint,
                args=(str(self.config.text_best_ckpt_path),),
            )
        if self.config.stage == "search":
            with self._record_artifact_error("best", "genotype"):
                genotype = rpc.rpc_sync(self.config.graph_worker_name, _rpc_export_genotype)
                self.config.genotype_best_file_path.parent.mkdir(parents=True, exist_ok=True)
                with self.config.genotype_best_file_path.open("w", encoding="utf-8") as handle:
                    json.dump(genotype, handle, indent=2)
        # Copy retrain genotype to the best path so eval --use-best can resolve it.
        elif self.config.stage == "retrain" and self.config.genotype_path:
            with self._record_artifact_error("best", "genotype"):
                src = Path(self.config.genotype_path).expanduser()
                dst = self.config.genotype_best_file_path
                if src.exists() and src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        if not self._best_artifact_save_errors:
            self._best_checkpoint_seconds = time.monotonic() - self._start_time

    def _limit_stop_reason(self, now: float) -> str | None:
        """Return a hard-stop reason when any configured limit is reached.

        Supported optional limits include max runtime, graph update steps, and text
        update steps.
        """
        elapsed = now - self._start_time
        if self.config.max_runtime_seconds is not None and elapsed >= float(self.config.max_runtime_seconds):
            return "max_runtime_seconds"

        with self._stats_lock:
            graph_updates = int(self.stats["graph_updates_total"])
            text_updates = int(self.stats["text_updates_total"])

        if self.config.max_graph_steps is not None and graph_updates >= int(self.config.max_graph_steps):
            return "max_graph_steps"
        if self.config.max_text_steps is not None and text_updates >= int(self.config.max_text_steps):
            return "max_text_steps"
        return None

    def _should_flush_metrics(self, now: float) -> bool:
        """Return whether metrics should be flushed based on the configured interval."""
        interval = float(self.config.metrics_interval_seconds)
        if interval <= 0:
            return False
        return (now - self._last_metrics_flush) >= interval

    def _should_save_checkpoint(self, now: float) -> bool:
        """Return whether a checkpoint should be saved.

        A save is triggered by either elapsed checkpoint time or enough successful
        updates since the last save.
        """
        time_interval = float(self.config.checkpoint_interval_seconds)
        if time_interval > 0 and (now - self._last_checkpoint_save) >= time_interval:
            return True

        step_interval = self.config.checkpoint_every_successful_updates
        if step_interval is None or int(step_interval) <= 0:
            return False
        return (self.successful_updates_total() - self._last_checkpoint_updates) >= int(step_interval)

    def _broadcast_targets_for_modality(self, modality: str) -> list[str]:
        """Return broadcast targets for a modality."""
        if modality == "graph":
            return [self.config.text_worker_name(index) for index in range(self.config.num_text_workers)]
        if modality == "text":
            return [self.config.graph_worker_name]
        raise ValueError(f"Unknown modality: {modality}")

    def _cache_for_modality(self, modality: str) -> AsyncEmbeddingCache:
        """Return the global cache object for a modality."""
        if modality == "graph":
            return self.graph_cache
        if modality == "text":
            return self.text_cache
        raise ValueError(f"Unknown modality: {modality}")

    # --- Validation early stopping ---

    def _should_run_val_eval(self, now: float) -> bool:
        """Return whether validation should run based on interval and update gates."""
        if (now - self._last_val_eval_time) < self.config.val_eval_interval_seconds:
            return False
        with self._stats_lock:
            graph_updates = int(self.stats["graph_updates_total"])
        return graph_updates >= self.config.val_eval_min_updates

    def _early_stop_blocked_by_text(self) -> bool:
        """True while the text tower has not yet performed enough aligned updates to stop on.

        A validation plateau is only evidence of convergence when both towers are training. Under
        drop-on-miss the text side stays at zero updates until the graph side has pushed an
        embedding for nearly every training node, so an early stop before that measures a frozen
        text tower.
        """
        minimum = int(self.config.val_early_stop_min_text_updates)
        if minimum <= 0:
            return False
        with self._stats_lock:
            text_updates = int(self.stats["text_updates_total"])
        return text_updates < minimum

    def _run_val_eval(self, now: float) -> dict[str, Any] | None:
        """Run validation via graph-worker RPC and update patience state."""
        self._last_val_eval_time = now
        _t_ev0 = time.time()
        try:
            result = rpc.rpc_sync(
                self.config.graph_worker_name,
                _rpc_run_val_eval,
                timeout=self.config.rpc_timeout_seconds,
            )
        except Exception:
            return None

        elapsed = now - self._start_time
        with self._stats_lock:
            graph_updates = int(self.stats["graph_updates_total"])
            text_updates = int(self.stats["text_updates_total"])

        entry = {
            "time": elapsed,
            # RPC round trip of this evaluation. It runs on the graph worker's RPC thread while the text
            # workers keep training, so it is recorded but does not pause the run.
            "eval_seconds": time.time() - _t_ev0,
            "graph_updates": graph_updates,
            "text_updates": text_updates,
            "auc": result["auc"],
            "ap": result["ap"],
            "recon": result["recon"],
            "num_val_nodes": result["num_val_nodes"],
            "num_val_edges": result["num_val_edges"],
            "best_auc": self._best_val_auc,
            "no_improve_count": self._val_no_improve_count,
        }
        # Search stage has extra supernet observations; retrain does not.
        if "super_auc" in result:
            entry["super_auc"] = result["super_auc"]
            entry["super_ap"] = result["super_ap"]
            entry["super_recon"] = result["super_recon"]

        if result["auc"] > self._best_val_auc:
            self._best_val_auc = result["auc"]
            self._best_selection_seconds = time.monotonic() - self._start_time
            # Save best artifacts separately from final checkpoints -- only when validation drives early
            # stopping. In curve-only mode (val_eval_always without early stop) the periodic check just
            # records the AUC, as the baselines' periodic sampling does, and checkpointing the text tower
            # every few seconds would slow the run it is measuring.
            if self.config.val_early_stop:
                self._save_best_artifacts()
        # Patience: a gain only counts if it exceeds val_min_delta over the last counted gain,
        # so a run that has plateaued (creeping up by 1e-3 per check) stops instead of refreshing
        # the patience forever. With val_min_delta=0 this is the plain "any strict improvement" rule.
        if result["auc"] - self._val_patience_ref > float(self.config.val_min_delta):
            self._val_patience_ref = result["auc"]
            self._val_no_improve_count = 0
        else:
            self._val_no_improve_count += 1

        entry["best_auc"] = self._best_val_auc
        entry["no_improve_count"] = self._val_no_improve_count
        self._val_history.append(entry)

        # Search reports greedy AUC as the primary metric plus optional supernet
        # observations; retrain reports a single AUC.
        super_str = (
            f"  super_auc={result['super_auc']:.4f}"
            if "super_auc" in result else ""
        )
        print(
            f"[Coordinator:val] auc={result['auc']:.4f}  ap={result['ap']:.4f}  "
            f"recon={result['recon']:.4f}  best_auc={self._best_val_auc:.4f}{super_str}  "
            f"patience={self._val_no_improve_count}/{self.config.val_patience}  "
            f"updates={graph_updates}  elapsed={elapsed:.1f}s"
        )

        # Append validation result to metrics.jsonl.
        self._append_metrics({**self.metrics_snapshot(now), "val_eval": entry})
        return result


# ──────────────────────────────────────────────────────────────────────
# Part 4: GraphWorkerTrainingService, graph encoding and architecture search
# ──────────────────────────────────────────────────────────────────────

class GraphWorkerTrainingService(BaseTrainingService):
    """Graph worker, usually rank 1, running the GNN encoder.

    Per-batch flow:
      1. Draw node_ids from the scheduler and build a k-hop subgraph.
      2. Run GNN forward to produce target_embeddings.
      3. Lookup matching text_embeddings from local/remote caches.
         - Full hit: compute contrastive loss, backward, and update parameters.
         - Miss: drop-on-miss skips the update.
      4. Push fresh graph embeddings to the coordinator regardless of update status.

    Architecture search:
      - Alternates weight_optimizer and arch_optimizer, following a DARTS-like
        bilevel schedule.
      - Odd successful updates use arch_optimizer; even updates use weight_optimizer.
      - current_phase() returns "weight" or "arch".
    """

    def __init__(
        self,
        *,
        config: AsyncTrainConfig,
        data_store: PubmedRuntimeStore,
        model_args: dict[str, Any],
    ) -> None:
        super().__init__(config=config, role_name=config.graph_worker_name, data_store=data_store, model_args=model_args)

        # Retrain loads the searched genotype to fix the architecture.
        genotype = None
        warm_graph_ckpt: Path | None = None
        if config.stage == "retrain":
            genotype = GraphGenotype.from_dict(_load_genotype_file(config.genotype_file_path))
            if config.retrain_warm_start:
                warm_graph_ckpt = _resolve_warm_start_path(config.warm_start_graph_ckpt)

        # --- Local cache for text embeddings broadcast by the coordinator ---
        self.text_cache = AsyncEmbeddingCache(
            device="cpu",
            capacity=config.text_cache_capacity,
            dtype=config.cache_dtype,
            name="graph-local-text-cache",
        )

        # --- GNN model ---
        # Build an unfixed tower first so search weights can be loaded before
        # genotype overwrites alpha as one-hot. Applying genotype before load_state_dict
        # would be undone by the continuous alpha values stored in the search checkpoint.
        # graph_device is the "without heterogeneous placement" ablation: the graph
        # tower can be moved onto the GPU shared with text workers. If CUDA is not
        # available, fall back to CPU without changing the full-system path.
        _graph_device = str(getattr(config, "graph_device", "cpu") or "cpu")
        if _graph_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[graph_worker] graph_device={_graph_device} but CUDA is unavailable -> cpu")
            _graph_device = "cpu"
        self.device = _graph_device
        if self.device != "cpu":
            print(f"[graph_worker] heterog-placement ablation: graph tower on {self.device}")
        self.graph_tower = GraphTower(model_args=model_args, genotype=None).to(self.device)
        self._warm_start_graph_source: str | None = None

        if warm_graph_ckpt is not None:
            payload = torch.load(warm_graph_ckpt, map_location="cpu", weights_only=False)
            model_state = payload.get("model_state", payload)
            # strict=False allows older checkpoints without mask_* buffers and newer
            # checkpoints with additional fields.
            result = self.graph_tower.load_state_dict(model_state, strict=False)
            _log_warm_start_load(
                warm_graph_ckpt,
                list(result.missing_keys),
                list(result.unexpected_keys),
                side="graph",
            )
            self._warm_start_graph_source = str(warm_graph_ckpt)

        if config.stage == "retrain":
            # Overwrite alpha as one-hot even if search checkpoint loaded continuous alpha.
            if genotype is not None:
                apply_graph_genotype(self.graph_tower.gnn, genotype, freeze_architecture=True)
            # Freeze alpha and keep ordinary weights/projection/tau trainable.
            self.graph_tower.freeze_architecture()

        # --- Optimizers ---
        # weight_optimizer updates ordinary GNN weights such as convolutions and MLPs.
        self.graph_weight_optimizer = torch.optim.AdamW(
            self.graph_tower.graph_weight_parameters(),
            lr=config.graph_weight_lr,
            weight_decay=float(model_args.get("weight_decay", 0.0)),
            eps=float(model_args.get("eps", 1.0e-8)),
            betas=tuple(model_args.get("betas", (0.9, 0.999))),
        )
        # arch_optimizer updates architecture parameters alpha during search only.
        self.graph_arch_optimizer = None
        if config.stage == "search":
            self.graph_arch_optimizer = torch.optim.AdamW(
                self.graph_tower.graph_arch_parameters(),
                lr=config.graph_arch_lr,
                weight_decay=0.0,  # Do not apply weight decay to architecture parameters.
                eps=float(model_args.get("eps", 1.0e-8)),
                betas=tuple(model_args.get("betas", (0.9, 0.999))),
            )

        # --- Batch scheduler ---
        # Batches cover the training seed-node set. completed_cycles tracks finished
        # epochs. scheduler_train_nodes() is the full train split by default or a
        # deterministic train_node_fraction subset. Graph and text sides share the same
        # seed-node set, improving push overlap and cache hit rate.
        _train_nodes = self.data_store.scheduler_train_nodes()
        if self.config.graph_cache_sampling:
            # Cover every training seed once per pass. Cached text seeds go first;
            # uncached remaining seeds still run forward and push without blocking.
            self.scheduler = CacheAwareNodeScheduler(
                node_ids=_train_nodes,
                batch_size=self.config.graph_batch_size,
                cached_node_ids=lambda: self.text_cache.node_ids("text"),
                seed=self.config.seed,
                explore_every=self.config.graph_cache_explore_every,
            )
        else:
            self.scheduler = StreamingNodeScheduler(
                node_ids=_train_nodes,
                batch_size=self.config.graph_batch_size,
                seed=self.config.seed,
            )

        # --- Version counters ---
        self._graph_version_counter = 0     # Monotonic version assigned to graph embeddings.
        self._successful_updates = 0        # Successful updates used for weight/arch alternation.

        # --- test6 telemetry: graph lane on CPU; async drop-on-miss has no barrier. ---
        self.telemetry: TelemetryRecorder | None = None
        if getattr(config, "profile", False):
            self.telemetry = TelemetryRecorder(
                system="tagnas",
                dataset=config.dataset,
                seed=config.seed,
                worker="graph",
                device=torch.device("cpu"),
                jsonl_path=derive_jsonl_path(
                    config.profile_out, system="tagnas", dataset=config.dataset,
                    seed=config.seed, worker=self.role_name,
                ),
                lit_module=None,            # Worker supplies perf_counter phases directly.
                rank=0,
                enabled=True,
                emit_barrier_sync_wait=False,
            )

    def run_training_loop(self) -> None:
        """Main graph-worker training loop.

        The worker keeps serving forward+push after local epochs complete so peer
        workers can keep receiving fresh embeddings until the coordinator shuts the
        job down.
        """
        epochs_done = False
        while not self.shutdown_event.is_set():
            batch = self.scheduler.next_batch()
            if self.config.sync_push:
                # In lock-step mode, False means shutdown and prevents deadlock.
                if not self.run_graph_batch_sync(batch):
                    break
            else:
                self.run_graph_batch(batch)
            if not epochs_done and self.scheduler.completed_cycles >= self.config.graph_num_epochs:
                epochs_done = True
                self._signal_finished()
                # Do not break; keep forward+push alive until coordinator shutdown.
        if self.telemetry is not None:
            self.telemetry.close(extra={"stage": self.config.stage,
                                        "num_text_workers": self.config.num_text_workers})

    def _signal_finished(self) -> None:
        """Notify the coordinator that this worker has completed all epochs.

        Exceptions are suppressed because the coordinator may already be tearing
        down RPC.
        """
        num = self.config.graph_num_epochs
        print(
            f"[{self.role_name}] Epoch [{num}/{num}] "
            f"all epochs completed, waiting for other workers to finish..."
        )
        with contextlib.suppress(Exception):
            rpc.rpc_sync(self.config.coordinator_name, _rpc_mark_worker_finished, args=(self.role_name,))

    def _anneal_arch_temperature(self) -> None:
        """Geometrically anneal architecture softmax temperature during search.

        tau moves from start to end by max_graph_steps, or 2000 if unset. tau < 1
        sharpens alpha, encourages commitment, enables faithful pruning, and makes
        greedy genotype extraction more stable. start==end is a no-op.
        """
        ts = float(self.config.graph_arch_temp_start)
        te = float(self.config.graph_arch_temp_end)
        if te == ts or self.config.stage != "search":
            return
        horizon = int(self.config.max_graph_steps or 2000)
        prog = min(1.0, self._successful_updates / max(1, horizon))
        tau = ts * (te / ts) ** prog
        self.graph_tower.gnn.set_arch_temperature(tau)

    def run_graph_batch(self, node_ids: list[int]) -> dict[str, Any]:
        """Run one full graph batch step.

        Args:
            node_ids: Node ids processed by this batch.

        Returns:
            Dictionary with loss, update status, and cache-hit statistics.

        Steps: choose the active optimizer, build a k-hop subgraph, run GNN
        forward, lookup peer text embeddings, optionally update on full hit, then
        push fresh graph embeddings to the coordinator.
        """
        self._batches_made = getattr(self, "_batches_made", 0) + 1   # producer_batch counter (lag.py)
        self.graph_tower.train(mode=True)
        _tel_t_start = time.time()   # Absolute iteration start for cross-process overlap telemetry.

        # Determine the current phase and configure GNN search behavior.
        phase = self.current_phase()
        if self.config.stage == "search":
            # set_search_phase selects architecture-mode vs weight-mode forward behavior.
            self.graph_tower.set_search_phase(train_architecture=phase == "arch")
        self._anneal_arch_temperature()

        # Select optimizer and clear gradients.
        optimizer = self._optimizer_for_phase(phase)
        optimizer.zero_grad()

        # --- Forward ---
        # Build the k-hop subgraph around target nodes; test6 buckets this with perf_counter.
        _t_b0 = time.perf_counter()
        batch = self.data_store.build_graph_batch(node_ids, self.config.graph_k_hop)
        extra_local, extra_node_ids = self._select_extra_push(batch)
        _t_b1 = time.perf_counter()
        # On CPU graph_device, .to() is a no-op; GPU ablation moves subgraph tensors.
        encoded = self.graph_tower.encode(
            batch.graph_x.to(self.device),          # Node features.
            batch.graph_edge_index.to(self.device),  # Edge indices [2, E].
            batch.graph_edge_attr.to(self.device) if batch.graph_edge_attr is not None else None,
            batch.target_index.to(self.device),      # Target-node indices in the subgraph.
            is_search=self.config.stage == "search",
            **({"extra_index": extra_local.to(self.device)} if extra_local is not None else {}),
        )
        target_embeddings, extra_embeddings = encoded if extra_local is not None else (encoded, None)
        _t_b2 = time.perf_counter()

        # --- Fetch peer embeddings ---
        # Cross-modal cache lookup is XferT; drop-on-miss does not block, so sync_wait is near zero.
        text_records, cache_stats = self._get_text_embeddings(node_ids)
        _t_b3 = time.perf_counter()

        # --- Backward + Update ---
        update_applied = 0
        loss_value = 0.0
        _prof_bw_ms = 0.0
        _prof_upd_ms = 0.0
        pruning_events: list[dict[str, Any]] = []
        if text_records is not None:
            # All text embeddings are available, so contrastive loss can be computed.
            loss = self._contrastive_loss(target_embeddings, text_records)
            _t_bw0 = time.perf_counter()
            loss.backward()
            # During arch phase, force pruned alpha entries to stay inactive before step.
            if self.config.use_adaptive_pruning and phase == "arch":
                self.graph_tower.gnn.zero_pruned_grads()
            parameters = [parameter for parameter in self.graph_tower.parameters() if parameter.requires_grad]
            if self.config.gradient_clip_norm > 0:
                nn_utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm)
            _t_bw1 = time.perf_counter()
            optimizer.step()
            _prof_bw_ms = (_t_bw1 - _t_bw0) * 1000.0
            _prof_upd_ms = (time.perf_counter() - _t_bw1) * 1000.0
            update_applied = 1
            loss_value = float(loss.detach().cpu().item())
            self._successful_updates += 1
            # Broadcast the learned temperature tau after a successful update.
            self._broadcast_temperature()
            # Periodic pruning runs only during search/arch phase. Pruned candidates
            # are permanently removed from forward/backward by zero probability and gradients.
            if (
                self.config.use_adaptive_pruning
                and self.config.stage == "search"
                and phase == "arch"
                and self._successful_updates >= self.config.pruning_min_updates_before_start
                and (self._successful_updates % max(1, self.config.pruning_check_interval_updates)) == 0
            ):
                events = self.graph_tower.gnn.prune_once(
                    threshold=float(self.config.pruning_threshold),
                    min_keep=int(self.config.pruning_min_candidates_per_layer),
                    mode=str(self.config.pruning_threshold_mode),
                    groups=tuple(self.config.pruning_groups),
                    threshold_abs=float(self.config.pruning_threshold_abs),
                )
                for event in events:
                    event["successful_updates"] = int(self._successful_updates)
                pruning_events = events

        # --- Push embeddings to coordinator ---
        # Fresh embeddings are pushed even when the weight update is skipped. Exact by-product
        # embeddings (graph_push_extra_max) ride along; they carry this forward's weights, like
        # the seeds' -- the post-step weights only reach the cache at the next batch either way.
        records = self._make_records(node_ids, target_embeddings)
        if extra_embeddings is not None:
            records += self._make_records(extra_node_ids, extra_embeddings)
        telemetry_payload: dict[str, Any] = {
            "worker_name": self.role_name,
            "batch_seen": 1.0,
            "seed_nodes": len(node_ids),
            "completed_cycles": self.scheduler.completed_cycles,
            "update_applied": float(update_applied),
            "drop_on_miss": float(1 - update_applied),
            "loss": loss_value,
            "epoch": self.scheduler.completed_cycles + 1,
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
            "pushed_records": float(len(records)),
            "pushed_extra_records": float(len(records) - len(node_ids)),
            # 0 while the local text cache holds fewer than one batch of training nodes
            # (the cold-start window), 1 once seeds are drawn from cached nodes; the
            # coordinator accumulates it as graph_cache_sampled_batches.
            "batch_from_cache": float(bool(getattr(self.scheduler, "last_batch_from_cache", False))),
            "phase": phase,
        }
        if pruning_events:
            # Attach pruning events; coordinator writes them to metrics.jsonl.
            telemetry_payload["pruning_events"] = pruning_events
        if update_applied and cache_stats.get("lag_used"):
            telemetry_payload["lag_used"] = cache_stats["lag_used"]      # peer embeddings this update used
        if "stale_T" in cache_stats:
            telemetry_payload["stale_T"] = cache_stats["stale_T"]
            telemetry_payload["stale_coverage"] = cache_stats["stale_coverage"]
        self.push_embeddings("graph", records, telemetry=telemetry_payload)

        # test6 telemetry: graph lane writes per-iteration JSONL directly.
        if self.telemetry is not None:
            _hit = float(cache_stats["local_hit_nodes"] + cache_stats["remote_hit_nodes"])
            _tot = _hit + float(cache_stats["miss_nodes"])
            self.telemetry.write_external_iter(
                {
                    "build_batch": _t_b1 - _t_b0,
                    "graph_fwd": _t_b2 - _t_b1,
                    "xfer": _t_b3 - _t_b2,
                    "backward": _prof_bw_ms / 1000.0,
                    "update": _prof_upd_ms / 1000.0,
                },
                t_start=_tel_t_start,
                t_end=time.time(),
                nodes_in_batch=len(node_ids),
                cache_hit=(_hit / _tot) if _tot > 0 else 1.0,
                staleness_steps=float(cache_stats.get("mean_staleness", 0.0)),
            )
        return {
            "loss": loss_value,
            "updated": float(update_applied),
            "dropped": float(1 - update_applied),
            "pushed": float(len(records)),
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
        }

    def run_graph_batch_sync(self, node_ids: list[int]) -> bool:
        """Run one lock-step sync_push graph batch.

        Order: forward, push local embeddings, wait for peer push, lookup peer
        embeddings, compute/update, report telemetry, then wait for both updates.
        Returns False when shutdown is requested.
        """
        self._batches_made = getattr(self, "_batches_made", 0) + 1   # producer_batch counter (lag.py)
        self.graph_tower.train(mode=True)
        phase = self.current_phase()
        if self.config.stage == "search":
            self.graph_tower.set_search_phase(train_architecture=phase == "arch")
        self._anneal_arch_temperature()
        optimizer = self._optimizer_for_phase(phase)
        optimizer.zero_grad()

        # --- Forward ---
        batch = self.data_store.build_graph_batch(node_ids, self.config.graph_k_hop)
        target_embeddings = self.graph_tower.encode(
            batch.graph_x, batch.graph_edge_index, batch.graph_edge_attr, batch.target_index,
            is_search=self.config.stage == "search",
        )

        # Push embeddings first with empty telemetry, then wait at the barrier.
        records = self._make_records(node_ids, target_embeddings)
        self.push_embeddings("graph", records, telemetry={})
        if not self._sync_barrier():
            return False

        # After the barrier, peer text embeddings must be in coordinator cache.
        text_records, cache_stats = self._get_text_embeddings(node_ids)
        update_applied = 0
        loss_value = 0.0
        pruning_events: list[dict[str, Any]] = []
        if text_records is not None:
            loss = self._contrastive_loss(target_embeddings, text_records)
            loss.backward()
            if self.config.use_adaptive_pruning and phase == "arch":
                self.graph_tower.gnn.zero_pruned_grads()
            parameters = [p for p in self.graph_tower.parameters() if p.requires_grad]
            if self.config.gradient_clip_norm > 0:
                nn_utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm)
            optimizer.step()
            update_applied = 1
            loss_value = float(loss.detach().cpu().item())
            self._successful_updates += 1
            self._broadcast_temperature()
            if (
                self.config.use_adaptive_pruning
                and self.config.stage == "search"
                and phase == "arch"
                and self._successful_updates >= self.config.pruning_min_updates_before_start
                and (self._successful_updates % max(1, self.config.pruning_check_interval_updates)) == 0
            ):
                events = self.graph_tower.gnn.prune_once(
                    threshold=float(self.config.pruning_threshold),
                    min_keep=int(self.config.pruning_min_candidates_per_layer),
                    mode=str(self.config.pruning_threshold_mode),
                    groups=tuple(self.config.pruning_groups),
                    threshold_abs=float(self.config.pruning_threshold_abs),
                )
                for event in events:
                    event["successful_updates"] = int(self._successful_updates)
                pruning_events = events

        # Report real telemetry separately after the update.
        telemetry_payload: dict[str, Any] = {
            "worker_name": self.role_name,
            "batch_seen": 1.0,
            "seed_nodes": len(node_ids),
            "completed_cycles": self.scheduler.completed_cycles,
            "update_applied": float(update_applied),
            "drop_on_miss": float(1 - update_applied),
            "loss": loss_value,
            "epoch": self.scheduler.completed_cycles + 1,
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
            "pushed_records": float(len(records)),
            "phase": phase,
        }
        if pruning_events:
            telemetry_payload["pruning_events"] = pruning_events
        # Reuse the async push accounting path with empty records. The older
        # _report_telemetry path left sync-mode counters at zero, which broke node
        # counts, max_graph_steps gates, and validation best-checkpoint tracking.
        self.push_embeddings("graph", [], telemetry=telemetry_payload)

        # Wait until both sides complete this batch update before moving on.
        return self._sync_barrier()

    def ingest_remote_push(self, modality: str, records: list[EmbeddingRecord], source_name: str) -> None:
        """Ingest text embeddings broadcast by the coordinator."""
        del source_name
        if modality != "text":
            raise ValueError(f"Graph worker expected text pushes, got {modality!r}")
        _note_peer_steps(self._peer_seen, records)
        self._stale_note(records)
        self.text_cache.put_many(records)

    def save_checkpoint(self, path: str) -> None:
        """Save a graph model checkpoint.

        Stores model weights, weight optimizer state, tau, and successful update
        count. Search additionally stores arch optimizer state and current genotype.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "stage": self.config.stage,
            "model_state": clone_state_to_cpu(self.graph_tower),
            "weight_optimizer": self.graph_weight_optimizer.state_dict(),
            "tau": float(self.graph_tower.tau.detach().cpu().item()),
            "successful_updates": int(self._successful_updates),
        }
        if self.graph_arch_optimizer is not None:
            payload["arch_optimizer"] = self.graph_arch_optimizer.state_dict()
            payload["genotype"] = self.export_genotype().to_dict()
        torch.save(payload, checkpoint_path)

    def export_genotype(self) -> GraphGenotype:
        """Export the current GNN genotype during search."""
        return export_graph_genotype(self.graph_tower.gnn)

    def run_validation_eval(self) -> dict[str, Any]:
        """Run link-prediction evaluation on the validation-induced subgraph.

        This zero-copy path reads self.graph_tower parameters directly from the RPC
        thread under no_grad. Occasional reads during an update are acceptable for
        convergence monitoring.

        Mode selection:
          - search: evaluate both greedy discrete architecture and supernet. Greedy
            metrics drive best tracking and early stopping; supernet metrics are
            recorded as observations.
          - retrain: alpha is fixed one-hot, so greedy evaluation is sufficient.
        """
        # Monitoring split: val by default, or test when val_eval_on_test is enabled.
        _eval_split = "test" if getattr(self.config, "val_eval_on_test", False) else "val"
        val_node_ids = self.data_store.split_nodes(_eval_split)
        empty_payload = {
            "auc": 0.0, "ap": 0.0, "recon": 0.0,
            "num_val_nodes": len(val_node_ids), "num_val_edges": 0,
        }
        if len(val_node_ids) < 2:
            return empty_payload

        val_indices = torch.tensor(
            [self.data_store.node_id_to_graph_index[nid] for nid in val_node_ids],
            dtype=torch.long,
        )
        val_ei = self.data_store.split_edge_index(_eval_split)
        val_ea = self.data_store.split_edge_attr(_eval_split)

        if val_ei.size(1) == 0:
            return empty_payload

        # Gather validation-node features.
        vx = self.data_store.graph_data.x[val_indices].float()

        # Map val_ei from global graph indices into local validation-subgraph indices.
        num_total_nodes = int(self.data_store.graph_data.node_ids.size(0))
        global_to_local = torch.full((num_total_nodes,), -1, dtype=torch.long)
        global_to_local[val_indices] = torch.arange(len(val_indices))
        local_ei = global_to_local[val_ei]

        # Match datamodule/final-eval/baseline semantics by dropping isolated nodes
        # in the induced subgraph. Otherwise high-order GIN on sparse test-test
        # subgraphs sees many zero-degree nodes and systematically underestimates
        # online AUC relative to final eval.
        local_ei, _, keep_mask = pyg_utils.remove_isolated_nodes(
            local_ei, num_nodes=len(val_indices)
        )
        vx = vx[keep_mask]

        # Edge features.
        edge_attr = val_ea
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros(local_ei.size(1), 1, dtype=vx.dtype)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Negatives follow the same rule as the final eval and the baselines' online monitors
        # (bin/ddp_train.py, UniTG): sample_negative_edges(mode=val_negative_sampling, default "auto"),
        # drawn once with a fixed seed and reused, so successive checks are comparable. Uniform
        # negatives used to be drawn here: on datasets whose final eval switches to type-aware
        # negatives (twitter, amazon-baby) they are separable by node type alone, so an untrained
        # model scored 0.95 online (twitter, 3 updates) against 0.53 in the final eval, and early
        # stopping / best tracking latched onto that meaningless peak.
        if getattr(self, "_val_neg_cache", None) is None or self._val_neg_cache[0] != _eval_split:
            from ..eval_neg_sampling import sample_negative_edges
            kept_ids = torch.tensor(val_node_ids, dtype=torch.long)[keep_mask]
            _neg_seed = 2969591811          # bin/ddp_train.py's _NEG_SEED
            torch.manual_seed(_neg_seed)
            neg, neg_info = sample_negative_edges(
                pos_edge_index=local_ei,
                num_nodes=vx.size(0),
                graph_node_ids=kept_ids,
                dataset_name=self.config.dataset,
                data_dir=str((Path("data") / self.config.dataset).resolve()),
                mode=str(getattr(self.config, "val_negative_sampling", "auto")),
                seed=_neg_seed,
                device=torch.device("cpu"),
            )
            self._val_neg_cache = (_eval_split, neg.cpu(), neg_info.actual_mode)
            print(f"[val-eval] negatives: requested={neg_info.requested_mode} actual={neg_info.actual_mode} "
                  f"reason={neg_info.reason} pos={local_ei.size(1)} neg={neg.size(1)}", flush=True)
        neg_ei = self._val_neg_cache[1]

        # graph_device ablation may place the tower on GPU; move eval tensors with it.
        _dev = self.device
        vx = vx.to(_dev)
        local_ei = local_ei.to(_dev)
        edge_attr = edge_attr.to(_dev)
        neg_ei = neg_ei.to(_dev)

        # Prepare multi-layer adjacency cache shared by both forwards.
        layers = self.graph_tower.gnn.num_layers
        self.graph_tower.gnn.adjs = (
            local_ei.unsqueeze(0).repeat(layers, 1, 1),
            edge_attr.unsqueeze(0).repeat(layers, 1, 1),
        )

        decoder = pyg_ae.InnerProductDecoder()
        pos_y = torch.ones(local_ei.size(1))
        neg_y = torch.zeros(neg_ei.size(1))
        y = torch.cat([pos_y, neg_y], dim=0).long().cpu().numpy()
        eps = 1e-15

        def _forward_and_score(is_search: bool) -> dict[str, float]:
            self.graph_tower.gnn.step = 1
            with torch.no_grad():
                out = self.graph_tower.gnn(vx, local_ei, edge_attr, is_search=is_search)
                h = out["output"] if out["output"] is not None else out["last_hidden_state"]
                h = self.graph_tower.proj(h)
                h = F.normalize(h, p=2, dim=1)
            pos_dec = decoder(h, local_ei, sigmoid=True)
            neg_dec = decoder(h, neg_ei, sigmoid=True)
            pred = torch.cat([pos_dec, neg_dec], dim=0).detach().cpu().numpy()
            return {
                "auc": float(sklearn_metrics.roc_auc_score(y, pred)),
                "ap": float(sklearn_metrics.average_precision_score(y, pred)),
                "recon": float(
                    (-torch.log(pos_dec + eps).mean()
                     - torch.log(1 - neg_dec + eps).mean()).item()
                ),
            }

        if self.config.stage == "search":
            # During search, greedy is the deployment architecture and primary metric.
            primary = _forward_and_score(is_search=False)
            secondary = _forward_and_score(is_search=True)
            return {
                "auc": primary["auc"],
                "ap": primary["ap"],
                "recon": primary["recon"],
                "super_auc": secondary["auc"],
                "super_ap": secondary["ap"],
                "super_recon": secondary["recon"],
                "num_val_nodes": int(vx.size(0)),
                "num_val_edges": int(local_ei.size(1)),
                "neg_mode": self._val_neg_cache[2],
            }

        # During retrain and other fixed modes, alpha is one-hot and greedy is enough.
        primary = _forward_and_score(is_search=False)
        return {
            "auc": primary["auc"],
            "ap": primary["ap"],
            "recon": primary["recon"],
            "num_val_nodes": int(vx.size(0)),
            "num_val_edges": int(local_ei.size(1)),
        }

    def current_phase(self) -> str:
        """Return the current graph optimization phase: weight or arch.

        Search alternates phases by successful update count; retrain is always
        weight phase.
        """
        if self.config.stage != "search":
            return "weight"
        return "arch" if self._successful_updates % 2 == 1 else "weight"

    def _optimizer_for_phase(self, phase: str) -> torch.optim.Optimizer:
        """Return the optimizer for the requested graph phase."""
        if phase == "arch":
            if self.graph_arch_optimizer is None:
                raise RuntimeError("arch optimizer is unavailable outside search stage")
            return self.graph_arch_optimizer
        return self.graph_weight_optimizer

    def _broadcast_temperature(self) -> None:
        """Broadcast graph_tower's learned log-temperature to all text workers.

        Text workers use the same contrastive temperature. rpc_async avoids blocking
        the current graph update path.
        """
        raw_tau = float(self.graph_tower.tau.detach().cpu().item())
        for worker_index in range(self.config.num_text_workers):
            rpc.rpc_async(self.config.text_worker_name(worker_index), _rpc_set_temperature, args=(raw_tau,))

    def _contrastive_loss(self, left: torch.Tensor, right_records: list[EmbeddingRecord]) -> torch.Tensor:
        """Compute graph-to-text contrastive loss.

        left is graph embeddings [B, D]; right_records are matching text
        EmbeddingRecord objects. The loss is symmetric square contrastive loss.
        """
        right = torch.stack([record.tensor.float() for record in right_records], dim=0).to(left.device)
        logits = left @ right.T * self.graph_tower.temperature()
        return ls.square_contrastive_loss(logits)

    def _select_extra_push(self, batch) -> tuple[torch.Tensor | None, list[int]]:
        """Pick up to graph_push_extra_max exact by-product rows of this batch to push.

        Within this batch's exact subgraph candidates, prefer never-pushed nodes,
        then least recently pushed; break ties randomly. These extra rows speed
        coverage but do not guarantee it: isolated nodes require seed traversal.
        Returns (local subgraph indices, node ids), or (None, []) when disabled.
        """
        cap = int(getattr(self.config, "graph_push_extra_max", 0) or 0)
        if cap <= 0 or batch.subset_index is None:
            return None, []
        exact_local = self.data_store.exact_train_nodes(batch, self.config.graph_k_hop)
        last = getattr(self, "_extra_last_push", None)
        if last is None:
            last = torch.full((self.data_store.graph_data.node_ids.size(0),), -1.0)
            self._extra_last_push = last
        self._extra_push_step = getattr(self, "_extra_push_step", 0) + 1
        last[batch.subset_index[batch.target_index]] = float(self._extra_push_step)
        if exact_local.numel() == 0:
            return None, []
        exact_global = batch.subset_index[exact_local]
        if exact_local.numel() > cap:
            key = last[exact_global] + torch.rand(exact_global.numel()) * 0.5
            pick = torch.topk(key, cap, largest=False).indices
            exact_local, exact_global = exact_local[pick], exact_global[pick]
        last[exact_global] = float(self._extra_push_step)
        node_ids = self.data_store.graph_data.node_ids[exact_global].tolist()
        return exact_local, [int(n) for n in node_ids]

    def _make_records(self, node_ids: list[int], embeddings: torch.Tensor) -> list[EmbeddingRecord]:
        """Wrap graph forward outputs into EmbeddingRecord objects.

        Each record stores modality+node key, tensor, monotonic version, producer,
        and producer step. Versions prevent stale records from replacing newer ones.
        """
        records: list[EmbeddingRecord] = []
        for index, node_id in enumerate(node_ids):
            records.append(
                EmbeddingRecord(
                    key=CacheKey("graph", int(node_id)),
                    tensor=embeddings[index].detach(),
                    version=self._graph_version_counter + index + 1,
                    producer=self.role_name,
                    producer_step=self._successful_updates,
                    producer_batch=getattr(self, "_batches_made", 0),
                )
            )
        self._graph_version_counter += len(records)
        return clone_embedding_records(records, "cpu")

    def _get_text_embeddings(self, node_ids: list[int]) -> tuple[list[EmbeddingRecord] | None, dict[str, int]]:
        """Fetch text embeddings for node_ids to compute contrastive loss.

        Lookup strategy:
          1. Local graph-local-text-cache.
          2. Coordinator global-cache RPC for local misses.
          3. Return None on remaining misses, triggering drop-on-miss.

        Returns:
            (records, stats). records is None when any node is missing.
        """
        keys = [CacheKey("text", int(node_id)) for node_id in node_ids]

        # w/o drop-on-miss ablation: poll until peer records arrive or timeout.
        _deadline: float | None = None
        if getattr(self.config, "block_on_miss", False):
            _deadline = time.monotonic() + float(getattr(self.config, "block_on_miss_timeout_seconds", 30.0))

        while True:
            # Level 1: local cache lookup.
            local_hits = self.text_cache.get_many(keys)
            misses = [key for key in keys if key not in local_hits]

            # Level 2: remote coordinator fallback.
            remote_hits = 0
            if misses and not self.shutdown_event.is_set():
                try:
                    fetched = rpc.rpc_sync(
                        self.config.coordinator_name,
                        _rpc_lookup_cache,
                        args=("text", [key.node_id for key in misses]),
                    )
                    remote_hits = len(fetched)
                    # Store remotely fetched records locally to avoid repeated RPC.
                    _note_peer_steps(self._peer_seen, fetched)
                    self._stale_note(fetched)
                    self.text_cache.put_many(fetched)
                except Exception:
                    # RPC can fail during shutdown; ignore quietly.
                    pass

            # Merge results.
            final_hits = self.text_cache.get_many(keys)
            remaining = [key for key in keys if key not in final_hits]
            stats = {
                "local_hit_nodes": len(local_hits),
                "remote_hit_nodes": remote_hits,
                "miss_nodes": len(remaining),
                # test6 staleness: peer embeddings' lag in the producer's updates (worker-side estimate).
                "mean_staleness": _mean_staleness(final_hits.values(), self._peer_seen),
            }
            if getattr(self.config, "staleness_stats", False):
                # lag_T of every hit, in the producing text replica's own successful updates.
                by_producer: dict[str, list[int]] = {}
                for r in final_hits.values():
                    if r.producer_step >= 0:
                        cur = int(self._peer_seen.get(str(r.producer), r.producer_step))
                        by_producer.setdefault(str(r.producer), []).append(max(0, cur - int(r.producer_step)))
                stats["stale_T"] = by_producer
                stats["stale_coverage"] = self._stale_coverage()
            if not remaining:
                # Return in original key order to stay aligned with node_ids.
                ordered = clone_embedding_records(
                    (final_hits[key] for key in keys), "cpu", dtype=self.config.cache_dtype,
                )
                stats["lag_used"] = used_lag_payload(final_hits.values())
                return ordered, stats
            if (
                _deadline is None
                or self.shutdown_event.is_set()
                or time.monotonic() >= _deadline
            ):
                # Missing nodes remain: drop-on-miss, or timeout fallback from block mode.
                return None, stats
            time.sleep(0.2)


# ──────────────────────────────────────────────────────────────────────
# Part 5: TextWorkerTrainingService, GPU text encoding
# ──────────────────────────────────────────────────────────────────────

class TextWorkerTrainingService(BaseTrainingService):
    """Text worker, usually rank 2+, running the LM encoder on GPU.

    The flow mirrors GraphWorker:
      1. Draw node_ids from the scheduler and build a tokenized text batch.
      2. Run LM forward to produce text_embeddings.
      3. Lookup matching graph_embeddings from local/remote caches.
      4. On hit, compute contrastive loss, backward, all_reduce, and optimizer.step.
         On miss, drop-on-miss skips the update, but collective synchronization may
         still need to participate depending on mode.
      5. Push fresh text embeddings to the coordinator.

    Multi-worker gradient synchronization:
      - In manual mode, text workers share model parameters through explicit all_reduce.
      - All text workers must participate in every collective.
      - If a worker drops a batch and has no gradient, it contributes zero tensors
        to preserve collective ordering.
    """

    def __init__(
        self,
        *,
        config: AsyncTrainConfig,
        role_name: str,
        data_store: PubmedRuntimeStore,
        model_args: dict[str, Any],
        text_group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__(config=config, role_name=role_name, data_store=data_store, model_args=model_args)

        # --- Worker identity and device ---
        self.worker_index = config.text_worker_index(role_name)  # Parse 0 from "text_worker_0".
        self.device = config.text_device(self.worker_index)      # Prefer GPU devices.
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.worker_index)

        # --- Local cache for graph embeddings broadcast by the coordinator ---
        self.graph_cache = AsyncEmbeddingCache(
            device=self.device,
            capacity=config.graph_cache_capacity,
            dtype=config.cache_dtype,
            name=f"{role_name}-local-graph-cache",
        )

        # --- LM model ---
        self.text_tower = TextTower(
            model_args={**model_args, "lm_grad_checkpointing": bool(config.text_grad_checkpointing)},
            local_files_only=config.local_files_only,
        ).to(self.device)
        self._warm_start_text_source: str | None = None

        # --- Temperature parameter ---
        # Learned by graph_worker and broadcast to text workers as read-only state.
        self.max_tau = model_args.get("max_tau")
        self.current_tau = torch.tensor(float(model_args.get("tau_init", 3.5)), device=self.device)

        # --- Retrain warm start: load search-stage text weights and tau ---
        warm_text_ckpt: Path | None = None
        if config.stage == "retrain" and config.retrain_warm_start:
            warm_text_ckpt = _resolve_warm_start_path(config.warm_start_text_ckpt)

        if warm_text_ckpt is not None:
            payload = torch.load(warm_text_ckpt, map_location=self.device, weights_only=False)
            model_state = payload.get("model_state", payload)
            result = self.text_tower.load_state_dict(model_state, strict=False)
            _log_warm_start_load(
                warm_text_ckpt,
                list(result.missing_keys),
                list(result.unexpected_keys),
                side="text",
            )
            self._warm_start_text_source = str(warm_text_ckpt)
            # Inherit final search tau so graph and text sides start consistently.
            if isinstance(payload, dict) and "tau" in payload:
                self.current_tau = torch.tensor(float(payload["tau"]), device=self.device)

        # --- Optimizer ---
        # Always start a fresh optimizer; warm start transfers model weights, not optimizer state.
        self.optimizer = torch.optim.AdamW(
            self.text_tower.parameters(),
            lr=config.lm_lr,
            weight_decay=float(model_args.get("weight_decay", 0.0)),
            eps=float(model_args.get("eps", 1.0e-8)),
            betas=tuple(model_args.get("betas", (0.9, 0.999))),
        )

        # --- Gradient synchronization ---
        # text_group is the ProcessGroup among text workers and is used for all_reduce
        # when text synchronization is enabled.
        self.text_group = text_group
        self.sync_world_size = 1
        self._ddp_text: DistributedDataParallel | None = None
        # --- LM periodic weighted averaging (post-local-SGD) state ---
        self._local_avg = False          # Whether periodic weighted averaging is enabled.
        self._local_avg_stop = False     # Set after rendezvous stop vote; main loop exits.
        self._batches_since_sync = 0     # Batches since last average, including drops.
        self._updates_since_sync = 0     # Real updates since last average, used as c_i.
        self._sync_count = 0             # Completed rendezvous count; workers must match exactly.
        if self.text_group is not None and self.config.text_sync_enabled:
            self.sync_world_size = dist.get_world_size(group=self.text_group)
            _use_local_avg = bool(getattr(self.config, "text_local_avg", False)) and not self.config.sync_push
            _use_ddp = bool(getattr(self.config, "text_ddp", False)) and not self.config.sync_push
            if _use_local_avg:
                # local_avg takes precedence over DDP: local optimizer steps run
                # normally and rendezvous performs weighted parameter averaging.
                self._local_avg = True
                self._broadcast_parameters_from_primary()
            elif _use_ddp:
                # True DDP uses buckets and overlaps communication with backward,
                # replacing manual per-parameter all_reduce. DDP broadcasts initial
                # parameters at construction. Constant LM buffers do not need sync.
                _dev_ids = [self.worker_index] if self.device.startswith("cuda") else None
                self._ddp_text = DistributedDataParallel(
                    self.text_tower,
                    device_ids=_dev_ids,
                    output_device=(_dev_ids[0] if _dev_ids else None),
                    process_group=self.text_group,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                )
            else:
                # Manual per-parameter all_reduce path: broadcast initial parameters.
                self._broadcast_parameters_from_primary()

        # --- Streaming batch scheduler ---
        # In async mode, offset seeds let text workers traverse batches differently.
        # In sync_push mode, text and graph must use identical order and batch size
        # so lookup must hit after the lock-step barrier.
        text_seed = self.config.seed if self.config.sync_push else (self.config.seed + 1000 + self.worker_index)
        # scheduler_train_nodes() is the same seed-node set used by graph side.
        # text_seed changes only order, not membership, so both sides cover the same nodes.
        _train_nodes = self.data_store.scheduler_train_nodes()
        if self.config.text_shard_nodes and self.config.num_text_workers > 1:
            # Each worker owns a disjoint 1/W shard, so the W workers together cover the
            # training set W times faster than when each shuffles the whole set; the graph
            # side's cross-modal cache therefore fills W times sooner.
            _shard = shard_node_ids(
                _train_nodes,
                num_shards=self.config.num_text_workers,
                shard_index=self.worker_index,
                seed=self.config.seed,
            )
            print(f"[{self.role_name}] node shard {self.worker_index + 1}/{self.config.num_text_workers}: "
                  f"{len(_shard)} of {len(_train_nodes)} training nodes", flush=True)
            _train_nodes = _shard
        _budget = int(getattr(self.config, "text_token_budget", 0) or 0)
        _probe = bool(getattr(self.config, "text_probe_worst_case", False))
        _lengths: dict[int, int] = {}
        if (_budget > 0 or _probe) and not self.config.sync_push:
            _t_len = time.perf_counter()
            # The probe needs the longest text of the WHOLE pool (every worker forces the same one,
            # as B5's LM_PROBE_WORST_CASE does on every rank); plain token budgeting only its shard.
            _lengths = self.data_store.text_token_lengths(
                self.data_store.scheduler_train_nodes() if _probe else _train_nodes)
            print(f"[{self.role_name}] token lengths of {len(_lengths)} nodes in "
                  f"{time.perf_counter() - _t_len:.1f}s (max {max(_lengths.values())})", flush=True)
        if _budget > 0 and not self.config.sync_push:
            # Length-bucketed chunks bounded by rows x padded length instead of a row count.
            self.scheduler = TokenBudgetNodeScheduler(
                node_ids=_train_nodes,
                lengths=_lengths,
                token_budget=_budget,
                max_rows=int(getattr(self.config, "text_max_chunk_rows", 1024)),
                seed=text_seed,
            )
            print(f"[{self.role_name}] token-budget chunks: budget={_budget} max_rows="
                  f"{self.scheduler.max_rows} -> {len(self.scheduler._chunks)} chunks/pass "
                  f"(mean {len(_train_nodes) / max(len(self.scheduler._chunks), 1):.1f} rows)", flush=True)
        else:
            self.scheduler = StreamingNodeScheduler(
                node_ids=_train_nodes,
                batch_size=self.config.text_chunk_size,
                seed=text_seed,
            )
        if _probe and _lengths and not self.config.sync_push:
            self._force_worst_case_first_chunk(_lengths, _budget)
        self._text_version_counter = 0  # Embedding version counter.
        self._successful_updates = 0    # Successful updates for producer_step/staleness.

        # --- test6 telemetry: text lane on GPU; all_reduce contributes to sync_wait. ---
        self.telemetry: TelemetryRecorder | None = None
        if getattr(config, "profile", False):
            self.telemetry = TelemetryRecorder(
                system="tagnas",
                dataset=config.dataset,
                seed=config.seed,
                worker="text",
                device=torch.device(self.device),
                jsonl_path=derive_jsonl_path(
                    config.profile_out, system="tagnas", dataset=config.dataset,
                    seed=config.seed, worker=self.role_name,
                ),
                lit_module=None,            # Worker supplies phases directly with CUDA events.
                rank=0,
                enabled=True,
                emit_barrier_sync_wait=False,
            )

    def run_training_loop(self) -> None:
        """Main text-worker training loop, symmetric to graph worker.

        After local epochs complete, the worker notifies the coordinator but keeps
        serving forward+push until shutdown.
        """
        epochs_done = False
        while True:
            # local_avg exits only after the collective rendezvous stop vote. A
            # worker must not exit alone before a collective, or peers can deadlock.
            # Other modes keep the regular shutdown check at loop head.
            if self._local_avg:
                if self._local_avg_stop:
                    break
            elif self.shutdown_event.is_set():
                break
            batch = self.scheduler.next_batch()
            if self.config.sync_push:
                if not self.run_text_batch_sync(batch):
                    break
            else:
                self.run_text_batch(batch)
            if not epochs_done and self.scheduler.completed_cycles >= self.config.text_num_epochs:
                epochs_done = True
                self._signal_finished()
                # Do not break; keep forward+push alive until coordinator shutdown.
        if self.telemetry is not None:
            self.telemetry.close(extra={"stage": self.config.stage,
                                        "num_text_workers": self.config.num_text_workers})

    def _signal_finished(self) -> None:
        """Notify the coordinator that this worker completed all epochs."""
        num = self.config.text_num_epochs
        print(
            f"[{self.role_name}] Epoch [{num}/{num}] "
            f"all epochs completed, waiting for other workers to finish..."
        )
        with contextlib.suppress(Exception):
            rpc.rpc_sync(self.config.coordinator_name, _rpc_mark_worker_finished, args=(self.role_name,))

    def run_text_batch(self, node_ids: list[int]) -> dict[str, Any]:
        """Run one full text batch step.

        Args:
            node_ids: Node ids processed by this batch.

        Returns:
            Dictionary with loss, update status, and cache-hit statistics.

        Steps: tokenize, run LM forward, fetch graph embeddings, optionally compute
        loss/backward/update on hit, synchronize gradients when required, and push
        fresh text embeddings to the coordinator.
        """
        self._batches_made = getattr(self, "_batches_made", 0) + 1   # producer_batch counter (lag.py)
        self.text_tower.train(mode=True)
        self.optimizer.zero_grad()
        _tel_t_start = time.time()   # Absolute iteration start for cross-process overlap telemetry.
        _prof = bool(self.config.profile)

        # Optional CUDA-event spans are diagnostic timings, not kernel activity.
        # Ordinary training must not synchronize the whole GPU just to read them.
        _use_cuda = str(self.device).startswith("cuda") and _prof
        gpu_active_ms = 0.0
        # test6 dual-lane phases: CPU/RPC spans use perf_counter; GPU spans use
        # CUDA events and are resolved after synchronize.
        _ph_build = _ph_xfer = _ph_sync = 0.0
        _prof_bw_ms = _prof_upd_ms = 0.0
        _ev_pbw0 = _ev_pbw1 = _ev_pup0 = _ev_pup1 = None
        if _use_cuda:
            _ev_f0 = torch.cuda.Event(enable_timing=True)
            _ev_f1 = torch.cuda.Event(enable_timing=True)

        # --- Forward ---
        _t_bld0 = time.perf_counter()
        batch = self.data_store.build_text_batch(node_ids)
        _ph_build = time.perf_counter() - _t_bld0
        _enc_ids = batch.input_ids.to(self.device)
        _enc_mask = batch.attention_mask.to(self.device)
        if _use_cuda:
            _ev_f0.record()
        # DDP mode must call the DDP wrapper so backward triggers bucketed all_reduce.
        if self._ddp_text is not None:
            text_embeddings = self._ddp_text(_enc_ids, _enc_mask)
        else:
            text_embeddings = self.text_tower.encode(_enc_ids, _enc_mask)
        if _use_cuda:
            _ev_f1.record()

        # --- Fetch peer embeddings; cross-modal cache lookup is XferT. ---
        _t_x0 = time.perf_counter()
        graph_records, cache_stats = self._get_graph_embeddings(node_ids)
        _ph_xfer = time.perf_counter() - _t_x0

        # --- Backward ---
        update_applied = 0
        loss_value = 0.0
        if _use_cuda:
            _ev_b0 = torch.cuda.Event(enable_timing=True)
            _ev_b1 = torch.cuda.Event(enable_timing=True)
            _ev_b0.record()
        if graph_records is not None:
            # All graph embeddings are available.
            graph_embeddings = torch.stack([record.tensor.float() for record in graph_records], dim=0).to(self.device)
            loss = self._contrastive_loss(text_embeddings, graph_embeddings)
            if _prof and _use_cuda:
                _ev_pbw0 = torch.cuda.Event(enable_timing=True)
                _ev_pbw1 = torch.cuda.Event(enable_timing=True)
                _ev_pbw0.record()
            loss.backward()
            if _ev_pbw1 is not None:
                _ev_pbw1.record()
        elif self._ddp_text is not None:
            # In DDP mode, misses still need a zero backward pass to trigger bucketed
            # all_reduce on every rank; otherwise peers can block in collectives.
            (text_embeddings.sum() * 0.0).backward()

        # Gradient synchronization:
        #   - DDP mode already performs bucketed all_reduce during backward.
        #   - Manual mode requires every text worker to participate every round;
        #     _synchronize_gradients fills zero gradients on misses.
        _t_sw0 = time.perf_counter()
        if self._local_avg:
            pass  # local_avg skips per-step grad sync and averages parameters every K batches.
        elif self._ddp_text is None:
            self._synchronize_gradients()
        _ph_sync = time.perf_counter() - _t_sw0

        if graph_records is not None:
            if self.config.gradient_clip_norm > 0:
                nn_utils.clip_grad_norm_(list(self.text_tower.parameters()), self.config.gradient_clip_norm)
            if _prof and _use_cuda:
                _ev_pup0 = torch.cuda.Event(enable_timing=True)
                _ev_pup1 = torch.cuda.Event(enable_timing=True)
                _ev_pup0.record()
            self.optimizer.step()
            if _ev_pup1 is not None:
                _ev_pup1.record()
            update_applied = 1
            self._successful_updates += 1

        # --- LM periodic weighted averaging (post-local-SGD) after local steps ---
        _sync_event = None
        if self._local_avg:
            self._batches_since_sync += 1
            if update_applied:
                self._updates_since_sync += 1
            _k_eff = 1 if self._sync_count < self.config.text_sync_warmup else max(1, self.config.text_sync_interval)
            if self._batches_since_sync >= _k_eff:
                _res = self._periodic_weighted_average()   # Collective: all text workers must call together.
                self._sync_count += 1
                if _res.get("stop", 0.0) > 0.0:
                    self._local_avg_stop = True            # Collective stop vote exits all workers together.
                if _res.get("did_sync", 0.0) > 0.0:
                    _sync_event = {
                        "worker": self.role_name,
                        "k_eff": float(_k_eff),
                        "div_l2": _res.get("div_l2"),
                        "div_l2_measured": bool(_res.get("div_l2_measured", False)),
                        "sync_updates": float(_res.get("sync_updates", 0.0)),
                        "weight": float(_res.get("sync_weight", 0.0)),
                    }

        _lm_ms = 0.0
        if _use_cuda:
            _ev_b1.record()
            torch.cuda.synchronize(self.device)
            _lm_ms = _ev_f0.elapsed_time(_ev_f1)   # LM forward GPU time for test6 LMComp.
            gpu_active_ms = _lm_ms + _ev_b0.elapsed_time(_ev_b1)
            if _ev_pbw0 is not None and _ev_pbw1 is not None:
                _prof_bw_ms = _ev_pbw0.elapsed_time(_ev_pbw1)
            if _ev_pup0 is not None and _ev_pup1 is not None:
                _prof_upd_ms = _ev_pup0.elapsed_time(_ev_pup1)

        # --- Push embeddings to coordinator ---
        records = self._make_records(node_ids, text_embeddings)
        # The batched RPC payload D2H has already completed the preceding work;
        # read the scalar afterwards rather than inserting another early fence.
        if update_applied:
            loss_value = float(loss.detach().cpu().item())
        _text_tel = {
            "worker_name": self.role_name,
            "batch_seen": 1.0,
            "update_applied": float(update_applied),
            "drop_on_miss": float(1 - update_applied),
            "loss": loss_value,
            "epoch": self.scheduler.completed_cycles + 1,
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
            "pushed_records": float(len(records)),
            "lag_used": cache_stats.get("lag_used") if update_applied else None,   # peer embeddings this update used
            "gpu_active_ms": float(gpu_active_ms),
            "cuda_event_timing_measured": _use_cuda,
        }
        if "stale_coverage" in cache_stats:
            _text_tel["stale_coverage"] = cache_stats["stale_coverage"]
            if cache_stats.get("stale_G"):
                _text_tel["stale_G"] = cache_stats["stale_G"]
        if self.config.profile:
            _text_tel["prof_lm_ms"] = float(_lm_ms)
        if _sync_event is not None:
            _text_tel["sync_events"] = [_sync_event]
        self.push_embeddings("text", records, telemetry=_text_tel)

        # test6 telemetry: text lane writes per-iteration JSONL directly.
        if self.telemetry is not None:
            _hit = float(cache_stats["local_hit_nodes"] + cache_stats["remote_hit_nodes"])
            _tot = _hit + float(cache_stats["miss_nodes"])
            self.telemetry.write_external_iter(
                {
                    "build_batch": _ph_build,
                    "lm_fwd": _lm_ms / 1000.0,
                    "xfer": _ph_xfer,
                    "backward": _prof_bw_ms / 1000.0,
                    "sync_wait": _ph_sync,
                    "update": _prof_upd_ms / 1000.0,
                },
                t_start=_tel_t_start,
                t_end=time.time(),
                nodes_in_batch=len(node_ids),
                cache_hit=(_hit / _tot) if _tot > 0 else 1.0,
                staleness_steps=float(cache_stats.get("mean_staleness", 0.0)),
            )
        return {
            "loss": loss_value,
            "updated": float(update_applied),
            "dropped": float(1 - update_applied),
            "pushed": float(len(records)),
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
        }

    def _encode_text_microbatched(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, micro_batch: int
    ) -> torch.Tensor:
        """Encode [B, L] text in micro-batches and concatenate embeddings.

        GPT-style models have no cross-sequence attention inside a batch, so chunked
        encoding is numerically equivalent to one full-batch encode. During training,
        checkpoint each chunk to reduce backward activation memory from O(B) to
        O(micro_batch), which prevents long-sequence sync_push OOM stalls.
        """
        total = int(input_ids.shape[0])
        micro = max(1, int(micro_batch))
        if total <= micro:
            return self.text_tower.encode(
                input_ids.to(self.device), attention_mask.to(self.device)
            )
        chunks: list[torch.Tensor] = []
        use_ckpt = torch.is_grad_enabled()
        for start in range(0, total, micro):
            ii = input_ids[start:start + micro].to(self.device)
            am = attention_mask[start:start + micro].to(self.device)
            if use_ckpt:
                emb = torch_checkpoint.checkpoint(
                    self.text_tower.encode, ii, am, use_reentrant=False
                )
            else:
                emb = self.text_tower.encode(ii, am)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    def run_text_batch_sync(self, node_ids: list[int]) -> bool:
        """Run one lock-step sync_push text batch.

        Order: forward, push local embeddings, wait at barrier, lookup peer graph
        embeddings, loss/backward, text-side gradient sync, optimizer step, telemetry,
        and final barrier. Returns False on shutdown.
        """
        self._batches_made = getattr(self, "_batches_made", 0) + 1   # producer_batch counter (lag.py)
        self.text_tower.train(mode=True)
        self.optimizer.zero_grad()

        _use_cuda = str(self.device).startswith("cuda") and bool(self.config.profile)
        gpu_active_ms = 0.0
        if _use_cuda:
            _ev_f0 = torch.cuda.Event(enable_timing=True)
            _ev_f1 = torch.cuda.Event(enable_timing=True)
            _ev_f0.record()

        # --- Forward with memory-safe micro-batching and checkpointing ---
        batch = self.data_store.build_text_batch(node_ids)
        text_embeddings = self._encode_text_microbatched(
            batch.input_ids, batch.attention_mask, self.config.sync_text_micro_batch
        )
        if _use_cuda:
            _ev_f1.record()

        # Push local embeddings first, then wait for peer push at the barrier.
        records = self._make_records(node_ids, text_embeddings)
        self.push_embeddings("text", records, telemetry={})
        if not self._sync_barrier():
            return False

        # After the barrier, peer graph embeddings must be in coordinator cache.
        graph_records, cache_stats = self._get_graph_embeddings(node_ids)

        update_applied = 0
        loss_value = 0.0
        if _use_cuda:
            _ev_b0 = torch.cuda.Event(enable_timing=True)
            _ev_b1 = torch.cuda.Event(enable_timing=True)
            _ev_b0.record()
        if graph_records is not None:
            graph_embeddings = torch.stack([record.tensor.float() for record in graph_records], dim=0).to(self.device)
            loss = self._contrastive_loss(text_embeddings, graph_embeddings)
            loss.backward()
        # Collective all_reduce: every text worker participates each round.
        self._synchronize_gradients()
        if graph_records is not None:
            if self.config.gradient_clip_norm > 0:
                nn_utils.clip_grad_norm_(list(self.text_tower.parameters()), self.config.gradient_clip_norm)
            self.optimizer.step()
            update_applied = 1
            loss_value = float(loss.detach().cpu().item())
        if _use_cuda:
            _ev_b1.record()
            torch.cuda.synchronize(self.device)
            gpu_active_ms = _ev_f0.elapsed_time(_ev_f1) + _ev_b0.elapsed_time(_ev_b1)

        # Report real telemetry after update through the reliable push accounting path.
        self.push_embeddings("text", [], telemetry={
            "worker_name": self.role_name,
            "batch_seen": 1.0,
            "update_applied": float(update_applied),
            "drop_on_miss": float(1 - update_applied),
            "loss": loss_value,
            "epoch": self.scheduler.completed_cycles + 1,
            "local_hit_nodes": float(cache_stats["local_hit_nodes"]),
            "remote_hit_nodes": float(cache_stats["remote_hit_nodes"]),
            "miss_nodes": float(cache_stats["miss_nodes"]),
            "pushed_records": float(len(records)),
            "lag_used": cache_stats.get("lag_used") if update_applied else None,   # peer embeddings this update used
            "gpu_active_ms": float(gpu_active_ms),
            "cuda_event_timing_measured": _use_cuda,
        })

        # Wait until both sides finish this batch update.
        return self._sync_barrier()

    def ingest_remote_push(self, modality: str, records: list[EmbeddingRecord], source_name: str) -> None:
        """Ingest graph embeddings broadcast by the coordinator."""
        del source_name
        if modality != "graph":
            raise ValueError(f"Text worker expected graph pushes, got {modality!r}")
        _note_peer_steps(self._peer_seen, records)
        self._stale_note(records)
        self.graph_cache.put_many(records)

    def set_temperature(self, raw_tau: float) -> None:
        """Receive the log-space temperature broadcast by graph_worker."""
        self.current_tau = torch.tensor(float(raw_tau), device=self.device)

    def save_checkpoint(self, path: str) -> None:
        """Save a text model checkpoint.

        Only worker_index=0 writes the file to avoid concurrent writes. The payload
        includes model weights, optimizer state, and current temperature.
        """
        if self.worker_index != 0:
            return
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": self.config.stage,
            "model_state": clone_state_to_cpu(self.text_tower),
            "optimizer": self.optimizer.state_dict(),
            "tau": float(self.current_tau.detach().cpu().item()),
        }
        torch.save(payload, checkpoint_path)

    def temperature(self) -> torch.Tensor:
        """Compute the actual temperature exp(tau), optionally clamped by max_tau.

        tau is stored in log-space. max_tau prevents extremely large temperatures
        from degrading the softmax.
        """
        value = self.current_tau.exp()
        if self.max_tau is None:
            return value
        max_value = torch.tensor(float(self.max_tau), device=self.device).exp()
        return value.clamp(max=max_value)

    def _contrastive_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute text-to-graph contrastive loss, mirroring graph worker."""
        logits = left @ right.T * self.temperature()
        return ls.square_contrastive_loss(logits)

    def _broadcast_parameters_from_primary(self) -> None:
        """Broadcast initial parameters and buffers from the primary text worker.

        This aligns manual-sync workers before training. DDP performs its own
        initial broadcast and does not use this path.
        """
        src_rank = self.config.text_worker_ranks[0]
        for parameter in self.text_tower.parameters():
            dist.broadcast(parameter.data, src=src_rank, group=self.text_group)
        for _, buffer in self.text_tower.named_buffers():
            dist.broadcast(buffer.data, src=src_rank, group=self.text_group)

    def _synchronize_gradients(self) -> None:
        """All-reduce text-worker gradients and average them.

        This is a collective operation: every text worker must participate for
        every parameter. When a worker drops a batch and has no gradient, it
        contributes a zero tensor to preserve collective ordering.
        """
        if self.text_group is None or not self.config.text_sync_enabled:
            return
        for parameter in self.text_tower.parameters():
            if parameter.grad is None:
                # A miss leaves grad as None; use zeros to preserve all_reduce alignment.
                parameter.grad = torch.zeros_like(parameter.data)
            dist.all_reduce(parameter.grad.data, op=dist.ReduceOp.SUM, group=self.text_group)
            parameter.grad.data.div_(self.sync_world_size)

    def _periodic_weighted_average(self) -> dict[str, Any]:
        """Run a post-local-SGD weighted-averaging rendezvous for the LM.

        Parameters and AdamW moments are averaged using each worker's successful
        update count c_i since the previous rendezvous. All text workers must call
        collectives in the same order; zero placeholders are used for missing state.
        Returns {did_sync, stop, sync_updates, sync_weight, div_l2}.
        """
        out = {"did_sync": 0.0, "stop": 0.0, "sync_updates": 0.0, "sync_weight": 0.0,
               "div_l2": None, "div_l2_measured": False}
        if self.text_group is None or not self.config.text_sync_enabled:
            self._batches_since_sync = 0
            self._updates_since_sync = 0
            return out
        grp = self.text_group
        dev = self.device
        c_local = float(self._updates_since_sync)

        # 1. Sum update counts and vote on shutdown. Any stop vote exits all workers after rendezvous.
        c_sum_t = torch.tensor([c_local], device=dev, dtype=torch.float32)
        dist.all_reduce(c_sum_t, op=dist.ReduceOp.SUM, group=grp)
        stop_t = torch.tensor([1.0 if self.shutdown_event.is_set() else 0.0], device=dev, dtype=torch.float32)
        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX, group=grp)
        out["stop"] = 1.0 if stop_t.item() > 0.0 else 0.0
        c_sum = float(c_sum_t.item())
        out["sync_updates"] = c_sum

        # If no worker updated in this period, skip averaging after aligned collectives.
        if c_sum <= 0.0:
            self._batches_since_sync = 0
            self._updates_since_sync = 0
            return out

        w = c_local / c_sum  # Worker weight; c_i=0 receives the global average without contributing.
        out["sync_weight"] = w

        # 2/3. Same weighted parameter and two-Adam-moment averages, communicated
        # in bounded flat buckets. FP64 divergence is optional profiling metadata;
        # it never affects an optimizer step or the averaging weights.
        out["div_l2"] = _bucketed_weighted_average(
            self.text_tower.parameters(), self.optimizer, w, grp,
            measure_divergence=bool(self.config.profile),
        )
        out["div_l2_measured"] = bool(self.config.profile)
        # Use the maximum optimizer step across workers so bias correction uses the same t.
        local_step = 0.0
        for p in self.text_tower.parameters():
            sv = self.optimizer.state.get(p, {}).get("step")
            if sv is not None:
                local_step = float(sv.item()) if torch.is_tensor(sv) else float(sv)
                break
        step_t = torch.tensor([local_step], device=dev, dtype=torch.float32)
        dist.all_reduce(step_t, op=dist.ReduceOp.MAX, group=grp)
        max_step = step_t.item()
        for p in self.text_tower.parameters():
            sv = self.optimizer.state.get(p, {}).get("step")
            if sv is not None:
                if torch.is_tensor(sv):
                    sv.fill_(max_step)
                else:
                    self.optimizer.state[p]["step"] = int(max_step)
        # LM buffers such as masks/positions are constant and already aligned.

        self._batches_since_sync = 0
        self._updates_since_sync = 0
        out["did_sync"] = 1.0
        return out

    def _force_worst_case_first_chunk(self, lengths: dict[int, int], budget: int) -> None:
        """Memory probe: make this worker's first chunk its worst case (the counterpart of B5's
        LM_PROBE_WORST_CASE), so a short probe window cannot miss the pool's longest texts.

        Fixed-row chunks: the pool's single longest text joins the first chunk, which is then padded
        to it. Token-budget chunks: the first chunk is the longest texts that fit the budget.
        """
        order = sorted(lengths, key=lengths.__getitem__, reverse=True)
        if budget > 0:
            first: list[int] = []
            for n in order:
                if first and (len(first) + 1) * lengths[order[0]] > budget:
                    break
                first.append(n)
            self.scheduler._chunks.append(first)          # next_batch pops from the end
        else:
            seq = self.scheduler.node_sequence
            longest = order[0]
            if longest in seq:
                i = seq.index(longest)
                seq[0], seq[i] = seq[i], seq[0]
            else:
                seq.insert(0, longest)
            first = seq[:self.scheduler.batch_size]
        print(f"[{self.role_name}] probe: first text chunk carries the pool's longest text "
              f"({lengths[order[0]]} tokens, {len(first)} rows)", flush=True)

    def _make_records(self, node_ids: list[int], embeddings: torch.Tensor) -> list[EmbeddingRecord]:
        """Wrap text forward outputs into EmbeddingRecord objects."""
        records: list[EmbeddingRecord] = []
        for index, node_id in enumerate(node_ids):
            records.append(
                EmbeddingRecord(
                    key=CacheKey("text", int(node_id)),
                    tensor=embeddings[index].detach(),
                    version=self._text_version_counter + index + 1,
                    producer=self.role_name,
                    producer_step=self._successful_updates,
                    producer_batch=getattr(self, "_batches_made", 0),
                )
            )
        self._text_version_counter += len(records)
        return clone_embedding_records(records, "cpu")

    def _get_graph_embeddings(self, node_ids: list[int]) -> tuple[list[EmbeddingRecord] | None, dict[str, int]]:
        """Fetch graph embeddings for node_ids to compute contrastive loss.

        This mirrors graph worker _get_text_embeddings: local cache, remote fallback,
        then None on remaining misses.
        """
        keys = [CacheKey("graph", int(node_id)) for node_id in node_ids]

        # w/o drop-on-miss ablation: symmetric blocking poll with timeout fallback.
        _deadline: float | None = None
        if getattr(self.config, "block_on_miss", False):
            _deadline = time.monotonic() + float(getattr(self.config, "block_on_miss_timeout_seconds", 30.0))

        while True:
            # Level 1: local cache.
            local_hits = self.graph_cache.get_many(keys)
            misses = [key for key in keys if key not in local_hits]

            # Level 2: remote fallback.
            remote_hits = 0
            if misses and not self.shutdown_event.is_set():
                try:
                    fetched = rpc.rpc_sync(
                        self.config.coordinator_name,
                        _rpc_lookup_cache,
                        args=("graph", [key.node_id for key in misses]),
                    )
                    remote_hits = len(fetched)
                    _note_peer_steps(self._peer_seen, fetched)
                    self._stale_note(fetched)
                    self.graph_cache.put_many(fetched)
                except Exception:
                    pass

            # Merge results.
            final_hits = self.graph_cache.get_many(keys)
            remaining = [key for key in keys if key not in final_hits]
            stats = {
                "local_hit_nodes": len(local_hits),
                "remote_hit_nodes": remote_hits,
                "miss_nodes": len(remaining),
                # test6 staleness: peer embeddings' lag in the producer's updates (worker-side estimate).
                "mean_staleness": _mean_staleness(final_hits.values(), self._peer_seen),
            }
            if getattr(self.config, "staleness_stats", False):
                # lag_G of every hit = graph worker's successful updates as of its latest push this replica
                # received - the entry's version (graph updates when it was pushed); counted only once this
                # replica's C_G covers staleness_cover_gate of the training nodes.
                cov = self._stale_coverage()
                stats["stale_coverage"] = cov
                if cov >= float(getattr(self.config, "staleness_cover_gate", 0.98)):
                    cur = int(self._peer_seen.get(self.config.graph_worker_name, -1))
                    stats["stale_G"] = [max(0, cur - int(r.producer_step))
                                        for r in final_hits.values() if r.producer_step >= 0 and cur >= 0]
            if not remaining:
                # Return in original key order to stay aligned with node_ids.
                ordered = clone_embedding_records(
                    (final_hits[key] for key in keys), self.device, dtype=self.config.cache_dtype,
                )
                stats["lag_used"] = used_lag_payload(final_hits.values())
                return ordered, stats
            if (
                _deadline is None
                or self.shutdown_event.is_set()
                or time.monotonic() >= _deadline
            ):
                return None, stats
            time.sleep(0.2)


# ──────────────────────────────────────────────────────────────────────
# Part 6: helper functions and service factory
# ──────────────────────────────────────────────────────────────────────

def _load_genotype_file(path: Path) -> dict[str, Any]:
    """Load a GNN architecture genotype JSON file for retrain."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _log_warm_start_load(
    path: Path,
    missing: list[str],
    unexpected: list[str],
    *,
    side: str,
) -> None:
    """Log warm-start load results for checkpoint compatibility diagnosis."""
    print(f"[warm-start:{side}] loaded model_state from {path}")
    if missing:
        head = missing[:10]
        tail = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        print(f"[warm-start:{side}]   missing keys (will stay at init): {head}{tail}")
    if unexpected:
        head = unexpected[:10]
        tail = f" ... (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else ""
        print(f"[warm-start:{side}]   unexpected keys (ignored): {head}{tail}")


def _resolve_warm_start_path(raw: str | None) -> Path | None:
    """Resolve a configured warm-start path, returning None when missing."""
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.exists():
        print(f"[warm-start] ckpt not found, falling back to cold start: {p}")
        return None
    return p


def _load_runtime_spec(config: AsyncTrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load data and model arguments from a YAML config.

    YAML structure, compatible with Lightning CLI:
      data:
        init_args:
          data_dir: ...
          ...
      model:
        init_args:
          gnn_class_name: GassoSpace
          gnn_params: { in_channels: ..., ... }
          ...

    Returns (data_args, model_args).

    Validation:
      - gnn_class_name must be "GassoSpace".
      - sim_smoothing must be 0 because this runtime does not support smoothing.
    """
    loaded = OmegaConf.to_container(OmegaConf.load(config.config_path), resolve=True)
    assert isinstance(loaded, dict)
    data_args = dict(loaded["data"]["init_args"])
    # Bind data_dir to this repository's data/<dataset> instead of trusting YAML absolute paths.
    data_args["data_dir"] = str((Path("data") / config.dataset).resolve())
    model_args = dict(loaded["model"]["init_args"])
    if str(model_args.get("gnn_class_name")) != "GassoSpace":
        raise ValueError("Async training runtime expects gnn_class_name = 'GassoSpace'")
    if float(model_args.get("sim_smoothing", 0.0)) != 0.0:
        raise ValueError("Async training runtime currently expects sim_smoothing = 0.0")
    # CLI/shell override: NUM_LAYERS / --num-layers overrides gnn_params.num_layers.
    if config.num_layers is not None:
        gnn_params = dict(model_args.get("gnn_params", {}))
        gnn_params["num_layers"] = int(config.num_layers)
        model_args["gnn_params"] = gnn_params
        print(f"[override] gnn_params.num_layers = {config.num_layers} (= GNN k_hop)")
    # CLI/shell override: weight_decay / eps. These affect graph weight optimizer
    # and text optimizer; architecture optimizer keeps weight_decay=0.
    if getattr(config, "weight_decay_override", None) is not None:
        model_args["weight_decay"] = float(config.weight_decay_override)
        print(f"[override] weight_decay = {config.weight_decay_override} (suppresses contrastive embedding collapse)")
    if getattr(config, "adam_eps_override", None) is not None:
        model_args["eps"] = float(config.adam_eps_override)
        print(f"[override] adam eps = {config.adam_eps_override}")
    return data_args, model_args


def build_training_service_for_role(
    *,
    config: AsyncTrainConfig,
    role_name: str,
    text_group: dist.ProcessGroup | None = None,
) -> BaseTrainingService:
    """Build the training service instance for a role name.

    Factory mapping used by each spawned process:
      - "coordinator"    → CoordinatorTrainingService
      - "graph_worker"   → GraphWorkerTrainingService
      - "text_worker_N"  → TextWorkerTrainingService

    All roles use the same PubmedRuntimeStore abstraction, but each process builds
    its own local object.

    Args:
        config: Global runtime configuration.
        role_name: Current process role name.
        text_group: Process group among text workers; used only by text workers.
    """
    data_args, model_args = _load_runtime_spec(config)
    data_store = PubmedRuntimeStore.build(
        data_args=data_args,
        split_proportions=config.split_proportions,
        seed=config.seed,
        local_files_only=config.local_files_only,
        feature_dim=int(model_args["gnn_params"]["in_channels"]),
        train_node_fraction=config.train_node_fraction,
    )
    if role_name == config.coordinator_name:
        return CoordinatorTrainingService(config=config, data_store=data_store, model_args=model_args)
    if role_name == config.graph_worker_name:
        return GraphWorkerTrainingService(config=config, data_store=data_store, model_args=model_args)
    if role_name.startswith("text_worker_"):
        return TextWorkerTrainingService(
            config=config,
            role_name=role_name,
            data_store=data_store,
            model_args=model_args,
            text_group=text_group,
        )
    raise ValueError(f"Unknown role name: {role_name}")
