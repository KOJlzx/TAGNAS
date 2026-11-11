"""Unified telemetry layer (used for the test6 time-breakdown): a single
per-iteration x per-worker timer shared by all 4 systems.

Replaces the two old diverging code paths (the synchronous systems' IterationProfiler
+ the async TAGNAS write_breakdown_json), eliminating inconsistent accounting (e.g.
ZeRO residual 0.025 vs DDP 0.13). Core conventions:

  * 7 mutually exclusive phase buckets (seconds): sync_wait / xfer / build_batch /
    lm_fwd / graph_fwd / backward / update
  * GPU segments use torch.cuda.Event(enable_timing=True); commit() does a single
    torch.cuda.synchronize() and then uniformly resolves elapsed_time() - this
    guarantees the synchronize semantics at phase boundaries are exactly consistent
    across all 4 systems.
  * CPU segments (build_batch / sync_wait) use time.perf_counter(); CPU-only devices
    (the TAGNAS graph worker) automatically take the perf_counter branch, and
    commit() skips cuda.synchronize.
  * Each iteration also records absolute time.time() timestamps t_start / t_end,
    used to reconstruct cross-process (dual-lane) overlap.
  * **Unified residual formula (consistent across all systems)**: residual =
    (t_end - t_start) - sum(phases).
    Since t_end - t_start is the iteration's true wall-clock time, sum(phases) +
    residual == wall-clock time (an identity); TAGNAS no longer zeroes it out.
  * Each post-warmup iteration writes one JSONL line (no all_reduce is performed;
    pooling across rank/worker is moved to an offline aggregator).
    DDP/ZeRO only write from rank0; TAGNAS has each worker process write its own
    file (the caller guarantees unique filenames).

LM/GNN forward passes are captured precisely with forward pre/post hooks (the
submodule-location logic reuses IterationProfiler's).
When emit_barrier_sync_wait=True (DDP), a dist.barrier() is timed after backward and
recorded as SyncWait; ZeRO/TAGNAS set this to False (gradient sync already happens
inside engine.backward / the async drop-on-miss path, so no barrier needs to be
injected).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:  # distributed support is optional
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


# Canonical phase keys (per spec naming: lm_comp->lm_fwd, graph_comp->graph_fwd)
PHASE_KEYS = (
    "sync_wait",
    "xfer",
    "build_batch",
    "lm_fwd",
    "graph_fwd",
    "backward",
    "update",
)

# Old schema (IterationProfiler) keys -> new keys, reused by the aggregator/dual-write shim
LEGACY_KEY_MAP = {"lm_comp": "lm_fwd", "graph_comp": "graph_fwd"}


@dataclass
class IterRecord:
    """One telemetry row per iteration x per worker."""

    system: str
    dataset: str
    seed: int
    worker: str                       # single | graph | text
    iter: int                         # monotonic count for this worker (post-warmup)
    t_start: float                    # time.time() at iter_begin
    t_end: float                      # time.time() at commit
    phases_s: Dict[str, float] = field(default_factory=dict)
    nodes_in_batch: int = 0
    cache_hit: float = 1.0            # hit rate: always 1.0 for synchronous systems; TAGNAS = hit/(hit+miss)
    staleness_steps: float = 0.0      # version staleness: 0 for synchronous systems; TAGNAS = mean consumer-producer step lag
    residual_s: float = 0.0           # (t_end - t_start) - sum(phases_s)


class TelemetryRecorder:
    """Per-iteration telemetry recorder shared by all 4 systems. Method names align with the old IterationProfiler for easy drop-in replacement."""

    LM_NAME_CANDIDATES = ("lm", "text_encoder", "text", "language_model", "lm_encoder", "text_tower")
    GNN_NAME_CANDIDATES = ("gnn", "graph_encoder", "graph", "gnn_encoder", "graph_tower", "gasso")

    def __init__(
        self,
        *,
        system: str,
        dataset: str,
        seed: int,
        worker: str,
        device: torch.device,
        jsonl_path: str,
        lit_module: Optional[nn.Module] = None,
        rank: int = 0,
        enabled: bool = True,
        warmup_iters: int = 1,
        emit_barrier_sync_wait: bool = False,
        legacy_json_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.system = str(system)
        self.dataset = str(dataset)
        self.seed = int(seed)
        self.worker = str(worker)
        self.device = device
        self.use_cuda = getattr(device, "type", None) == "cuda"
        self.rank = int(rank)
        self.warmup_iters = int(warmup_iters)
        self.emit_barrier_sync_wait = bool(emit_barrier_sync_wait)
        self.jsonl_path = jsonl_path
        self.legacy_json_path = legacy_json_path
        # Only rank0 writes to disk (for TAGNAS, each worker process has its own
        # rank0 file; the caller guarantees the path is unique)
        self._write_enabled = self.enabled and (self.rank == 0)

        # Scratch state for the current iteration
        self._cur: Dict[str, float] = defaultdict(float)
        self._prev_end_perf: Optional[float] = None
        self._prev_end_wall: Optional[float] = None
        self._iter_t0_wall: Optional[float] = None
        self._iter_build = 0.0
        self._active = False
        self._skip_warmup_left = int(warmup_iters)   # used by the externally-fed path: skip the first batch (CUDA init, etc.)
        self._cur_nodes = 0
        self._cur_cache_hit = 1.0
        self._cur_staleness = 0.0
        self._iter_idx = 0            # post-warmup count

        # CUDA Event pairs (xfer / backward / update)
        self._ev_start: Dict[str, torch.cuda.Event] = {}
        self._ev_end: Dict[str, torch.cuda.Event] = {}
        self._t_start: Dict[str, float] = {}
        # forward hook event lists
        self._lm_starts: List[torch.cuda.Event] = []
        self._lm_ends: List[torch.cuda.Event] = []
        self._gnn_starts: List[torch.cuda.Event] = []
        self._gnn_ends: List[torch.cuda.Event] = []
        self._lm_t0: Optional[float] = None
        self._gnn_t0: Optional[float] = None

        # accumulators for the legacy mean dual-write
        self._sum: Dict[str, float] = defaultdict(float)
        self._wall_sum = 0.0
        self._count = 0

        self.lm_attached = False
        self.gnn_attached = False
        if self.enabled and lit_module is not None:
            self._attach_hooks(lit_module, verbose=verbose)

        self._fh = None
        if self._write_enabled:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
            self._fh = open(jsonl_path, "w", encoding="utf-8")
            if verbose:
                print(f"[Telemetry] {system}/{dataset} worker={worker} -> {jsonl_path}")

    # ---------- submodule forward hooks ----------

    @staticmethod
    def _find(module: nn.Module, candidates) -> Optional[nn.Module]:
        for name in candidates:
            sub = getattr(module, name, None)
            if isinstance(sub, nn.Module):
                return sub
        return None

    def _attach_hooks(self, lit_module: nn.Module, *, verbose: bool) -> None:
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
                print(f"[Telemetry] attached LM hook -> {type(lm_module).__name__}")
        if gnn_module is not None:
            gnn_module.register_forward_pre_hook(self._gnn_pre)
            gnn_module.register_forward_hook(self._gnn_post)
            self.gnn_attached = True
            if verbose:
                print(f"[Telemetry] attached GNN hook -> {type(gnn_module).__name__}")

    def _event(self) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _lm_pre(self, module, inputs):  # noqa: ANN001
        if not (self.enabled and self._active):
            return
        if self.use_cuda:
            self._lm_starts.append(self._event())
        else:
            self._lm_t0 = time.perf_counter()

    def _lm_post(self, module, inputs, output):  # noqa: ANN001
        if not (self.enabled and self._active):
            return
        if self.use_cuda:
            self._lm_ends.append(self._event())
        elif self._lm_t0 is not None:
            self._cur["lm_fwd"] += time.perf_counter() - self._lm_t0

    def _gnn_pre(self, module, inputs):  # noqa: ANN001
        if not (self.enabled and self._active):
            return
        if self.use_cuda:
            self._gnn_starts.append(self._event())
        else:
            self._gnn_t0 = time.perf_counter()

    def _gnn_post(self, module, inputs, output):  # noqa: ANN001
        if not (self.enabled and self._active):
            return
        if self.use_cuda:
            self._gnn_ends.append(self._event())
        elif self._gnn_t0 is not None:
            self._cur["graph_fwd"] += time.perf_counter() - self._gnn_t0

    # ---------- generic span (xfer / backward / update) ----------

    def _span_start(self, key: str) -> None:
        if not self.enabled:
            return
        if self.use_cuda:
            self._ev_start[key] = self._event()
        else:
            self._t_start[key] = time.perf_counter()

    def _span_end(self, key: str) -> None:
        if not self.enabled:
            return
        if self.use_cuda:
            self._ev_end[key] = self._event()
        elif key in self._t_start:
            self._cur[key] += time.perf_counter() - self._t_start[key]

    def xfer_start(self) -> None:
        self._span_start("xfer")

    def xfer_end(self) -> None:
        self._span_end("xfer")

    def bwd_start(self) -> None:
        self._span_start("backward")

    def bwd_end(self) -> None:
        self._span_end("backward")

    def upd_start(self) -> None:
        self._span_start("update")

    def upd_end(self) -> None:
        self._span_end("update")

    # ---------- sync wait (timed barrier, DDP only) ----------

    def sync_wait(self) -> None:
        """DDP: after synchronizing this rank's backward pass, times a
        dist.barrier() call to get the load-imbalance wait time.
        No-op when emit_barrier_sync_wait=False (ZeRO/TAGNAS)."""
        if not (self.enabled and self.emit_barrier_sync_wait):
            return
        if not (dist is not None and dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            return
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        dist.barrier()
        self._cur["sync_wait"] += time.perf_counter() - t0

    # ---------- iteration lifecycle ----------

    def epoch_start(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._prev_end_perf = now
        self._prev_end_wall = time.time()
        self._skip_warmup_left = self.warmup_iters

    def iter_begin(self, *, nodes_in_batch: int = 0) -> None:
        """Call as the first line of the iteration loop body: finalizes build_batch (prefetch wait), records t_start."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._prev_end_perf is None:
            self._prev_end_perf = now
            self._prev_end_wall = time.time()
        self._iter_t0_wall = self._prev_end_wall
        self._iter_build = now - self._prev_end_perf
        self._active = True
        self._cur_nodes = int(nodes_in_batch)
        self._cur_cache_hit = 1.0
        self._cur_staleness = 0.0

    def set_iter_meta(
        self,
        *,
        nodes_in_batch: Optional[int] = None,
        cache_hit: Optional[float] = None,
        staleness_steps: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        if nodes_in_batch is not None:
            self._cur_nodes = int(nodes_in_batch)
        if cache_hit is not None:
            self._cur_cache_hit = float(cache_hit)
        if staleness_steps is not None:
            self._cur_staleness = float(staleness_steps)

    def commit(self) -> None:
        """End of iteration: synchronize + pair up events into buckets + record t_end + compute the unified residual + write a JSONL row."""
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        # sum up the forward hook events (ms -> s)
        self._cur["lm_fwd"] += self._pair_ms(self._lm_starts, self._lm_ends) / 1000.0
        self._cur["graph_fwd"] += self._pair_ms(self._gnn_starts, self._gnn_ends) / 1000.0
        for key in ("xfer", "backward", "update"):
            s = self._ev_start.get(key)
            e = self._ev_end.get(key)
            if s is not None and e is not None:
                try:
                    self._cur[key] += s.elapsed_time(e) / 1000.0
                except Exception:
                    pass
        self._cur["build_batch"] += self._iter_build

        now_perf = time.perf_counter()
        now_wall = time.time()
        t_start = self._iter_t0_wall if self._iter_t0_wall is not None else now_wall
        wall = now_wall - t_start
        phases = {k: float(self._cur.get(k, 0.0)) for k in PHASE_KEYS}
        residual = wall - sum(phases.values())

        if self._skip_warmup_left > 0:
            self._skip_warmup_left -= 1  # discard the warmup iteration, only update the baseline
        else:
            self._iter_idx += 1
            if self._write_enabled and self._fh is not None:
                rec = IterRecord(
                    system=self.system,
                    dataset=self.dataset,
                    seed=self.seed,
                    worker=self.worker,
                    iter=self._iter_idx,
                    t_start=t_start,
                    t_end=now_wall,
                    phases_s=phases,
                    nodes_in_batch=self._cur_nodes,
                    cache_hit=self._cur_cache_hit,
                    staleness_steps=self._cur_staleness,
                    residual_s=residual,
                )
                self._fh.write(json.dumps(asdict(rec)) + "\n")
            # accumulators for the legacy mean dual-write
            for k in PHASE_KEYS:
                self._sum[k] += phases[k]
            self._wall_sum += wall
            self._count += 1

        # reset scratch state
        self._prev_end_perf = now_perf
        self._prev_end_wall = now_wall
        self._active = False
        self._cur = defaultdict(float)
        self._ev_start.clear()
        self._ev_end.clear()
        self._t_start.clear()
        self._lm_starts.clear()
        self._lm_ends.clear()
        self._gnn_starts.clear()
        self._gnn_ends.clear()

    def write_external_iter(
        self,
        phases_s: Dict[str, float],
        *,
        t_start: float,
        t_end: float,
        nodes_in_batch: int = 0,
        cache_hit: float = 1.0,
        staleness_steps: float = 0.0,
    ) -> None:
        """Feed-in path for phases that have already been measured externally (used
        by TAGNAS async workers).

        The worker has already timed each phase with perf_counter / cuda.Event; this
        method is only responsible for: computing the unified residual formula
        (residual = (t_end - t_start) - sum(phases)), discarding warmup, writing
        JSONL, and accumulating the legacy mean.
        The phase-collection mechanism can differ from the synchronous systems, but
        **the output schema and residual formula are exactly the same**.
        """
        if not self.enabled:
            return
        phases = {k: float(phases_s.get(k, 0.0)) for k in PHASE_KEYS}
        wall = float(t_end) - float(t_start)
        residual = wall - sum(phases.values())
        if self._skip_warmup_left > 0:
            self._skip_warmup_left -= 1
            return
        self._iter_idx += 1
        if self._write_enabled and self._fh is not None:
            rec = IterRecord(
                system=self.system,
                dataset=self.dataset,
                seed=self.seed,
                worker=self.worker,
                iter=self._iter_idx,
                t_start=float(t_start),
                t_end=float(t_end),
                phases_s=phases,
                nodes_in_batch=int(nodes_in_batch),
                cache_hit=float(cache_hit),
                staleness_steps=float(staleness_steps),
                residual_s=residual,
            )
            self._fh.write(json.dumps(asdict(rec)) + "\n")
        for k in PHASE_KEYS:
            self._sum[k] += phases[k]
        self._wall_sum += wall
        self._count += 1

    @staticmethod
    def _pair_ms(starts: List[torch.cuda.Event], ends: List[torch.cuda.Event]) -> float:
        n = min(len(starts), len(ends))
        total = 0.0
        for i in range(n):
            try:
                total += starts[i].elapsed_time(ends[i])
            except Exception:
                pass
        return total

    # ---------- finalization ----------

    def close(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Closes the JSONL handle; if legacy_json_path is provided, also writes a legacy mean breakdown JSON (transitional dual-write)."""
        if not self.enabled:
            return
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self._write_enabled and self.legacy_json_path:
            self._write_legacy_json(self.legacy_json_path, extra=extra)

    def _write_legacy_json(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Compatibility output in the old schema (lm_comp/graph_comp keys) for downstream consumers that haven't switched yet; a rank0-local approximation."""
        n = float(self._count) if self._count > 0 else 1.0
        per_iter_new = {k: self._sum.get(k, 0.0) / n for k in PHASE_KEYS}
        # new keys -> old keys
        inv = {v: k for k, v in LEGACY_KEY_MAP.items()}
        per_iter_legacy = {inv.get(k, k): v for k, v in per_iter_new.items()}
        sum_buckets = sum(per_iter_new.values())
        e2e_per = self._wall_sum / n
        obj: Dict[str, Any] = {
            "system": self.system,
            "dataset": self.dataset,
            "seed": self.seed,
            "num_iterations_averaged": int(self._count),
            "per_iteration_seconds": per_iter_legacy,
            "per_iteration_total_seconds": e2e_per,
            "sum_of_buckets_seconds": sum_buckets,
            "residual_seconds": e2e_per - sum_buckets,
            "lm_attached": self.lm_attached,
            "gnn_attached": self.gnn_attached,
            "worker": self.worker,
            "notes": "legacy dual-write from TelemetryRecorder; canonical source is per-iter JSONL.",
        }
        if extra:
            obj.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"[Telemetry] legacy breakdown -> {path}")


def derive_jsonl_path(profile_out: Optional[str], *, system: str, dataset: str,
                      seed: int, worker: str, out_dir: str = "results/test6") -> str:
    """Derives the new JSONL path from the old --profile-out: <dir>/telemetry/<sys>_<ds>_seed<seed>_<worker>.jsonl.

    When profile_out is given, uses the telemetry/ subdirectory of its containing
    directory; otherwise uses out_dir/telemetry/.
    """
    if profile_out:
        base_dir = os.path.dirname(os.path.abspath(profile_out)) or out_dir
    else:
        base_dir = out_dir
    tdir = os.path.join(base_dir, "telemetry")
    return os.path.join(tdir, f"{system}_{dataset}_seed{seed}_{worker}.jsonl")
