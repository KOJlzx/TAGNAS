"""Unified per-iteration time-breakdown profiler (used for the test6 time-breakdown).

Splits a single training iteration's wall-clock time into 7 mutually exclusive
buckets (seconds):

    build_batch  fetch data + build batch (the CPU gap between two iterations of the main loop)
    xfer         host<->device transfer (_move_batch_to_device, CUDA Event)
    lm_comp      LM forward pass (forward hook, CUDA Event)
    graph_comp   GNN/supernet forward pass (forward hook, CUDA Event)
    backward     single fused backward pass loss.backward() (CUDA Event)
    sync_wait    gradient-sync wait (a timed dist.barrier() placed after backward and before step)
    update       optimizer.step() (CUDA Event)

Design notes:
  - GPU segments use torch.cuda.Event(enable_timing=True); commit() calls
    torch.cuda.synchronize() and then takes the elapsed_time difference (ms -> s).
    Non-blocking .to(non_blocking=True) async copies are correctly reflected by the
    event on the stream once synchronize has completed.
  - CPU segments (build_batch / sync_wait) use time.perf_counter().
  - LM/GNN forward passes are captured precisely with forward pre/post hooks (the
    submodule-location logic reuses the _PhaseTimer verified in graphNAS_deepspeed_b4:
    it looks for a submodule by candidate name on lit_module and lit_module.model).
  - When enabled=False, all methods are no-ops (zero overhead, does not affect normal training).
  - The first iteration of each epoch is discarded as warmup (CUDA init / cudnn
    autotune / DataLoader's wait for the first batch when workers restart each epoch).
  - write_json() first pools each bucket's total time and count across ranks via
    all_reduce(SUM) under DDP, then rank0 writes the JSON in the unified schema.

LMComp/GraphComp only contain forward passes; Backward is the single backward pass
for the fused two-tower model. The gap between the sum of the columns and the
measured end-to-end per-iteration time (contrastive-loss/projection-head forward
residual + framework overhead) is recorded explicitly as residual_seconds.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:  # distributed support is optional
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


# Bucket order = table column order (SyncWait is placed as its own leading column
# in write_json; here grouped by what's measured)
BUCKETS = (
    "build_batch",
    "xfer",
    "lm_comp",
    "graph_comp",
    "backward",
    "sync_wait",
    "update",
)


class IterationProfiler:
    """7-bucket timer shared by the three synchronous systems (fixed_gat_ddp / gnas_ddp / gnas_zero)."""

    LM_NAME_CANDIDATES = ("lm", "text_encoder", "text", "language_model", "lm_encoder", "text_tower")
    GNN_NAME_CANDIDATES = ("gnn", "graph_encoder", "graph", "gnn_encoder", "graph_tower", "gasso")

    def __init__(
        self,
        lit_module: Optional[nn.Module],
        device: torch.device,
        *,
        enabled: bool = True,
        world_size: int = 1,
        is_distributed: bool = False,
        warmup_iters_per_epoch: int = 1,
        verbose: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.use_cuda = getattr(device, "type", None) == "cuda"
        self.world_size = int(world_size)
        self.is_distributed = bool(is_distributed)
        self.warmup_iters_per_epoch = int(warmup_iters_per_epoch)

        # Accumulated (seconds), only counts iterations after warmup
        self._sum: Dict[str, float] = defaultdict(float)
        self._end_to_end_sum = 0.0
        self._count = 0          # number of iterations counted (this rank)
        self._seen = 0           # total number of iterations seen (including warmup, this rank)

        # Scratch state for the current iteration
        self._cur: Dict[str, float] = defaultdict(float)
        self._prev_end: Optional[float] = None      # perf_counter at the previous iteration's commit
        self._iter_t0: Optional[float] = None        # start of this iteration (= end of the previous iteration)
        self._iter_build = 0.0
        self._active = False                          # only record forward hooks while inside a training iteration
        self._skip_this_epoch_first = False           # discard the first iteration of each epoch

        # CUDA Event pairs (xfer / backward / update)
        self._ev_start: Dict[str, torch.cuda.Event] = {}
        self._ev_end: Dict[str, torch.cuda.Event] = {}
        # CPU timing start point (used when use_cuda=False)
        self._t_start: Dict[str, float] = {}
        # forward hook event lists (an iteration may have multiple forward passes, summed)
        self._lm_starts: List[torch.cuda.Event] = []
        self._lm_ends: List[torch.cuda.Event] = []
        self._gnn_starts: List[torch.cuda.Event] = []
        self._gnn_ends: List[torch.cuda.Event] = []
        self._lm_t0: Optional[float] = None
        self._gnn_t0: Optional[float] = None

        self.lm_attached = False
        self.gnn_attached = False
        if self.enabled and lit_module is not None:
            self._attach_hooks(lit_module, verbose=verbose)

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
                print(f"[Profiler] attached LM hook -> {type(lm_module).__name__}")
        elif verbose:
            print("[Profiler] WARN: LM submodule not found; lm_comp disabled")

        if gnn_module is not None:
            gnn_module.register_forward_pre_hook(self._gnn_pre)
            gnn_module.register_forward_hook(self._gnn_post)
            self.gnn_attached = True
            if verbose:
                print(f"[Profiler] attached GNN hook -> {type(gnn_module).__name__}")
        elif verbose:
            print("[Profiler] WARN: GNN submodule not found; graph_comp disabled")

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
            self._cur["lm_comp"] += time.perf_counter() - self._lm_t0

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
            self._cur["graph_comp"] += time.perf_counter() - self._gnn_t0

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

    # ---------- sync wait (timed barrier) ----------

    def sync_wait(self) -> None:
        """Measures gradient-sync wait time: first synchronize to make sure this
        rank's own GPU backward pass has finished, then time a dist.barrier() call;
        the result is the time spent waiting on the slowest rank due to load
        imbalance (roughly the synchronization stall). It is 0 for a single process
        or when not running distributed. Only appears in dedicated --profile runs
        (introduces a slight perturbation, which is acceptable).
        """
        if not self.enabled:
            return
        if not (self.is_distributed and dist is not None
                and dist.is_available() and dist.is_initialized() and self.world_size > 1):
            return
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        dist.barrier()
        self._cur["sync_wait"] += time.perf_counter() - t0

    # ---------- iteration lifecycle ----------

    def epoch_start(self) -> None:
        """Call before each epoch's training loop starts: resets the timing baseline and marks this epoch's first iteration as warmup."""
        if not self.enabled:
            return
        self._prev_end = time.perf_counter()
        self._skip_this_epoch_first = True

    def iter_begin(self) -> None:
        """Call as the first line of each iteration's loop body: finalizes build_batch (= the wait for fetching the prefetched batch)."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._prev_end is None:
            self._prev_end = now
        self._iter_t0 = self._prev_end
        self._iter_build = now - self._prev_end
        self._active = True

    def commit(self) -> None:
        """Call at the end of the iteration: synchronize + pair up CUDA Events into buckets + end-to-end timing + warmup discard."""
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        # sum up the forward hook events
        self._cur["lm_comp"] += self._pair_ms(self._lm_starts, self._lm_ends) / 1000.0
        self._cur["graph_comp"] += self._pair_ms(self._gnn_starts, self._gnn_ends) / 1000.0
        # generic span events
        for key in ("xfer", "backward", "update"):
            s = self._ev_start.get(key)
            e = self._ev_end.get(key)
            if s is not None and e is not None:
                try:
                    self._cur[key] += s.elapsed_time(e) / 1000.0
                except Exception:
                    pass
        # build_batch(CPU)
        self._cur["build_batch"] += self._iter_build

        now = time.perf_counter()
        e2e = now - self._iter_t0 if self._iter_t0 is not None else 0.0

        self._seen += 1
        if self._skip_this_epoch_first:
            # discard this epoch's first iteration (warmup), but still update the baseline
            self._skip_this_epoch_first = False
        else:
            for k in BUCKETS:
                self._sum[k] += self._cur.get(k, 0.0)
            self._end_to_end_sum += e2e
            self._count += 1

        # reset scratch state
        self._prev_end = now
        self._active = False
        self._cur = defaultdict(float)
        self._ev_start.clear()
        self._ev_end.clear()
        self._t_start.clear()
        self._lm_starts.clear()
        self._lm_ends.clear()
        self._gnn_starts.clear()
        self._gnn_ends.clear()

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

    # ---------- aggregate + write to disk ----------

    def write_json(
        self,
        path: str,
        system: str,
        dataset: str,
        *,
        rank: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Pools (SUM) each bucket's total time and count across ranks under DDP, then rank0 writes the unified-schema JSON."""
        if not self.enabled:
            return None

        sums = {k: float(self._sum.get(k, 0.0)) for k in BUCKETS}
        e2e = float(self._end_to_end_sum)
        count = float(self._count)

        if (self.is_distributed and dist is not None and dist.is_available()
                and dist.is_initialized() and self.world_size > 1):
            vec = torch.tensor(
                [sums[k] for k in BUCKETS] + [e2e, count],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            vals = vec.tolist()
            for i, k in enumerate(BUCKETS):
                sums[k] = vals[i]
            e2e = vals[len(BUCKETS)]
            count = vals[len(BUCKETS) + 1]

        if rank != 0:
            return None

        n = count if count > 0 else 1.0
        per_iter = {k: sums[k] / n for k in BUCKETS}
        sum_buckets = sum(per_iter.values())
        e2e_per = e2e / n

        obj: Dict[str, Any] = {
            "system": system,
            "dataset": dataset,
            "num_iterations_averaged": int(round(count / max(self.world_size, 1))),
            "num_iterations_total_all_ranks": int(round(count)),
            "per_iteration_seconds": per_iter,
            "per_iteration_total_seconds": e2e_per,
            "sum_of_buckets_seconds": sum_buckets,
            "residual_seconds": e2e_per - sum_buckets,
            "world_size": self.world_size,
            "lm_attached": self.lm_attached,
            "gnn_attached": self.gnn_attached,
            "notes": (
                "lm_comp/graph_comp=forward only; backward=single fused two-tower backward; "
                "sync_wait=timed dist.barrier() after backward (load-imbalance wait); "
                "residual=contrastive-loss forward + framework overhead."
            ),
        }
        if extra:
            obj.update(extra)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"[Profiler] wrote breakdown -> {path}")
        return obj


def write_breakdown_json(
    path: str,
    system: str,
    dataset: str,
    per_iteration_seconds: Dict[str, float],
    *,
    per_iteration_total_seconds: Optional[float] = None,
    num_iterations: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Free-assembly variant for the async TAGNAS system: feed in the 7 bucket
    seconds directly, written to the same schema.

    TAGNAS produces a merged row where "the graph-side timeline is primary and the
    text-side LMComp is merged in"; each bucket comes from perf_counter telemetry on
    the graph/text workers, aggregated at the coordinator - it does not go through
    IterationProfiler's hook path.
    """
    per_iter = {k: float(per_iteration_seconds.get(k, 0.0)) for k in BUCKETS}
    sum_buckets = sum(per_iter.values())
    e2e_per = (per_iteration_total_seconds
               if per_iteration_total_seconds is not None else sum_buckets)
    obj: Dict[str, Any] = {
        "system": system,
        "dataset": dataset,
        "num_iterations_averaged": int(num_iterations),
        "per_iteration_seconds": per_iter,
        "per_iteration_total_seconds": e2e_per,
        "sum_of_buckets_seconds": sum_buckets,
        "residual_seconds": e2e_per - sum_buckets,
        "notes": (
            "TAGNAS merged row: graph-worker timeline (build/graph_comp/backward/update/xfer); "
            "lm_comp grafted from text worker GPU telemetry; sync_wait~=0 by async drop-on-miss."
        ),
    }
    if extra:
        obj.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[Profiler] wrote breakdown -> {path}")
    return obj
