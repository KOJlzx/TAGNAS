"""Inter-layer flow parallelism (GPipe-style F-then-B) with bubble gating.

Each rank owns one :class:`RevGNNStage`. A mini-batch of ``M`` microbatches is
run forward-then-backward: activations flow ``rank -> rank+1`` and gradients
``rank+1 -> rank`` as point-to-point sends. Around every piece of GPU work the
engine flips an :class:`LMGate`: the colocated LM thread pauses at its next
transformer-layer boundary while the stage computes and resumes when the stage
is waiting for its peer (the pipeline bubble). This is the Bubble Interleaver
of the paper implemented cooperatively inside one process.

The stage-to-stage transfers go through **gloo via pinned host buffers**, not
NCCL. They are small (``nodes x hidden`` floats, a few MB), and keeping the
GNN off NCCL matters: a NCCL kernel that spin-waits for a peer starves every
other kernel on that GPU, so with the LM's NCCL collectives in the same
process the two roles deadlocked each other. With gloo P2P the only NCCL user
per GPU is the LM, whose collectives can only ever wait on a peer's *local*
compute, which nothing blocks.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from .partition import MicroBatch


class LMGate:
    """Cooperative pause/resume flag between the GNN thread and the LM thread."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._ev = threading.Event()
        self._ev.set()
        self.pause_count = 0
        self.wait_seconds = 0.0
        self.wait_count = 0

    def pause(self) -> None:
        if self.enabled:
            self._ev.clear()
            self.pause_count += 1

    def resume(self) -> None:
        self._ev.set()

    def is_paused(self) -> bool:
        return not self._ev.is_set()

    def wait(self, stall_seconds: float = 120.0) -> None:
        """Called by the LM thread; blocks while the GNN stage is busy.

        A gate is only ever closed for the duration of *local* GNN compute, so a wait longer
        than ``stall_seconds`` means something is wrong; dump every thread's stack once so the
        hang can be diagnosed post-mortem."""
        if self._ev.is_set():
            return
        t = time.perf_counter()
        dumped = False
        while not self._ev.wait(timeout=stall_seconds):
            if not dumped:
                dumped = True
                self._dump_stacks(time.perf_counter() - t)
        self.wait_seconds += time.perf_counter() - t
        self.wait_count += 1

    @staticmethod
    def _dump_stacks(waited: float) -> None:
        import faulthandler
        import sys
        path = os.environ.get("UNITG_STACK_FILE")
        try:
            f = open(path, "a") if path else sys.stderr
            f.write(f"\n[LMGate] LM thread has waited {waited:.0f}s on the gate; dumping all thread stacks\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            if path:
                f.close()
        except Exception:  # noqa: BLE001
            pass


class PipelineEngine:
    def __init__(
        self,
        stage,
        rank: int,
        world_size: int,
        group,
        device: torch.device,
        gate: Optional[LMGate] = None,
    ):
        self.stage = stage
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.group = group
        self.device = device
        self.gate = gate or LMGate(enabled=False)
        self.prev = rank - 1 if rank > 0 else None
        self.next = rank + 1 if rank < world_size - 1 else None
        self.is_first = rank == 0
        self.is_last = rank == world_size - 1
        self.hidden = int(stage.hidden_channels)
        self.out_dim = int(stage.out_channels)
        self.comm_seconds = 0.0
        self.compute_seconds = 0.0
        # Per-mini-batch phase accumulator for clip_graph.telemetry (reset by train_minibatch).
        # Every span boundary below already drains the device (_sync / _wait_event), so plain
        # perf_counter deltas are GPU-complete; the graph lane needs no CUDA events.
        self._phase = dict.fromkeys(("sync_wait", "xfer", "build_batch", "graph_fwd",
                                     "backward", "update"), 0.0)
        self._pending = []
        self.trace = bool(os.environ.get("UNITG_TRACE"))
        self._trace_budget = 200
        self._t0 = time.perf_counter()

    def _tr(self, msg: str) -> None:
        if self.trace and self._trace_budget > 0:
            self._trace_budget -= 1
            print(f"[trace r{self.rank} gnn +{time.perf_counter() - self._t0:8.3f}s] {msg}", flush=True)

    # ------------------------------------------------------------- primitives
    def _host_buffer(self, shape) -> torch.Tensor:
        pin = self.device.type == "cuda"
        return torch.empty(shape, dtype=torch.float32, pin_memory=pin)

    def _recv(self, src: int, shape) -> torch.Tensor:
        """Blocking receive (gloo, host staging); the gate is open while we wait for the peer.

        The ``dist.recv`` wait and the host-to-device copy are billed separately: the wait is this
        stage's share of the pipeline bubble (it is idle because a peer has not finished), which is
        the same thing the DDP baselines measure with a timed barrier, whereas only the copy is
        actual transfer work.
        """
        host = self._host_buffer(shape)
        t = time.perf_counter()
        dist.recv(host, src=src, group=self.group)
        t_recv = time.perf_counter()
        buf = host.to(self.device, non_blocking=True)
        self._sync()
        now = time.perf_counter()
        self.comm_seconds += now - t
        self._phase["sync_wait"] += t_recv - t
        self._phase["xfer"] += now - t_recv
        return buf

    def _send(self, tensor: torch.Tensor, dst: int) -> None:
        """Send through a host copy. ``.cpu()`` waits for the local compute that produced the
        tensor (which is exactly when the gate may reopen); the transfer itself is asynchronous
        and ``_drain`` waits for completion before the buffers are released."""
        t = time.perf_counter()
        host = tensor.detach().contiguous().float().cpu()
        work = dist.isend(host, dst=dst, group=self.group)
        self._pending.append((work, host))
        self._phase["xfer"] += time.perf_counter() - t

    def _drain(self) -> None:
        t = time.perf_counter()
        for work, _ in self._pending:
            work.wait()
        self._pending.clear()
        self._sync()
        self._phase["sync_wait"] += time.perf_counter() - t

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _event(self):
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
            return ev
        return None

    def _wait_event(self, ev) -> None:
        if ev is not None:
            ev.synchronize()

    # ---------------------------------------------------------------- training
    def train_minibatch(
        self,
        microbatches: List[MicroBatch],
        features_fn: Callable[[MicroBatch], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, MicroBatch], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: Optional[float] = None,
    ) -> Optional[float]:
        """One F-then-B pass over ``microbatches``. Returns the mean loss on the last stage."""
        stage = self.stage
        stage.train()
        M = len(microbatches)
        ins: List[torch.Tensor] = []
        outs: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []
        for k in self._phase:
            self._phase[k] = 0.0
        self.last_iter_t_start = time.time()

        # ---- forward: all microbatches, in order --------------------------
        for m, mb in enumerate(microbatches):
            ei = mb.edge_index
            if self.is_first:
                t = time.perf_counter()
                x = features_fn(mb)
                self._phase["build_batch"] += time.perf_counter() - t
            else:
                self._tr(f"fwd m{m}: recv from {self.prev} ({mb.num_nodes}x{self.hidden})")
                x = self._recv(self.prev, (mb.num_nodes, self.hidden)).requires_grad_(True)
                self._tr(f"fwd m{m}: recv done")
            self.gate.pause()
            t = time.perf_counter()
            out = stage(x, ei)
            ev = self._event()
            # Wait for the stage kernel BEFORE _send: _send starts with a blocking .cpu(), so
            # without this the kernel time would be billed to xfer instead of graph_fwd. On CPU
            # _wait_event is a no-op and .cpu() is a copy, so the order is inert there.
            self._wait_event(ev)
            self._phase["graph_fwd"] += time.perf_counter() - t
            if self.is_last:
                t_loss = time.perf_counter()
                losses.append(loss_fn(out, mb))
                # The loss forward has no bucket of its own; leave it in the residual.
                self.compute_seconds += time.perf_counter() - t_loss
            else:
                self._tr(f"fwd m{m}: send to {self.next}")
                self._send(out.detach(), self.next)
            self._tr(f"fwd m{m}: compute done")
            self.compute_seconds += time.perf_counter() - t
            self.gate.resume()
            ins.append(x)
            outs.append(out)

        # ---- backward: same order ------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        for m in range(M):
            if self.is_last:
                self.gate.pause()
                t = time.perf_counter()
                self._tr(f"bwd m{m}: backward")
                (losses[m] / M).backward()
            else:
                self._tr(f"bwd m{m}: recv grad from {self.next}")
                g = self._recv(self.next, outs[m].shape)
                self._tr(f"bwd m{m}: recv done; backward")
                self.gate.pause()
                t = time.perf_counter()
                outs[m].backward(g)
            ev = self._event()
            self._wait_event(ev)          # same reason as the forward: bill the kernel to backward
            self._phase["backward"] += time.perf_counter() - t
            if not self.is_first:
                self._tr(f"bwd m{m}: send grad to {self.prev}")
                self._send(ins[m].grad, self.prev)
            self._tr(f"bwd m{m}: done")
            self.compute_seconds += time.perf_counter() - t
            self.gate.resume()

        self.gate.pause()
        t = time.perf_counter()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(stage.parameters(), grad_clip)
        optimizer.step()
        ev = self._event()
        self._wait_event(ev)
        self._phase["update"] += time.perf_counter() - t
        self.gate.resume()
        self._drain()         # every in-flight send has completed before its buffer dies
        self.last_iter_t_end = time.time()
        self._tr("minibatch done (optimizer step)")

        if self.is_last:
            return float(torch.stack([l.detach() for l in losses]).mean().item())
        return None

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def eval_forward(
        self,
        chunks: List[MicroBatch],
        features_fn: Callable[[MicroBatch], torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Pipeline forward in eval mode; the last stage returns the concatenated embeddings."""
        stage = self.stage
        stage.eval()
        outs = []
        for ch in chunks:
            if self.is_first:
                x = features_fn(ch)
            else:
                x = self._recv(self.prev, (ch.num_nodes, self.hidden))
            self.gate.pause()
            out = stage(x, ch.edge_index)
            ev = self._event()
            if self.is_last:
                outs.append(out)
            else:
                self._send(out, self.next)
            self._wait_event(ev)
            self.gate.resume()
        self._drain()
        stage.train()
        if self.is_last:
            return torch.cat(outs, dim=0)
        return None
