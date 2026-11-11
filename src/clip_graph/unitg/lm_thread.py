"""LM fine-tuning thread (data-parallel replica colocated with one GNN stage).

The thread fine-tunes the early-exit LM for as long as the GNN trains, on its
own CUDA stream, pausing at transformer-layer boundaries whenever the
colocated GNN stage is computing (see :class:`LMGate`).

Objectives (``config.objective``):

* ``contrastive`` (default, shared with the other systems): a batch of train
  nodes is encoded and aligned with the GNN embeddings of the same nodes
  (``clip_graph.losses.square_contrastive_loss``), read from the GNN table
  the GNN thread publishes at every Δt sync. Until the first table produced by
  a *trained* GNN arrives the LM only encodes (``lm_warmup_hold``).
* ``link`` (default): text-side edge InfoNCE — endpoint texts of train edges are
  positives, the other endpoints in the batch negatives. Anchors the LM in text.

At every step boundary the replicas agree (all-reduce MAX) on three flags:
*stop*, the latest *sync request id*, and *hold*. A sync request makes every
replica encode its shard of all nodes with the current LM (early exits
included), all-gather the ``[N, D]`` table and publish it. Gradients are
averaged with one flat all-reduce issued from this thread (no DDP), so the
LM's collectives never depend on anything but a peer's local compute.
"""

from __future__ import annotations

import contextlib
import math
import os
import random as _random
import threading
import time
import traceback
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from clip_graph import losses as ls

from .channel import SyncChannel
from .config import UniTGConfig
from .data import UniTGData
from .lazy_tuner import EarlyExitLM
from .pipeline import LMGate


def load_pretrained_lm(lm_name: str, local_files_only: bool = True):
    import transformers as tf

    lm = tf.AutoModel.from_pretrained(lm_name, local_files_only=local_files_only)
    if hasattr(lm, "pooler") and getattr(lm, "pooler") is not None:
        lm.pooler = None
    return lm



def subsample_pool(nodes: torch.Tensor, fraction, seed: int) -> torch.Tensor:
    """A seeded ``fraction`` of ``nodes`` (sorted), or ``nodes`` unchanged when fraction is None/>=1.

    Same rule as TAGNAS's PubmedRuntimeStore for its seed nodes: ``random.Random(seed).sample`` over
    the sorted pool, keeping ``max(1, int(len * fraction))``. Every rank computes the same subset.
    """
    if fraction is None or not 0.0 < float(fraction) < 1.0 or nodes.numel() == 0:
        return nodes
    pool = sorted(int(v) for v in nodes.tolist())
    keep = max(1, int(len(pool) * float(fraction)))
    chosen = sorted(_random.Random(int(seed)).sample(pool, keep))
    return torch.tensor(chosen, dtype=nodes.dtype, device=nodes.device)

class _Span:
    """Context manager used by LMTrainer._span; CUDA events on CUDA, perf_counter on CPU."""

    __slots__ = ("owner", "key", "t0", "ev0", "ev1")

    def __init__(self, owner: "LMTrainer", key: str):
        self.owner, self.key = owner, key

    def __enter__(self) -> "_Span":
        if self.owner.device.type == "cuda":
            self.ev0 = torch.cuda.Event(enable_timing=True)
            self.ev0.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.owner.device.type == "cuda":
            self.ev1 = torch.cuda.Event(enable_timing=True)
            self.ev1.record()
            self.owner._pending_events.append((self.key, self.ev0, self.ev1))
        else:
            self.owner._phase[self.key] = self.owner._phase.get(self.key, 0.0) + (time.perf_counter() - self.t0)
        return None


class LMTrainer(threading.Thread):
    def __init__(
        self,
        config: UniTGConfig,
        data: UniTGData,
        rank: int,
        world_size: int,
        device: torch.device,
        lm_pg,
        gate: LMGate,
        channel: SyncChannel,
        log: Callable[[str], None] = print,
        local_files_only: bool = True,
        telemetry=None,
        lm_factory: Optional[Callable[[], torch.nn.Module]] = None,
    ):
        super().__init__(name=f"lm-rank{rank}", daemon=True)
        self.config = config
        self.data = data
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.lm_pg = lm_pg
        self.gate = gate
        self.channel = channel
        self.log = log
        self.local_files_only = local_files_only
        self.telemetry = telemetry     # clip_graph.telemetry.TelemetryRecorder or None
        self._phase = {}               # per-step phase accumulator (see _train_step)
        self._pending_events = []      # [(phase_key, start_event, end_event)] resolved per step
        self.lm_factory = lm_factory
        self.contrastive = config.lm_objective == "contrastive"     # how the LM itself is trained
        self.error: Optional[BaseException] = None
        self.model: Optional[EarlyExitLM] = None
        self.stats = {
            "steps": 0, "nodes_processed": 0, "loss_ema": None, "encodes": 0,
            "encode_seconds": 0.0, "train_seconds": 0.0, "gate_wait_seconds": 0.0, "hold_seconds": 0.0,
        }
        self.trace = bool(os.environ.get("UNITG_TRACE"))
        self._trace_budget = 200
        self._t0 = time.perf_counter()

    def _tr(self, msg: str) -> None:
        if self.trace and self._trace_budget > 0:
            self._trace_budget -= 1
            print(f"[trace r{self.rank} lm  +{time.perf_counter() - self._t0:8.3f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ thread
    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            self.log(f"[lm rank{self.rank}] FAILED: {exc}\n{traceback.format_exc()}")
            self.channel.mark_error(exc)

    def _stream_ctx(self):
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
            return torch.cuda.stream(self.stream)
        self.stream = None
        return contextlib.nullcontext()

    def _span(self, key: str):
        """Time one span into self._phase[key].

        On CUDA the LM never drains the device mid-step, so wall time would only measure kernel
        launches; use events, which land on this thread's own stream (torch.cuda.stream is
        thread-local). They are resolved in _train_step against self.stream ONLY -- never
        torch.cuda.synchronize(device), which would also wait on the colocated GNN stage's
        default stream and corrupt both lanes' numbers.
        """
        return _Span(self, key)

    def _autocast(self):
        if self.device.type == "cuda" and self.config.lm_autocast:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _run(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        with self._stream_ctx():
            self._build()
            self._loop()

    # ------------------------------------------------------------------- build
    def _build(self) -> None:
        cfg, data = self.config, self.data
        torch.manual_seed(cfg.seed)
        lm = self.lm_factory() if self.lm_factory is not None else load_pretrained_lm(data.lm_name, self.local_files_only)
        model = EarlyExitLM(
            lm,
            num_exits=cfg.lm_num_exits,
            embed_dim=cfg.embed_dim,
            pooling=str(data.model_args.get("lm_pooling_mode", "mean")),
            normalize=True,
            lazy=cfg.lazy_tuner,
            tau_init=cfg.tau_init,
            max_tau=cfg.max_tau,
        ).to(self.device)
        for layer in model.layer_modules():
            layer.register_forward_pre_hook(lambda m, args: self.gate.wait())
        self.model = model
        # Data parallelism is done by hand (flat gradient all-reduce at the step boundary, issued
        # from this thread) rather than with DistributedDataParallel: DDP's reducer runs inside the
        # autograd engine and entangles the LM's collectives with the GNN stage sharing this
        # device, which deadlocked in practice. Same pattern as the TAGNAS text workers.
        self.params = [p for p in model.parameters() if p.requires_grad]
        self._numels = [p.numel() for p in self.params]
        self._flat = torch.zeros(sum(self._numels), dtype=torch.float32, device=self.device)
        self._broadcast_params()

        lr = cfg.lm_lr if cfg.lm_lr is not None else float(data.model_args.get("lr", 1e-4))
        eps = cfg.lm_adam_eps if cfg.lm_adam_eps is not None else float(data.model_args.get("eps", 1e-8))
        wd = cfg.lm_weight_decay if cfg.lm_weight_decay is not None else float(data.model_args.get("weight_decay", 0.0))
        betas = tuple(data.model_args.get("betas", (0.9, 0.999)))
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

        # The LM only ever trains on text-bearing nodes (B1/TAGNAS likewise): contrastive anchors are
        # drawn from them and a train edge is a text-side positive only if both endpoints have text.
        train_has_text = data.has_text[data.train.rows]
        self.train_text_nodes = torch.nonzero(train_has_text, as_tuple=False).flatten().to(self.device)
        self.train_text_nodes = subsample_pool(self.train_text_nodes, cfg.lm_train_node_fraction, cfg.seed)
        # Memory-probe mode: the pool entry with the longest text, forced into the first LM batch.
        self._worst_case_pending = bool(cfg.lm_probe_worst_case) and self.train_text_nodes.numel() > 0
        self._worst_case_node = None
        self._worst_case_len = 0
        if self._worst_case_pending:
            lens = data.attention_mask[data.train.rows[self.train_text_nodes.cpu()]].sum(dim=1)
            k = int(torch.argmax(lens))
            self._worst_case_node = int(self.train_text_nodes[k])
            self._worst_case_len = int(lens[k])
        self.full_text_nodes = torch.nonzero(data.has_text, as_tuple=False).flatten().to(self.device)
        ei = data.train.edge_index
        und = ei[:, ei[0] < ei[1]]
        und = und[:, train_has_text[und[0]] & train_has_text[und[1]]]
        self.train_pairs = und.to(self.device)                # 'link' objective
        self.num_train_nodes = data.train.num_nodes
        self.train_rows = data.train.rows.to(self.device)
        self.train_group = data.train.group.to(self.device)
        self.input_ids = data.input_ids.to(self.device)
        self.attention_mask = data.attention_mask.to(self.device)
        self.group_full = data.group_full.to(self.device)
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1000 + self.rank)
        self.log(f"[lm rank{self.rank}] {data.lm_name}: {model.num_layers} layers, exits={model.exit_layers}, "
                 f"lm_objective={cfg.lm_objective} lr={lr} eps={eps} wd={wd} batch={cfg.lm_batch_size} lazy={cfg.lazy_tuner}")

    # ---------------------------------------------------------------- dp sync
    @torch.no_grad()
    def _broadcast_params(self) -> None:
        """Make every replica start from rank 0's weights (what DDP does at construction)."""
        if self.world_size <= 1:
            return
        off = 0
        for p, n in zip(self.params, self._numels):
            self._flat[off:off + n].copy_(p.data.view(-1))
            off += n
        dist.broadcast(self._flat, src=0, group=self.lm_pg)
        off = 0
        for p, n in zip(self.params, self._numels):
            p.data.copy_(self._flat[off:off + n].view_as(p))
            off += n

    @torch.no_grad()
    def _allreduce_grads(self) -> None:
        """Average gradients across replicas with one flat all-reduce (None grads count as zero)."""
        if self.world_size <= 1:
            return
        off = 0
        for p, n in zip(self.params, self._numels):
            seg = self._flat[off:off + n]
            if p.grad is None:
                seg.zero_()
            else:
                seg.copy_(p.grad.view(-1))
            off += n
        dist.all_reduce(self._flat, op=dist.ReduceOp.SUM, group=self.lm_pg)
        self._flat.div_(self.world_size)
        off = 0
        for p, n in zip(self.params, self._numels):
            seg = self._flat[off:off + n].view_as(p)
            if p.grad is None:
                p.grad = seg.clone()
            else:
                p.grad.copy_(seg)
            off += n

    # -------------------------------------------------------------------- loop
    def _local_hold(self) -> int:
        """1 if this replica must not train yet (no usable GNN table under the contrastive objective)."""
        if not self.contrastive:
            return 0
        table, version = self.channel.current_gnn_table()
        if table is None:
            return 1
        if self.config.lm_warmup_hold and version < 2:
            return 1
        return 0

    def _flags(self):
        stop, rid = self.channel.snapshot_flags()
        t = torch.tensor([stop, rid, self._local_hold()], dtype=torch.long, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.lm_pg)
        return int(t[0].item()), int(t[1].item()), int(t[2].item())

    def _loop(self) -> None:
        served = 0
        loss_ema = None
        while True:
            # A telemetry iteration is one loop turn that ends in a training step. Reset the
            # accumulators at the TOP of every turn (encode/hold turns `continue` without writing,
            # and their spans must not leak into the next record) and take the wall-clock start
            # here so that sum(phases) can never exceed t_end - t_start.
            self._phase.clear()
            self._pending_events.clear()
            t_wall0 = time.time()
            self._tr("flags all_reduce")
            with self._span("sync_wait"):
                stop, rid, hold = self._flags()
            self._tr(f"flags -> stop={stop} rid={rid} hold={hold}")
            if stop:
                break
            if rid > served:
                t = time.perf_counter()
                self._encode_all(rid)
                served = rid
                self.stats["encodes"] += 1
                self.stats["encode_seconds"] += time.perf_counter() - t
                continue
            if hold:
                time.sleep(0.05)
                self.stats["hold_seconds"] += 0.05
                continue
            # One iteration = one loop turn that actually trains. Encode turns and warm-up holds
            # are skipped above and never recorded, as TAGNAS does with its non-training turns.
            with self._span("sync_wait"):
                self.gate.wait()
            t = time.perf_counter()
            self._tr("train step: forward")
            loss, n_nodes = self._train_step()
            self._tr("train step: done")
            self.stats["train_seconds"] += time.perf_counter() - t
            if self.telemetry is not None:
                self.telemetry.write_external_iter(
                    dict(self._phase), t_start=t_wall0, t_end=time.time(),
                    nodes_in_batch=int(n_nodes),
                )
            self.stats["steps"] += 1
            self.stats["nodes_processed"] += int(n_nodes)
            loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
            self.stats["loss_ema"] = loss_ema
            self.stats["gate_wait_seconds"] = self.gate.wait_seconds
            self.channel.lm_stats = dict(self.stats)

    # -------------------------------------------------------------------- step
    # (the contrastive batch is built inline in _train_step so that drawing the nodes and
    #  fetching the peer-modality table land in different telemetry phases)
    def _train_step(self):
        cfg = self.config
        self.model.train()
        gate0 = self.gate.wait_seconds
        if self.contrastive:
            with self._span("build_batch"):
                pool = self.train_text_nodes
                B = min(int(cfg.lm_batch_size), int(pool.numel()))
                nodes = pool[torch.randperm(int(pool.numel()), generator=self.gen)[:B].to(self.device)]
                if self._worst_case_pending:
                    self._worst_case_pending = False
                    if not bool((nodes == self._worst_case_node).any()):
                        nodes = nodes.clone()
                        nodes[0] = self._worst_case_node
                    self.log(f"[lm rank{self.rank}] probe: first LM batch carries the pool's longest text "
                             f"({self._worst_case_len} tokens, {int(nodes.numel())} rows)")
                rows = self.train_rows[nodes]
                ids, am = self._batch_tokens(rows)
                grp = self.train_group[nodes]
            with self._span("xfer"):
                gnn_table, _ = self.channel.current_gnn_table()
                targets = gnn_table[nodes].float()
            with self._span("lm_fwd"):
                with self._autocast():
                    _, logits = self.model(ids, am, grp, pairs=targets)
            loss = ls.square_contrastive_loss(logits.float())
            n_nodes = int(nodes.numel())
        else:
            # Text-side edge InfoNCE: the texts of the two endpoints of a train edge are positives,
            # the other endpoints in the batch are negatives (same loss family as the GNN side).
            with self._span("build_batch"):
                E = min(int(cfg.lm_batch_size), self.train_pairs.size(1))
                idx = torch.randperm(self.train_pairs.size(1), generator=self.gen)[:E].to(self.device)
                u, v = self.train_pairs[0, idx], self.train_pairs[1, idx]
                nodes = torch.cat([u, v])
                rows = self.train_rows[nodes]
                ids, am = self._batch_tokens(rows)
                grp = self.train_group[nodes]
            with self._span("lm_fwd"):
                with self._autocast():
                    emb = self.model(ids, am, grp)
            logits = self.model.contrastive_logits(emb[:E].float(), emb[E:].float())
            loss = ls.square_contrastive_loss(logits)
            n_nodes = int(nodes.numel())

        self.gate.wait()
        self._tr("train step: backward")
        with self._span("backward"):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        self._tr("train step: backward done; allreduce")
        self.gate.wait()
        with self._span("sync_wait"):
            self._allreduce_grads()
        with self._span("update"):
            if cfg.lm_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.params, cfg.lm_grad_clip)
            self.optimizer.step()
        self._tr("train step: optimizer done")
        out = float(loss.detach().item()), n_nodes
        self._resolve_phases(gate0)
        return out

    def _resolve_phases(self, gate_before: float) -> None:
        """Turn this step's CUDA events into seconds and fold in the bubble-gate wait.

        The gate stalls this thread mid-forward while the colocated GNN stage computes, so that
        idle time sits inside the forward event pair; move it from lm_fwd to sync_wait. Kernels
        queued before the pause keep running during it, so the subtraction can over-correct --
        the clamp is a red flag, counted in stats["lm_fwd_clamped"].
        """
        if self.stream is not None:
            self.stream.synchronize()          # this lane's stream only -- never the whole device
            for key, ev0, ev1 in self._pending_events:
                self._phase[key] = self._phase.get(key, 0.0) + ev0.elapsed_time(ev1) / 1000.0
            self._pending_events.clear()
        gate_delta = self.gate.wait_seconds - gate_before
        if gate_delta > 0:
            fwd = self._phase.get("lm_fwd", 0.0)
            if gate_delta > fwd:
                self.stats["lm_fwd_clamped"] = self.stats.get("lm_fwd_clamped", 0) + 1
            self._phase["lm_fwd"] = max(0.0, fwd - gate_delta)
            self._phase["sync_wait"] = self._phase.get("sync_wait", 0.0) + gate_delta

    def _batch_tokens(self, rows: torch.Tensor):
        """Token ids and mask for ``rows``; trimmed to the batch's longest real row when
        lm_dynamic_padding is on (the table is right-padded, so trimming drops padding only)."""
        ids = self.input_ids[rows].long()
        am = self.attention_mask[rows].long()
        if self.config.lm_dynamic_padding and am.numel():
            longest = int(am.sum(dim=1).max())
            if 0 < longest < am.size(1):
                ids, am = ids[:, :longest], am[:, :longest]
        return ids, am

    # ----------------------------------------------------------------- encode
    @torch.no_grad()
    def _refresh_standardization_stats(self, num_batches: int = 8) -> None:
        """Update the exit heads' standardisation statistics on a random sample of nodes
        (train-mode BatchNorm under no_grad) so the encode that follows is centred even before
        the LM has taken any optimisation step (initial sync)."""
        cfg, model = self.config, self.model
        model.train()
        B = int(cfg.lm_encode_batch)
        pool = self.full_text_nodes                     # statistics of real texts, not of the blank
        for _ in range(num_batches):
            rows = pool[torch.randperm(int(pool.numel()), generator=self.gen)[:B].to(self.device)]
            with self._autocast():
                ids, am = self._batch_tokens(rows)
                model.encode(ids, am, None if cfg.encode_exit == "full" else self.group_full[rows])
        model.eval()

    @torch.no_grad()
    def _encode_all(self, rid: int) -> None:
        cfg, data, model = self.config, self.data, self.model
        self._refresh_standardization_stats()
        model.eval()
        N = data.num_nodes
        per = int(math.ceil(N / self.world_size))
        start, end = self.rank * per, min(N, (self.rank + 1) * per)
        D = int(cfg.embed_dim)
        gather_dtype = torch.float16 if self.device.type == "cuda" else torch.float32   # gloo: gather in fp32
        local = torch.zeros((per, D), dtype=gather_dtype, device=self.device)
        if end > start:
            rows = torch.arange(start, end, device=self.device)
            grp = self.group_full[rows]
            if cfg.lm_dynamic_padding:
                # Sort by (exit group, length) so each encode batch shares an exit depth AND a similar
                # length; with a fixed L this ordering changes nothing, with dynamic padding it keeps a
                # 1024-token outlier from padding a whole batch of short rows. Output is unchanged.
                lens = self.attention_mask[rows].sum(dim=1).long()
                order = torch.argsort(grp * (int(self.attention_mask.size(1)) + 1) + lens)
            else:
                order = torch.argsort(grp)                # group-sorted so batches share an exit depth
            rows_sorted = rows[order]
            B = int(cfg.lm_encode_batch)
            full_depth = cfg.encode_exit == "full"
            for i in range(0, rows_sorted.numel(), B):
                r = rows_sorted[i:i + B]
                with self._autocast():
                    ids, am = self._batch_tokens(r)
                    emb = model.encode(ids, am, None if full_depth else self.group_full[r])
                local[r - start] = emb.to(gather_dtype)
        if self.world_size > 1:
            gathered = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(gathered, local, group=self.lm_pg)
            table = torch.cat(gathered, dim=0)[:N]
        else:
            table = local[:N]
        table = table.to(torch.float16)
        if self.stream is not None:
            self.stream.synchronize()
        model.train()
        self.channel.publish(table.contiguous(), rid)
        self.log(f"[lm rank{self.rank}] sync #{rid}: encoded {N} nodes at {cfg.encode_exit} exit "
                 f"(shard {end - start}, steps so far {self.stats['steps']})")

    def state_dict_cpu(self, dtype: torch.dtype = torch.float16) -> dict:
        """CPU copy of the LM weights (fp16 by default: it is only kept for inspection/re-encoding)."""
        return {k: (v.detach().to(dtype) if v.is_floating_point() else v.detach()).cpu().clone()
                for k, v in self.model.state_dict().items()}
