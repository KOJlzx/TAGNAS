"""GNN training loop for one pipeline stage (runs in the main thread of each rank).

Rank 0 owns every time- or loss-based decision (Δt schedule, validation
cadence, early stopping, runtime cap) and broadcasts it over the gloo world
group so that all stages take identical control paths; the last stage owns
the loss and the validation AUC and broadcasts those back.

Objectives (``config.objective``):

* ``contrastive`` (default) – the objective shared with B1/B2/B3/TAGNAS: the
  GNN reads the adjacency-SVD node features and its embeddings are aligned
  with the LM text embeddings of the same nodes through
  ``clip_graph.losses.square_contrastive_loss``. At every Δt sync the LM table
  is refreshed (LM -> GNN) and the GNN publishes its own embeddings of all
  train nodes (GNN -> LM) as the LM's alignment targets.
* ``link`` – the original UniTG task-loss port (BCE on edges, LM table as GNN
  input). Kept for reference; see ``config.py`` for why it is not comparable.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from clip_graph import losses as ls

from .channel import SyncChannel
from .collaborator import SyncScheduler
from .config import UniTGConfig
from .data import UniTGData
from .deep_gnn import RevGNNStage
from .eval_utils import link_pred_metrics, sample_eval_negatives
from .partition import MicroBatch, build_microbatches, chunk_for_eval
from .pipeline import LMGate, PipelineEngine


def _bcast_floats(values: List[float], src: int, group) -> List[float]:
    t = torch.tensor(values, dtype=torch.float64)
    dist.broadcast(t, src=src, group=group)
    return t.tolist()


def _bcast_object(obj, src: int, group):
    holder = [obj]
    dist.broadcast_object_list(holder, src=src, group=group)
    return holder[0]


class GNNStageRunner:
    def __init__(
        self,
        config: UniTGConfig,
        data: UniTGData,
        rank: int,
        world_size: int,
        device: torch.device,
        world_pg,
        gnn_pg,
        gate: LMGate,
        channel: SyncChannel,
        log: Callable[[str], None] = print,
        telemetry=None,
    ):
        self.cfg = config
        self.data = data
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.world_pg = world_pg
        self.gnn_pg = gnn_pg
        self.gate = gate
        self.channel = channel
        self.log = log
        self.is_root = rank == 0
        self.is_last = rank == world_size - 1
        self._logged_stop_defer = False      # one-shot log when the LM guard defers an early stop
        self.run_dir = config.run_dir
        self.events: List[dict] = []
        self.contrastive = config.objective == "contrastive"
        self.telemetry = telemetry      # clip_graph.telemetry.TelemetryRecorder or None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        cfg, data = self.cfg, self.data
        if cfg.objective not in ("contrastive", "link"):
            raise ValueError(f"unknown objective {cfg.objective!r}")
        if not self.contrastive and int(data.in_channels) != int(cfg.embed_dim):
            # Under 'link' the GNN input alternates between the numerical fallback features and
            # the LM table, so both must have the same width.
            raise ValueError(f"feature width mismatch: data features are {data.in_channels}-d but "
                             f"embed_dim={cfg.embed_dim}; set --embed-dim to the feature width")
        if self.contrastive and not cfg.initial_sync:
            raise ValueError("the contrastive objective needs the LM table from the start; keep --initial-sync")
        if cfg.lm_objective not in ("contrastive", "link"):
            raise ValueError(f"unknown lm_objective {cfg.lm_objective!r}")
        if not self.contrastive and cfg.lm_objective != "link":
            raise ValueError("objective=link requires lm_objective=link")
        torch.manual_seed(cfg.seed + 17 * self.rank)
        self.stage = RevGNNStage(
            stage_index=self.rank, num_stages=self.world_size,
            in_channels=data.in_channels, hidden_channels=cfg.hidden_channels,
            out_channels=cfg.embed_dim, num_layers=cfg.num_layers,
            dropout=cfg.gnn_dropout, num_groups=cfg.num_groups,
            tau_init=cfg.tau_init, max_tau=cfg.max_tau,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.stage.parameters(), lr=cfg.gnn_lr, weight_decay=cfg.gnn_weight_decay)
        self.engine = PipelineEngine(self.stage, self.rank, self.world_size, self.gnn_pg, self.device, self.gate)

        # Partition once on rank 0 and share, so every stage sees identical microbatches.
        train, val = data.train, data.val
        eval_split = data.test if cfg.val_eval_on_test else data.val
        if self.is_root:
            minibatches = build_microbatches(train.edge_index, train.num_nodes, cfg.microbatches,
                                             cfg.max_nodes_per_microbatch, min_microbatches=None,
                                             halo_hops=int(cfg.halo_hops))
            eval_chunks = chunk_for_eval(eval_split.edge_index, eval_split.num_nodes, cfg.max_nodes_per_microbatch)
            payload = (minibatches, eval_chunks)
        else:
            payload = None
        minibatches, eval_chunks = _bcast_object(payload, 0, self.world_pg)
        self.minibatches = [[self._to_device(mb) for mb in mini] for mini in minibatches]
        self.eval_chunks = [self._to_device(ch) for ch in eval_chunks]
        self.train_chunks = [mb for mini in self.minibatches for mb in mini]
        # Seed rows only: with halo_hops > 0 a node also appears as another microbatch's halo, where its
        # neighbourhood is truncated; the published table must hold its own (seed) embedding.
        self.train_perm = torch.cat([mb.node_index[:mb.seeds] for mb in self.train_chunks]).to(self.device)
        self._seed_rows = torch.cat([torch.arange(mb.seeds) + off for mb, off in zip(
            self.train_chunks, torch.cumsum(torch.tensor([0] + [mb.num_nodes for mb in self.train_chunks[:-1]]), 0).tolist())
        ]).to(self.device)
        self.eval_split = eval_split
        self.nodes_per_epoch = sum(mb.num_nodes for mb in self.train_chunks)

        self.train_rows = train.rows.to(self.device)
        self.train_has_text = data.has_text[train.rows].to(self.device)   # contrastive targets exist only for these
        # Seed nodes per epoch: the nodes that actually enter the contrastive objective, which is what
        # B1/B2/B3 and TAGNAS put in their throughput numerator (TAGNAS: batches x graph_batch_size).
        # It mirrors loss_fn() below exactly, and every rank can compute it without communication
        # because the microbatch partition, has_text and contrastive_nodes are identical on all stages
        # (only the last stage runs the loss). Under the link objective the loss has no anchors, so the
        # metric degenerates to the full pass.
        if self.contrastive:
            cap = int(self.cfg.contrastive_nodes)
            self.seed_nodes_per_minibatch = []
            for mini in self.minibatches:
                tot = 0
                for mb in mini:
                    n = int(self.train_has_text[mb.node_index[:mb.seeds]].sum())
                    tot += 0 if n < 2 else min(cap, n)
                self.seed_nodes_per_minibatch.append(tot)
        else:
            self.seed_nodes_per_minibatch = [sum(mb.seeds for mb in mini) for mini in self.minibatches]
        self.seed_nodes_per_epoch = sum(self.seed_nodes_per_minibatch)
        self.eval_rows = eval_split.rows.to(self.device)
        if self.engine.is_first:
            self.train_x = train.x.to(self.device)
            self.eval_x = eval_split.x.to(self.device)
        if self.is_last:
            self.eval_pos = eval_split.edge_index.to(self.device)
            self.eval_neg, self.neg_info = sample_eval_negatives(
                self.eval_pos, eval_split.num_nodes, eval_split.node_ids, data.dataset, data.data_dir,
                mode=cfg.link_neg_sampling, device=self.device,
            )
            self.eval_perm = torch.cat([ch.node_index for ch in self.eval_chunks]).to(self.device)
        if self.is_root:
            unit, budget = cfg.sync_budget()
            self.scheduler = SyncScheduler(fractions=tuple(cfg.sync_schedule), budget=budget, unit=unit,
                                           tighten_after=cfg.sync_tighten_epochs, ema=cfg.sync_ema)
            self.log(f"[gnn] objective={cfg.objective} lm_objective={cfg.lm_objective} stages={self.world_size} layers/stage={self.stage.num_local_layers} "
                     f"minibatches={len(self.minibatches)} microbatches={[mb.num_nodes for mb in self.minibatches[0]]} "
                     f"seeds={[mb.seeds for mb in self.minibatches[0]]} halo_hops={cfg.halo_hops} "
                     f"eval_chunks={[ch.num_nodes for ch in self.eval_chunks]} sync={unit} budget={budget} "
                     f"intervals={self.scheduler.intervals}")

    def _to_device(self, mb: MicroBatch) -> MicroBatch:
        return MicroBatch(node_index=mb.node_index.to(self.device), edge_index=mb.edge_index.to(self.device),
                          num_seeds=mb.num_seeds)

    # --------------------------------------------------------------- features
    def _lm_rows(self, rows: torch.Tensor) -> Optional[torch.Tensor]:
        table = self.channel.current_table()
        if table is None:
            return None
        return table[rows].float()

    def train_features(self, mb: MicroBatch) -> torch.Tensor:
        if self.contrastive:
            return self.train_x[mb.node_index]
        feats = self._lm_rows(self.train_rows[mb.node_index])
        return self.train_x[mb.node_index] if feats is None else feats

    def eval_features(self, ch: MicroBatch) -> torch.Tensor:
        if self.contrastive:
            return self.eval_x[ch.node_index]
        feats = self._lm_rows(self.eval_rows[ch.node_index])
        return self.eval_x[ch.node_index] if feats is None else feats

    # ------------------------------------------------------------------- loss
    def loss_fn(self, z: torch.Tensor, mb: MicroBatch) -> torch.Tensor:
        if self.contrastive:
            # Only text-bearing nodes can be anchors of the graph-text InfoNCE (as in B1/TAGNAS);
            # a blank text would otherwise be one shared target for every textless node.
            # With halo_hops > 0 only the seeds (the first mb.seeds rows) are anchors, as in TAGNAS.
            cand = torch.nonzero(self.train_has_text[mb.node_index[:mb.seeds]], as_tuple=False).flatten()
            n = int(cand.numel())
            if n < 2:
                return z.sum() * 0.0
            k = min(int(self.cfg.contrastive_nodes), n)
            idx = cand[torch.randperm(n, device=z.device)[:k]] if k < n else cand
            targets = self._lm_rows(self.train_rows[mb.node_index[idx]])
            if targets is None:
                raise RuntimeError("no LM table available for the contrastive loss")
            logits = self.stage.contrastive_logits(z[idx], targets)
            return ls.square_contrastive_loss(logits)
        ei = mb.edge_index
        pos = ei[:, ei[0] < ei[1]]
        cap = int(self.cfg.max_pos_edges_per_microbatch)
        if cap and pos.size(1) > cap:
            pos = pos[:, torch.randperm(pos.size(1), device=pos.device)[:cap]]
        neg = negative_sampling(ei, num_nodes=mb.num_nodes, num_neg_samples=pos.size(1))
        logits = torch.cat([self.stage.link_logits(z, pos), self.stage.link_logits(z, neg)])
        y = torch.cat([torch.ones(pos.size(1), device=z.device), torch.zeros(neg.size(1), device=z.device)])
        return F.binary_cross_entropy_with_logits(logits, y)

    # ------------------------------------------------------------------- eval
    def evaluate(self) -> float:
        z = self.engine.eval_forward(self.eval_chunks, self.eval_features if self.engine.is_first else None)
        auc = 0.0
        if self.is_last:
            full = torch.empty((self.eval_split.num_nodes, z.size(1)), device=z.device, dtype=z.dtype)
            full[self.eval_perm] = z
            metrics = link_pred_metrics(full, self.eval_pos, self.eval_neg)
            auc = float(metrics["auc"])
        (auc,) = _bcast_floats([auc], self.world_size - 1, self.world_pg)
        return auc

    # ------------------------------------------------------------------- sync
    def sync_features(self, rid: int) -> None:
        """Blocking LM -> GNN sync: every LM replica re-encodes its shard of all nodes."""
        self.channel.request_sync(rid)
        self.channel.wait_served(rid)

    def publish_gnn_table(self, version: int) -> None:
        """GNN -> LM sync (contrastive objective): embed every train node with the current GNN in
        eval mode and hand the table to the LM threads of all ranks."""
        n_train, d = self.data.train.num_nodes, int(self.cfg.embed_dim)
        z = self.engine.eval_forward(self.train_chunks, self.train_features if self.engine.is_first else None)
        table = torch.empty((n_train, d), dtype=torch.float16)
        if self.is_last:
            full = torch.empty((n_train, z.size(1)), device=z.device, dtype=z.dtype)
            full[self.train_perm] = z[self._seed_rows]
            table = full.to(torch.float16).cpu()
        dist.broadcast(table, src=self.world_size - 1, group=self.world_pg)
        self.channel.publish_gnn(table.to(self.device), version)

    # ------------------------------------------------------------------- train
    def train(self) -> Dict:
        cfg = self.cfg
        t_start = time.perf_counter()
        elapsed = lambda: time.perf_counter() - t_start  # noqa: E731
        min_lm_texts = int(cfg.min_lm_texts_before_stop)
        if min_lm_texts == -1:
            # Resolve against this run's actual split, not another seed's cached metadata.
            min_lm_texts = int(self.data.has_text[self.data.train.rows].sum())
        if self.is_root:
            self._open_logs()
            self._event({"event": "early_stop_guard", "configured_min_lm_texts_before_stop": cfg.min_lm_texts_before_stop,
                         "min_lm_texts_before_stop": min_lm_texts})
        next_rid = 1
        num_syncs = 0
        sync_seconds = 0.0
        last_eval_t = 0.0
        best_auc, best_epoch, best_elapsed = -1.0, -1, 0.0
        no_improve = 0
        best_state: Optional[dict] = None
        best_table: Optional[torch.Tensor] = None
        stop_reason = "all_epochs_completed"
        epoch_times: List[float] = []
        nodes_total = 0
        seed_nodes_total = 0
        evals = 0
        epoch = -1
        val_auc = None

        def do_sync(rid: int, epoch_no: int, initial: bool = False):
            nonlocal num_syncs, sync_seconds
            t = time.perf_counter()
            self.sync_features(rid)
            num_syncs += 1
            if self.contrastive and cfg.lm_objective == "contrastive":
                self.publish_gnn_table(num_syncs)
            sync_seconds += time.perf_counter() - t
            if self.is_root:
                # In wall-clock mode Δt starts after the blocking encode/table exchange, including
                # the initial sync; a long encode must not make the next sync immediately due.
                self.scheduler.mark_synced(elapsed() if self.scheduler.unit == "seconds" else float(epoch_no))
                ev = {"event": "sync", "rid": rid, "epoch": epoch_no, "elapsed": elapsed(),
                      "seconds": time.perf_counter() - t, "lm_steps": self.channel.lm_stats.get("steps")}
                if initial:
                    ev["initial"] = True
                else:
                    ev["interval"] = self.scheduler.current_interval
                self._event(ev)

        def run_eval(epoch_no: int):
            """Collective validation + best-checkpoint bookkeeping (rank 0 decides, all ranks snapshot)."""
            nonlocal evals, last_eval_t, best_auc, best_epoch, best_elapsed, no_improve, best_state, best_table
            _t_ev0 = time.perf_counter()     # the GNN pipeline is paused for the evaluation
            auc = self.evaluate()
            eval_s = time.perf_counter() - _t_ev0
            evals += 1
            is_best, stop = 0.0, 0.0
            if self.is_root:
                last_eval_t = elapsed()
                if auc > best_auc + 1e-6:
                    best_auc, best_epoch, best_elapsed = auc, epoch_no, elapsed()
                    no_improve = 0
                    is_best = 1.0
                else:
                    no_improve += 1
                # The LM counter is per rank; replicas train in lock-step. Only rank 0 decides
                # whether the plateau is meaningful, then broadcasts the result to every stage.
                lm_texts = int(self.channel.lm_stats.get("nodes_processed", 0)) * self.world_size
                text_ok = lm_texts >= min_lm_texts
                patience_exhausted = no_improve >= cfg.early_stopping_patience
                deferred = patience_exhausted and not text_ok
                if deferred and not self._logged_stop_defer:
                    self._logged_stop_defer = True
                    self.log(f"[gnn] patience exhausted but the LM has trained on {lm_texts} texts < "
                             f"{min_lm_texts}; deferring early stop")
                stop = 1.0 if patience_exhausted and text_ok else 0.0
                self._timeline(elapsed(), epoch_no, auc, eval_s)
                self._event({"event": "val_eval", "epoch": epoch_no, "elapsed": elapsed(), "auc": auc,
                             "best_auc": best_auc, "no_improve": no_improve, "lm_steps": self.channel.lm_stats.get("steps"),
                             "lm_texts": lm_texts, "min_lm_texts_before_stop": min_lm_texts,
                             "early_stop_deferred": deferred, "early_stop": bool(stop)})
            is_best, stop = _bcast_floats([is_best, stop], 0, self.world_pg)
            if is_best:
                best_state = {k: v.detach().cpu().clone() for k, v in self.stage.state_dict().items()}
                if self.is_root:
                    tab = self.channel.current_table()
                    best_table = tab.detach().cpu().clone() if tab is not None else None
            return auc, is_best, stop

        if cfg.initial_sync:
            do_sync(next_rid, 0, initial=True)
            next_rid += 1
        if cfg.eval_at_start:
            val_auc, _, _ = run_eval(0)

        for epoch in range(cfg.num_epochs):
            t_ep = time.perf_counter()
            losses = []
            for _mb_i, mini in enumerate(self.minibatches):
                loss = self.engine.train_minibatch(
                    mini,
                    self.train_features if self.engine.is_first else None,
                    self.loss_fn if self.is_last else None,
                    self.optimizer,
                    cfg.gnn_grad_clip,
                )
                if loss is not None:
                    losses.append(loss)
                if self.telemetry is not None:
                    # One iteration = one mini-batch = one optimizer step, the same unit the
                    # DDP baselines and the TAGNAS graph lane use.
                    self.telemetry.write_external_iter(
                        dict(self.engine._phase),
                        t_start=self.engine.last_iter_t_start,
                        t_end=self.engine.last_iter_t_end,
                        nodes_in_batch=self.seed_nodes_per_minibatch[_mb_i],
                    )
            nodes_total += self.nodes_per_epoch
            seed_nodes_total += self.seed_nodes_per_epoch
            epoch_loss = float(sum(losses) / len(losses)) if losses else 0.0
            (epoch_loss,) = _bcast_floats([epoch_loss], self.world_size - 1, self.world_pg)
            epoch_times.append(time.perf_counter() - t_ep)

            # ---- rank-0 decisions --------------------------------------------
            if self.is_root:
                ev = self.scheduler.observe_epoch_loss(epoch_loss, epoch + 1)
                if ev is not None:
                    ev["elapsed"] = elapsed()
                    self._event(ev)
                mark = elapsed() if self.scheduler.unit == "seconds" else float(epoch + 1)
                sync_due = 1.0 if self.scheduler.due(mark) else 0.0
                eval_due = 1.0 if (elapsed() - last_eval_t >= cfg.val_eval_interval_seconds
                                   and (epoch + 1) >= cfg.val_eval_min_epochs) else 0.0
                over_time = 1.0 if (cfg.max_runtime_seconds and elapsed() >= cfg.max_runtime_seconds) else 0.0
                last_epoch = 1.0 if epoch + 1 >= cfg.num_epochs else 0.0
                if over_time or last_epoch:
                    eval_due = 1.0        # always score the final state
                dec = [sync_due, float(next_rid), eval_due, over_time]
            else:
                dec = [0.0, 0.0, 0.0, 0.0]
            sync_due, rid, eval_due, over_time = _bcast_floats(dec, 0, self.world_pg)

            is_best = 0.0
            if eval_due:
                val_auc, is_best, stop_es = run_eval(epoch + 1)
            else:
                stop_es = 0.0

            # Validate before refreshing the tables: a terminal epoch has no next optimizer step
            # that could consume them. The first broadcast can itself cross the cap even when
            # no evaluation was scheduled; every rank must then score the final state together.
            final_eval_due = 0.0
            if self.is_root:
                if cfg.max_runtime_seconds and elapsed() >= cfg.max_runtime_seconds:
                    over_time = 1.0
                if over_time and not eval_due:
                    final_eval_due = 1.0
                if stop_es or over_time or last_epoch:
                    sync_due = 0.0
            sync_due, over_time, final_eval_due = _bcast_floats(
                [sync_due, over_time, final_eval_due], 0, self.world_pg,
            )
            if final_eval_due:
                val_auc, is_best, stop_es = run_eval(epoch + 1)
                eval_due = 1.0
            if sync_due:
                do_sync(int(rid), epoch + 1)
                next_rid = int(rid) + 1

            if self.is_root:
                self._metrics_row(epoch + 1, epoch_loss, val_auc if eval_due else None, best_auc, epoch_times[-1], num_syncs)
                if (epoch + 1) % cfg.log_every_epochs == 0 or eval_due:
                    lm = self.channel.lm_stats
                    lm_loss = lm.get("loss_ema")
                    self.log(f"[gnn] epoch {epoch + 1} loss={epoch_loss:.4f} ema={self.scheduler.ema_loss:.4f} "
                             f"dt={self.scheduler.current_interval:.0f}{self.scheduler.unit[0]} syncs={num_syncs} "
                             f"val_auc={'%.4f' % val_auc if val_auc is not None else '-'} best={best_auc:.4f} "
                             f"ep={epoch_times[-1]:.2f}s t={elapsed():.0f}s lm_steps={lm.get('steps')} "
                             f"lm_loss={'-' if lm_loss is None else round(lm_loss, 4)}")

            if stop_es:
                stop_reason = "val_early_stop"
                break
            if over_time:
                stop_reason = "max_runtime_seconds"
                break

        total = elapsed()
        final_state = {k: v.detach().cpu().clone() for k, v in self.stage.state_dict().items()}
        if best_state is None:
            best_state, best_table = final_state, None
        final_table = self.channel.current_table()
        final_table = final_table.detach().cpu().clone() if final_table is not None else None
        if best_table is None:
            best_table = final_table

        gathered_best = [None] * self.world_size if self.is_root else None
        gathered_final = [None] * self.world_size if self.is_root else None
        dist.gather_object(best_state, gathered_best, dst=0, group=self.world_pg)
        dist.gather_object(final_state, gathered_final, dst=0, group=self.world_pg)

        summary = {
            "objective": cfg.objective,
            "total_seconds": total,
            "epochs_completed": epoch + 1 if epoch >= 0 else 0,
            "epoch_seconds": epoch_times,
            "stop_reason": stop_reason,
            "min_lm_texts_before_stop": min_lm_texts,
            "best_val_auc": best_auc if best_auc >= 0 else None,
            "best_epoch": best_epoch if best_epoch >= 0 else None,
            "best_elapsed_seconds": best_elapsed,
            "num_syncs": num_syncs,
            "sync_seconds": sync_seconds,
            "num_val_evals": evals,
            "gnn_nodes_processed": nodes_total,
            "gnn_seed_nodes_processed": seed_nodes_total,
            "gnn_compute_seconds": self.engine.compute_seconds,
            "gnn_comm_seconds": self.engine.comm_seconds,
            "gate_pause_count": self.gate.pause_count,
            "final_val_auc": val_auc,
        }
        if self.is_root:
            summary["scheduler"] = self.scheduler.state()
            summary["events"] = self.events
            summary["eval_split"] = self.eval_split.name
            self._save_models(gathered_best, gathered_final, best_table, final_table, summary)
        return summary

    # ---------------------------------------------------------------- artifacts
    def _open_logs(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.run_dir, "metrics.csv")
        with open(self.metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "val_auc", "best_val_auc",
                                    "epoch_seconds", "timestamp", "ema_loss", "sync_interval", "num_syncs", "lm_steps"])
        self.timeline_csv = os.path.join(self.run_dir, "val_timeline.csv")
        with open(self.timeline_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["elapsed_seconds", "epoch", "global_step", "val_auc", "timestamp", "eval_seconds"])
        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        open(self.events_path, "w").close()

    def _event(self, ev: dict) -> None:
        ev = dict(ev)
        ev.setdefault("timestamp", datetime.now().isoformat())
        self.events.append(ev)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        self.log(f"[event] {json.dumps(ev)}")

    def _timeline(self, elapsed: float, epoch: int, auc: float, eval_seconds: Optional[float] = None) -> None:
        with open(self.timeline_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{elapsed:.3f}", epoch, epoch * len(self.minibatches), f"{auc:.6f}",
                                    datetime.now().isoformat(), "" if eval_seconds is None else f"{eval_seconds:.3f}"])

    def _metrics_row(self, epoch, loss, val_auc, best_auc, ep_s, num_syncs) -> None:
        with open(self.metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{loss:.6f}", "", "", "" if val_auc is None else f"{val_auc:.6f}",
                "" if best_auc < 0 else f"{best_auc:.6f}", f"{ep_s:.4f}", datetime.now().isoformat(),
                f"{self.scheduler.ema_loss:.6f}", f"{self.scheduler.current_interval:.3f}", num_syncs,
                self.channel.lm_stats.get("steps", 0),
            ])

    def _save_models(self, best_states, final_states, best_table, final_table, summary) -> None:
        cfg = self.cfg
        common = {"config": cfg.to_dict(), "num_stages": self.world_size, "in_channels": self.data.in_channels,
                  "objective": cfg.objective}
        torch.save({**common, "stages": best_states, "epoch": summary["best_epoch"], "val_auc": summary["best_val_auc"]},
                   os.path.join(self.run_dir, "best_model.pth"))
        torch.save({**common, "stages": final_states, "epoch": summary["epochs_completed"], "val_auc": summary["final_val_auc"]},
                   os.path.join(self.run_dir, "final_model.pth"))
        if cfg.save_features or cfg.objective == "link":
            for tag, tab in (("best", best_table), ("final", final_table)):
                torch.save({"node_ids": self.data.full_node_ids, "table": tab},
                           os.path.join(self.run_dir, f"features_{tag}.pt"))
        self.log(f"[gnn] saved best/final models to {self.run_dir}")
