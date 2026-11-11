#!/usr/bin/env python
"""Staleness experiment B: TAGNAS's two engines simulated in one process on one GPU, with a controlled lag.

Takes exactly bin/async_coexec_train.py's command line (run_sys.sh with TRAIN_ENTRY=bin/staleness_sim.py),
plus the --sim-* options below; the graph tower is placed on the same GPU (--graph-device cuda:0) and one
text replica is used (--num-text-workers 1). The real GraphWorkerTrainingService / TextWorkerTrainingService
do the work: their pushes go straight into the peer's local cache instead of over RPC, and the learned
temperature is handed to the text side directly. Nothing about the model, loss, optimisers or
hyperparameters changes.

Loop: before every LM step, run r GNN steps (r accumulated and floored when fractional). A GNN step is the
graph worker's normal step: read C_T (drop-on-miss), update w or alpha (alternating), prune on its schedule,
push C_G. An LM step is the text worker's normal step except for where z_G comes from:
  --sim-mode cache   C_G as usual (drop-on-miss) -- the control;
  --sim-mode fixed   C_G is not read: z_G of the batch's nodes is recomputed from the supernet snapshot taken
                     s GNN updates ago (weights, alpha, pruning masks, search phase and arch temperature of the
                     next step), in train mode under no_grad, on the batch's k-hop subgraph exactly as a push
                     computes it. A snapshot is kept every ceil(s/16) updates (at most 17, plus the initial one,
                     which serves the first s updates), so the actual lag lies in [s, s + ceil(s/16) - 1];
  --sim-mode frozen  the LM is never updated: one pass of its initial weights over every training node fills C_T
                     (each LM batch misses on purpose, so it only pushes z_T), then the GNN takes as many steps as
                     the r-schedule gives the other modes, always against those z_T -- the most stale z_T can be.
The run stops after --sim-lm-steps LM steps; the graph checkpoint and genotype are then written where the
real search writes them, so run_sys.sh's post-run evaluation applies unchanged.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clip_graph.async_coexec import train_runtime as tr  # noqa: E402
from clip_graph.async_coexec.pubmed_runtime import PubmedRuntimeStore  # noqa: E402
from clip_graph.async_coexec.types import CacheKey, EmbeddingRecord  # noqa: E402


def _entry_module():
    spec = importlib.util.spec_from_file_location("async_coexec_train", REPO / "bin" / "async_coexec_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sim_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sim-mode", choices=("fixed", "cache", "frozen"), required=True)
    p.add_argument("--sim-s", type=int, default=0, help="fixed mode: lag of z_G in GNN updates")
    p.add_argument("--sim-ratio", type=float, required=True, help="GNN steps per LM step (r)")
    p.add_argument("--sim-lm-steps", type=int, required=True, help="LM step budget")
    p.add_argument("--sim-log-every", type=int, default=200)
    p.add_argument("--sim-genotype", default=None,
                   help="fixed architecture (e.g. results/staleness/genotype_gat.json): the retrain stage -- alpha one-hot "
                        "and frozen, no arch updates, no pruning -- trained from scratch")
    sim, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    return sim


class Snapshots:
    """Supernet snapshots keyed by the graph worker's successful-update count."""

    def __init__(self, graph_svc, s: int, horizon_cfg) -> None:
        self.g = graph_svc
        self.s = int(s)
        self.every = max(1, math.ceil(self.s / 16)) if self.s > 0 else 0
        self.store: collections.deque = collections.deque(maxlen=17)
        self.cfg = horizon_cfg
        self.initial = (0, self._take(0))
        self.lags: collections.Counter = collections.Counter()

    def _take(self, u: int):
        """Copy of the live tower, set up as the graph worker's NEXT step (after update u) would run it."""
        snap = copy.deepcopy(self.g.graph_tower)
        if self.g.config.stage == "search":
            snap.set_search_phase(train_architecture=(u % 2 == 1))
            ts, te = float(self.cfg.graph_arch_temp_start), float(self.cfg.graph_arch_temp_end)
            if te != ts:
                horizon = int(self.cfg.max_graph_steps or 2000)
                snap.gnn.set_arch_temperature(ts * (te / ts) ** min(1.0, u / max(1, horizon)))
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap

    def after_graph_step(self) -> None:
        u = int(self.g._successful_updates)
        if self.every and u > 0 and u % self.every == 0 and (not self.store or self.store[-1][0] != u):
            self.store.append((u, self._take(u)))

    def tower_for_now(self):
        """(version, tower) whose version is the newest <= now - s (s = 0: the live state, copied)."""
        u = int(self.g._successful_updates)
        if self.s == 0:
            return u, self._take(u)
        target = u - self.s
        if target <= 0:
            return self.initial
        best = None
        for v, t in self.store:
            if v <= target:
                best = (v, t)
        if best is None:          # cannot happen with 17 snapshots every ceil(s/16) updates; be safe
            best = self.store[0] if self.store else self.initial
        return best


def main() -> None:
    sim = _sim_args()
    entry = _entry_module()
    args = entry.parse_args()
    entry.configure_hf_runtime(local_files_only=args.offline_models_only)
    stage = "retrain" if sim.sim_genotype else "search"
    config = entry.build_stage_config(args, stage=stage, dist_port=args.dist_port, rpc_port=args.rpc_port,
                                      run_name=entry.build_run_name(),
                                      genotype_path=str(Path(sim.sim_genotype).resolve()) if sim.sim_genotype else None)
    if config.num_text_workers != 1 or not str(config.graph_device).startswith("cuda"):
        raise SystemExit("staleness_sim: run with NUM_TEXT_WORKERS=1 and GRAPH_DEVICE=cuda:0 (one GPU for both engines)")
    entry.seed_everything(config.seed)
    config.train_output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        entry.dump_cli_args(config, args)

    data_args, model_args = tr._load_runtime_spec(config)
    store = PubmedRuntimeStore.build(
        data_args=data_args, split_proportions=config.split_proportions, seed=config.seed,
        local_files_only=config.local_files_only, feature_dim=int(model_args["gnn_params"]["in_channels"]),
        train_node_fraction=config.train_node_fraction)
    g = tr.GraphWorkerTrainingService(config=config, data_store=store, model_args=model_args)
    t = tr.TextWorkerTrainingService(config=config, role_name=config.text_worker_name(0), data_store=store,
                                     model_args=model_args, text_group=None)
    dev = t.device

    # --- in-process "network": pushes land in the peer's local cache; tau goes straight to the text side ---
    counts = collections.Counter()

    def g_push(modality, records, *, telemetry):
        counts["g_batches"] += 1
        counts["g_updates"] += int(telemetry.get("update_applied", 0))
        if records:
            t.ingest_remote_push("graph", records, config.graph_worker_name)
        return True

    def t_push(modality, records, *, telemetry):
        counts["t_batches"] += 1
        counts["t_updates"] += int(telemetry.get("update_applied", 0))
        if records:
            g.ingest_remote_push("text", records, t.role_name)
        return True

    g.push_embeddings = g_push
    t.push_embeddings = t_push
    g._broadcast_temperature = lambda: t.set_temperature(float(g.graph_tower.tau.detach().cpu().item()))
    g._broadcast_temperature()

    snaps = Snapshots(g, sim.sim_s, config) if sim.sim_mode == "fixed" else None
    if snaps is not None:
        def fixed_lookup(node_ids):
            version, tower = snaps.tower_for_now()
            snaps.lags[int(g._successful_updates) - int(version)] += 1
            batch = store.build_graph_batch(node_ids, config.graph_k_hop)
            tower.train(mode=True)
            with torch.no_grad():
                z = tower.encode(
                    batch.graph_x.to(g.device), batch.graph_edge_index.to(g.device),
                    batch.graph_edge_attr.to(g.device) if batch.graph_edge_attr is not None else None,
                    batch.target_index.to(g.device), is_search=config.stage == "search")
            recs = [EmbeddingRecord(key=CacheKey("graph", int(n)), tensor=z[i].detach().to(dev, dtype=config.cache_dtype),
                                    version=1, producer=config.graph_worker_name, producer_step=int(version))
                    for i, n in enumerate(node_ids)]
            return recs, {"local_hit_nodes": len(recs), "remote_hit_nodes": 0, "miss_nodes": 0,
                          "mean_staleness": float(int(g._successful_updates) - int(version))}
        t._get_graph_embeddings = fixed_lookup
    if sim.sim_mode == "frozen":
        def no_lookup(node_ids):     # every LM batch misses: no update, but its z_T is pushed as usual
            return None, {"local_hit_nodes": 0, "remote_hit_nodes": 0, "miss_nodes": len(node_ids), "mean_staleness": 0.0}
        t._get_graph_embeddings = no_lookup

    # --- the loop ---
    t0 = time.time()
    acc = 0.0
    if sim.sim_mode == "frozen":
        n_train, pushed = len(store.scheduler_train_nodes()), set()
        while len(pushed) < n_train and counts["t_batches"] < 2 * math.ceil(n_train / config.text_chunk_size):
            ids = t.scheduler.next_batch()
            t.run_text_batch(ids)
            pushed.update(int(i) for i in ids)
        print(f"[sim] frozen LM: z_T of {len(pushed)}/{n_train} training nodes from {counts['t_batches']} batches, "
              f"{counts['t_updates']} updates  {time.time() - t0:.0f}s", flush=True)
        g_steps = 0
        for _ in range(sim.sim_lm_steps):     # the GNN steps the r-schedule gives the other modes
            acc += sim.sim_ratio
            n = int(math.floor(acc + 1e-9))
            acc -= n
            g_steps += n
        for i in range(1, g_steps + 1):
            g.run_graph_batch(g.scheduler.next_batch())
            if i % sim.sim_log_every == 0 or i == g_steps:
                print(f"[sim] gnn step {i}/{g_steps}  updates {counts['g_updates']}  {time.time() - t0:.0f}s", flush=True)
    else:
        for step in range(1, sim.sim_lm_steps + 1):
            acc += sim.sim_ratio
            n = int(math.floor(acc + 1e-9))
            acc -= n
            for _ in range(n):
                g.run_graph_batch(g.scheduler.next_batch())
                if snaps is not None:
                    snaps.after_graph_step()
            t.run_text_batch(t.scheduler.next_batch())
            if step % sim.sim_log_every == 0 or step == sim.sim_lm_steps:
                lag = (sum(k * v for k, v in snaps.lags.items()) / max(1, sum(snaps.lags.values()))) if snaps else None
                print(f"[sim] lm_step {step}/{sim.sim_lm_steps}  gnn batches {counts['g_batches']} updates "
                      f"{counts['g_updates']}  lm updates {counts['t_updates']}  mean z_G lag "
                      f"{'-' if lag is None else f'{lag:.1f}'}  {time.time() - t0:.0f}s", flush=True)

    # --- artifacts where the real search writes them (run_sys.sh's post-run eval reads these) ---
    g.save_checkpoint(str(config.graph_ckpt_path))
    with (config.train_output_dir / "genotype.json").open("w", encoding="utf-8") as fh:   # never the input genotype file
        json.dump(g.export_genotype().to_dict(), fh, indent=2)
    t.save_checkpoint(str(config.text_ckpt_path))
    lag_hist = dict(sorted(snaps.lags.items())) if snaps else {}
    summary = {
        "mode": sim.sim_mode, "s": sim.sim_s, "stage": stage, "genotype": sim.sim_genotype, "r": sim.sim_ratio, "lm_steps": sim.sim_lm_steps,
        "seed": config.seed, "dataset": config.dataset, "seconds": time.time() - t0,
        "gnn_batches": counts["g_batches"], "gnn_updates": counts["g_updates"],
        "lm_batches": counts["t_batches"], "lm_updates": counts["t_updates"],
        "n_train": len(store.scheduler_train_nodes()),
        "zg_lag_hist": {str(k): v for k, v in lag_hist.items()},
        "zg_lag_mean": (sum(k * v for k, v in lag_hist.items()) / max(1, sum(lag_hist.values()))) if lag_hist else None,
        "pruning_masks": {k: v.tolist() for k, v in g.graph_tower.gnn.state_dict().items() if k.startswith("mask_")},
    }
    with (config.train_output_dir / "sim_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("[sim] done:", json.dumps({k: v for k, v in summary.items() if k not in ("zg_lag_hist", "pruning_masks")}))


if __name__ == "__main__":
    main()
