#!/usr/bin/env python3
"""UniTG-style baseline (B5) launcher: deep RevGNN flow parallelism + interleaved LM fine-tuning.

One process per GPU; each process hosts one pipeline stage of a deep reversible
GNN and one data-parallel replica of the text encoder that is fine-tuned in the
pipeline bubbles. By default both are trained with the same graph-text
contrastive objective as B1/B2/B3/TAGNAS, with the LM and GNN embedding tables
exchanged at every Δt sync (``--objective link`` selects the original UniTG
task-loss port, reference only). See ``src/clip_graph/unitg/``.

Artifacts (``model/runpytorch/unitg_<dataset>_<lm>_<timestamp>/``) follow the
other baselines: ``run_args.json``, ``model_config_snapshot.yaml``,
``metrics.csv``, ``val_timeline.csv``, ``events.jsonl``, ``training_time.json``,
``manifest.json``, ``best_model.pth`` / ``final_model.pth`` (+ ``features_*.pt``),
and after training ``link-prediction-eval-unitg.csv`` produced by
``bin/eval_link_pred_unitg.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from clip_graph.unitg.config import UniTGConfig  # noqa: E402
from clip_graph.unitg.runner import worker_main  # noqa: E402


def _lm_tag(config_path: str) -> str:
    try:
        snap = OmegaConf.load(config_path)
        name = str(snap.model.init_args.lm_name)
    except Exception:  # noqa: BLE001
        name = "lm"
    return name.split("/")[-1]


def make_run_dir(config: UniTGConfig) -> str:
    root = os.path.abspath(os.path.expanduser(config.output_root))
    os.makedirs(root, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"unitg_{config.dataset}_{_lm_tag(config.config_path)}_{ts}"
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_run_metadata(config: UniTGConfig) -> None:
    with open(os.path.join(config.run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump({**config.to_dict(), "system": "unitg", "gnn_type": "revgnn"}, f, indent=2)
    shutil.copyfile(config.config_path, os.path.join(config.run_dir, "model_config_snapshot.yaml"))


def post_eval(config: UniTGConfig) -> None:
    script = REPO_ROOT / "bin" / "eval_link_pred_unitg.py"
    status_path = Path(config.run_dir) / "post_eval_status.json"
    status = {"eval_ok": False, "split": config.post_eval_split, "checkpoints": {},
              "paper_v3_attempt": config.paper_v3_attempt, "paper_v3_cell": config.paper_v3_cell}

    def save_status() -> None:
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    save_status()
    for ckpt in ("best", "final"):
        record = {"status": "pending", "attempts": [], "returncode": None}
        status["checkpoints"][ckpt] = record
        if not os.path.isfile(os.path.join(config.run_dir, f"{ckpt}_model.pth")):
            record["status"] = "missing_checkpoint"
            save_status()
            continue
        cmd = [sys.executable, str(script), "--run-dir", config.run_dir, "--checkpoint", ckpt,
               "--split", config.post_eval_split, "--device", "cuda" if torch.cuda.is_available() else "cpu",
               "--negative-sampling", config.link_neg_sampling, "--dataset-name", config.dataset]
        print(f"\n[post-eval] {' '.join(cmd)}", flush=True)
        rc = 1
        for attempt in range(4):
            record["status"] = "running"
            save_status()
            rc = subprocess.run(cmd).returncode
            record["attempts"].append({"attempt": attempt + 1, "returncode": rc})
            record["returncode"] = rc
            save_status()
            if rc == 0:
                break
            # Exclusive_Process GPUs are released a few seconds after the workers exit.
            if attempt < 3:
                print(f"[post-eval] checkpoint={ckpt} attempt {attempt + 1} failed (exit {rc}); retrying in 15s",
                      flush=True)
                time.sleep(15)
        record["status"] = "passed" if rc == 0 else "failed"
        save_status()
    status["eval_ok"] = all(record["status"] == "passed" for record in status["checkpoints"].values())
    save_status()
    if not status["eval_ok"]:
        raise RuntimeError(f"post-eval failed; see {status_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default="pubmed", help="comma-separated dataset names under data/")
    p.add_argument("--config-path", type=str, default="configs/clip-graph/inductive-causal/pubmed/base.yaml")
    p.add_argument("--output-root", type=str, default="model/runpytorch")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--world-size", type=int, default=None, help="pipeline stages = GPUs (default: visible GPU count)")
    p.add_argument("--master-port", type=int, default=30500)
    p.add_argument("--seed", type=int, default=42)
    # deep GNN
    p.add_argument("--num-layers", type=int, default=112)
    p.add_argument("--hidden-channels", type=int, default=224)
    p.add_argument("--num-groups", type=int, default=2)
    p.add_argument("--gnn-lr", type=float, default=1e-3)
    p.add_argument("--gnn-dropout", type=float, default=0.2)
    p.add_argument("--gnn-weight-decay", type=float, default=0.0)
    p.add_argument("--microbatches", type=int, default=3)
    p.add_argument("--max-nodes-per-microbatch", type=int, default=100_000)
    p.add_argument("--halo-hops", type=int, default=0,
                   help="> 0: each microbatch = its METIS cluster (seeds) + their k-hop neighbourhood, "
                        "TAGNAS's subgraph extraction; 0 = UniTG's induced cluster subgraph")
    p.add_argument("--max-pos-edges-per-microbatch", type=int, default=200_000)
    # objective
    p.add_argument("--objective", choices=["contrastive", "link"], default="contrastive",
                   help="contrastive = the graph-text InfoNCE every other system trains (default, comparable AUC); "
                        "link = UniTG-style BCE on edges (reference only, exploits the eval protocol)")
    p.add_argument("--lm-objective", choices=["link", "contrastive"], default="link",
                   help="link = LM fine-tuned on its own text-pair link loss (anchored, one-directional LM->GNN flow; default); "
                        "contrastive = LM aligned back to the GNN table (bidirectional; collapses in practice)")
    p.add_argument("--contrastive-nodes", type=int, default=1024, help="nodes per microbatch in the GNN-side InfoNCE")
    p.add_argument("--tau-init", type=float, default=3.5)
    p.add_argument("--max-tau", type=float, default=4.605)
    p.add_argument("--no-lm-warmup-hold", action="store_true", help="let the LM train against the untrained GNN's table from the start")
    # LM
    p.add_argument("--lm-lr", type=float, default=None)
    p.add_argument("--lm-adam-eps", type=float, default=None)
    p.add_argument("--lm-weight-decay", type=float, default=None)
    p.add_argument("--lm-batch-size", type=int, default=64, help="contrastive: nodes per LM step; link: edges per step")
    p.add_argument("--lm-max-length", type=int, default=128)
    p.add_argument("--text-per-node", choices=["joined", "tagnas"], default="joined",
                   help="joined = all of a node's text rows (UniTG artifact); tagnas = exactly the "
                        "string TAGNAS's text worker feeds for each node")
    p.add_argument("--tagnas-text-config", default="configs/clip-graph/inductive-causal/pubmed/gassobase.yaml",
                   help="TAGNAS YAML whose data path defines the text in --text-per-node tagnas")
    p.add_argument("--lm-train-node-fraction", type=float, default=None,
                   help="draw LM contrastive anchors from a seeded fraction of the train text nodes "
                        "(TAGNAS's train_node_fraction rule); default = all")
    p.add_argument("--lm-probe-worst-case", action="store_true",
                   help="memory probes: force the pool's longest text into the first LM batch")
    p.add_argument("--lm-dynamic-padding", action="store_true",
                   help="trim each LM batch to its longest real row (pad-to-longest, as B1-B4/TAGNAS)")
    p.add_argument("--lm-num-exits", type=int, default=4)
    p.add_argument("--no-lazy-tuner", action="store_true", help="disable centrality-guided early exit")
    p.add_argument("--encode-exit", choices=["full", "group"], default="full",
                   help="exit used for the synchronised table: full depth for all nodes (consistent contrastive target, default) "
                        "or each node's centrality exit (paper-literal; use with --objective link)")
    p.add_argument("--no-lm-autocast", action="store_true")
    p.add_argument("--lm-encode-batch", type=int, default=256)
    p.add_argument("--no-bubble-interleave", action="store_true", help="let the LM run without pausing during GNN compute")
    # collaborator
    p.add_argument("--sync-schedule", type=str, default="0.3,0.2,0.1,0.02")
    p.add_argument("--sync-tighten-epochs", type=int, default=20)
    p.add_argument("--sync-ema", type=float, default=0.8)
    p.add_argument("--sync-unit", choices=["auto", "seconds", "epochs"], default="auto")
    p.add_argument("--no-initial-sync", action="store_true")
    # budget / eval
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--max-runtime-seconds", type=float, default=3000.0, help="<=0 disables the cap")
    p.add_argument("--val-eval-interval-seconds", type=float, default=60.0)
    p.add_argument("--val-eval-min-epochs", type=int, default=5)
    p.add_argument("--profile", action="store_true",
                   help="record per-iteration phase timings for experiments/04_time_breakdown.sh")
    p.add_argument("--profile-out", type=str, default=None,
                   help="legacy breakdown JSON path; telemetry/*.jsonl is written next to its directory")
    p.add_argument("--no-eval-at-start", action="store_true", help="skip scoring the untrained model at epoch 0")
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--min-lm-texts-before-stop", type=int, default=0,
                   help="cumulative trained-text budget: -1 = this run's training text pool size, 0 = off")
    p.add_argument("--val-eval-on-test", action="store_true")
    p.add_argument("--negative-sampling", choices=["default", "type_aware", "auto"], default="auto")
    p.add_argument("--no-post-eval", action="store_true")
    p.add_argument("--post-eval-split", choices=["train", "val", "test"], default="test")
    # parity-only knobs
    p.add_argument("--text-subsample-fraction", type=float, default=None)
    p.add_argument("--early-stop-metric", choices=["auc", "loss"], default="auc")
    p.add_argument("--log-every-epochs", type=int, default=10)
    p.add_argument("--save-features", action="store_true", help="also save the LM table next to best/final")
    return p.parse_args()


def build_config(args: argparse.Namespace, dataset: str) -> UniTGConfig:
    world_size = args.world_size
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        raise RuntimeError("no GPU visible; set --world-size or CUDA_VISIBLE_DEVICES")
    max_rt = args.max_runtime_seconds if args.max_runtime_seconds and args.max_runtime_seconds > 0 else None
    return UniTGConfig(
        dataset=dataset,
        config_path=str(Path(args.config_path).resolve()),
        output_root=args.output_root,
        run_name=args.run_name,
        world_size=int(world_size),
        master_port=int(args.master_port),
        seed=int(args.seed),
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        num_groups=args.num_groups,
        gnn_lr=args.gnn_lr,
        gnn_dropout=args.gnn_dropout,
        gnn_weight_decay=args.gnn_weight_decay,
        microbatches=args.microbatches,
        max_nodes_per_microbatch=args.max_nodes_per_microbatch,
        halo_hops=args.halo_hops,
        max_pos_edges_per_microbatch=args.max_pos_edges_per_microbatch,
        objective=args.objective,
        lm_objective=args.lm_objective,
        contrastive_nodes=args.contrastive_nodes,
        tau_init=args.tau_init,
        max_tau=args.max_tau,
        lm_warmup_hold=not args.no_lm_warmup_hold,
        lm_lr=args.lm_lr,
        lm_adam_eps=args.lm_adam_eps,
        lm_weight_decay=args.lm_weight_decay,
        lm_batch_size=args.lm_batch_size,
        lm_max_length=args.lm_max_length,
        text_per_node=args.text_per_node,
        tagnas_text_config=args.tagnas_text_config,
        lm_dynamic_padding=args.lm_dynamic_padding,
        lm_train_node_fraction=args.lm_train_node_fraction,
        lm_probe_worst_case=args.lm_probe_worst_case,
        lm_num_exits=args.lm_num_exits,
        lazy_tuner=not args.no_lazy_tuner,
        encode_exit=args.encode_exit,
        lm_autocast=not args.no_lm_autocast,
        lm_encode_batch=args.lm_encode_batch,
        bubble_interleave=not args.no_bubble_interleave,
        sync_schedule=tuple(float(x) for x in args.sync_schedule.split(",") if x.strip()),
        sync_tighten_epochs=args.sync_tighten_epochs,
        sync_ema=args.sync_ema,
        sync_unit=args.sync_unit,
        initial_sync=not args.no_initial_sync,
        num_epochs=args.num_epochs,
        max_runtime_seconds=max_rt,
        val_eval_interval_seconds=args.val_eval_interval_seconds,
        val_eval_min_epochs=args.val_eval_min_epochs,
        eval_at_start=not args.no_eval_at_start,
        profile=args.profile,
        profile_out=args.profile_out,
        early_stopping_patience=args.early_stopping_patience,
        min_lm_texts_before_stop=args.min_lm_texts_before_stop,
        val_eval_on_test=args.val_eval_on_test,
        link_neg_sampling=args.negative_sampling,
        post_eval=not args.no_post_eval,
        post_eval_split=args.post_eval_split,
        text_subsample_fraction=args.text_subsample_fraction,
        early_stop_metric=args.early_stop_metric,
        log_every_epochs=args.log_every_epochs,
        save_features=args.save_features,
    )


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset in datasets:
        config = build_config(args, dataset)
        config.run_dir = make_run_dir(config)
        write_run_metadata(config)
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = str(config.master_port)
        print("=" * 60)
        print(f"[launcher] UniTG baseline  dataset={dataset}  stages/GPUs={config.world_size}  run_dir={config.run_dir}")
        print(f"[launcher] objective={config.objective} lm_objective={config.lm_objective}  GNN: RevGNN L={config.num_layers} H={config.hidden_channels} "
              f"groups={config.num_groups} M={config.microbatches}  LM: {config.config_path} exits={config.lm_num_exits} "
              f"lazy={config.lazy_tuner} batch={config.lm_batch_size}")
        print(f"[launcher] budget: epochs={config.num_epochs} max_runtime={config.max_runtime_seconds}s "
              f"sync={config.sync_schedule} tighten={config.sync_tighten_epochs} initial_sync={config.initial_sync}")
        print("=" * 60, flush=True)
        mp.spawn(worker_main, args=(config,), nprocs=config.world_size, join=True)
        if config.post_eval:
            post_eval(config)
        print(f"[launcher] done: {config.run_dir}", flush=True)


if __name__ == "__main__":
    main()
