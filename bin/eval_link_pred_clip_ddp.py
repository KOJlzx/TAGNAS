#!/usr/bin/env python3
"""Link-prediction evaluation for bin/congrat_train.py training artifacts (non-GASSO).

This script is a sibling of ``bin/eval_link_pred_ddp.py``:
  - eval_link_pred_ddp.py     -> GASSO model (bin/ddp_train.py / bin/zero_offload_train.py),
                                  uses ``clip_graph.gassolit.LitClipGraph`` + alphas
  - this script               -> non-GASSO (bin/congrat_train.py),
                                  uses ``clip_graph.lit.LitClipGraph`` + a fixed GNN

The ckpt dict looks like:
  {"model_state_dict": <LitClipGraph state>, "epoch": ..., "global_step": ...}

Output: ``<run-dir>/link-prediction-eval-clip.csv``, with one appended row per ckpt.
The gnn_model column looks like ``ddp_clip_GAT@best_model``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pydoc import locate

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as mt
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from omegaconf import OmegaConf

from clip_graph.eval_neg_sampling import NEGATIVE_SAMPLING_CHOICES, sample_negative_edges
from clip_graph.lit import LitClipGraph


# ---------------------------------------------------------------------------
# Checkpoint resolution (identical semantics to the GASSO version: supports
# best/final living either at the run-dir root or under the checkpoints/ subdir)
# ---------------------------------------------------------------------------


def resolve_checkpoint(run_dir: Path, spec: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"

    if spec.startswith("/") or spec.endswith(".pth"):
        p = Path(spec)
        if not p.is_absolute():
            p = run_dir / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    if spec == "best":
        for p in (ckpt_dir / "best_model.pth", run_dir / "best_model.pth"):
            if p.exists():
                return p
        raise FileNotFoundError(f"{ckpt_dir / 'best_model.pth'} / {run_dir / 'best_model.pth'}")

    if spec == "final":
        for p in (ckpt_dir / "final_model.pth", run_dir / "final_model.pth"):
            if p.exists():
                return p
        raise FileNotFoundError(f"{ckpt_dir / 'final_model.pth'} / {run_dir / 'final_model.pth'}")

    if spec.startswith("epoch_"):
        n = spec.split("_", 1)[1]
        for p in (ckpt_dir / f"checkpoint_epoch_{n}.pth", run_dir / f"checkpoint_epoch_{n}.pth"):
            if p.exists():
                return p
        raise FileNotFoundError(f"checkpoint_epoch_{n}.pth")

    if spec == "auto":
        for p in (ckpt_dir / "best_model.pth", run_dir / "best_model.pth",
                  ckpt_dir / "final_model.pth", run_dir / "final_model.pth"):
            if p.exists():
                return p
        cand = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if cand:
            return cand[-1]
        raise FileNotFoundError(f"No ckpt found under {run_dir}")

    raise ValueError(f"Cannot parse --checkpoint={spec!r}")


# ---------------------------------------------------------------------------
# DataModule + split-induced subgraph (mostly identical to the GASSO version,
# minus one k_hop argument -- non-GASSO datamodules may not accept k_hop, so
# we make the call adaptively)
# ---------------------------------------------------------------------------


def _twitter_bipartite_negative_sampling(
    pos_ei: torch.Tensor,
    is_user: torch.Tensor,
    is_tweet: torch.Tensor,
    num_neg: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Twitter-specific type-aware negative sampling: samples only within (user, tweet)
    cross-type pairs, automatically avoiding positive edges.

    Unlike the random full-graph negative sampling in ``pyg.utils.negative_sampling``,
    this aligns with the TEG-Benchmark "predict whether a user will post a tweet" task
    definition -- negatives must also be user-tweet pairs, not tweet-tweet or user-user
    edges, which would be trivially impossible.
    """
    user_idx = is_user.nonzero(as_tuple=False).flatten().to(device)
    tweet_idx = is_tweet.nonzero(as_tuple=False).flatten().to(device)
    if user_idx.numel() == 0 or tweet_idx.numel() == 0:
        raise RuntimeError(
            f"Could not find user({user_idx.numel()}) or tweet({tweet_idx.numel()}) nodes; cannot do type-aware negative sampling"
        )

    # Add both directions of each positive edge to the blacklist (undirected).
    pos_set = set()
    for s, t in pos_ei.t().tolist():
        pos_set.add((s, t))
        pos_set.add((t, s))

    g = torch.Generator(device=device).manual_seed(seed)
    sampled: list = []
    seen: set = set()

    for _ in range(20):  # At most 20 rounds; normally 1-2 rounds suffice.
        n_try = max(num_neg * 2, num_neg + 100)
        s = user_idx[torch.randint(len(user_idx), (n_try,), generator=g, device=device)]
        t = tweet_idx[torch.randint(len(tweet_idx), (n_try,), generator=g, device=device)]
        for ss, tt in zip(s.tolist(), t.tolist()):
            key = (ss, tt)
            if key in pos_set or key in seen:
                continue
            seen.add(key)
            sampled.append([ss, tt])
            if len(sampled) >= num_neg:
                break
        if len(sampled) >= num_neg:
            break

    if len(sampled) < num_neg:
        print(f"[warn] Only sampled {len(sampled)}/{num_neg} type-aware negatives (the graph may be very dense)")

    return torch.tensor(sampled, dtype=torch.long, device=device).t()


def build_eval_graph(
    snapshot,
    data_dir_override: Path | None,
    dataset_name: str,
    split: str,
    seed: int,
    device: torch.device,
    negative_sampling: str,
):
    data_args = OmegaConf.to_container(snapshot.data.init_args, resolve=True)
    data_dir = str(data_dir_override or REPO_ROOT / "data" / dataset_name)
    data_args["data_dir"] = data_dir

    DataModuleCls = locate(snapshot.data.class_path)
    if DataModuleCls is None:
        raise ImportError(f"Could not locate datamodule class: {snapshot.data.class_path}")

    dm = DataModuleCls(**data_args)
    dm.prepare_data()

    # Non-GASSO datamodules may also accept k_hop; try passing it, and fall back if not.
    k_hop = int(snapshot.model.init_args.gnn_params.num_layers)
    try:
        dm.setup(node=0, world_size=1, stage="fit", k_hop=k_hop)
    except TypeError:
        dm.setup(node=0, world_size=1, stage="fit")

    split_to_ds = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    split_ds = split_to_ds[split]
    if split_ds is None:
        raise RuntimeError(f"datamodule has no {split}_dataset")

    graph_data = split_ds.dataset.graph_data
    vx = graph_data.x.float().to(device)
    vei = graph_data.edge_index.to(device)

    if vei.size(1) == 0:
        raise RuntimeError(f"split={split} subgraph has no edges; link-prediction is impossible")

    torch.manual_seed(seed)

    vnei, neg_info = sample_negative_edges(
        pos_edge_index=vei,
        num_nodes=vx.size(0),
        graph_node_ids=getattr(graph_data, "node_ids", None),
        dataset_name=dataset_name,
        data_dir=data_dir,
        mode=negative_sampling,
        seed=seed,
        device=device,
    )

    print(f"[data] split={split}  num_nodes={vx.size(0)}  pos_edges={vei.size(1)}  neg_edges={vnei.size(1)}")
    print(
        f"[data] negative_sampling requested={neg_info.requested_mode} actual={neg_info.actual_mode} "
        f"suffix={neg_info.label_suffix or '<none>'} reason={neg_info.reason}"
    )
    return vx, vei, vnei, neg_info


# ---------------------------------------------------------------------------
# Model load + forward
# ---------------------------------------------------------------------------


def load_lit_model(snapshot, ckpt_path: Path, device: torch.device) -> LitClipGraph:
    model_args = OmegaConf.to_container(snapshot.model.init_args, resolve=True)
    lit_model = LitClipGraph(**model_args).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    lit_model.load_state_dict(state, strict=False)  # Training may also have wrapped it in a DDP wrapper; disabling strict is more robust
    lit_model.eval()
    print(f"[ckpt] loaded {ckpt_path.name} (epoch={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')})")
    return lit_model


def encode_split_graph(gnn, gnn_proj, vx: torch.Tensor, vei: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = gnn(vx, vei)
        # GNNMod.forward returns a dict; retain the same "output first" logic as the GASSO version
        if isinstance(out, dict):
            h = out.get("output")
            if h is None:
                h = out.get("last_hidden_state")
        else:
            h = out
        h = gnn_proj(h)
        h = F.normalize(h, p=2, dim=1)
    return h


def link_pred_metrics(z: torch.Tensor, pos_ei: torch.Tensor, neg_ei: torch.Tensor, eps: float = 1e-15) -> dict:
    pos_y = z.new_ones(pos_ei.size(1))
    neg_y = z.new_zeros(neg_ei.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).long().detach().cpu().numpy()

    decoder = pyg.nn.models.autoencoder.InnerProductDecoder()
    pos_dec = decoder(z, pos_ei, sigmoid=True)
    neg_dec = decoder(z, neg_ei, sigmoid=True)
    pred = torch.cat([pos_dec, neg_dec], dim=0).detach().cpu().numpy()

    return {
        "auc": mt.roc_auc_score(y, pred),
        "ap": mt.average_precision_score(y, pred),
        "recon": (
            -torch.log(pos_dec + eps).mean() + -torch.log(1 - neg_dec + eps).mean()
        ).item(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training artifact directory produced by bin/congrat_train.py")
    parser.add_argument("--checkpoint", type=str, default="auto",
                        help="best | final | epoch_N | <abs path> | auto")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2969591811)
    parser.add_argument("--dataset-name", default=None,
                        help="CSV 'dataset' column; when omitted, reads 'dataset' from run_args.json")
    parser.add_argument("--gnn-type", default=None,
                        help="GNN name to display in the CSV gnn_model prefix (e.g. GAT/GCN); "
                             "when omitted, reads 'gnn_type' from run_args.json, falling back to 'unknown'")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override data.init_args.data_dir from the snapshot YAML. "
                             "Defaults to <repo>/data/<dataset_name>.")
    parser.add_argument("--negative-sampling", choices=NEGATIVE_SAMPLING_CHOICES, default="auto",
                        help="Negative-sampling convention for link prediction. auto automatically uses type-aware sampling for bipartite graphs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    device = torch.device(args.device)

    snapshot_path = args.run_dir / "model_config_snapshot.yaml"
    run_args_path = args.run_dir / "run_args.json"
    for required in (snapshot_path, run_args_path):
        if not required.exists():
            raise FileNotFoundError(required)
    snapshot = OmegaConf.load(str(snapshot_path))
    with open(run_args_path) as f:
        run_args = json.load(f)

    dataset_name = args.dataset_name or run_args.get("dataset")
    if not dataset_name:
        raise ValueError("Could not determine dataset_name (run_args.json has no 'dataset' and --dataset-name was not passed)")

    gnn_type = args.gnn_type or run_args.get("gnn_type") or "unknown"

    ckpt_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    print(f"[ckpt] resolved -> {ckpt_path}")

    lit_model = load_lit_model(snapshot, ckpt_path, device)
    gnn = lit_model.model.gnn
    gnn_proj = lit_model.model.gnn_proj

    vx, vei, vnei, neg_info = build_eval_graph(
        snapshot=snapshot,
        data_dir_override=args.data_dir,
        dataset_name=dataset_name,
        split=args.split,
        seed=args.seed,
        device=device,
        negative_sampling=args.negative_sampling,
    )

    ckpt_tag = ckpt_path.stem  # e.g. best_model / final_model / checkpoint_epoch_3
    label = f"ddp_clip_{gnn_type}@{ckpt_tag}{neg_info.label_suffix}"
    z = encode_split_graph(gnn, gnn_proj, vx, vei)
    metrics = link_pred_metrics(z, vei, vnei)
    print(f"[score] {label:<40} auc={metrics['auc']:.4f}  ap={metrics['ap']:.4f}  recon={metrics['recon']:.4f}")
    rows = [{"dataset": dataset_name, "gnn_model": label, **metrics}]

    df = pd.DataFrame(rows).set_index(["dataset", "gnn_model"]).sort_index()
    out_path = args.run_dir / "link-prediction-eval-clip.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path).set_index(["dataset", "gnn_model"])
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(out_path)
    print(f"\n[out] saved -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
