#!/usr/bin/env python3
"""Link-prediction evaluation for the DDP-trained GASSO checkpoints.

Designed for ckpts from `bin/ddp_train.py` (mp.spawn DDP training); this is a parallel
sibling of the heterogeneous-distributed script `bin/eval_link_pred_distributed.py`:
  - heterogeneous eval: loads from GraphTower + genotype.json + PubmedRuntimeStore
  - this script (DDP eval): loads from the LitClipGraph state_dict + the same
    datamodule, deriving the architecture genotype on the fly via alphas argmax

The ckpt dict looks like:
  {"model_state_dict": <LitClipGraph state>, "optimizer_state_dict": ..., "epoch": ..., "global_step": ...}

Expected run-dir layout (as produced by bin/ddp_train.py):
  <run-dir>/
    checkpoints/{best_model.pth, final_model.pth, checkpoint_epoch_*.pth}
    model_config_snapshot.yaml
    run_args.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from clip_graph.async_coexec.training_models import export_graph_genotype
from clip_graph.eval_neg_sampling import NEGATIVE_SAMPLING_CHOICES, sample_negative_edges
from clip_graph.gassolit import LitClipGraph
from clip_graph.gassomodels import ACT_LIST, NORM_LIST


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint(run_dir: Path, spec: str) -> Path:
    """Resolve --checkpoint into an absolute path.

    spec: "auto" | "best" | "final" | "epoch_N" | <abs path>
    """
    ckpt_dir = run_dir / "checkpoints"

    if spec.startswith("/") or spec.endswith(".pth"):
        # Absolute path or an explicit .pth file.
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
        raise FileNotFoundError(
            f"{ckpt_dir / 'best_model.pth'} / {run_dir / 'best_model.pth'}"
            " (this run may never have triggered a best-model save -- early-stop was not enabled)"
        )

    if spec == "final":
        for p in (ckpt_dir / "final_model.pth", run_dir / "final_model.pth"):
            if p.exists():
                return p
        raise FileNotFoundError(
            f"{ckpt_dir / 'final_model.pth'} / {run_dir / 'final_model.pth'} (training may not have finished normally)"
        )

    epoch_match = re.fullmatch(r"epoch_(\d+)", spec)
    if epoch_match:
        n = int(epoch_match.group(1))
        for p in (ckpt_dir / f"checkpoint_epoch_{n}.pth", run_dir / f"checkpoint_epoch_{n}.pth"):
            if p.exists():
                return p
        raise FileNotFoundError(ckpt_dir / f"checkpoint_epoch_{n}.pth")

    if spec == "auto":
        # Priority: best -> final -> latest checkpoint_epoch_N (each name is checked
        # in both checkpoints/ and the run-dir root).
        for name in ("best_model.pth", "final_model.pth"):
            for candidate in (ckpt_dir / name, run_dir / name):
                if candidate.exists():
                    return candidate
        epoch_pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
        epochs = []
        for f in ckpt_dir.iterdir():
            m = epoch_pattern.match(f.name)
            if m:
                epochs.append((int(m.group(1)), f))
        if not epochs:
            raise FileNotFoundError(f"No ckpt found under {ckpt_dir}")
        epochs.sort(reverse=True)
        return epochs[0][1]

    raise ValueError(f"Cannot parse --checkpoint={spec!r}")


# ---------------------------------------------------------------------------
# DataModule + split-induced subgraph
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
    cross-type pairs.

    This aligns with the TEG-Benchmark "predict whether a user will post a tweet" task
    definition -- negatives must also be user-tweet pairs, not tweet-tweet or user-user
    edges, which would be trivially impossible.
    """
    user_idx = is_user.nonzero(as_tuple=False).flatten().to(device)
    tweet_idx = is_tweet.nonzero(as_tuple=False).flatten().to(device)
    if user_idx.numel() == 0 or tweet_idx.numel() == 0:
        raise RuntimeError(
            f"Could not find user({user_idx.numel()}) or tweet({tweet_idx.numel()}) nodes; cannot do type-aware negative sampling"
        )

    pos_set = set()
    for s, t in pos_ei.t().tolist():
        pos_set.add((s, t))
        pos_set.add((t, s))

    g = torch.Generator(device=device).manual_seed(seed)
    sampled: list = []
    seen: set = set()

    for _ in range(20):
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
    """Reuse the same datamodule used during training to obtain split-induced graph_data.

    This matches the bin/ddp_train.py:_setup_data path exactly: override data_dir,
    setup(stage='fit', k_hop), then pull the already split-induced + drop_isolates
    + z-scored graph_data from <split>_dataset.dataset.
    """
    data_args = OmegaConf.to_container(snapshot.data.init_args, resolve=True)
    data_dir = str(data_dir_override or REPO_ROOT / "data" / dataset_name)
    data_args["data_dir"] = data_dir

    DataModuleCls = locate(snapshot.data.class_path)
    if DataModuleCls is None:
        raise ImportError(f"Could not locate datamodule class: {snapshot.data.class_path}")

    dm = DataModuleCls(**data_args)
    dm.prepare_data()
    k_hop = int(snapshot.model.init_args.gnn_params.num_layers)
    dm.setup(node=0, world_size=1, stage="fit", k_hop=k_hop)

    split_to_ds = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    split_ds = split_to_ds[split]
    if split_ds is None:
        raise RuntimeError(f"datamodule has no {split}_dataset")

    # BatchGraphTextDataset wraps a layer around it; graph_data lives on .dataset.
    graph_data = split_ds.dataset.graph_data
    vx = graph_data.x.float().to(device)
    vei = graph_data.edge_index.to(device)

    edge_attr = getattr(graph_data, "edge_attr", None)
    if edge_attr is None or edge_attr.numel() == 0:
        edge_attr = torch.zeros(vei.size(1), 1, dtype=vx.dtype)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    edge_attr = edge_attr.float().to(device)

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
    print(f"[debug] vx mean={vx.mean():.4f} std={vx.std():.4f} "
          f"min={vx.min():.4f} max={vx.max():.4f} shape={tuple(vx.shape)}")
    print(f"[debug] edge_attr nonzero ratio={(edge_attr != 0).float().mean().item():.4f} "
          f"shape={tuple(edge_attr.shape)}")
    return vx, vei, vnei, edge_attr, neg_info


# ---------------------------------------------------------------------------
# Model load + forward
# ---------------------------------------------------------------------------


def load_lit_model(snapshot, ckpt_path: Path, device: torch.device) -> LitClipGraph:
    model_args = OmegaConf.to_container(snapshot.model.init_args, resolve=True)
    lit_model = LitClipGraph(**model_args).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    # strict=False tolerates missing adaptive-pruning mask buffers (mask_op_*/mask_norm_*/mask_act_*):
    # these buffers were added later, so older ckpts (from before the pruning feature) don't have them.
    # In a freshly built model they default to all True (= no pruning), which is exactly what we want.
    # But any other missing/unexpected real weights must raise, not be silently ignored.
    result = lit_model.load_state_dict(state, strict=False)
    _MASK_PREFIXES = ("model.gnn.mask_op_", "model.gnn.mask_norm_", "model.gnn.mask_act_")
    bad_missing = [k for k in result.missing_keys if not k.startswith(_MASK_PREFIXES)]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch (not a mask buffer):\n"
            f"  missing   = {bad_missing}\n"
            f"  unexpected= {list(result.unexpected_keys)}"
        )
    if result.missing_keys:
        print(f"[ckpt] Tolerating {len(result.missing_keys)} missing pruning mask buffers (old ckpt, defaults to no pruning)")
    lit_model.eval()
    print(f"[ckpt] loaded {ckpt_path.name} (epoch={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')})")
    return lit_model


def encode_split_graph(
    gnn,
    gnn_proj,
    vx: torch.Tensor,
    vei: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    is_search: bool,
) -> torch.Tensor:
    layers = gnn.num_layers
    gnn.adjs = (
        vei.unsqueeze(0).repeat(layers, 1, 1),
        edge_attr.unsqueeze(0).repeat(layers, 1, 1),
    )
    gnn.step = 1

    with torch.no_grad():
        out = gnn(vx, vei, edge_attr, is_search=is_search)
        h = out["output"] if out["output"] is not None else out["last_hidden_state"]
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


def print_alpha_breakdown(gnn) -> None:
    print("\n[alpha] per-layer softmax over (op / norm / act):")
    ops = list(gnn.ops)
    for layer_idx, (op_a, norm_a, act_a) in enumerate(
        zip(gnn.alphas_normal, gnn.alphas_norm, gnn.alphas_act)
    ):
        op_w = torch.softmax(op_a.detach().cpu(), dim=-1)
        norm_w = torch.softmax(norm_a.detach().cpu(), dim=-1)
        act_w = torch.softmax(act_a.detach().cpu(), dim=-1)
        op_str = ", ".join(f"{n}:{w:.3f}" for n, w in zip(ops, op_w))
        norm_str = ", ".join(f"{n}:{w:.3f}" for n, w in zip(NORM_LIST, norm_w))
        act_str = ", ".join(f"{n}:{w:.3f}" for n, w in zip(ACT_LIST, act_w))
        print(f"  layer {layer_idx}:")
        print(f"    op   -> {op_str}  (argmax={ops[int(op_w.argmax())]})")
        print(f"    norm -> {norm_str}  (argmax={NORM_LIST[int(norm_w.argmax())]})")
        print(f"    act  -> {act_str}  (argmax={ACT_LIST[int(act_w.argmax())]})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="DDP training artifact directory, e.g. model/runpytorch/gasso_arxiv_ddp_20260507_224955")
    parser.add_argument("--checkpoint", type=str, default="auto",
                        help="best | final | epoch_N | <abs path>. "
                             "auto: best->final->latest checkpoint_epoch_*.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2969591811,
                        help="Random seed for negative sampling (does not affect the split -- the split seed comes from the snapshot YAML)")
    parser.add_argument("--dataset-name", default=None,
                        help="Label for the CSV 'dataset' column; when omitted, reads the 'dataset' field from run_args.json")
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
        raise ValueError("Could not determine dataset_name (run_args.json has no 'dataset' field and --dataset-name was not passed)")

    ckpt_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    print(f"[ckpt] resolved -> {ckpt_path}")

    lit_model = load_lit_model(snapshot, ckpt_path, device)
    gnn = lit_model.model.gnn
    gnn_proj = lit_model.model.gnn_proj

    # Derive the genotype on the fly and persist it.
    genotype = export_graph_genotype(gnn)
    geno_dump = args.run_dir / "genotype_derived.json"
    with open(geno_dump, "w", encoding="utf-8") as f:
        json.dump(genotype.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[genotype] derived genotype written -> {geno_dump}")
    print_alpha_breakdown(gnn)

    vx, vei, vnei, edge_attr, neg_info = build_eval_graph(
        snapshot=snapshot,
        data_dir_override=args.data_dir,
        dataset_name=dataset_name,
        split=args.split,
        seed=args.seed,
        device=device,
        negative_sampling=args.negative_sampling,
    )

    rows = []
    ckpt_tag = ckpt_path.stem  # e.g. best_model / final_model / checkpoint_epoch_3
    for is_search, base_label in [(True, "ddp_GASSO_search"), (False, "ddp_GASSO_greedy")]:
        label = f"{base_label}@{ckpt_tag}{neg_info.label_suffix}"
        z = encode_split_graph(gnn, gnn_proj, vx, vei, edge_attr, is_search=is_search)
        metrics = link_pred_metrics(z, vei, vnei)
        print(f"[score] {label:<40} auc={metrics['auc']:.4f}  ap={metrics['ap']:.4f}  recon={metrics['recon']:.4f}")
        rows.append({"dataset": dataset_name, "gnn_model": label, **metrics})

    df = pd.DataFrame(rows).set_index(["dataset", "gnn_model"]).sort_index()
    out_path = args.run_dir / "link-prediction-eval-gnn.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path).set_index(["dataset", "gnn_model"])
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(out_path)
    print(f"\n[out] saved -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
