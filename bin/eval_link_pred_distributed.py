#!/usr/bin/env python3
"""Link-prediction evaluation for the async-distributed search/retrain runs.

Loads a `GraphTower` checkpoint produced by `bin/async_coexec_train.py`, runs
the GNN encoder on the PubMed test graph in both `is_search=True` (soft alpha)
and `is_search=False` (greedy/argmax) modes, and writes AUC / AP / recon to
`<run-dir>/link-prediction-eval-gasso.csv`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

from pydoc import locate

import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as mt
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from omegaconf import OmegaConf

from clip_graph.async_coexec.training_models import GraphGenotype, GraphTower
from clip_graph.eval_neg_sampling import NEGATIVE_SAMPLING_CHOICES, sample_negative_edges
from clip_graph.gassomodels import ACT_LIST, NORM_LIST, GassoSpace


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
    data_args: dict,
    class_path: str,
    num_layers: int,
    device: torch.device,
    seed: int,
    split: str,
    dataset_name: str | None = None,
    negative_sampling: str = "auto",
):
    """Exactly matches `eval_link_pred_ddp.py:build_eval_graph`: uses the same
    datamodule as training (PubmedGraphTextDataModule from gassobase.yaml),
    setup(stage='fit', k_hop=num_layers), and reads
    <split>_dataset.dataset.graph_data (already split-induced + drop_isolates + z-scored).

    This puts TAGNAS and B1/B2/B3 on the same footing so the AUC table is comparable,
    and it is consistent with historical evaluation conventions.
    (Previously using PubmedRuntimeStore's strict split_edge_index convention
    systematically deflated AUC.)
    """
    DataModuleCls = locate(class_path)
    if DataModuleCls is None:
        raise ImportError(f"Could not locate datamodule class: {class_path}")

    dm = DataModuleCls(**data_args)
    dm.prepare_data()
    dm.setup(node=0, world_size=1, stage="fit", k_hop=num_layers)

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

    data_dir = str(data_args.get("data_dir", ""))
    vnei, neg_info = sample_negative_edges(
        pos_edge_index=vei,
        num_nodes=vx.size(0),
        graph_node_ids=getattr(graph_data, "node_ids", None),
        dataset_name=dataset_name or "",
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


def filter_gnn_params(gnn_params: dict) -> dict:
    valid_keys = set(inspect.signature(GassoSpace.__init__).parameters) - {"self"}
    return {k: v for k, v in gnn_params.items() if k in valid_keys}


def load_graph_tower(
    config_path: Path,
    ckpt_path: Path,
    genotype_path: Path,
    device: torch.device,
) -> tuple[GraphTower, dict]:
    conf = OmegaConf.load(str(config_path))
    raw_model_args = OmegaConf.to_container(conf.model.init_args, resolve=True)
    model_args = dict(raw_model_args)
    model_args["gnn_params"] = filter_gnn_params(dict(model_args.get("gnn_params") or {}))

    with open(genotype_path) as f:
        genotype = GraphGenotype.from_dict(json.load(f))

    # The genotype holds the num_layers actually persisted during training, and
    # authoritatively overrides the YAML config -- otherwise, if training changed
    # the layer count via NUM_LAYERS=1, eval using the original YAML would build an
    # N-layer GNN inconsistent with genotype.layers, and apply_graph_genotype would
    # raise a ValueError.
    genotype_num_layers = len(genotype.layers)
    yaml_num_layers = int(model_args["gnn_params"].get("num_layers", genotype_num_layers))
    if genotype_num_layers != yaml_num_layers:
        print(f"[override] gnn_params.num_layers: {yaml_num_layers} (YAML) -> "
              f"{genotype_num_layers} (genotype.layers)")
        model_args["gnn_params"]["num_layers"] = genotype_num_layers
        # Sync a copy into raw_model_args too, since build_eval_graph uses it
        # (even though it only reads in_channels, keeping it consistent is safer).
        raw_model_args["gnn_params"] = dict(raw_model_args.get("gnn_params") or {})
        raw_model_args["gnn_params"]["num_layers"] = genotype_num_layers

    tower = GraphTower(model_args, genotype=genotype).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    tower.load_state_dict(state, strict=True)
    tower.eval()
    # Keys like in_channels were stripped by filter_gnn_params but build_eval_graph
    # still needs them, so return the full version.
    return tower, raw_model_args


def encode_full_graph(
    tower: GraphTower,
    vx: torch.Tensor,
    vei: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    is_search: bool,
) -> torch.Tensor:
    layers = tower.gnn.num_layers
    tower.gnn.adjs = (
        vei.unsqueeze(0).repeat(layers, 1, 1),
        edge_attr.unsqueeze(0).repeat(layers, 1, 1),
    )
    tower.gnn.step = 1

    with torch.no_grad():
        out = tower.gnn(vx, vei, edge_attr, is_search=is_search)
        h = out["output"] if out["output"] is not None else out["last_hidden_state"]
        h = tower.proj(h)
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


def print_alpha_breakdown(tower: GraphTower) -> None:
    print("\n[alpha] per-layer softmax over (op / norm / act):")
    ops = list(tower.gnn.ops)
    for layer_idx, (op_a, norm_a, act_a) in enumerate(
        zip(tower.gnn.alphas_normal, tower.gnn.alphas_norm, tower.gnn.alphas_act)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["search", "retrain"], default="search")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "configs/clip-graph/inductive-causal/pubmed/gassobase.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2969591811)
    parser.add_argument("--dataset-name", default="pubmed",
                        help="Label used in the output CSV's 'dataset' column.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override the data.init_args.data_dir from the YAML config.")
    parser.add_argument("--use-best", action="store_true",
                        help="Load the best val-AUC checkpoint (*_graph_best.pt + genotype_best.json); "
                             "loads the final checkpoint by default.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split to evaluate on. For diagnostics: with val, this should match the val_eval AUC seen during training.")
    parser.add_argument("--split-proportions", type=str, default="0.7,0.1,0.2",
                        help="train/val/test proportions, comma-separated. Must match the training cli_args.json.")
    parser.add_argument("--offline-models-only", action="store_true",
                        help="Only allow loading HF models from the local cache (passed through to PubmedRuntimeStore).")
    parser.add_argument("--negative-sampling", choices=NEGATIVE_SAMPLING_CHOICES, default="auto",
                        help="Negative-sampling convention for link prediction. auto automatically uses type-aware sampling for bipartite graphs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    device = torch.device(args.device)

    phase_dir = args.run_dir / args.phase
    suffix = "_best" if args.use_best else ""
    ckpt_path = phase_dir / f"{args.phase}_graph{suffix}.pt"
    genotype_path = phase_dir / f"genotype{suffix}.json"

    for required in (ckpt_path, genotype_path, args.config_path):
        if not required.exists():
            raise FileNotFoundError(required)

    conf = OmegaConf.load(str(args.config_path))
    data_args = OmegaConf.to_container(conf.data.init_args, resolve=True)
    if args.data_dir is not None:
        data_args["data_dir"] = str(args.data_dir)

    print(f"[ckpt] loading {ckpt_path}")
    tower, full_model_args = load_graph_tower(args.config_path, ckpt_path, genotype_path, device)
    print_alpha_breakdown(tower)

    # The true layer count persisted in the genotype (already used by load_graph_tower
    # to override the YAML) = the k_hop for the eval subgraph, consistent with
    # eval_link_pred_ddp.py.
    eval_num_layers = int(full_model_args["gnn_params"]["num_layers"])
    vx, vei, vnei, edge_attr, neg_info = build_eval_graph(
        data_args=data_args,
        class_path=conf.data.class_path,
        num_layers=eval_num_layers,
        device=device,
        seed=args.seed,
        split=args.split,
        dataset_name=args.dataset_name,
        negative_sampling=args.negative_sampling,
    )

    # Diagnostic: check whether the alphas argmax in the ckpt actually matches what genotype.json records.
    with open(genotype_path) as f:
        gj = json.load(f)
    ops_list = list(tower.gnn.ops)
    for i, layer_g in enumerate(gj["layers"]):
        op_arg = ops_list[int(torch.argmax(tower.gnn.alphas_normal[i]).item())]
        norm_arg = NORM_LIST[int(torch.argmax(tower.gnn.alphas_norm[i]).item())]
        act_arg = ACT_LIST[int(torch.argmax(tower.gnn.alphas_act[i]).item())]
        match_op = "OK" if op_arg == layer_g["op"] else "MISMATCH"
        match_norm = "OK" if norm_arg == layer_g["norm"] else "MISMATCH"
        match_act = "OK" if act_arg == layer_g["act"] else "MISMATCH"
        print(f"[geno-check] layer {i}: ckpt argmax=({op_arg}/{norm_arg}/{act_arg}) "
              f"vs genotype=({layer_g['op']}/{layer_g['norm']}/{layer_g['act']}) "
              f"-> op:{match_op} norm:{match_norm} act:{match_act}")

    rows = []
    label_suffix = ("_best" if args.use_best else "") + neg_info.label_suffix
    # Prefix the label with phase (search/retrain), otherwise row keys from the two stages
    # would collide and overwrite each other.
    phase_prefix = f"{args.phase}_"
    for is_search, label in (
        (True, f"{phase_prefix}GASSO_search{label_suffix}"),
        (False, f"{phase_prefix}GASSO_greedy{label_suffix}"),
    ):
        z = encode_full_graph(tower, vx, vei, edge_attr, is_search=is_search)
        metrics = link_pred_metrics(z, vei, vnei)
        print(
            f"[score] {label:<20} auc={metrics['auc']:.4f}  "
            f"ap={metrics['ap']:.4f}  recon={metrics['recon']:.4f}"
        )
        rows.append({"dataset": args.dataset_name, "gnn_model": label, **metrics})

    df = pd.DataFrame(rows).set_index(["dataset", "gnn_model"]).sort_index()
    out_path = args.run_dir / "link-prediction-eval-gasso.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path).set_index(["dataset", "gnn_model"])
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(out_path)
    print(f"\n[out] saved -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
