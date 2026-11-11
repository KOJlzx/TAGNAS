#!/usr/bin/env python3
"""Link-prediction evaluation for the UniTG-style baseline (bin/unitg_train.py artifacts).

Same protocol as ``bin/eval_link_pred_clip_ddp.py`` (whose ``build_eval_graph``
and ``link_pred_metrics`` are imported verbatim): split-induced test subgraph,
seeded negative sampling (type-aware on bipartite graphs), InnerProductDecoder
on L2-normalised embeddings. The model is the deep RevGNN reassembled from its
pipeline stages on one device; its input features are the LM feature table
saved with the checkpoint (``features_<best|final>.pt``), looked up by node id.

Output: ``<run-dir>/link-prediction-eval-unitg.csv`` with rows such as
``unitg_revgnn@best_model`` (``dataset,gnn_model,auc,ap,recon``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "bin")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from eval_link_pred_clip_ddp import link_pred_metrics  # noqa: E402
from clip_graph.eval_neg_sampling import NEGATIVE_SAMPLING_CHOICES, sample_negative_edges  # noqa: E402
from clip_graph.unitg.config import UniTGConfig  # noqa: E402
from clip_graph.unitg.deep_gnn import build_stages, run_stages  # noqa: E402


def build_eval_graph(snapshot, data_dir_override, dataset_name, split, seed, device, negative_sampling):
    """Same steps as ``eval_link_pred_clip_ddp.build_eval_graph`` (datamodule, split-induced
    subgraph, seeded negatives) but also returns the subgraph's node ids, which are needed to
    look up the LM feature table."""
    from pydoc import locate

    data_args = OmegaConf.to_container(snapshot.data.init_args, resolve=True)
    data_dir = str(data_dir_override or REPO_ROOT / "data" / dataset_name)
    data_args["data_dir"] = data_dir
    data_args["compute_sim_mutual"] = False
    DataModuleCls = locate(snapshot.data.class_path)
    if DataModuleCls is None:
        raise ImportError(f"Could not locate datamodule class: {snapshot.data.class_path}")
    dm = DataModuleCls(**data_args)
    dm.prepare_data()
    k_hop = int(snapshot.model.init_args.gnn_params.num_layers)
    try:
        dm.setup(node=0, world_size=1, stage="fit", k_hop=k_hop)
    except TypeError:
        dm.setup(node=0, world_size=1, stage="fit")
    split_ds = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[split]
    graph_data = split_ds.dataset.graph_data
    vx = graph_data.x.float().to(device)
    vei = graph_data.edge_index.to(device)
    if vei.size(1) == 0:
        raise RuntimeError(f"split={split} subgraph has no edges; link-prediction is impossible")
    torch.manual_seed(seed)
    vnei, neg_info = sample_negative_edges(
        pos_edge_index=vei, num_nodes=vx.size(0), graph_node_ids=getattr(graph_data, "node_ids", None),
        dataset_name=dataset_name, data_dir=data_dir, mode=negative_sampling, seed=seed, device=device,
    )
    print(f"[data] split={split}  num_nodes={vx.size(0)}  pos_edges={vei.size(1)}  neg_edges={vnei.size(1)}")
    print(f"[data] negative_sampling requested={neg_info.requested_mode} actual={neg_info.actual_mode} "
          f"suffix={neg_info.label_suffix or '<none>'} reason={neg_info.reason}")
    return vx, vei, vnei, neg_info, graph_data.node_ids.long()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2969591811)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--negative-sampling", choices=NEGATIVE_SAMPLING_CHOICES, default="auto")
    return p.parse_args()


def load_config(run_dir: Path) -> UniTGConfig:
    with open(run_dir / "run_args.json", encoding="utf-8") as f:
        d = json.load(f)
    fields = {k: v for k, v in d.items() if k in UniTGConfig.__dataclass_fields__}
    fields["sync_schedule"] = tuple(fields.get("sync_schedule", (0.3, 0.2, 0.1, 0.02)))
    return UniTGConfig(**fields)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = args.run_dir
    config = load_config(run_dir)
    dataset_name = args.dataset_name or config.dataset
    snapshot = OmegaConf.load(str(run_dir / "model_config_snapshot.yaml"))

    ckpt_path = run_dir / f"{args.checkpoint}_model.pth"
    feat_path = run_dir / f"features_{args.checkpoint}.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    objective = ckpt.get("objective", getattr(config, "objective", "contrastive"))
    feats = torch.load(str(feat_path), map_location="cpu", weights_only=False) if (objective == "link" and feat_path.is_file()) else {}
    print(f"[ckpt] {ckpt_path.name}: objective={objective} epoch={ckpt.get('epoch')} val_auc={ckpt.get('val_auc')} stages={len(ckpt['stages'])}")

    vx, vei, vnei, neg_info, eval_ids = build_eval_graph(
        snapshot=snapshot, data_dir_override=args.data_dir, dataset_name=dataset_name,
        split=args.split, seed=args.seed, device=device, negative_sampling=args.negative_sampling,
    )

    # Input features: under the contrastive objective the GNN reads the adjacency-SVD features
    # (exactly as B1); under 'link' it reads the LM table saved with the checkpoint.
    table = feats.get("table")
    if table is not None:
        node_ids = feats["node_ids"].long()
        row_of = {int(v): i for i, v in enumerate(node_ids.tolist())}
        rows = torch.tensor([row_of[int(v)] for v in eval_ids.tolist()], dtype=torch.long)
        x = table[rows].float().to(device)
        src = "lm_table"
    else:
        x = vx
        src = "svd (datamodule features)"
    print(f"[feat] input features from {src}: {tuple(x.shape)}")

    config.world_size = int(ckpt.get("num_stages", config.world_size))
    stages = build_stages(config, in_channels=int(ckpt.get("in_channels", x.size(1))), device=device)
    for st, sd in zip(stages, ckpt["stages"]):
        st.load_state_dict(sd, strict=True)
    z = run_stages(stages, x, vei)
    metrics = link_pred_metrics(z, vei, vnei)

    label = f"unitg_revgnn@{args.checkpoint}_model{neg_info.label_suffix}"
    print(f"[score] {label:<40} auc={metrics['auc']:.4f}  ap={metrics['ap']:.4f}  recon={metrics['recon']:.4f}")
    df = pd.DataFrame([{"dataset": dataset_name, "gnn_model": label, **metrics}]).set_index(["dataset", "gnn_model"])
    out_path = run_dir / "link-prediction-eval-unitg.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path).set_index(["dataset", "gnn_model"])
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(out_path)
    print(f"[out] saved -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
