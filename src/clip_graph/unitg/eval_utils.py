"""Link-prediction scoring helpers shared by training-time validation and the evaluator.

The protocol mirrors the other systems' evaluators (``bin/eval_link_pred_*.py``
and ``bin/congrat_train.py::_build_val_auc_cache``): positives are all edges of the
split-induced subgraph, negatives are sampled once with
``clip_graph.eval_neg_sampling.sample_negative_edges`` (type-aware on bipartite
graphs) under a fixed seed, and scores are ``InnerProductDecoder(sigmoid=True)``
on L2-normalised embeddings.
"""

from __future__ import annotations

import os
from typing import Optional

import sklearn.metrics as mt
import torch
import torch_geometric as pyg

from clip_graph.eval_neg_sampling import sample_negative_edges

EVAL_NEG_SEED = 2969591811


def sample_eval_negatives(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    node_ids: Optional[torch.Tensor],
    dataset_name: str,
    data_dir: str,
    mode: str = "auto",
    seed: int = EVAL_NEG_SEED,
    device: Optional[torch.device] = None,
):
    device = device or pos_edge_index.device
    torch.manual_seed(seed)
    neg, info = sample_negative_edges(
        pos_edge_index=pos_edge_index,
        num_nodes=num_nodes,
        graph_node_ids=node_ids,
        dataset_name=dataset_name,
        data_dir=data_dir,
        mode=mode,
        seed=seed,
        device=device,
    )
    return neg.to(device), info


@torch.no_grad()
def link_pred_metrics(z: torch.Tensor, pos_ei: torch.Tensor, neg_ei: torch.Tensor, eps: float = 1e-15) -> dict:
    pos_y = z.new_ones(pos_ei.size(1))
    neg_y = z.new_zeros(neg_ei.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).long().detach().cpu().numpy()

    decoder = pyg.nn.models.autoencoder.InnerProductDecoder()
    pos_dec = decoder(z, pos_ei, sigmoid=True)
    neg_dec = decoder(z, neg_ei, sigmoid=True)
    pred = torch.cat([pos_dec, neg_dec], dim=0).detach().float().cpu().numpy()

    return {
        "auc": float(mt.roc_auc_score(y, pred)),
        "ap": float(mt.average_precision_score(y, pred)),
        "recon": float((-torch.log(pos_dec + eps).mean() + -torch.log(1 - neg_dec + eps).mean()).item()),
    }


def data_dir_for(dataset: str) -> str:
    return os.path.join("data", dataset)
