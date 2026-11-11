"""Negative sampling helpers for link-prediction evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch_geometric as pyg


NEGATIVE_SAMPLING_CHOICES = ("default", "type_aware", "auto")
TYPE_AWARE_LABEL_SUFFIX = "__neg_type_aware"
AUTO_TYPE_AWARE_THRESHOLD = 0.95


@dataclass(frozen=True)
class NegativeSamplingInfo:
    requested_mode: str
    actual_mode: str
    label_suffix: str
    cross_type_ratio: float | None = None
    reason: str = ""


def sample_negative_edges(
    *,
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    graph_node_ids: Iterable[int] | torch.Tensor | None,
    dataset_name: str,
    data_dir: str | Path,
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, NegativeSamplingInfo]:
    """Sample negative edges for link-prediction evaluation.

    ``type_aware`` preserves each positive edge's source node and target node type.
    This is important for bipartite datasets such as Amazon/Goodreads, where
    random all-node negative sampling creates many same-type negatives that do
    not match the task semantics.
    """
    if mode not in NEGATIVE_SAMPLING_CHOICES:
        raise ValueError(f"negative sampling mode must be one of {NEGATIVE_SAMPLING_CHOICES}, got {mode!r}")

    type_info = _load_local_node_types(data_dir, graph_node_ids)
    if mode == "default":
        return _default_negative_sampling(pos_edge_index, num_nodes, device), NegativeSamplingInfo(
            requested_mode=mode,
            actual_mode="default",
            label_suffix="",
            reason="requested default",
        )

    if type_info is None:
        if mode == "type_aware":
            raise RuntimeError(
                f"negative_sampling=type_aware requires {Path(data_dir) / 'raw' / 'node-data.csv'} "
                "with node_id/node_type columns and graph_data.node_ids"
            )
        return _default_negative_sampling(pos_edge_index, num_nodes, device), NegativeSamplingInfo(
            requested_mode=mode,
            actual_mode="default",
            label_suffix="",
            reason="node_type unavailable",
        )

    local_types, type_to_nodes = type_info
    cross_type_ratio = _cross_type_ratio(pos_edge_index, local_types)
    use_type_aware = mode == "type_aware" or cross_type_ratio >= AUTO_TYPE_AWARE_THRESHOLD
    if not use_type_aware:
        return _default_negative_sampling(pos_edge_index, num_nodes, device), NegativeSamplingInfo(
            requested_mode=mode,
            actual_mode="default",
            label_suffix="",
            cross_type_ratio=cross_type_ratio,
            reason=f"cross_type_ratio={cross_type_ratio:.4f} < {AUTO_TYPE_AWARE_THRESHOLD:.2f}",
        )

    neg_edge_index = _type_aware_negative_sampling(
        pos_edge_index=pos_edge_index,
        local_types=local_types,
        type_to_nodes=type_to_nodes,
        seed=seed,
        device=device,
    )
    return neg_edge_index, NegativeSamplingInfo(
        requested_mode=mode,
        actual_mode="type_aware",
        label_suffix=TYPE_AWARE_LABEL_SUFFIX,
        cross_type_ratio=cross_type_ratio,
        reason=f"cross_type_ratio={cross_type_ratio:.4f}",
    )


def _default_negative_sampling(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    return pyg.utils.negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    ).to(device)


def _load_local_node_types(
    data_dir: str | Path,
    graph_node_ids: Iterable[int] | torch.Tensor | None,
) -> tuple[list[object], dict[object, list[int]]] | None:
    if graph_node_ids is None:
        return None
    node_data_path = Path(data_dir) / "raw" / "node-data.csv"
    if not node_data_path.is_file():
        return None

    node_data = pd.read_csv(node_data_path, sep="\t")
    if "node_id" not in node_data.columns or "node_type" not in node_data.columns:
        return None

    node_type_by_id = dict(zip(node_data["node_id"].astype(int).tolist(), node_data["node_type"].tolist()))
    if isinstance(graph_node_ids, torch.Tensor):
        ids = graph_node_ids.detach().cpu().tolist()
    else:
        ids = list(graph_node_ids)

    local_types: list[object] = []
    type_to_nodes: dict[object, list[int]] = {}
    for local_idx, node_id in enumerate(ids):
        node_type = node_type_by_id.get(int(node_id))
        local_types.append(node_type)
        if node_type is not None:
            type_to_nodes.setdefault(node_type, []).append(local_idx)

    if len(type_to_nodes) < 2:
        return None
    return local_types, type_to_nodes


def _cross_type_ratio(pos_edge_index: torch.Tensor, local_types: list[object]) -> float:
    edges = pos_edge_index.detach().cpu().t().tolist()
    typed = 0
    cross = 0
    for src, dst in edges:
        src_type = local_types[src]
        dst_type = local_types[dst]
        if src_type is None or dst_type is None:
            continue
        typed += 1
        if src_type != dst_type:
            cross += 1
    return cross / typed if typed else 0.0


def _type_aware_negative_sampling(
    *,
    pos_edge_index: torch.Tensor,
    local_types: list[object],
    type_to_nodes: dict[object, list[int]],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    pos_edges = [(int(s), int(t)) for s, t in pos_edge_index.detach().cpu().t().tolist()]
    blocked = set(pos_edges)
    blocked.update((t, s) for s, t in pos_edges)

    rng = random.Random(seed)
    sampled: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for src, dst in pos_edges:
        dst_type = local_types[dst]
        candidates = type_to_nodes.get(dst_type, [])
        if not candidates:
            continue
        accepted = False
        for _ in range(200):
            neg_dst = candidates[rng.randrange(len(candidates))]
            key = (src, neg_dst)
            if neg_dst == src or key in blocked or key in seen:
                continue
            seen.add(key)
            sampled.append(key)
            accepted = True
            break
        if accepted:
            continue

    # Rare dense-node fallback: keep target-type distribution, but no longer keep
    # the exact source from the positive edge.
    target = len(pos_edges)
    attempts = 0
    by_type = list(type_to_nodes.items())
    while len(sampled) < target and attempts < target * 100:
        _, src_candidates = by_type[rng.randrange(len(by_type))]
        dst_type, dst_candidates = by_type[rng.randrange(len(by_type))]
        if not src_candidates or not dst_candidates:
            attempts += 1
            continue
        src = src_candidates[rng.randrange(len(src_candidates))]
        dst = dst_candidates[rng.randrange(len(dst_candidates))]
        key = (src, dst)
        if src == dst or local_types[src] == dst_type or key in blocked or key in seen:
            attempts += 1
            continue
        seen.add(key)
        sampled.append(key)
        attempts += 1

    if len(sampled) < target:
        print(f"[warn] only sampled {len(sampled)}/{target} type-aware negative edges")

    if not sampled:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(sampled, dtype=torch.long, device=device).t().contiguous()
