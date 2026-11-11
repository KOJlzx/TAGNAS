"""Affinity-aware node reordering and microbatch construction.

UniTG reorders nodes with a (modified) METIS partitioning so that densely
connected nodes get contiguous ids, and cuts the graph into a small number of
microbatches along the same cluster boundaries so that pipelined execution
does not destroy connectivity. We use the METIS binding shipped with
``torch-sparse`` through :class:`torch_geometric.loader.ClusterData`: its
``node_perm`` is the affinity-aware permutation and ``partptr`` the cluster
boundaries. A microbatch is one cluster, i.e. a Cluster-GCN style induced
subgraph; a mini-batch is ``M`` consecutive microbatches.

With ``halo_hops = k > 0`` a microbatch is instead the cluster's nodes (its *seeds*) plus their
k-hop neighbourhood, i.e. the induced subgraph of ``k_hop_subgraph(seeds, k)`` -- the same
extraction TAGNAS's graph worker uses for its seed batch. Seeds come first in ``node_index``
(``num_seeds`` of them); only seeds are loss anchors and only seed rows are published, the halo
nodes are neighbourhood context. ``halo_hops = 0`` is UniTG's shipped behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, k_hop_subgraph, subgraph


@dataclass
class MicroBatch:
    node_index: torch.Tensor   # [n] indices into the split graph's node arrays (original local ids)
    edge_index: torch.Tensor   # [2, e] relabelled to 0..n-1 (induced subgraph, both directions)
    num_seeds: Optional[int] = None   # seeds are node_index[:num_seeds]; None = every node is a seed

    @property
    def num_nodes(self) -> int:
        return int(self.node_index.numel())

    @property
    def seeds(self) -> int:
        return self.num_nodes if self.num_seeds is None else int(self.num_seeds)


def with_halo(nodes: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, hops: int) -> MicroBatch:
    """Seeds ``nodes`` plus their ``hops``-hop neighbourhood, seeds first, edges induced and relabelled."""
    subset, ei, mapping, _ = k_hop_subgraph(nodes, hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
    is_seed = torch.zeros(subset.numel(), dtype=torch.bool)
    is_seed[mapping] = True
    order = torch.cat([mapping, torch.nonzero(~is_seed, as_tuple=False).flatten()])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    return MicroBatch(node_index=subset[order], edge_index=inv[ei], num_seeds=int(nodes.numel()))


def metis_permutation(edge_index: torch.Tensor, num_nodes: int, num_parts: int):
    """Return ``(node_perm, partptr)`` from METIS (via ClusterData).

    ``node_perm[i]`` is the original id of the node placed at position ``i`` in
    the reordered graph; cluster ``k`` occupies positions
    ``partptr[k]:partptr[k+1]``. For ``num_parts == 1`` the identity is returned
    without invoking METIS.
    """
    if num_parts <= 1:
        return torch.arange(num_nodes), torch.tensor([0, num_nodes])
    from torch_geometric.loader import ClusterData  # local import: pulls torch-sparse

    data = Data(edge_index=edge_index.cpu(), num_nodes=num_nodes)
    cd = ClusterData(data, num_parts=num_parts, log=False)
    return cd.partition.node_perm.clone(), cd.partition.partptr.clone()


def build_microbatches(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_microbatches: int,
    max_nodes_per_microbatch: int,
    *,
    min_microbatches: Optional[int] = None,
    halo_hops: int = 0,
) -> List[List[MicroBatch]]:
    """Cut a (split-induced) graph into mini-batches of ``M`` microbatches each.

    The number of mini-batches ``B`` is the smallest value such that each
    microbatch holds at most ``max_nodes_per_microbatch`` nodes (UniTG's Auto
    Scheduler analogue). ``B*M`` METIS clusters are computed once; microbatch
    ``k`` is cluster ``k`` and mini-batch ``b`` is clusters ``[b*M, (b+1)*M)``.
    """
    M = max(1, int(num_microbatches))
    if min_microbatches is not None:
        M = max(M, int(min_microbatches))
    B = max(1, math.ceil(num_nodes / float(M * max_nodes_per_microbatch)))
    num_parts = B * M
    perm, partptr = metis_permutation(edge_index, num_nodes, num_parts)

    out: List[List[MicroBatch]] = []
    for b in range(B):
        mini: List[MicroBatch] = []
        for m in range(M):
            k = b * M + m
            nodes = perm[int(partptr[k]):int(partptr[k + 1])]
            if nodes.numel() == 0:
                continue
            if halo_hops > 0:
                mini.append(with_halo(nodes, edge_index, num_nodes, int(halo_hops)))
            else:
                ei, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=num_nodes)
                mini.append(MicroBatch(node_index=nodes, edge_index=ei))
        if mini:
            out.append(mini)
    return out


def chunk_for_eval(edge_index: torch.Tensor, num_nodes: int, max_nodes: int) -> List[MicroBatch]:
    """Chunk an evaluation graph into METIS clusters small enough for one forward pass.

    A single chunk (the whole graph) is returned when it fits, which is the
    common case for the val/test splits.
    """
    parts = max(1, math.ceil(num_nodes / float(max_nodes)))
    if parts == 1:
        return [MicroBatch(node_index=torch.arange(num_nodes), edge_index=edge_index)]
    perm, partptr = metis_permutation(edge_index, num_nodes, parts)
    chunks = []
    for k in range(parts):
        nodes = perm[int(partptr[k]):int(partptr[k + 1])]
        if nodes.numel() == 0:
            continue
        ei, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        chunks.append(MicroBatch(node_index=nodes, edge_index=ei))
    return chunks


def degree_groups(
    deg: torch.Tensor,
    num_groups: int,
    thresholds: Optional[torch.Tensor] = None,
):
    """Assign each node to one of ``num_groups`` centrality groups by degree quantile.

    Returns ``(group, thresholds)`` where ``group[i] in [0, num_groups)`` (0 =
    least central) and ``thresholds`` are the ``num_groups-1`` upper bounds
    used, so the same cut points can be applied to other splits.
    """
    deg = deg.float()
    if thresholds is None:
        qs = torch.tensor([(i + 1) / num_groups for i in range(num_groups - 1)])
        thresholds = torch.quantile(deg, qs) if deg.numel() > 0 else torch.zeros(num_groups - 1)
    # torch.bucketize with right=True: group = number of thresholds strictly less than deg,
    # i.e. deg <= t0 -> 0, t0 < deg <= t1 -> 1, ...
    group = torch.bucketize(deg, thresholds, right=False)
    group = torch.clamp(group, 0, num_groups - 1)
    return group.to(torch.long), thresholds


def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
