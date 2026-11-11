"""Deep reversible GNN split into pipeline stages.

The backbone is the grouped-reversible GNN of *Training Graph Neural Networks
with 1000 Layers* (Li et al., ICML'21), which is also UniTG's default backbone
(``revgnn``, 2 groups, 112 layers, 224 hidden). The layer stack is cut into
``num_stages`` contiguous chunks; stage 0 additionally owns the input
projection and the last stage owns the output head (LayerNorm + linear +
L2-normalisation) plus a learnable link-score temperature.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GroupAddRev, LayerNorm, SAGEConv


def stage_layer_ranges(num_layers: int, num_stages: int) -> List[range]:
    """Balanced contiguous split of ``range(num_layers)`` into ``num_stages`` ranges."""
    base, extra = divmod(num_layers, num_stages)
    ranges, start = [], 0
    for s in range(num_stages):
        n = base + (1 if s < extra else 0)
        ranges.append(range(start, start + n))
        start += n
    return ranges


class GNNBlock(nn.Module):
    """Pre-norm SAGE block used inside each reversible group (as in PyG's rev_gnn example)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm(in_channels, affine=True)
        self.conv = SAGEConv(in_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)


class RevGNNStage(nn.Module):
    """One pipeline stage of the deep reversible GNN."""

    def __init__(
        self,
        stage_index: int,
        num_stages: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.2,
        num_groups: int = 2,
        tau_init: float = 3.5,
        max_tau: Optional[float] = 4.605,
    ):
        super().__init__()
        if hidden_channels % num_groups != 0:
            raise ValueError("hidden_channels must be divisible by num_groups")
        self.stage_index = int(stage_index)
        self.num_stages = int(num_stages)
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.dropout = float(dropout)
        self.layer_range = stage_layer_ranges(num_layers, num_stages)[stage_index]

        if self.is_first:
            self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in self.layer_range:
            block = GNNBlock(hidden_channels // num_groups, hidden_channels // num_groups)
            self.convs.append(GroupAddRev(block, num_groups=num_groups))
        self.max_tau = max_tau
        if self.is_last:
            self.norm_out = LayerNorm(hidden_channels, affine=True)
            self.lin_out = nn.Linear(hidden_channels, out_channels)
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)))   # logit scale = exp(log_tau), as B1/TAGNAS

    @property
    def num_local_layers(self) -> int:
        return len(self.convs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            x = self.lin_in(x)
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False) / (1 - self.dropout)
        for conv in self.convs:
            x = conv(x, edge_index, mask)
        if self.is_last:
            x = self.norm_out(x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin_out(x)
            x = F.normalize(x, p=2, dim=1)
        return x

    def temperature(self) -> torch.Tensor:
        value = self.log_tau.exp()
        if self.max_tau is None:
            return value
        return value.clamp(max=math.exp(float(self.max_tau)))

    def link_logits(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine score for the given (src, dst) pairs ('link' objective)."""
        if not self.is_last:
            raise RuntimeError("link_logits is only defined on the last stage")
        score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return score * self.temperature()

    def contrastive_logits(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Square graph-vs-text similarity matrix scaled by the temperature ('contrastive')."""
        if not self.is_last:
            raise RuntimeError("contrastive_logits is only defined on the last stage")
        return z @ targets.T * self.temperature()


def build_stages(config, in_channels: int, device: Optional[torch.device] = None) -> List[RevGNNStage]:
    """Instantiate every stage (used by the single-GPU evaluator)."""
    stages = []
    for s in range(config.num_stages):
        st = RevGNNStage(
            stage_index=s,
            num_stages=config.num_stages,
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.embed_dim,
            num_layers=config.num_layers,
            dropout=config.gnn_dropout,
            num_groups=config.num_groups,
            tau_init=getattr(config, "tau_init", 3.5),
            max_tau=getattr(config, "max_tau", 4.605),
        )
        if device is not None:
            st = st.to(device)
        stages.append(st)
    return stages


@torch.no_grad()
def run_stages(stages: List[RevGNNStage], x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Sequentially run all stages on one device (evaluation only)."""
    h = x
    for st in stages:
        st.eval()
        h = st(h, edge_index)
    return h
