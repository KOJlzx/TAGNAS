"""Centrality-guided Lazy Tuner: early-exit sentence encoder.

Nodes are grouped by degree centrality into ``num_exits`` groups; group ``g``
leaves the transformer after ``exit_layers[g]`` layers and is pooled and
projected by its own exit head φ_g. The truncation is real (deeper layers are
not executed), implemented by temporarily swapping the model's layer
``ModuleList`` for a slice of itself, which works for the GPT-2 family
(``model.h`` + ``model.ln_f``) and the BERT family (``model.encoder.layer``).
"""

from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def exit_layers_for(num_layers: int, num_exits: int) -> List[int]:
    """Evenly spaced exit depths ending at the full depth, e.g. 6 layers/4 exits -> [2, 3, 5, 6]."""
    num_exits = max(1, min(int(num_exits), int(num_layers)))
    exits = [max(1, math.ceil(num_layers * (i + 1) / num_exits)) for i in range(num_exits)]
    exits[-1] = num_layers
    # Ensure strictly increasing depths.
    for i in range(1, len(exits)):
        exits[i] = max(exits[i], exits[i - 1] + 1) if exits[i - 1] + 1 <= num_layers else num_layers
    return exits


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ExitHead(nn.Module):
    """Per-exit projection φ_i with the structure of the text towers of B1/TAGNAS
    (``gassomodels.ProjectHead``: Linear → GELU → Linear → LayerNorm). Random initialisation
    matters: a plain identity/linear head emits the raw mean-pooled LM state, whose rows are
    almost collinear (anisotropy), which leaves the GNN-side InfoNCE at chance with no gradient."""

    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, target_dim)
        self.linear2 = nn.Linear(target_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear2(F.gelu(self.linear1(x))))


class EarlyExitLM(nn.Module):
    """Sentence encoder with centrality-guided early exits.

    ``forward(input_ids, attention_mask, group)`` returns L2-normalised
    ``[batch, embed_dim]`` embeddings in the input order; rows with group ``g``
    were computed with only the first ``exit_layers[g]`` transformer layers.
    """

    def __init__(
        self,
        lm: nn.Module,
        num_exits: int = 4,
        embed_dim: int = 768,
        pooling: str = "mean",
        normalize: bool = True,
        lazy: bool = True,
        tau_init: float = 3.5,
        max_tau: Optional[float] = 4.605,
    ):
        super().__init__()
        self.lm = lm
        self.pooling = pooling
        self.normalize = normalize
        self.lazy = bool(lazy)

        owner, attr = self._locate_layers(lm)
        # Keep only a *path* to the layer list: assigning the owner module to an attribute would
        # register it as a second child and duplicate every LM tensor in state_dict().
        self._owner_is_lm = owner is lm
        self._layers_attr = attr
        self.num_layers = len(getattr(owner, attr))
        self.exit_layers = exit_layers_for(self.num_layers, num_exits) if lazy else [self.num_layers]
        self.num_exits = len(self.exit_layers)
        hidden = int(lm.config.hidden_size)
        self.heads = nn.ModuleList([ExitHead(hidden, embed_dim) for _ in range(self.num_exits)])
        # Per-dimension standardisation across nodes (non-affine), shared by all exits and applied
        # to the assembled batch. Pretrained LM sentence states share a large common component; in
        # the joint training of B1/TAGNAS the spread GNN partner supplies the InfoNCE gradient that
        # removes it, but in this decoupled design both sides of each InfoNCE would be near-collinear
        # and the loss would sit at ln(N) with no gradient. Standardising the text embeddings
        # restores the signal on both sides.
        self.standardize = nn.BatchNorm1d(embed_dim, affine=False, momentum=0.05)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)))   # logit scale = exp(log_tau)
        self.max_tau = max_tau

    # ------------------------------------------------------------------ layers
    @staticmethod
    def _locate_layers(lm: nn.Module):
        if hasattr(lm, "h") and isinstance(getattr(lm, "h"), nn.ModuleList):
            return lm, "h"                       # GPT-2 / distilgpt2
        enc = getattr(lm, "encoder", None)
        if enc is not None and hasattr(enc, "layer") and isinstance(enc.layer, nn.ModuleList):
            return enc, "layer"                  # BERT / LinkBERT / RoBERTa
        raise TypeError(f"Unsupported LM architecture for early exit: {type(lm).__name__}")

    @property
    def _layers_owner(self) -> nn.Module:
        return self.lm if self._owner_is_lm else self.lm.encoder

    @property
    def transformer_layers(self) -> nn.ModuleList:
        return getattr(self._layers_owner, self._layers_attr)

    @contextlib.contextmanager
    def _truncated(self, k: int):
        full = self.transformer_layers
        if k >= len(full):
            yield
            return
        try:
            setattr(self._layers_owner, self._layers_attr, nn.ModuleList(list(full)[:k]))
            yield
        finally:
            setattr(self._layers_owner, self._layers_attr, full)

    def _run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, k: int) -> torch.Tensor:
        with self._truncated(k):
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state

    # ----------------------------------------------------------------- forward
    def group_for_exit(self, group: torch.Tensor) -> torch.Tensor:
        if not self.lazy:
            return torch.zeros_like(group)
        return group.clamp(0, self.num_exits - 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                group: Optional[torch.Tensor] = None, pairs=None):
        """Encode a batch; with ``pairs=(src, dst)`` also return temperature-scaled link logits.

        ``pairs`` may be ``(src, dst)`` index tensors (link objective) or a
        ``[batch, D]`` target matrix (contrastive objective).
        """
        emb = self.encode(input_ids, attention_mask, group)
        if pairs is None:
            return emb
        if isinstance(pairs, torch.Tensor):
            return emb, self.contrastive_logits(emb.float(), pairs.float())
        src, dst = pairs
        return emb, self.link_logits(emb.float(), src, dst)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               group: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = input_ids.size(0)
        if group is None or not self.lazy:
            group = torch.full((n,), self.num_exits - 1, dtype=torch.long, device=input_ids.device)
        group = self.group_for_exit(group.to(input_ids.device))
        out = None
        for g in torch.unique(group).tolist():
            idx = (group == g).nonzero(as_tuple=True)[0]
            hidden = self._run(input_ids[idx], attention_mask[idx], self.exit_layers[g])
            pooled = masked_mean_pool(hidden, attention_mask[idx]) if self.pooling == "mean" else hidden[:, 0]
            emb = self.heads[g](pooled.to(self.heads[g].linear1.weight.dtype))
            if out is None:
                out = emb.new_zeros((n, emb.size(1)))
            out = out.index_copy(0, idx, emb.to(out.dtype))
        if self.training and n < 2:
            # BatchNorm cannot estimate statistics from one row; use the running ones.
            self.standardize.eval()
            out = self.standardize(out.float())
            self.standardize.train()
        else:
            out = self.standardize(out.float())
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def temperature(self) -> torch.Tensor:
        value = self.log_tau.exp()
        if self.max_tau is None:
            return value
        return value.clamp(max=math.exp(float(self.max_tau)))

    def link_logits(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (e[src] * e[dst]).sum(dim=-1) * self.temperature()

    def contrastive_logits(self, e: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return e @ targets.T * self.temperature()

    def layer_modules(self) -> List[nn.Module]:
        return list(self.transformer_layers)
