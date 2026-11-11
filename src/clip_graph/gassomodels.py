from typing import Optional, Dict, Any, Callable, Union

import io
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch_geometric.nn as gnn

import transformers as tf

from . import utils as ut

#
# Base classes
#
import typing as _typ
# Import List directly for compatibility with existing annotations.
from typing import List
from torch.autograd import Variable
from .gasso_space import *

class ModelBase(nn.Module):
    '''
    Base class for models.
    '''

    pass


#
# Graph models
#

GNN_LIST = [
    "gat",     # GAT 1-head
    "gat_2h",  # GAT 2-head (concat)
    "gat_4h",  # GAT 4-head (concat)
    "gcn",
    "gin",
    "sage",
    "linear",
]

NORM_LIST = ["layernorm", "batchnorm", "graphnorm", "identity"]
ACT_LIST  = ["relu", "gelu", "elu", "identity"]



class GraphNorm(nn.Module):
    """Graph-level mean-subtraction normalization + learnable scale/bias."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (var + self.eps).sqrt() + self.bias


class MixedNorm(nn.Module):
    """Searchable normalization: softmax-weighted mixture of 4 candidates."""
    def __init__(self, channels: int):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(channels),
            nn.BatchNorm1d(channels),
            GraphNorm(channels),
            nn.Identity(),
        ])

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        selected_idx: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if selected_idx is not None:
            return self.norms[int(selected_idx)](x)
        assert weights is not None
        out = None
        for k, (w, n) in enumerate(zip(weights, self.norms)):
            if mask is not None and not bool(mask[k]):
                continue  # Pruned norm candidate: skip forward and avoid building a gradient graph.
            term = w * n(x)
            out = term if out is None else out + term
        return out if out is not None else torch.zeros_like(x)


class MixedAct(nn.Module):
    """Searchable activation: softmax-weighted mixture of 4 candidates."""
    def __init__(self):
        super().__init__()
        self.acts = nn.ModuleList([
            nn.ReLU(),
            nn.GELU(),
            nn.ELU(),
            nn.Identity(),
        ])

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        selected_idx: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if selected_idx is not None:
            return self.acts[int(selected_idx)](x)
        assert weights is not None
        out = None
        for k, (w, a) in enumerate(zip(weights, self.acts)):
            if mask is not None and not bool(mask[k]):
                continue
            term = w * a(x)
            out = term if out is None else out + term
        return out if out is not None else torch.zeros_like(x)


class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, gnn_list=GNN_LIST):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.gnn_list = gnn_list
        for action in gnn_list:
            self._ops.append(gnn_map(action, in_c, out_c))
        self.norm = MixedNorm(out_c)
        self.act = MixedAct()

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        op_weights,
        norm_weights,
        act_weights,
        selected_op_idx: Optional[int] = None,
        selected_norm_idx: Optional[int] = None,
        selected_act_idx: Optional[int] = None,
        op_mask: Optional[torch.Tensor] = None,
        norm_mask: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None,
    ):
        if selected_op_idx is None:
            # Search mode: mix all surviving GNN candidates with softmax(alpha) weights.
            # Masked candidates are skipped entirely, so they allocate no activation memory
            # and do not participate in autograd.
            fin = None
            for k, (w, op) in enumerate(zip(op_weights, self._ops)):
                if op_mask is not None and not bool(op_mask[k]):
                    continue
                term = w * op(x, edge_index)
                fin = term if fin is None else fin + term
            if fin is None:
                fin = torch.zeros_like(x)
            out = self.norm(fin, weights=norm_weights, mask=norm_mask)
            out = self.act(out, weights=act_weights, mask=act_mask)
            return out
        # Retrain / discrete mode: execute only the argmax-selected branch.
        out = self._ops[int(selected_op_idx)](x, edge_index)
        out = self.norm(out, selected_idx=int(selected_norm_idx))
        out = self.act(out, selected_idx=int(selected_act_idx))
        return out


def Get_edges(adjs,):
    edges = []
    edges_weights = []
    edge_index_all = adjs[0]  # [num_layers, 2, E]
    edge_weight_all = adjs[1]  # [num_layers, E, 1] or [num_layers, E]
    
    num_layers = edge_index_all.shape[0]
    for i in range(num_layers):
        edges.append(edge_index_all[i])  # [2, E]
        ew = edge_weight_all[i]  # [E, 1] or [E]
        # Keep edge weights in the 1D [E] format expected by downstream layers.
        if ew.ndim > 1:
            ew = ew.view(-1)  # [E]
        edges_weights.append(torch.sigmoid(ew))
        # print(ew.shape)
        # print(ew)
        # print("X"*100)
    return edges, edges_weights


class CellWS(nn.Module):
    def __init__(self, num_layers, his_dim, hidden_dim, out_dim, dp, training=True, bias=True):
        super(CellWS, self).__init__()
        self.num_layers = num_layers
        self._ops = nn.ModuleList()
        self.dp = 0.3
        self.training = training
        for i in range(self.num_layers):
            op = MixedOp(hidden_dim, hidden_dim)
            self._ops.append(op)

    def forward(
        self,
        x,
        adjs,
        edge_index,
        op_weights,
        norm_weights,
        act_weights,
        selection=None,
        op_masks=None,
        norm_masks=None,
        act_masks=None,
    ):
        """
        selection: None means continuous relaxation with soft mixtures.
        Otherwise this is (op_idx, norm_idx, act_idx), where each tensor has shape
        [num_layers] and stores the discrete GNN / normalization / activation branch
        selected by argmax for that layer.
        op_masks/norm_masks/act_masks are optional per-layer BoolTensors where
        True means the candidate is still active. They only affect the continuous
        relaxation path; discrete selection has already avoided pruned entries.
        """
        if adjs is None:
            ews = [torch.ones(edge_index.size(1), device=edge_index.device)] * self.num_layers
        else:
            _, ews = Get_edges(adjs)
        for i in range(self.num_layers):
            if selection is not None:
                # In a discrete architecture, each layer executes exactly one
                # op/norm/activation path.
                op_i = int(selection[0][i].item())
                norm_i = int(selection[1][i].item())
                act_i = int(selection[2][i].item())
                x = x + self._ops[i](
                    x,
                    edge_index,
                    ews[i],
                    None,
                    None,
                    None,
                    selected_op_idx=op_i,
                    selected_norm_idx=norm_i,
                    selected_act_idx=act_i,
                )
            else:
                # During continuous search, each layer is a residual mixed operation.
                x = x + self._ops[i](
                    x,
                    edge_index,
                    ews[i],
                    op_weights[i],
                    norm_weights[i],
                    act_weights[i],
                    op_mask=(op_masks[i] if op_masks is not None else None),
                    norm_mask=(norm_masks[i] if norm_masks is not None else None),
                    act_mask=(act_masks[i] if act_masks is not None else None),
                )
        return x

class GassoSpace(ModelBase):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        in_channels: int = 768,
        out_channels: Optional[int] = None,
        ops: _typ.Tuple = GNN_LIST,
        training: bool = True,
    ):

        assert num_layers >= 1

        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.ops = ops
        self.training = training
        self.adjs = None
        self.step = 0
        self.dropout = nn.Dropout(dropout)
        if self.in_channels is not None:
            self.embed = nn.Linear(self.in_channels, self.hidden_channels)
        else:  # self.num_nodes is not None
            self.embed = nn.Embedding(self.num_nodes, self.hidden_channels)
        # print("embed", self.embed.weight)
        self.build_graph() # build the graph for gnn modules
        if self.out_channels is not None:
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        else:
            self.head = None
        self.initialize_alphas() # initialize alphas for gnn modules
        # self.apply(self._init_weights_zero)



    def _init_weights_zero(self, m):
        """
        Set every supported parameter to zero and force dropout layers off.
        """
        # 1. Linear
        if isinstance(m, nn.Linear):
            init.constant_(m.weight, 0)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
        # 2. Embedding
        elif isinstance(m, nn.Embedding):
            init.constant_(m.weight, 0)
            
        # 3. LayerNorm
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 0)
            init.constant_(m.bias, 0)

        # 4. GATConv (PyG)
        elif isinstance(m, gnn.GATConv):
            if hasattr(m, 'att_src') and m.att_src is not None:
                init.constant_(m.att_src, 0)
            if hasattr(m, 'att_dst') and m.att_dst is not None:
                init.constant_(m.att_dst, 0)
            if hasattr(m, 'att') and m.att is not None: # Backward compatibility with older PyG versions.
                init.constant_(m.att, 0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)
            
            # GATConv commonly stores its own dropout probability; set it explicitly
            # so deterministic zero-initialization tests are not affected by attention dropout.
            if hasattr(m, 'dropout'):
                m.dropout = 0.0

        # 5. Standalone Dropout layers.
        elif isinstance(m, nn.Dropout):
            # Dropout has no weight, only probability p. Setting p=0 makes it a
            # deterministic pass-through layer.
            m.p = 0.0

    
    def build_graph(self):
        # print(self.dropout)
        # print("X"*100)
        self.cell = CellWS( # cell -> gnn module
            self.num_layers, self.in_channels, self.hidden_channels, self.out_channels, self.dropout, self.training
        )
        

    # def forward(self, x, adjs):
    def forward(self, x, edge_index, edge_weight = None, is_search = True):
        # if self.step <= 1:
        #     edge_index = edge_index.unsqueeze(0).repeat(self.num_layers, 1, 1)
        #     edge_weight = edge_weight.unsqueeze(0).repeat(self.num_layers, 1, 1)
        #     self.adjs = (edge_index, edge_weight)
        #     self.step += 1

        #     return None
        # print(self.adjs[0], self.adjs[1])
        # print(self.alphas_normal)
        # print("X"*100)
        # print("x[0][0]:",x[0][0])
        self.step += 1
        if self.in_channels is not None:
            x = self.dropout(x)
            x = self.embed(x) * (self.hidden_channels ** 0.5)
        else:
            x = self.embed(x) * (self.hidden_channels ** 0.5)
            x = self.dropout(x)
        op_weights = []
        norm_weights = []
        act_weights = []
        op_masks, norm_masks, act_masks = self._get_layer_masks()
        for j in range(self.num_layers):
            # Each layer owns three alpha groups:
            # 1. GNN operator selection
            # 2. normalization selection
            # 3. activation selection
            # _masked_softmax uses a fast path when all masks are True, preserving
            # byte-equivalent behavior to F.softmax. Pruned positions receive zero
            # probability, so argmax naturally avoids them.
            op_weights.append(self._masked_softmax(self.alphas_normal[j], op_masks[j]))
            norm_weights.append(self._masked_softmax(self.alphas_norm[j], norm_masks[j]))
            act_weights.append(self._masked_softmax(self.alphas_act[j], act_masks[j]))

        if is_search:
            # The search phase keeps the continuous relaxation so alpha can be updated.
            selection = None
        else:
            # Discrete architecture: select GNN / Norm / Act independently per layer
            # using the same masked softmax distribution used during training.
            wt_op = torch.stack(op_weights).detach()
            wt_norm = torch.stack(norm_weights).detach()
            wt_act = torch.stack(act_weights).detach()
            op_idx = torch.argmax(wt_op, dim=-1)
            norm_idx = torch.argmax(wt_norm, dim=-1)
            act_idx = torch.argmax(wt_act, dim=-1)
            selection = (op_idx, norm_idx, act_idx)

        x = self.cell(
            x, self.adjs, edge_index, op_weights, norm_weights, act_weights, selection,
            op_masks=op_masks, norm_masks=norm_masks, act_masks=act_masks,
        )
        # print("x", x)
        ret = {'last_hidden_state': x}
        if self.head is not None:
            ret['output'] = self.head(x)
        else:
            ret['output'] = None

        return ret

    def keep_prediction(self):
        self.prediction = self.current_pred

    """def to(self, *args, **kwargs):
        fin = super().to(*args, **kwargs)
        device = next(fin.parameters()).device
        fin.alphas_normal = [i.to(device) for i in self.alphas_normal]
        return fin"""

    def initialize_alphas(self):
        num_ops = len(self.ops)
        num_norms = len(NORM_LIST)
        num_acts = len(ACT_LIST)

        # ParameterList ensures these tensors are tracked by named_parameters(),
        # which lets the optimizer code split architecture and weight parameters.
        self.alphas_normal = nn.ParameterList()
        self.alphas_norm = nn.ParameterList()
        self.alphas_act = nn.ParameterList()
        for i in range(self.num_layers):
            self.alphas_normal.append(nn.Parameter(1e-3 * torch.randn(num_ops)))
            self.alphas_norm.append(nn.Parameter(1e-3 * torch.randn(num_norms)))
            self.alphas_act.append(nn.Parameter(1e-3 * torch.randn(num_acts)))

        self._arch_parameters = [self.alphas_normal, self.alphas_norm, self.alphas_act]

        # --- Mask buffers for adaptive pruning ---
        # True means the candidate is active; False means it has been pruned.
        # register_buffer keeps masks in state_dict and moves them with .to().
        # When adaptive pruning is disabled, masks remain all True and forward
        # follows the fast path, matching the original implementation.
        for i in range(self.num_layers):
            self.register_buffer(f"mask_op_{i}",   torch.ones(num_ops,   dtype=torch.bool))
            self.register_buffer(f"mask_norm_{i}", torch.ones(num_norms, dtype=torch.bool))
            self.register_buffer(f"mask_act_{i}",  torch.ones(num_acts,  dtype=torch.bool))
        self._pruning_events: List[dict] = []
        # Architecture softmax temperature for annealing. tau=1.0 matches the
        # original untempered behavior; the training loop can lower it to encourage
        # earlier commitment to a small set of operators.
        self.arch_temperature = 1.0

    # ------------------------------------------------------------------
    # Adaptive pruning helpers
    # ------------------------------------------------------------------
    def set_arch_temperature(self, tau: float) -> None:
        """Set the architecture softmax temperature.

        tau < 1 sharpens the softmax, encouraging alpha to commit and pushing weak
        operators toward zero. tau = 1 preserves the original untempered behavior.
        """
        self.arch_temperature = max(float(tau), 1e-4)

    def _masked_softmax(
        self, alpha: torch.Tensor, mask: torch.Tensor, tau: Optional[float] = None
    ) -> torch.Tensor:
        """Apply softmax(alpha / tau) after setting masked positions to -inf.

        When tau=1 and every mask entry is True, this is equivalent to F.softmax.
        tau=None uses the current arch_temperature.
        """
        tau = float(getattr(self, "arch_temperature", 1.0)) if tau is None else float(tau)
        logits = alpha / tau if tau != 1.0 else alpha
        if bool(mask.all()):
            return F.softmax(logits, dim=-1)
        return F.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)

    def _get_layer_masks(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        op_masks = [getattr(self, f"mask_op_{i}")   for i in range(self.num_layers)]
        norm_masks = [getattr(self, f"mask_norm_{i}") for i in range(self.num_layers)]
        act_masks = [getattr(self, f"mask_act_{i}")  for i in range(self.num_layers)]
        return op_masks, norm_masks, act_masks

    def zero_pruned_grads(self) -> None:
        """Zero gradients for masked alpha entries and push their values to -1e4.

        Call this before graph_arch_optimizer.step() so pruned candidates cannot
        recover through stale gradients or future argmax selection.
        """
        for alphas, prefix in [
            (self.alphas_normal, "mask_op"),
            (self.alphas_norm,   "mask_norm"),
            (self.alphas_act,    "mask_act"),
        ]:
            for i in range(self.num_layers):
                m = getattr(self, f"{prefix}_{i}")
                if alphas[i].grad is not None:
                    alphas[i].grad[~m] = 0.0
                with torch.no_grad():
                    alphas[i].data[~m] = -1e4

    def prune_once(
        self,
        threshold: float,
        min_keep: int = 1,
        mode: str = "absolute",
        groups: Optional[_typ.Iterable[str]] = None,
        threshold_abs: float = 0.0,
    ) -> list[dict]:
        """Scan alpha groups and mark low-probability candidates as pruned.

        Args:
            threshold: Pruning threshold. Its semantics depend on mode.
            min_keep: Minimum number of active candidates retained in every
                layer/group so the search space cannot become empty.
            mode:
                - "absolute": prune only when weight < threshold.
                - "relative": prune when weight < threshold * max(weights), the
                  fraction-of-group-maximum rule of PruneOnce. The cutoff adapts
                  to how peaked the distribution already is.
            threshold_abs: Absolute floor combined with the per-mode cutoff as
                max(threshold_abs, cutoff) -- the delta term of PruneOnce's
                max(delta, rho * p_max). 0.0 (default) disables the floor.

        Pruning decisions use the temperature-free softmax(alpha) (tau=1), so the
        annealing schedule (tau > 1 early in search) cannot flatten the weights
        above any fixed cutoff; the forward mixture keeps the annealed softmax.

        Returns the pruning events added by this call. Each event records layer,
        group, index, name, weight_at_removal, effective_threshold, and mode.
        """
        if mode not in ("absolute", "relative"):
            raise ValueError(f"Unknown pruning mode: {mode!r}")
        allowed_groups: Optional[set[str]] = None
        if groups is not None:
            allowed_groups = {str(g) for g in groups}
            unknown = allowed_groups - {"op", "norm", "act"}
            if unknown:
                raise ValueError(f"Unknown pruning groups: {sorted(unknown)}")
        events: list[dict] = []
        group_specs = [
            (self.alphas_normal, "mask_op",   "op",   list(self.ops)),
            (self.alphas_norm,   "mask_norm", "norm", list(NORM_LIST)),
            (self.alphas_act,    "mask_act",  "act",  list(ACT_LIST)),
        ]
        for alphas, prefix, group_name, name_list in group_specs:
            if allowed_groups is not None and group_name not in allowed_groups:
                continue  # This group was excluded by the caller.
            for i in range(self.num_layers):
                m = getattr(self, f"{prefix}_{i}")
                weights = self._masked_softmax(alphas[i], m, tau=1.0).detach()
                order = torch.argsort(weights)  # Ascending order.
                # Relative mode: cutoff is a fraction of the current group maximum
                # (masked entries softmax to 0, so max() is over active candidates).
                # threshold_abs is the delta floor of max(delta, rho * p_max).
                if mode == "relative":
                    effective_threshold = float(threshold) * float(weights.max())
                else:
                    effective_threshold = float(threshold)
                effective_threshold = max(float(threshold_abs), effective_threshold)
                for k in order.tolist():
                    if not bool(m[k]):
                        continue  # Already pruned.
                    if int(m.sum().item()) <= min_keep:
                        break
                    if float(weights[k]) >= effective_threshold:
                        break  # The remaining candidates are larger because order is ascending.
                    m[k] = False
                    events.append({
                        "type": "pruning",
                        "layer": i,
                        "group": group_name,
                        "index": int(k),
                        "name": str(name_list[k]),
                        "weight_at_removal": float(weights[k]),
                        "effective_threshold": effective_threshold,
                        "mode": mode,
                    })
        self._pruning_events.extend(events)
        return events

    def pruning_summary(self) -> dict:
        """Return pruned candidate names for every layer/group for manifests."""
        summary: dict[str, dict[str, list[str]]] = {}
        groups = [
            ("mask_op",   "ops",   list(self.ops)),
            ("mask_norm", "norms", list(NORM_LIST)),
            ("mask_act",  "acts",  list(ACT_LIST)),
        ]
        for i in range(self.num_layers):
            summary[f"layer_{i}"] = {}
            for prefix, key, names in groups:
                m = getattr(self, f"{prefix}_{i}")
                summary[f"layer_{i}"][key] = [
                    names[k] for k in range(len(names)) if not bool(m[k])
                ]
        return summary

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()

    def weight_parameters(self):
        """Return all GNN model weights, excluding architecture parameters."""
        return [p for n, p in self.named_parameters() if "alphas" not in n]

    def arch_parameters(self):
        """Return architecture parameters only: ops, normalization, and activation alphas."""
        return [p for n, p in self.named_parameters() if "alphas" in n]

    def set_alternating_phase(self, phase: str):
        if phase == 'train_weights':
            # Freeze alphas while training ordinary network weights.
            for p in self.arch_parameters():
                p.requires_grad = False
            for p in self.weight_parameters():
                p.requires_grad = True
        elif phase == 'train_arch':
            # Freeze ordinary network weights while training architecture parameters.
            for p in self.arch_parameters():
                p.requires_grad = True
            for p in self.weight_parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown phase: {phase}")

#
# Language models
#


class LMForSentenceEmbedding(ModelBase):
    def __init__(self,
        model: Union[str, torch.nn.Module],
        pooling_mode: str = 'cls',
        normalize: bool = True
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            self.model = tf.AutoModel.from_pretrained(model)

            # we don't use this and it causes unused-parameters errors on DDP
            if hasattr(self.model, 'pooler') and self.model.pooler is not None:
                self.model.pooler = None
        else:
            self.model = model

        self.pooling_mode = pooling_mode
        self.normalize = normalize

    def _get_pooler(self, pooling_mode: str) -> Callable:
        if self.pooling_mode == 'mean':
            return self.mean_pool
        elif self.pooling_mode == 'cls':
            return self.cls_pool
        elif self.pooling_mode == 'max':
            return self.max_pool
        else:
            raise ValueError('Unsupported pooling mode')

    # NOTE all three of these poolers pool over the second (sequence) dimension
    @staticmethod
    def mean_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(embeds.size())

        # mask.sum(dim=1) has the same dimension as sum(embeds * mask, dim=1)
        # so the division is elementwise, no broadcasting, no opportunity for
        # e.g. unexpected batch size to screw it up
        ret = torch.sum(embeds * mask, dim=1)
        ret /= torch.clamp(mask.sum(dim=1), min=1e-9)

        return ret

    @staticmethod
    def cls_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return embeds[:, 0, ...]

    @staticmethod
    def max_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(embeds.size())

        if copy:
            embeds = torch.clone(embeds)

        embeds[mask == 0] = -1e9  # never choose a padding token

        return torch.max(embeds, 1)[0]

    def sentence_embeddings(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: Optional[str] = None,
        normalize: Optional[bool] = None
    ) -> torch.Tensor:
        pooling_mode = ut.coalesce(pooling_mode, self.pooling_mode)
        normalize = ut.coalesce(normalize, self.normalize)

        embeds = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )['last_hidden_state']

        pooler = self._get_pooler(pooling_mode)
        embeds = pooler(embeds, attention_mask)

        if normalize:
            embeds = F.normalize(embeds, p=2, dim=1)

        return embeds

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.sentence_embeddings(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

    @property
    def config(self) -> Union[tf.PretrainedConfig, Any]:
        return self.model.config


class LMForPretrain(ModelBase):
    def __init__(self,
        model: Union[str, torch.nn.Module],
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert mode is None or mode in ('masked', 'causal')
        self.mode = mode

        if isinstance(model, str) and self.mode == 'masked':
            self.model = tf.AutoModelForMaskedLM.from_pretrained(model)
        elif isinstance(model, str):  # self.mode == 'causal'
            self.model = tf.AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )['logits']

    def to_sentence_embedding(self, **kwargs: Any) -> LMForSentenceEmbedding:
        return LMForSentenceEmbedding(self.model.base_model, **kwargs)

    @property
    def config(self) -> Union[tf.PretrainedConfig, Any]:
        return self.model.config

# configs/clip-graph/inductive-causal/pubmed/base.yaml#
#
# CLIP-graph model
#


class ProjectHead(ModelBase):
    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        self.linear1 = nn.Linear(input_dim, target_dim)
        self.linear2 = nn.Linear(target_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.norm(x)

        return x


class ClipGraph(ModelBase):
    def __init__(self,
        lm: ModelBase,
        gnn: ModelBase,

        embed_dim: int = 768,
        dropout: float = 0,
        tau_init: float = 2.7,
        max_tau: Optional[float] = None,

        debug_checks: bool = False
    ) -> None:
        super().__init__()

        self.lm = lm
        self.gnn = gnn

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.tau_init = tau_init
        self.max_tau = max_tau

        self.debug_checks = debug_checks

        self.lm_proj = ProjectHead(
            self.lm.config.hidden_size,
            self.embed_dim,
        )

        if self.gnn.out_channels is not None:
            gnn_out_channels = self.gnn.out_channels
        else:
            gnn_out_channels = self.gnn.hidden_channels

        self.gnn_proj = ProjectHead(
            gnn_out_channels,
            self.embed_dim,
        )

        self.register_parameter(
            name='tau',
            param=nn.Parameter(torch.tensor(self.tau_init, requires_grad=True))
        )

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if "alphas" not in n]

    def arch_parameters(self):
        return [p for n, p in self.named_parameters() if "alphas" in n]



    def embed_text(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.lm(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        if self.debug_checks:
            assert len(embeds.shape) == 2
            assert not torch.isinf(embeds).any().item()
            assert not torch.isnan(embeds).any().item()

        embeds = self.lm_proj(embeds)

        return embeds

    def embed_nodes(self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        is_search: bool = True,
    ) -> torch.Tensor:
        embeds = self.gnn(x, edge_index, edge_attr, is_search)
        if embeds['output'] is not None:
            embeds = embeds['output']
        else:
            embeds = embeds['last_hidden_state']

        if self.debug_checks:
            assert len(embeds.shape) == 2
            assert not torch.isinf(embeds).any().item()
            assert not torch.isnan(embeds).any().item()

        embeds = self.gnn_proj(embeds)

        return embeds

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                graph_x: torch.Tensor, graph_edge_index: torch.Tensor, graph_edge_attr: torch.Tensor,
                node_index: torch.Tensor, is_search: bool = True) -> torch.Tensor:
        if self.debug_checks:
            assert not torch.isinf(input_ids).any().item()
            assert not torch.isnan(input_ids).any().item()

            assert not torch.isinf(attention_mask).any().item()
            assert not torch.isnan(attention_mask).any().item()

            assert not torch.isinf(graph_x).any().item()
            assert not torch.isnan(graph_x).any().item()

            assert not torch.isinf(graph_edge_index).any().item()
            assert not torch.isnan(graph_edge_index).any().item()

            assert not torch.isinf(node_index).any().item()
            assert not torch.isnan(node_index).any().item()

        lm_embeds = self.embed_text(input_ids, attention_mask)
        lm_embeds = F.normalize(lm_embeds, p=2, dim=1)
        lm_embeds = self.dropout(lm_embeds)

        if self.debug_checks:
            assert not torch.isinf(lm_embeds).any().item()
            assert not torch.isnan(lm_embeds).any().item()

        gnn_embeds = self.embed_nodes(graph_x, graph_edge_index, graph_edge_attr, is_search)

        gnn_embeds = F.normalize(gnn_embeds, p=2, dim=1)
        gnn_embeds = gnn_embeds[node_index, ...]
        gnn_embeds = self.dropout(gnn_embeds)

        if self.debug_checks:
            assert not torch.isinf(gnn_embeds).any().item()
            assert not torch.isnan(gnn_embeds).any().item()

        temperature = self.tau.exp()
        if self.max_tau is not None:
            max_temp = torch.tensor(self.max_tau).exp().item()
            temperature = temperature.clamp(-max_temp, max_temp)

        ret = lm_embeds @ gnn_embeds.T * temperature

        if self.debug_checks:
            assert not torch.isinf(ret).any().item()
            assert not torch.isnan(ret).any().item()

        return ret
