"""Real GraphNAS / LM model helpers for async PubMed training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as tf

from .. import gassomodels as md


@dataclass(frozen=True)
class LayerGenotype:
    """Discrete architecture choice for one GraphNAS layer."""

    op: str
    norm: str
    act: str


@dataclass(frozen=True)
class GraphGenotype:
    """Discrete architecture exported from the searchable GraphNAS supernet."""

    layers: tuple[LayerGenotype, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [
                {"op": layer.op, "norm": layer.norm, "act": layer.act}
                for layer in self.layers
            ]
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphGenotype":
        layers = tuple(
            LayerGenotype(
                op=str(layer["op"]),
                norm=str(layer["norm"]),
                act=str(layer["act"]),
            )
            for layer in payload["layers"]
        )
        return cls(layers=layers)


def export_graph_genotype(gnn: md.GassoSpace) -> GraphGenotype:
    """Read the current soft architecture weights and export their argmax path."""

    # Search alpha values are continuous; export compresses each layer to argmax choices.
    layers = []
    for layer_index in range(gnn.num_layers):
        op_idx = int(torch.argmax(gnn.alphas_normal[layer_index]).item())
        norm_idx = int(torch.argmax(gnn.alphas_norm[layer_index]).item())
        act_idx = int(torch.argmax(gnn.alphas_act[layer_index]).item())
        layers.append(
            LayerGenotype(
                op=str(gnn.ops[op_idx]),
                norm=str(md.NORM_LIST[norm_idx]),
                act=str(md.ACT_LIST[act_idx]),
            )
        )
    return GraphGenotype(layers=tuple(layers))


def apply_graph_genotype(
    gnn: md.GassoSpace,
    genotype: GraphGenotype,
    *,
    freeze_architecture: bool = True,
) -> None:
    """Overwrite alpha tensors with a discrete architecture selection."""

    if len(genotype.layers) != gnn.num_layers:
        raise ValueError(
            f"Genotype has {len(genotype.layers)} layers, expected {gnn.num_layers}"
        )

    with torch.no_grad():
        for layer_index, layer in enumerate(genotype.layers):
            # This does not rebuild modules; one-hot alpha is equivalent to selecting one branch.
            op_idx = list(gnn.ops).index(layer.op)
            norm_idx = list(md.NORM_LIST).index(layer.norm)
            act_idx = list(md.ACT_LIST).index(layer.act)

            gnn.alphas_normal[layer_index].zero_()
            gnn.alphas_normal[layer_index][op_idx] = 1.0

            gnn.alphas_norm[layer_index].zero_()
            gnn.alphas_norm[layer_index][norm_idx] = 1.0

            gnn.alphas_act[layer_index].zero_()
            gnn.alphas_act[layer_index][act_idx] = 1.0

    if freeze_architecture:
        for parameter in gnn.arch_parameters():
            parameter.requires_grad = False


class GraphTower(nn.Module):
    """GraphNAS encoder plus projection head and trainable temperature."""

    def __init__(
        self,
        model_args: Mapping[str, Any],
        *,
        genotype: GraphGenotype | None = None,
    ) -> None:
        super().__init__()

        gnn_params = dict(model_args["gnn_params"])
        self.gnn = md.GassoSpace(**gnn_params)

        # embed_dim is the shared graph/text contrastive space, not necessarily GNN hidden size.
        self.embed_dim = int(model_args["embed_dim"])
        self.dropout = nn.Dropout(float(model_args.get("dropout", 0.0)))
        self.max_tau = model_args.get("max_tau")

        gnn_out_channels = self.gnn.out_channels or self.gnn.hidden_channels
        self.proj = md.ProjectHead(gnn_out_channels, self.embed_dim)
        self.tau = nn.Parameter(torch.tensor(float(model_args.get("tau_init", 3.5))))

        if genotype is not None:
            # Retrain loads the searched architecture after discretization.
            apply_graph_genotype(self.gnn, genotype, freeze_architecture=True)

    def set_search_phase(self, train_architecture: bool) -> None:
        """Switch GraphNAS between weight training and alpha training."""

        if train_architecture:
            # During architecture training, update only alpha and freeze projector/tau.
            self.gnn.set_alternating_phase("train_arch")
            for parameter in self.proj.parameters():
                parameter.requires_grad = False
            self.tau.requires_grad = False
            return

        # During weight training, freeze alpha and update graph encoder, projector, and tau.
        self.gnn.set_alternating_phase("train_weights")
        for parameter in self.proj.parameters():
            parameter.requires_grad = True
        self.tau.requires_grad = True

    def freeze_architecture(self) -> None:
        """Freeze GraphNAS alpha tensors for retraining with a fixed genotype."""

        for parameter in self.gnn.arch_parameters():
            parameter.requires_grad = False
        for parameter in self.gnn.weight_parameters():
            parameter.requires_grad = True
        for parameter in self.proj.parameters():
            parameter.requires_grad = True
        self.tau.requires_grad = True

    def graph_weight_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.gnn.weight_parameters())
        parameters.extend(self.proj.parameters())
        parameters.append(self.tau)
        return parameters

    def graph_arch_parameters(self) -> list[nn.Parameter]:
        return list(self.gnn.arch_parameters())

    def temperature(self) -> torch.Tensor:
        value = self.tau.exp()
        if self.max_tau is None:
            return value
        max_value = torch.tensor(float(self.max_tau), device=value.device).exp()
        return value.clamp(max=max_value)

    def encode(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_attr: torch.Tensor | None,
        target_index: torch.Tensor,
        *,
        is_search: bool,
        extra_index: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a k-hop subgraph and return embeddings for the requested nodes.

        With ``extra_index`` also return the (detached) embeddings of those subgraph rows -- the
        same forward's by-products, see ``PubmedRuntimeStore.exact_train_nodes``.
        """

        # GassoSpace expects per-layer edge_index / edge_attr, so replicate this subgraph.
        layers = self.gnn.num_layers
        edge_index = graph_edge_index
        edge_attr = graph_edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=edge_index.device, dtype=graph_x.dtype)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        self.gnn.adjs = (
            edge_index.unsqueeze(0).repeat(layers, 1, 1),
            edge_attr.unsqueeze(0).repeat(layers, 1, 1),
        )

        outputs = self.gnn(graph_x, edge_index, edge_attr, is_search=is_search)
        # GassoSpace may return output or last_hidden_state depending on mode; normalize it here.
        node_embeddings = outputs["output"] if outputs["output"] is not None else outputs["last_hidden_state"]
        node_embeddings = self.proj(node_embeddings)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)
        node_embeddings = self.dropout(node_embeddings)
        if extra_index is not None:
            return node_embeddings[target_index], node_embeddings[extra_index].detach()
        return node_embeddings[target_index]


class TextTower(nn.Module):
    """Sentence LM plus projection head used on the text workers."""

    def __init__(
        self,
        model_args: Mapping[str, Any],
        *,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = int(model_args["embed_dim"])
        self.dropout = nn.Dropout(float(model_args.get("dropout", 0.0)))
        # The text tower trains locally on this worker; cross-process exchange sends
        # final embeddings only, never autograd graphs.
        lm_model = tf.AutoModel.from_pretrained(
            str(model_args["lm_name"]),
            local_files_only=local_files_only,
        )
        if hasattr(lm_model, "pooler") and lm_model.pooler is not None:
            lm_model.pooler = None
        # The LM is only an encoder here: a causal LM's default use_cache=True makes every
        # training forward also build and hold a KV cache that nothing reads (2.6 GiB at 80 rows
        # x 945 tokens on distilgpt2). Outputs are identical either way.
        if hasattr(lm_model.config, "use_cache"):
            lm_model.config.use_cache = False
        # Optional activation checkpointing: recompute each transformer block in backward instead
        # of storing its activations (memory for compute; off unless asked for).
        if bool(model_args.get("lm_grad_checkpointing", False)):
            lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lm = md.LMForSentenceEmbedding(
            model=lm_model,
            pooling_mode=str(model_args.get("lm_pooling_mode", "mean")),
            normalize=bool(model_args.get("lm_normalize", True)),
        )
        # Projector maps LM hidden size into the shared contrastive space.
        self.proj = md.ProjectHead(self.lm.config.hidden_size, self.embed_dim)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_embeddings = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.proj(text_embeddings)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        text_embeddings = self.dropout(text_embeddings)
        return text_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # DDP installs backward hooks through __call__ -> forward, so forward delegates to encode.
        return self.encode(input_ids, attention_mask)


def clone_state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    """Create a CPU copy of a module state dict for checkpointing."""

    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def zero_gradients(parameters: Iterable[nn.Parameter]) -> None:
    """Detach and clear gradients on an arbitrary parameter list."""

    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter.grad.detach_()
        parameter.grad.zero_()
