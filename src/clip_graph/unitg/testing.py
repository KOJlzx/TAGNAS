"""Synthetic data and tiny models for exercising the UniTG baseline on CPU."""

from __future__ import annotations

import torch
from torch_geometric.utils import subgraph, to_undirected

from .data import SplitGraph, UniTGData
from .partition import degree_groups, node_degrees


def _split_graph(name, node_ids_global, global_rows, edge_index_full, num_nodes_full, x_full, thresholds, num_groups):
    ei, _ = subgraph(global_rows, edge_index_full, relabel_nodes=True, num_nodes=num_nodes_full)
    n = global_rows.numel()
    deg = node_degrees(ei, n)
    group, thr = degree_groups(deg, num_groups, thresholds)
    return SplitGraph(name=name, edge_index=ei, x=x_full[global_rows].clone(), node_ids=node_ids_global[global_rows].clone(),
                      rows=global_rows.clone(), degree=deg, group=group), thr


def synthetic_data(num_nodes: int = 150, in_channels: int = 16, seq_len: int = 8, vocab: int = 50,
                   num_groups: int = 4, seed: int = 0, p: float = 0.06) -> UniTGData:
    g = torch.Generator().manual_seed(seed)
    # Erdős–Rényi graph drawn from the *seeded* generator (never the global RNG), so every
    # spawned rank builds exactly the same graph.
    mask = torch.triu(torch.rand(num_nodes, num_nodes, generator=g) < p, diagonal=1)
    ei = to_undirected(mask.nonzero().t().contiguous())
    x = torch.randn(num_nodes, in_channels, generator=g)
    node_ids = torch.arange(num_nodes) * 7 + 1000          # arbitrary, non-contiguous global ids
    perm = torch.randperm(num_nodes, generator=g)
    n_tr, n_va = int(0.7 * num_nodes), int(0.1 * num_nodes)
    rows = {"train": perm[:n_tr].sort().values, "val": perm[n_tr:n_tr + n_va].sort().values,
            "test": perm[n_tr + n_va:].sort().values}
    train, thr = _split_graph("train", node_ids, rows["train"], ei, num_nodes, x, None, num_groups)
    val, _ = _split_graph("val", node_ids, rows["val"], ei, num_nodes, x, thr, num_groups)
    test, _ = _split_graph("test", node_ids, rows["test"], ei, num_nodes, x, thr, num_groups)
    group_full = torch.full((num_nodes,), num_groups - 1, dtype=torch.long)
    for sg in (train, val, test):
        group_full[sg.rows] = sg.group
    input_ids = torch.randint(1, vocab, (num_nodes, seq_len), generator=g).to(torch.int32)
    attention_mask = torch.ones(num_nodes, seq_len, dtype=torch.uint8)
    attention_mask[:, seq_len - 2:] = 0
    return UniTGData(
        dataset="synthetic", data_dir="/nonexistent", snapshot=None,
        model_args={"lm_name": "tiny-gpt2", "lr": 1e-3, "eps": 1e-8, "weight_decay": 0.0, "lm_pooling_mode": "mean",
                    "betas": [0.9, 0.999]},
        lm_name="tiny-gpt2", tokenizer_name="tiny-gpt2", in_channels=in_channels, full_node_ids=node_ids,
        train=train, val=val, test=test, input_ids=input_ids, attention_mask=attention_mask,
        group_full=group_full, group_thresholds=thr,
        has_text=torch.ones(group_full.numel(), dtype=torch.bool), meta={"synthetic": True},
    )


def synthetic_data_builder(config, rank, log):
    return synthetic_data(seed=config.seed, in_channels=config.embed_dim)


def tiny_gpt2_factory():
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(0)
    return GPT2Model(GPT2Config(n_layer=4, n_embd=16, n_head=2, vocab_size=64, n_positions=32))


def tiny_bert_factory():
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    return BertModel(BertConfig(num_hidden_layers=4, hidden_size=16, num_attention_heads=2,
                                intermediate_size=32, vocab_size=64, max_position_embeddings=32))
