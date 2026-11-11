"""PubMed runtime data access for async GraphNAS + LM training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import torch
from torch_geometric.utils import k_hop_subgraph

from ..data.gassodatamodule import PubmedGraphTextDataModule
from ..data.gassodataset import GraphTextDataset
from ..utils import _train_val_test_split


@dataclass
class GraphBatch:
    """One graph-side training batch built from a list of target node ids."""

    target_node_ids: torch.Tensor
    target_index: torch.Tensor
    graph_node_ids: torch.Tensor
    graph_x: torch.Tensor
    graph_edge_index: torch.Tensor
    graph_edge_attr: torch.Tensor
    # Global graph indices of the subgraph's nodes (sorted ascending, as k_hop_subgraph returns
    # them); graph_x row i is node subset_index[i]. Used to find by-product embeddings to push.
    subset_index: torch.Tensor | None = None


@dataclass
class TextBatch:
    """One text-side training batch aligned to a node-id list."""

    text_node_ids: torch.Tensor
    text: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class PubmedRuntimeStore:
    """Holds the full PubMed graph, texts, tokenizer, and split metadata."""

    def __init__(
        self,
        dataset: GraphTextDataset,
        split_nodes: dict[str, list[int]],
        train_node_fraction: float | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.split_nodes_map = split_nodes
        self.graph_data = dataset.graph_data
        self.node_id_to_graph_index = {
            int(node_id): index
            for index, node_id in enumerate(self.graph_data.node_ids.tolist())
        }
        self.text_index_by_node: dict[int, int] = {}
        for index, node_id in enumerate(self.dataset.text_node_ids.tolist()):
            self.text_index_by_node.setdefault(int(node_id), index)
        # ``node_ids`` is the legal node universe for training and scheduling.
        # Keep only graph-text intersections so every scheduled node has both modalities.
        self.node_ids = [
            int(node_id)
            for node_id in self.aligned_node_ids(self.graph_data.node_ids, self.dataset.text_node_ids).tolist()
        ]
        # Inductive split: each split keeps only edges whose endpoints both
        # belong to that split, preventing k-hop training subgraphs from
        # traversing validation or test nodes.
        self._split_edge_index, self._split_edge_attr = self._build_split_edges()

        # Optional training seed subset: the scheduler samples batches only from
        # this subset, so only these nodes are pushed into the cache. This reduces
        # cache pressure and drop rates while preserving the full train-induced
        # graph for each seed's k-hop receptive field.
        #
        # Sampling is deterministic over sorted train nodes. All processes use
        # the same seed, so graph and text schedulers draw from the same node
        # universe and can hit each other's pushed cache entries. ``None`` or
        # fractions >= 1.0 keep the full train split.
        full_train = list(self.split_nodes_map.get("train", []))
        if train_node_fraction is not None and 0.0 < train_node_fraction < 1.0 and full_train:
            pop = sorted(full_train)
            keep = max(1, int(len(pop) * train_node_fraction))
            self.train_seed_nodes = random.Random(seed).sample(pop, keep)
        else:
            self.train_seed_nodes = full_train

    @staticmethod
    def aligned_node_ids(graph_node_ids: torch.Tensor, text_node_ids: torch.Tensor) -> torch.Tensor:
        """Return sorted node ids that are present in both the graph and text tables."""

        # This is the legal node universe for all splits, schedulers, and cache keys.
        graph_nodes = torch.unique(graph_node_ids.long(), sorted=True)
        text_nodes = torch.unique(text_node_ids.long(), sorted=True)
        return torch.tensor(
            sorted(set(graph_nodes.tolist()) & set(text_nodes.tolist())),
            dtype=torch.long,
        )

    @classmethod
    def build(
        cls,
        *,
        data_args: Mapping[str, Any],
        split_proportions: tuple[float, float, float],
        seed: int,
        local_files_only: bool | None = None,
        feature_dim: int | None = None,
        train_node_fraction: float | None = None,
    ) -> "PubmedRuntimeStore":
        init_args = dict(data_args)
        init_args["batch_size"] = 1
        init_args["num_workers"] = 0
        init_args["pin_memory"] = False

        datamodule = PubmedGraphTextDataModule(**init_args)
        datamodule.prepare_data()
        split_tag = "-".join(str(int(part * 100)) for part in split_proportions)
        cache_dir = Path(init_args["data_dir"]).expanduser().resolve() / "derived"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Version the runtime cache separately to avoid stale split or feature
        # generation logic contaminating current runs.
        cache_path = cache_dir / f"async_runtime_cache_v3_seed{seed}_split{split_tag}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            graph_data = cached["graph_data"]
            split_nodes = cached["split_nodes"]
            dataset = GraphTextDataset(
                graph_data=graph_data,
                drop_isolates=datamodule.drop_isolates,
                text=datamodule._get_text(),
                tokenizer_name=datamodule.tokenizer_name,
                local_files_only=local_files_only,
                mlm=datamodule.mlm,
                mlm_probability=datamodule.mlm_probability,
                random_data_debug=datamodule.random_data_debug,
                max_texts_per_node=datamodule.max_texts_per_node,
                transductive=datamodule.transductive,
                transductive_identity_features=datamodule.transductive_identity_features,
                device="cpu",
            )
            # Sanitize splits again even when a cache file exists. Older cache
            # versions or changed graph-text coverage should not silently reintroduce
            # illegal nodes into training.
            split_nodes = cls._sanitize_split_nodes(dataset, split_nodes, split_proportions, seed)
            cls._ensure_full_graph_features(
                dataset,
                torch.tensor(split_nodes["train"], dtype=torch.long),
                feature_dim=feature_dim,
            )
            cls._normalize_graph_features(dataset, torch.tensor(split_nodes["train"], dtype=torch.long))
            return cls(dataset=dataset, split_nodes=split_nodes,
                       train_node_fraction=train_node_fraction, seed=seed)

        dataset = GraphTextDataset(
            graph_data=datamodule._get_graph_object(),
            drop_isolates=datamodule.drop_isolates,
            text=datamodule._get_text(),
            tokenizer_name=datamodule.tokenizer_name,
            local_files_only=local_files_only,
            mlm=datamodule.mlm,
            mlm_probability=datamodule.mlm_probability,
            random_data_debug=datamodule.random_data_debug,
            max_texts_per_node=datamodule.max_texts_per_node,
            transductive=datamodule.transductive,
            transductive_identity_features=datamodule.transductive_identity_features,
            device="cpu",
        )
        aligned_node_ids = cls.aligned_node_ids(dataset.graph_data.node_ids, dataset.text_node_ids)
        if aligned_node_ids.numel() == 0:
            raise ValueError("PubMed runtime store found no node ids shared by graph and text data")

        datamodule.dataset = dataset
        # The async runtime owns the split construction instead of relying on
        # the older datamodule setup path.
        datamodule._split_nodes = _train_val_test_split(
            aligned_node_ids,
            torch.tensor(split_proportions),
            seed,
        )
        # Ensure every graph node has a feature vector before normalization.
        cls._ensure_full_graph_features(
            dataset,
            datamodule._split_nodes["train"],
            feature_dim=feature_dim,
        )
        cls._normalize_graph_features(dataset, datamodule._split_nodes["train"])
        cls._ensure_edge_attributes(dataset)

        split_nodes = {
            split: [int(node_id) for node_id in nodes.tolist()]
            for split, nodes in datamodule._split_nodes.items()
        }
        torch.save(
            {
                "graph_data": dataset.graph_data,
                "split_nodes": split_nodes,
            },
            cache_path,
        )
        return cls(dataset=dataset, split_nodes=split_nodes,
                   train_node_fraction=train_node_fraction, seed=seed)

    @classmethod
    def _sanitize_split_nodes(
        cls,
        dataset: GraphTextDataset,
        split_nodes: dict[str, list[int]],
        split_proportions: tuple[float, float, float],
        seed: int,
    ) -> dict[str, list[int]]:
        aligned = cls.aligned_node_ids(dataset.graph_data.node_ids, dataset.text_node_ids)
        allowed = set(int(node_id) for node_id in aligned.tolist())
        # Reuse cached splits when possible. If coverage is incomplete, rebuild
        # the splits from the aligned node universe.
        requested = {
            split: [int(node_id) for node_id in node_ids if int(node_id) in allowed]
            for split, node_ids in split_nodes.items()
        }
        recovered = sum(len(node_ids) for node_ids in requested.values())
        if recovered == len(allowed):
            return requested

        rebuilt = _train_val_test_split(aligned, torch.tensor(split_proportions), seed)
        return {
            split: [int(node_id) for node_id in node_ids.tolist()]
            for split, node_ids in rebuilt.items()
        }

    @staticmethod
    def _normalize_graph_features(dataset: GraphTextDataset, train_nodes: torch.Tensor) -> None:
        if not hasattr(dataset.graph_data, "x") or dataset.graph_data.x is None:
            raise ValueError("PubMed runtime store expected graph_data.x to be populated")

        # Normalize with train-node statistics only to avoid leaking validation
        # or test information back into training.
        train_mask = torch.isin(dataset.graph_data.node_ids, train_nodes)
        train_x = dataset.graph_data.x[train_mask].float()
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0)
        std[std == 0] = 1.0
        dataset.graph_data.x = (dataset.graph_data.x.float() - mean) / std

    @staticmethod
    def _ensure_full_graph_features(
        dataset: GraphTextDataset,
        train_nodes: torch.Tensor,
        *,
        feature_dim: int | None,
    ) -> None:
        current_x = getattr(dataset.graph_data, "x", None)
        node_count = int(dataset.graph_data.node_ids.size(0))
        if current_x is not None and current_x.dim() == 2 and current_x.size(0) == node_count:
            # The cache already stores full-graph features; reuse them directly.
            dataset.graph_data.x = current_x.float()
            return

        if feature_dim is None:
            if current_x is None or current_x.dim() != 2:
                raise ValueError("feature_dim is required when graph_data.x is missing or partial")
            feature_dim = int(current_x.size(1))

        all_nodes = dataset.graph_data.node_ids.long()
        train_mask = torch.isin(all_nodes, train_nodes.long())
        if not bool(train_mask.any()):
            raise ValueError("Cannot build graph features without at least one training node")

        # Learn the feature basis from the train-by-train sub-adjacency, then
        # project the full N-by-|train| matrix into that basis. Keep the entire
        # pipeline in scipy sparse format to avoid materializing an N-by-N dense
        # matrix, which is infeasible for large graphs.
        edge_index_np = dataset.graph_data.edge_index.cpu().numpy()
        rows = edge_index_np[0]
        cols = edge_index_np[1]
        data = np.ones(rows.shape[0], dtype=np.float32)
        adjacency_sp = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(node_count, node_count),
        ).tocsr()

        train_mask_np = train_mask.cpu().numpy()
        train_idx_np = np.flatnonzero(train_mask_np)

        train_projection_sp = adjacency_sp[:, train_idx_np]                # [N, |train|] sparse
        train_square_sp = train_projection_sp[train_idx_np, :]             # [|train|, |train|] sparse

        max_components = min(train_square_sp.shape[0], train_square_sp.shape[1]) - 1
        if max_components <= 0:
            raise ValueError("Randomized SVD requires at least two training nodes")

        effective_dim = min(int(feature_dim), max_components)
        # Fit randomized SVD on the train-by-train sparse matrix to obtain
        # V with shape |train| by k.
        _, _, vt = randomized_svd(train_square_sp, n_components=effective_dim, random_state=0)
        # Project the full graph matrix [N, |train|] onto the train basis:
        # features = train_projection @ V.
        features_np = train_projection_sp @ vt.T                            # [N, k] dense with small k
        features = torch.from_numpy(np.asarray(features_np, dtype=np.float32))
        if effective_dim < int(feature_dim):
            padding = torch.zeros((features.size(0), int(feature_dim) - effective_dim), dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)
        dataset.graph_data.x = features

    @staticmethod
    def _ensure_edge_attributes(dataset: GraphTextDataset) -> None:
        edge_attr = getattr(dataset.graph_data, "edge_attr", None)
        if edge_attr is None:
            dataset.graph_data.edge_attr = torch.zeros(
                (dataset.graph_data.edge_index.size(1), 1),
                dtype=torch.float32,
            )
            return
        if edge_attr.dim() == 1:
            dataset.graph_data.edge_attr = edge_attr.unsqueeze(-1).float()
        else:
            dataset.graph_data.edge_attr = edge_attr.float()

    def split_nodes(self, split: str) -> list[int]:
        if split not in self.split_nodes_map:
            raise KeyError(f"Unknown split: {split}")
        return list(self.split_nodes_map[split])

    def scheduler_train_nodes(self) -> list[int]:
        """Return the train seed nodes actually used by the scheduler.

        By default this is the full train split. When ``train_node_fraction`` is
        active, it becomes a deterministic random subset that is identical across
        processes because they share the same seed. This only affects training;
        validation and test still use ``split_nodes('val')`` and ``split_nodes('test')``.
        """
        return list(self.train_seed_nodes)

    def split_edge_index(self, split: str) -> torch.Tensor:
        """Return the induced edge_index for a given split (both endpoints in split)."""
        if split not in self._split_edge_index:
            raise KeyError(f"Unknown split: {split}")
        return self._split_edge_index[split]

    def split_edge_attr(self, split: str) -> torch.Tensor:
        """Return the edge_attr corresponding to ``split_edge_index(split)``."""
        if split not in self._split_edge_attr:
            raise KeyError(f"Unknown split: {split}")
        return self._split_edge_attr[split]

    def _build_split_edges(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute per-split induced edge indices.

        For inductive node splits, each split contains only original graph edges
        whose endpoints both belong to that split. Training k-hop subgraphs then
        traverse train nodes only and avoid leaking validation/test structure.
        """
        edge_index = self.graph_data.edge_index  # [2, E]
        edge_attr = self.graph_data.edge_attr     # [E, D]
        node_ids = self.graph_data.node_ids       # [N]

        split_ei: dict[str, torch.Tensor] = {}
        split_ea: dict[str, torch.Tensor] = {}

        for split, node_list in self.split_nodes_map.items():
            # Convert external node ids to contiguous graph indices.
            split_indices = torch.tensor(
                [self.node_id_to_graph_index[int(nid)] for nid in node_list],
                dtype=torch.long,
            )
            # Build a graph-index mask for nodes that belong to this split.
            in_split = torch.zeros(node_ids.size(0), dtype=torch.bool)
            in_split[split_indices] = True
            # Keep edges whose endpoints are both inside the split.
            mask = in_split[edge_index[0]] & in_split[edge_index[1]]
            split_ei[split] = edge_index[:, mask]
            split_ea[split] = edge_attr[mask]

        return split_ei, split_ea

    def build_graph_batch(
        self, node_ids: list[int], k_hop: int, *, split: str = "train",
    ) -> GraphBatch:
        target_ids = [int(node_id) for node_id in node_ids]
        global_indices = torch.tensor(
            [self.node_id_to_graph_index[node_id] for node_id in target_ids],
            dtype=torch.long,
        )
        # Inductive mode samples k-hop subgraphs from the split-induced graph,
        # ensuring training never traverses validation/test edges.
        split_ei = self._split_edge_index[split]
        split_ea = self._split_edge_attr[split]
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            global_indices,
            k_hop,
            split_ei,
            relabel_nodes=True,
            num_nodes=self.graph_data.node_ids.size(0),
        )
        edge_attr = split_ea[edge_mask]
        return GraphBatch(
            target_node_ids=torch.tensor(target_ids, dtype=torch.long),
            target_index=mapping.long(),
            graph_node_ids=self.graph_data.node_ids[subset].long(),
            graph_x=self.graph_data.x[subset].float(),
            graph_edge_index=edge_index.long(),
            graph_edge_attr=edge_attr.float(),
            subset_index=subset,
        )

    def exact_train_nodes(self, batch: GraphBatch, k_hop: int, *, split: str = "train") -> torch.Tensor:
        """Local (subgraph) indices of training nodes whose embedding in this batch is as good as a seed's.

        The graph forward computes an embedding for every node of the k-hop subgraph, but only
        the seeds are guaranteed their whole receptive field. A non-seed node v is accepted when
        the nearest node missing from the subgraph is at least ``max(k_hop, 2)`` hops upstream of
        it. Measured against a forward over the whole training graph with the same weights
        (results/text_mem_gap/truncation_error.py, 3 layers): cosine p1 of the accepted nodes is
        0.994-0.999 on reddit / amazon-baby / citation -- the same as the seeds' own (0.995-0.999;
        the supernet has subgraph-level operators, so even a seed's embedding depends on the batch)
        -- while nodes 1-2 hops from the boundary fall to p1 0.71-0.97 and are left out. On large
        graphs the 3-hop subgraph of 1024 seeds is most of the training graph, so this accepts
        40K (reddit) to 118K (amazon-baby) nodes per batch. Seeds themselves are excluded.
        """
        if batch.subset_index is None:
            return torch.empty(0, dtype=torch.long)
        num_nodes = self.graph_data.node_ids.size(0)
        split_ei = self._split_edge_index[split]
        inside = torch.zeros(num_nodes, dtype=torch.bool)
        inside[batch.subset_index] = True
        tainted = ~inside
        src, dst = split_ei[0], split_ei[1]
        for _ in range(max(int(k_hop), 2) - 1):
            reached = torch.zeros_like(tainted)
            reached[dst[tainted[src]]] = True
            tainted = tainted | reached
        exact = inside & ~tainted & self._train_seed_mask()
        exact[batch.subset_index[batch.target_index]] = False
        exact_global = exact.nonzero(as_tuple=True)[0]
        return torch.searchsorted(batch.subset_index, exact_global)

    def _train_seed_mask(self) -> torch.Tensor:
        mask = getattr(self, "_train_seed_mask_cache", None)
        if mask is None:
            mask = torch.zeros(self.graph_data.node_ids.size(0), dtype=torch.bool)
            idx = [self.node_id_to_graph_index[int(n)] for n in self.train_seed_nodes
                   if int(n) in self.node_id_to_graph_index]
            mask[torch.tensor(idx, dtype=torch.long)] = True
            self._train_seed_mask_cache = mask
        return mask

    def text_token_lengths(self, node_ids: list[int], chunk: int = 2048) -> dict[int, int]:
        """Token count (after the tokenizer's cap) of each node's training text, as build_text_batch feeds it."""
        out: dict[int, int] = {}
        ids = [int(n) for n in node_ids]
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            lens = self.build_text_batch(part).attention_mask.sum(dim=1).tolist()
            out.update(zip(part, (int(v) for v in lens)))
        return out

    def build_text_batch(self, node_ids: list[int]) -> TextBatch:
        texts: list[str] = []
        ordered_ids: list[int] = []
        for node_id in node_ids:
            node_id = int(node_id)
            if node_id not in self.text_index_by_node:
                raise KeyError(f"Node {node_id} has no text in the PubMed runtime store")
            # If a node has multiple texts, use the first one indexed for that node.
            sample_index = self.text_index_by_node[node_id]
            texts.append(str(self.dataset.text[sample_index]))
            ordered_ids.append(node_id)

        # Reuse GraphTextDataset's tokenizer and collator so runtime batches
        # remain behaviorally consistent with the repository's original data path.
        encoded = self.dataset.tokenize(texts)
        encoded = self.dataset.collate_targets(encoded)
        return TextBatch(
            text_node_ids=torch.tensor(ordered_ids, dtype=torch.long),
            text=texts,
            input_ids=encoded["input_ids"].long(),
            attention_mask=encoded["attention_mask"].long(),
        )
