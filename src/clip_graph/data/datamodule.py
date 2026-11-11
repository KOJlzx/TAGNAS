'''
The pytorch-lightning LightningDataModules encapsulating our datasets and their
splits, loaders, etc.
'''

from typing import List, Optional, Any, Union

import os
import pickle
import collections as cl
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.utils.data as td
import transformers as tf

import sentence_transformers as st

import pytorch_lightning as pl

from sklearn.decomposition import TruncatedSVD

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj, to_undirected

from .dataset import (
    BaseDataset,
    GraphDataset, TextDataset, GraphTextDataset,
    BatchGraphTextDataset
)

from ..utils import _train_val_test_split
import time

#
# Base data module functionality
#


class BaseDataModule(ABC, pl.LightningDataModule):
    def __init__(self,
        data_dir: str,
        batch_size: int = 32,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: int = 42,
        pin_memory: bool = True,
        num_workers: int = 0,
        device: Union[torch.device, str] = 'cuda',

        **kwargs: Any
    ) -> None:
        if isinstance(split_proportions, (list, tuple)):
            split_proportions = torch.tensor(split_proportions)
        assert torch.isclose(split_proportions.sum(), torch.tensor(1.0))

        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_proportions = split_proportions  # [train, val, test]
        self.seed = seed

        self.pin_memory = pin_memory
        self.num_workers = mp.cpu_count() if num_workers == -1 else num_workers
        self.device = device

        self._edgelist_path = os.path.join(self.data_dir, 'raw', 'graph.csv')
        self._text_path = os.path.join(self.data_dir, 'raw', 'texts.csv')
        self._node_data_path = os.path.join(self.data_dir, 'raw', 'node-data.csv')
        self._derived_path = os.path.join(self.data_dir, 'derived', 'node-properties.csv')

        self._split_nodes = None

    @abstractmethod
    def prepare_data(self) -> "BaseDataModule":
        raise NotImplementedError()

    @abstractmethod
    def setup(self) -> "BaseDataModule":
        raise NotImplementedError()

    def post_compute_split_nodes_hook(self) -> "BaseDataModule":
        pass

    def post_split_hook(self) -> "BaseDataModule":
        pass

    def _get_edgelist(self) -> pd.DataFrame:
        return pd.read_csv(self._edgelist_path, sep='\t')

    def _get_text(self) -> pd.DataFrame:
        ret = pd.read_csv(self._text_path, sep='\t')

        # A handful of rows can have content == NaN (43 rows in goodreads-crime).
        # Mixing a float NaN into an object-dtype array later trips up np.unique
        # with '<' not supported between float and str. Just drop the NaN rows —
        # the node_id is still preserved via its title row.
        if 'content' in ret.columns:
            ret = ret.dropna(subset=['content']).reset_index(drop=True)

        if 'id' not in ret.columns:
            ret['id'] = np.arange(ret.shape[0])

        return ret

    def _get_node_data(self) -> pd.DataFrame:
        node_data = pd.read_csv(self._node_data_path, sep='\t')

        if os.path.exists(self._derived_path):
            derived = pd.read_csv(self._derived_path, sep='\t')
            node_data = node_data.merge(derived, on='node_id', how='left')

        return node_data

    def _dataloader(self, dataset: BaseDataset, **kwargs: Any
                   ) -> td.DataLoader:
        print("in_dataloader", "-" * 50)
        return td.DataLoader(
            dataset,
            batch_size = 1,
            num_workers = 1,
            pin_memory = self.pin_memory,
            collate_fn = dataset.__collate__,  # *not* self.dataset
            **kwargs
        )

    def train_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False, **kwargs)

    def compute_split_nodes(self) -> "BaseDataModule":
        self._split_nodes = _train_val_test_split(
            self.dataset.unique_node_ids,
            self.split_proportions,
            self.seed,
        )

        return self

    def split(self) -> "BaseDataModule":
        if self._split_nodes is None:
            self.compute_split_nodes()

        print('in_split_compute_split_nodes')
        self.post_compute_split_nodes_hook()

        print('in_split_splits')
        splits = self.dataset.split(splits=self._split_nodes)
        self.train_dataset = splits['train']
        self.val_dataset = splits['val']
        self.test_dataset = splits['test']
    
        print('in_split_splits_train_dataset')
        self.post_split_hook()

        return self


#
# Text-only data module
#


class TextDataModule(BaseDataModule):
    def __init__(self,
        tokenizer_name: str,
        mlm: bool = False,
        mlm_probability: float = 0.15,

        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.tokenizer_name = tokenizer_name
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def prepare_data(self) -> "TextDataModule":
        return self  # nothing to do

    def setup(self, stage: Optional[str] = None) -> "TextDataModule":
        self.dataset = TextDataset(
            text = self._get_text(),
            tokenizer_name = self.tokenizer_name,
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            device = self.device,
        )

        self.split()

        return self


#
# Graph-only data module
#


class GraphDataModule(BaseDataModule):
    def __init__(self,
        drop_isolates: bool = True,
        directed: bool = False,

        **kwargs: Any
    ) -> None:
        if 'batch_size' in kwargs.keys() and kwargs['batch_size'] != 1:
            raise ValueError('batch_size must be 1 for GraphDataModule')
        elif 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = 1

        super().__init__(**kwargs)

        self.drop_isolates = drop_isolates
        self.directed = directed

        self._graph_data_cache_path = os.path.join(self.data_dir, 'graph_data.pkl')

    @abstractmethod
    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        raise NotImplementedError()

    def _ensure_graph_object(self) -> None:
        if os.path.exists(self._graph_data_cache_path):
            return

        graph_data = self.prep_graph(
            edgelist = self._get_edgelist(),
            node_data = self._get_node_data()
        )

        tmp_path = f"{self._graph_data_cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(graph_data, f)
        os.replace(tmp_path, self._graph_data_cache_path)

    def _get_graph_object(self) -> Data:
        print('in_get_graph_object')
        with open(self._graph_data_cache_path, 'rb') as f:
            ret = pickle.load(f)
        # print('in_get_graph_object_pickle_load')
        if not self.directed:
            if hasattr(ret, 'num_nodes'):
                num_nodes = ret.num_nodes
            elif hasattr(ret, 'x'):
                num_nodes = ret.x.shape[0]
            else:
                num_nodes = None

            ret.edge_index = to_undirected(ret.edge_index, num_nodes=num_nodes)

        return ret

    def prepare_data(self) -> "GraphDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, stage: Optional[str] = None) -> "BaseDataModule":
        self.dataset = GraphDataset(
            graph_data=self._get_graph_object(),
            drop_isolates=self.drop_isolates,
            device = self.device,
        )

        self.split()

        return self

    def count_edges(self) -> int:
        return self.dataset.graph_data.edge_index.shape[1]
#
# Graph+text data module
#


class GraphTextDataModule(TextDataModule, GraphDataModule):
    def __init__(self,
        random_data_debug: bool = False,
        max_texts_per_node: int = 0,
        transductive: bool = False,
        transductive_identity_features: bool = False,
        compute_sim_mutual: bool = True,

        **kwargs: Any
    ) -> None:
        assert not (not transductive and transductive_identity_features), \
            "transductive_identity_features requires transductive = True"

        batch_size = kwargs.pop('batch_size', 32)
        print('batch_size', batch_size, flush=True)
        super().__init__(**kwargs)

        self.max_texts_per_node = max_texts_per_node
        self.transductive = transductive
        self.transductive_identity_features = transductive_identity_features
        self.compute_sim_mutual = compute_sim_mutual

        self.random_data_debug = random_data_debug
        self._real_batch_size = batch_size

    def prepare_data(self) -> "GraphTextDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, node: int = 0, world_size: int = 1, k_hop: int = 1, include_self: bool = True, idx: Optional[int] = None, stage: Optional[str] = None) -> "GraphTextDataModule":
        self.dataset = GraphTextDataset(
            graph_data = self._get_graph_object(),
            drop_isolates = self.drop_isolates,
            text = self._get_text(),
            tokenizer_name = self.tokenizer_name,
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            random_data_debug = self.random_data_debug,
            max_texts_per_node = self.max_texts_per_node,
            transductive = self.transductive,
            transductive_identity_features = self.transductive_identity_features,
            device = self.device,
        )

        print("in_setup_graph_text_dataset_extract_subgraph_with_texts")
        total_samples = self.dataset.graph_data.node_ids.shape[0]
        samples_per_shard = total_samples // world_size
        start_idx = node * samples_per_shard
        end_idx = start_idx + samples_per_shard if node < world_size - 1 else total_samples
        node_mask = torch.zeros(total_samples, dtype=torch.bool)
        node_mask[start_idx:end_idx] = True


        self.dataset.extract_subgraph_with_texts(node_mask, k_hop, include_self)
        print("in_setup_graph_text_dataset")
        self.split()
        print("in_setup_graph_text_dataset_split")
        # wrap the datasets in batching logic
        datasets = ['train_dataset', 'val_dataset', 'test_dataset']
        for dataset_name in datasets:
            dataset = getattr(self, dataset_name)
            
            # Skip compute_mutuals once the node count exceeds this threshold, to avoid OOM.
            if self.compute_sim_mutual and dataset.graph_data.num_nodes <= 20000:
                dataset.compute_mutuals()
                print('in_setup_graph_text_dataset_compute_mutuals')
            else:
                reason = "disabled" if not self.compute_sim_mutual else f"large graph ({dataset.graph_data.num_nodes} nodes)"
                print(f'Skipping compute_mutuals: {reason}')
                # Leave this as a placeholder (None) instead of a zero matrix, to keep the interface consistent.

                dataset.graph_data.sim_mutual = None

            dataset = BatchGraphTextDataset(
                dataset,
                batch_size=self._real_batch_size,
                seed=self.seed
            )
            setattr(self, dataset_name, dataset)
        return self


#
# Data-specific prep code
#


class SVDMixin:
    @property
    def svd_vectors_name(self):
        raise NotImplementedError()

    def _svd_cache_path(self, svd_dim: int, all_nodes: torch.Tensor) -> str:
        name = f"{self.svd_vectors_name}_svd_seed{int(self.seed)}_dim{svd_dim}_nodes{int(all_nodes.numel())}.pt"
        return os.path.join(self.data_dir, "derived", name)

    def _load_cached_svd(self, cache_path: str, all_nodes: torch.Tensor) -> bool:
        if not os.path.isfile(cache_path):
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception as exc:
            print(f"[svd-cache] failed to load {cache_path}: {exc}")
            return False
        node_ids = payload.get("node_ids") if isinstance(payload, dict) else None
        vectors = payload.get("vectors") if isinstance(payload, dict) else None
        if not isinstance(node_ids, torch.Tensor) or not isinstance(vectors, torch.Tensor):
            print(f"[svd-cache] ignore malformed cache: {cache_path}")
            return False
        if not torch.equal(node_ids.cpu().long(), all_nodes.cpu().long()):
            print(f"[svd-cache] ignore stale cache with different node order: {cache_path}")
            return False
        setattr(self.dataset.graph_data, self.svd_vectors_name, vectors)
        print(f"[svd-cache] loaded {cache_path}")
        return True

    def _save_cached_svd(self, cache_path: str, all_nodes: torch.Tensor, vectors: torch.Tensor) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        torch.save({"node_ids": all_nodes.cpu(), "vectors": vectors.cpu()}, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"[svd-cache] saved {cache_path}")

    def _wait_for_cached_svd(self, cache_path: str, lock_path: str, all_nodes: torch.Tensor) -> "SVDMixin":
        timeout = float(os.environ.get("SVD_CACHE_TIMEOUT_SECONDS", "7200"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._load_cached_svd(cache_path, all_nodes):
                return self
            if not os.path.isdir(lock_path):
                break
            time.sleep(2.0)
        if self._load_cached_svd(cache_path, all_nodes):
            return self
        raise TimeoutError(f"Timed out waiting for SVD cache: {cache_path}")

    def post_compute_split_nodes_hook(self) -> "SVDMixin":
        svd_dim = 768
        all_nodes = self.dataset.graph_data.node_ids
        n_nodes = all_nodes.shape[0]
        cache_path = self._svd_cache_path(svd_dim, all_nodes)
        lock_path = f"{cache_path}.lock"

        if self._load_cached_svd(cache_path, all_nodes):
            return self
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            os.mkdir(lock_path)
            have_lock = True
        except FileExistsError:
            stale_after = float(os.environ.get("SVD_CACHE_STALE_LOCK_SECONDS", "3600"))
            lock_age = time.time() - os.path.getmtime(lock_path)
            if lock_age > stale_after:
                print(f"[svd-cache] removing stale lock ({lock_age:.0f}s): {lock_path}")
                os.rmdir(lock_path)
                os.mkdir(lock_path)
                have_lock = True
            else:
                have_lock = False
        if not have_lock:
            return self._wait_for_cached_svd(cache_path, lock_path, all_nodes)

        # Sparse adjacency: build a CSR matrix directly from edge_index, never
        # to_dense_adj. For a large graph (goodreads-crime, ~420K nodes), the
        # dense adjacency would be ~420K^2 x 4B ~= 714GB, which blows up memory.
        # TruncatedSVD natively supports sparse input with numerically equivalent
        # results; the explicit shape also stays compatible with isolated nodes.
        try:
            import scipy.sparse as sp
            ei = self.dataset.graph_data.edge_index.cpu().numpy()
            adj = sp.csr_matrix(
                (np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])),
                shape=(n_nodes, n_nodes),
            )

            train_nodes = self._split_nodes['train']
            train_mask = torch.isin(all_nodes, train_nodes)
            train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            train_adj = adj[train_idx][:, train_idx]

            # use of train columns is not an error! need same number of features
            # as for train because we want to project the validation and test
            # data points into the training data's space
            val_nodes = self._split_nodes['val']
            val_mask = torch.isin(all_nodes, val_nodes)
            val_idx = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            val_adj = adj[val_idx][:, train_idx]

            test_nodes = self._split_nodes['test']
            test_mask = torch.isin(all_nodes, test_nodes)
            test_idx = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            test_adj = adj[test_idx][:, train_idx]

            # FIXME should we use algorithm='arpack' to make this completely
            # deterministic?
            print("in_post_compute_split_nodes_hook_TruncatedSVD")
            mod = TruncatedSVD(svd_dim, algorithm='randomized').fit(train_adj)
            print("in_post_compute_split_nodes_hook_TruncatedSVD_fit")
            train_svd = torch.from_numpy(mod.transform(train_adj))
            val_svd = torch.from_numpy(mod.transform(val_adj))
            test_svd = torch.from_numpy(mod.transform(test_adj))

            svd = torch.cat([train_svd, val_svd, test_svd], dim=0)
            svd_nodes = torch.cat([
                all_nodes[train_mask],
                all_nodes[val_mask],
                all_nodes[test_mask]
            ], dim=0)

            # has to be same order as in the dataset object
            node_to_pos = {int(v): i for i, v in enumerate(svd_nodes.tolist())}
            svd_inds = torch.tensor([node_to_pos[int(v)] for v in all_nodes.tolist()], dtype=torch.long)
            vectors = svd[svd_inds, :]

            setattr(self.dataset.graph_data, self.svd_vectors_name, vectors)
            self._save_cached_svd(cache_path, all_nodes, vectors)
        finally:
            try:
                os.rmdir(lock_path)
            except OSError:
                pass

        return self


class PubmedDataMixin(SVDMixin):
    svd_vectors_name = 'x'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        # Some datasets (e.g. T-REx) only have node_id plus feature/multi-hot columns
        # and no supervised label; contrastive learning does not depend on this column.
        if 'label' not in node_data.columns:
            node_data = node_data.copy()
            node_data['label'] = 0

        tmp = node_data[['node_id', 'label']].set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)
        
        del ret.num_nodes

        return ret



#
# Put the data mixins and the other classes together
#

class PubmedGraphDataModule(PubmedDataMixin, GraphDataModule):
    pass


class PubmedGraphTextDataModule(PubmedDataMixin, GraphTextDataModule):
    pass


# --- OGB-Arxiv data module ---

class ArxivDataMixin:
    """
    Adapter for ogbn-arxiv data.
    - Nodes come with 128-dimensional word2vec features, used directly, no SVD needed.
    - Uses the official ogbn-arxiv train/val/test split rather than a random split.
    """

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelist does not include isolated nodes; fill them in from node_data.
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        # Set node attributes: node_ids, label, and the 128-dimensional feature x.
        tmp = node_data[['node_id', 'label']].set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)
        del ret.num_nodes

        # Write the 128-dimensional word2vec feature into ret.x.
        # node_data must include columns feat_0 through feat_127 (see the data prep script).
        feat_cols = [c for c in node_data.columns if c.startswith('feat_')]
        if feat_cols:
            node_data_indexed = node_data.set_index('node_id')
            # Align with the node order used by networkx.
            node_order = list(G.nodes())
            feats = node_data_indexed.loc[node_order, feat_cols].values
            ret.x = torch.from_numpy(feats).float()

        return ret

    def compute_split_nodes(self) -> "ArxivDataMixin":
        node_data = pd.read_csv(self._node_data_path, sep='\t', usecols=['node_id', 'split'])
        all_node_ids = self.dataset.graph_data.node_ids

        split_map = node_data.set_index('node_id')['split']

        def get_split_nodes(split_name):
            ids = split_map[split_map == split_name].index.tolist()
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            return ids_tensor[torch.isin(ids_tensor, all_node_ids)]

        self._split_nodes = {
            'train': get_split_nodes('train'),
            'val':   get_split_nodes('valid'),
            'test':  get_split_nodes('test'),
        }
        return self


class ArxivGraphDataModule(ArxivDataMixin, GraphDataModule):
    pass


class ArxivGraphTextDataModule(ArxivDataMixin, GraphTextDataModule):
    def __init__(self, **kwargs):
        kwargs['drop_isolates'] = False
        super().__init__(**kwargs)

    def setup(self, node=0, world_size=1, k_hop=1, include_self=True, 
              idx=None, stage=None):
        # Call the parent setup.
        super().setup(node=node, world_size=world_size, k_hop=k_hop,
                     include_self=include_self, idx=idx, stage=stage)
        return self
