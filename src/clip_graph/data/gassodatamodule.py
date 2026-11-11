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
from torch_geometric.utils import from_networkx, to_undirected
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from .gassodataset import (
    GraphDataset, GraphTextDataset,
    BatchGraphTextDataset
)

from .datamodule import BaseDataModule, TextDataModule
from ..utils import _train_val_test_split
import time

#
# Base data module functionality
#


#
# Text-only data module
#




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
        # print('in_get_graph_object')
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

        # print("in_setup_graph_text_dataset_extract_subgraph_with_texts")
        total_samples = self.dataset.graph_data.node_ids.shape[0]
        samples_per_shard = total_samples // world_size
        start_idx = node * samples_per_shard
        # Normalize node and world_size before using them in split arithmetic.
        node_int = int(node) if node is not None else 0
        world_size_int = int(world_size) if world_size is not None else 1

        end_idx = start_idx + samples_per_shard if node_int < world_size_int - 1 else total_samples
                
        node_mask = torch.zeros(total_samples, dtype=torch.bool)
        node_mask[start_idx:end_idx] = True


        self.dataset.extract_subgraph_with_texts(node_mask, k_hop, include_self)
        # print("in_setup_graph_text_dataset")
        self.split()
        # print("in_setup_graph_text_dataset_split")
        # wrap the datasets in batching logic
        datasets = ['train_dataset', 'val_dataset', 'test_dataset']
        for dataset_name in datasets:
            dataset = getattr(self, dataset_name)
            if self.compute_sim_mutual:
                dataset.compute_mutuals()
                # print('in_setup_graph_text_dataset_compute_mutuals')
            else:
                dataset.graph_data.sim_mutual = None
            dataset = BatchGraphTextDataset(
                dataset,
                batch_size = self._real_batch_size,
                seed = self.seed
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
        # The old to_dense_adj(...) path allocated N x N x 4 bytes. At 422K nodes,
        # that is about 715 GB, so this implementation stays in scipy sparse
        # matrices and uses randomized_svd throughout.
        all_nodes = self.dataset.graph_data.node_ids
        node_count = int(all_nodes.size(0))
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

        try:
            edge_index_np = self.dataset.graph_data.edge_index.cpu().numpy()
            rows = edge_index_np[0]
            cols = edge_index_np[1]
            data = np.ones(rows.shape[0], dtype=np.float32)
            adj_sp = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(node_count, node_count),
            ).tocsr()

            train_nodes = self._split_nodes['train']
            train_mask = torch.isin(all_nodes, train_nodes)
            train_idx = np.flatnonzero(train_mask.cpu().numpy())
            # Keep the column space fixed to train_idx: every split is projected
            # into the feature space defined by training nodes.
            train_adj = adj_sp[train_idx, :][:, train_idx]

            val_nodes = self._split_nodes['val']
            val_mask = torch.isin(all_nodes, val_nodes)
            val_idx = np.flatnonzero(val_mask.cpu().numpy())
            val_adj = adj_sp[val_idx, :][:, train_idx]

            test_nodes = self._split_nodes['test']
            test_mask = torch.isin(all_nodes, test_nodes)
            test_idx = np.flatnonzero(test_mask.cpu().numpy())
            test_adj = adj_sp[test_idx, :][:, train_idx]

            # Fit randomized SVD on the train-by-train adjacency block to learn
            # V (|train| x k), then right-multiply train/val/test projection
            # matrices by V to obtain k-dimensional structural features.
            effective_dim = min(int(svd_dim), min(train_adj.shape) - 1)
            if effective_dim <= 0:
                raise ValueError("Randomized SVD requires at least two training nodes")
            _, _, vt = randomized_svd(train_adj, n_components=effective_dim, random_state=0)
            v_t = vt.T  # [|train|, k]
            train_svd = torch.from_numpy(np.asarray(train_adj @ v_t, dtype=np.float32))
            val_svd = torch.from_numpy(np.asarray(val_adj @ v_t, dtype=np.float32))
            test_svd = torch.from_numpy(np.asarray(test_adj @ v_t, dtype=np.float32))
            if effective_dim < int(svd_dim):
                pad_train = torch.zeros((train_svd.size(0), int(svd_dim) - effective_dim), dtype=train_svd.dtype)
                pad_val   = torch.zeros((val_svd.size(0),   int(svd_dim) - effective_dim), dtype=val_svd.dtype)
                pad_test  = torch.zeros((test_svd.size(0),  int(svd_dim) - effective_dim), dtype=test_svd.dtype)
                train_svd = torch.cat([train_svd, pad_train], dim=1)
                val_svd   = torch.cat([val_svd,   pad_val],   dim=1)
                test_svd  = torch.cat([test_svd,  pad_test],  dim=1)

            svd = torch.cat([train_svd, val_svd, test_svd], dim=0)
            svd_nodes = torch.cat([
                all_nodes[train_mask],
                all_nodes[val_mask],
                all_nodes[test_mask]
            ], dim=0)

            # The old (all_nodes[:, None] == svd_nodes).nonzero() path built an
            # N x N boolean matrix (about 178 GB at 422K nodes). A dict mapping
            # node id to SVD row position gives the same result with linear memory.
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



# ─────────────────────────────────────────────
# OGB dataset adapters
# ─────────────────────────────────────────────
from ogb.nodeproppred import NodePropPredDataset

class OGBDataMixin:
    """
    Adapter for ogb nodeproppred datasets such as ogbn-arxiv and ogbn-products.
    These datasets provide node features directly, so SVDMixin is not required.
    
    prepare_data / _ensure_graph_object caches the graph as graph_data.pkl, matching
    the PubMed data flow so downstream setup() code does not need special cases.
    """

    # Subclasses can override this dataset name.
    ogb_dataset_name: str = "ogbn-arxiv"

    def prep_graph(self,
        edgelist: pd.DataFrame,   # Unused: OGB provides edge_index directly.
        node_data: pd.DataFrame   # Unused: OGB provides node_feat directly.
    ) -> Data:
        raise RuntimeError("OGBDataMixin.prep_graph should not be called; "
                           "please confirm _ensure_graph_object has been overridden correctly.")

    def _ensure_graph_object(self) -> None:
        """Build PyG Data directly from OGB and skip edgelist/node_data files."""
        if os.path.exists(self._graph_data_cache_path):
            return

        print(f"[OGBDataMixin] Downloading/loading {self.ogb_dataset_name} ...")
        dataset = NodePropPredDataset(
            name=self.ogb_dataset_name,
            root=self.data_dir
        )
        graph, labels = dataset[0]           # graph: dict, labels: np.ndarray (N,1)
        split_idx = dataset.get_idx_split()  # {'train': ..., 'valid': ..., 'test': ...}

        # Node features.
        x = torch.from_numpy(graph['node_feat']).float()  # (N, feat_dim)
        num_nodes = x.shape[0]

        # Edges. OGB stores edge_index as (2, E), dtype int64.
        edge_index = torch.from_numpy(graph['edge_index']).long()

        # Labels. Convert labels from shape (N, 1) to (N,).
        y = torch.from_numpy(labels.squeeze()).long()

        # Node ids. Use contiguous integer ids to match the PubMed-style pipeline.
        node_ids = torch.arange(num_nodes, dtype=torch.long)

        # Train/validation/test masks.
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']]   = True
        test_mask[split_idx['test']]   = True

        data = Data(
            x          = x,
            edge_index = edge_index,
            y          = y,
            node_ids   = node_ids,
            train_mask = train_mask,
            val_mask   = val_mask,
            test_mask  = test_mask,
            num_nodes  = num_nodes,
        )

        with open(self._graph_data_cache_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"[OGBDataMixin] Graph cache written to {self._graph_data_cache_path}")

    def _get_text(self):
        text_path = os.path.join(
            self.data_dir,
            self.ogb_dataset_name.replace("-", "_"),
            "titleabs.tsv.gz"
        )
        mapping_path = os.path.join(
            self.data_dir,
            self.ogb_dataset_name.replace("-", "_"),
            "mapping", "nodeidx2paperid.csv.gz"
        )

        # Read the node_idx -> paper_id mapping, then invert it to paper_id -> node_idx.
        mapping = pd.read_csv(mapping_path)
        mapping.columns = ['node_idx', 'paper_id']
        paperid2nodeidx = dict(zip(mapping['paper_id'], mapping['node_idx']))

        print(f"[OGBDataMixin] Reading text from {text_path}")
        df = pd.read_csv(
            text_path,
            sep='\t',
            skiprows=1,
            header=None,
            names=['paper_id', 'title', 'abstract'],
            compression='gzip'
        )
        df = df.dropna(subset=['paper_id'])
        df['paper_id'] = df['paper_id'].astype(float).astype(np.int64)

        # Map paper_id values to OGB node indices.
        df['node_id'] = df['paper_id'].map(paperid2nodeidx)
        df = df.dropna(subset=['node_id'])           # Drop rows without a matching graph node.
        df['node_id'] = df['node_id'].astype(np.int64)

        df['node_id'] = df['node_id']                # OGB node index.
        df['id'] = df['node_id']                     # GraphTextDataset expects an id column.
        df['content'] = df['title'].fillna('') + '. ' + df['abstract'].fillna('')

        print(f"[OGBDataMixin] Text rows: {len(df)}, node coverage: {df['node_id'].nunique()}")
        return df[['node_id', 'id', 'content']]

    def compute_split_nodes(self) -> "OGBDataMixin":
        dataset = NodePropPredDataset(name=self.ogb_dataset_name, root=self.data_dir)
        split_idx = dataset.get_idx_split()
        
        # Parent classes expect {'train': tensor, 'val': tensor, 'test': tensor}.
        # OGB uses 'valid', so convert it to 'val' here.
        self._split_nodes = {
            'train': torch.from_numpy(split_idx['train']).squeeze().long(),
            'val':   torch.from_numpy(split_idx['valid']).squeeze().long(),
            'test':  torch.from_numpy(split_idx['test']).squeeze().long(),
        }
        return self
class ArxivDataMixin:
    """
    Adapter for ogbn-arxiv data exported into the repository raw/*.csv format.
    - Uses the provided 128-dimensional word2vec node features directly.
    - Reads raw/graph.csv, raw/texts.csv, and raw/node-data.csv.
    - Uses the official split stored in node-data.csv instead of online OGB APIs.
    """

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        tmp = node_data[['node_id', 'label']].set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)
        del ret.num_nodes

        feat_cols = [c for c in node_data.columns if c.startswith('feat_')]
        if feat_cols:
            node_data_indexed = node_data.set_index('node_id')
            node_order = list(G.nodes())
            feats = node_data_indexed.loc[node_order, feat_cols].values
            ret.x = torch.from_numpy(feats).float()

        return ret

    def compute_split_nodes(self) -> "ArxivDataMixin":
        node_data = pd.read_csv(self._node_data_path, sep='	')
        print("reading from:", self._node_data_path)

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

class OGBArxivGraphTextDataModule(ArxivDataMixin, GraphTextDataModule):
    """Graph+text data module for ogbn-arxiv using repository raw/*.csv files."""
    def __init__(self, **kwargs):
        kwargs['drop_isolates'] = False
        super().__init__(**kwargs)
        
#class OGBArxivGraphTextDataModule(OGBDataMixin, GraphTextDataModule):
#    """Graph+text data module for ogbn-arxiv as a PubmedGraphTextDataModule replacement."""
#    ogb_dataset_name = "ogbn-arxiv"


class OGBProductsGraphTextDataModule(OGBDataMixin, GraphTextDataModule):
    """Graph+text data module for ogbn-products; prefer arxiv for smaller smoke runs."""
    ogb_dataset_name = "ogbn-products"
