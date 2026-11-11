"""Data preparation for the UniTG-style baseline.

Everything is derived from the same datamodule the other baselines and every
evaluator use (``clip_graph.data.datamodule.PubmedGraphTextDataModule`` for the
B1 YAMLs), so the node universe, the 0.7/0.1/0.2 node split (seed from the
YAML) and the z-scored adjacency-SVD node features are identical:

* ``train/val/test`` are the split-induced subgraphs (local ids, undirected,
  isolates dropped) exactly as ``<split>_dataset.dataset.graph_data``;
* the LM feature table is indexed by the *full* (unsplit) node order; each
  split carries ``rows`` mapping its local ids into that table;
* one text per node (its text rows joined) is tokenised once to a fixed
  length and cached under ``data/<ds>/derived/``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from pydoc import locate
from typing import Callable, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .partition import degree_groups, node_degrees

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_TEXT_ROWS_PER_NODE = 8


@dataclass
class SplitGraph:
    name: str
    edge_index: torch.Tensor      # [2, E] local ids, both directions
    x: torch.Tensor               # [n, in_channels] float32 (SVD, z-scored)
    node_ids: torch.Tensor        # [n] global dataset node ids
    rows: torch.Tensor            # [n] row of each node in the full feature/token table
    degree: torch.Tensor          # [n]
    group: torch.Tensor           # [n] centrality group (0 = least central)

    @property
    def num_nodes(self) -> int:
        return int(self.node_ids.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


@dataclass
class UniTGData:
    dataset: str
    data_dir: str
    snapshot: object                      # OmegaConf of the YAML
    model_args: Dict
    lm_name: str
    tokenizer_name: str
    in_channels: int
    full_node_ids: torch.Tensor           # [N]
    train: SplitGraph
    val: SplitGraph
    test: SplitGraph
    input_ids: torch.Tensor               # [N, L] int32
    attention_mask: torch.Tensor          # [N, L] uint8
    group_full: torch.Tensor              # [N] centrality group of every node (by its own split)
    group_thresholds: torch.Tensor
    has_text: torch.Tensor                # [N] bool: the node has at least one non-blank text row
    meta: Dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.full_node_ids.numel())

    def split(self, name: str) -> SplitGraph:
        return {"train": self.train, "val": self.val, "test": self.test}[name]


def _lm_tag(lm_name: str) -> str:
    return lm_name.replace("/", "__")


def _join_texts_per_node(text_node_ids: np.ndarray, texts: np.ndarray, node_ids: np.ndarray) -> list:
    """One string per node in ``node_ids`` order: its text rows (file order) joined by newlines."""
    order = np.argsort(text_node_ids, kind="stable")
    sorted_ids = text_node_ids[order]
    starts = np.searchsorted(sorted_ids, node_ids, side="left")
    ends = np.searchsorted(sorted_ids, node_ids, side="right")
    out = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e <= s:
            out.append(" ")
            continue
        rows = order[s:min(e, s + MAX_TEXT_ROWS_PER_NODE)]
        parts = [str(texts[r]) for r in rows]
        joined = "\n".join(p for p in parts if p)
        out.append(joined if joined.strip() else " ")
    return out


def _first_text_per_node(text_node_ids: np.ndarray, texts: np.ndarray, node_ids: np.ndarray) -> list:
    """One string per node in ``node_ids`` order: the FIRST of its rows in the given table's order.

    This is TAGNAS's selection rule verbatim -- PubmedRuntimeStore indexes
    ``setdefault(node_id, index)`` over ``dataset.text_node_ids`` and ``build_text_batch`` feeds that
    row. Applied to TAGNAS's own text table (see :func:`_tagnas_text_per_node`) it reproduces TAGNAS's
    input exactly; applied to another table it only reproduces the rule. A node without rows gets a
    single space, as in the joined mode.
    """
    first: Dict[int, int] = {}
    for index, node_id in enumerate(text_node_ids.tolist()):
        first.setdefault(int(node_id), index)
    out = []
    for node_id in node_ids.tolist():
        index = first.get(int(node_id))
        text = "" if index is None else str(texts[index])
        out.append(text if text.strip() else " ")
    return out


def _tagnas_text_per_node(dataset: str, tagnas_config: str, node_ids: np.ndarray, seed: int = 42) -> list:
    """The exact per-node text TAGNAS's text worker feeds, one string per node in ``node_ids`` order.

    TAGNAS's config samples at most ``max_texts_per_node`` rows per node (gassobase.yaml: 5) and
    regroups them, so "first row in file order" is NOT what it reads (it agrees on only ~84% of
    citation's nodes). The selection is deterministic, so building TAGNAS's own store and applying its
    setdefault-first rule to that store's table reproduces it exactly.
    """
    from clip_graph.async_coexec.config import AsyncTrainConfig
    from clip_graph.async_coexec.pubmed_runtime import PubmedRuntimeStore
    from clip_graph.async_coexec.train_runtime import _load_runtime_spec

    cfg = AsyncTrainConfig(output_dir=os.path.join(REPO_ROOT, "model", "unitg_tagnas_text_probe"),
                           config_path=str(tagnas_config), dataset=dataset, seed=seed,
                           local_files_only=True)
    data_args, model_args = _load_runtime_spec(cfg)
    store = PubmedRuntimeStore.build(data_args=data_args, split_proportions=cfg.split_proportions,
                                     seed=cfg.seed, local_files_only=True,
                                     feature_dim=int(model_args["gnn_params"]["in_channels"]))
    return _first_text_per_node(store.dataset.text_node_ids.numpy(), store.dataset.text, node_ids)


def _tokenize_all(tokenizer, texts: list, max_length: int, batch: int = 4096):
    ids_chunks, mask_chunks = [], []
    for i in range(0, len(texts), batch):
        enc = tokenizer(
            texts[i:i + batch],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids_chunks.append(enc["input_ids"].to(torch.int32))
        mask_chunks.append(enc["attention_mask"].to(torch.uint8))
    return torch.cat(ids_chunks, 0), torch.cat(mask_chunks, 0)


def _split_graph(name: str, ds, row_of: Dict[int, int], thresholds: Optional[torch.Tensor], num_groups: int):
    gd = ds.dataset.graph_data
    node_ids = gd.node_ids.clone().long()
    ei = gd.edge_index.clone().long()
    x = gd.x.clone().float()
    rows = torch.tensor([row_of[int(v)] for v in node_ids.tolist()], dtype=torch.long)
    deg = node_degrees(ei, node_ids.numel())
    group, thr = degree_groups(deg, num_groups, thresholds)
    return SplitGraph(name=name, edge_index=ei, x=x, node_ids=node_ids, rows=rows, degree=deg, group=group), thr


def build_data(
    dataset: str,
    config_path: str,
    *,
    lm_max_length: int = 128,
    num_groups: int = 4,
    log: Callable[[str], None] = print,
    tokenizer_cache: bool = True,
    text_per_node: str = "joined",
    tagnas_text_config: str = "configs/clip-graph/inductive-causal/pubmed/gassobase.yaml",
) -> UniTGData:
    t0 = time.time()
    snapshot = OmegaConf.load(str(config_path))
    data_args = OmegaConf.to_container(snapshot.data.init_args, resolve=True)
    data_dir = str(REPO_ROOT / "data" / dataset)
    data_args["data_dir"] = data_dir
    data_args["compute_sim_mutual"] = False       # the dense mutual matrix is not needed here (as in B1 with sim_smoothing=0)
    model_args = OmegaConf.to_container(snapshot.model.init_args, resolve=True)

    raw_dir = os.path.join(data_dir, "raw")
    for fn in ("graph.csv", "node-data.csv", "texts.csv"):
        if not os.path.isfile(os.path.join(raw_dir, fn)):
            raise FileNotFoundError(f"{raw_dir}/{fn} is missing; is data/{dataset} downloaded?")

    DataModuleCls = locate(snapshot.data.class_path)
    if DataModuleCls is None:
        raise ImportError(f"cannot locate datamodule class {snapshot.data.class_path}")
    dm = DataModuleCls(**data_args)
    dm.prepare_data()
    dm.setup(node=0, world_size=1, stage="fit")
    log(f"[data] datamodule ready in {time.time() - t0:.1f}s")

    full = dm.dataset
    full_node_ids = full.graph_data.node_ids.clone().long()
    row_of = {int(v): i for i, v in enumerate(full_node_ids.tolist())}

    train, thr = _split_graph("train", dm.train_dataset, row_of, None, num_groups)
    val, _ = _split_graph("val", dm.val_dataset, row_of, thr, num_groups)
    test, _ = _split_graph("test", dm.test_dataset, row_of, thr, num_groups)
    group_full = torch.full((full_node_ids.numel(),), num_groups - 1, dtype=torch.long)
    for sg in (train, val, test):
        group_full[sg.rows] = sg.group
    in_channels = int(train.x.size(1))

    # Nodes without any text (PubMed: only 27% of the nodes carry a title/abstract, Twitter: 58%)
    # take no part in the graph-text contrastive losses, exactly as in B1/TAGNAS where the text
    # side only ever sees text-bearing nodes. They still take part in message passing.
    text_ids = full.text_node_ids.numpy()
    nonblank = np.array([bool(str(s).strip()) for s in full.text], dtype=bool)
    ids_with_text = set(text_ids[nonblank].tolist())
    has_text = torch.tensor([int(v) in ids_with_text for v in full_node_ids.tolist()], dtype=torch.bool)

    lm_name = str(model_args["lm_name"])
    tokenizer_name = str(data_args.get("tokenizer_name", lm_name))
    if text_per_node not in ("joined", "tagnas"):
        raise ValueError(f"text_per_node must be 'joined' or 'tagnas', got {text_per_node!r}")
    mode_tag = "" if text_per_node == "joined" else f"_{text_per_node}"
    cache_path = Path(data_dir) / "derived" / f"unitg_tokens_{_lm_tag(tokenizer_name)}_L{int(lm_max_length)}{mode_tag}.pt"
    input_ids = attention_mask = None
    if tokenizer_cache and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if torch.equal(payload["node_ids"].long(), full_node_ids):
                input_ids, attention_mask = payload["input_ids"], payload["attention_mask"]
                log(f"[data] loaded token cache {cache_path}")
        except Exception as exc:  # noqa: BLE001
            log(f"[data] ignoring bad token cache {cache_path}: {exc}")
    if input_ids is None:
        t1 = time.time()
        if text_per_node == "tagnas":
            texts = _tagnas_text_per_node(dataset, str(REPO_ROOT / tagnas_text_config)
                                          if not os.path.isabs(tagnas_text_config) else tagnas_text_config,
                                          full_node_ids.numpy())
        else:
            texts = _join_texts_per_node(full.text_node_ids.numpy(), full.text, full_node_ids.numpy())
        input_ids, attention_mask = _tokenize_all(full.tokenizer, texts, int(lm_max_length))
        log(f"[data] tokenised {len(texts)} node texts (L={lm_max_length}) in {time.time() - t1:.1f}s")
        if tokenizer_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            torch.save({"node_ids": full_node_ids, "input_ids": input_ids, "attention_mask": attention_mask}, tmp)
            os.replace(tmp, cache_path)

    meta = {
        "num_nodes_full": int(full_node_ids.numel()),
        "train_nodes": train.num_nodes, "train_edges": train.num_edges,
        "val_nodes": val.num_nodes, "val_edges": val.num_edges,
        "test_nodes": test.num_nodes, "test_edges": test.num_edges,
        "group_thresholds": [float(t) for t in thr.tolist()],
        "group_sizes_train": torch.bincount(train.group, minlength=num_groups).tolist(),
        "nodes_with_text": int(has_text.sum()),
        "text_per_node": text_per_node,
        "lm_max_length": int(lm_max_length),
        "token_length_mean": float(attention_mask.float().sum(1).mean()),
        "token_length_max": int(attention_mask.sum(1).max()),
        "train_nodes_with_text": int(has_text[train.rows].sum()),
        "seconds": time.time() - t0,
    }
    log(f"[data] {dataset}: N={meta['num_nodes_full']} train={train.num_nodes}/{train.num_edges}e "
        f"val={val.num_nodes}/{val.num_edges}e test={test.num_nodes}/{test.num_edges}e "
        f"groups(train)={meta['group_sizes_train']} thr={meta['group_thresholds']} "
        f"with_text={meta['nodes_with_text']}/{meta['num_nodes_full']} (train {meta['train_nodes_with_text']})")
    return UniTGData(
        dataset=dataset, data_dir=data_dir, snapshot=snapshot, model_args=model_args,
        lm_name=lm_name, tokenizer_name=tokenizer_name, in_channels=in_channels,
        full_node_ids=full_node_ids, train=train, val=val, test=test,
        input_ids=input_ids, attention_mask=attention_mask,
        group_full=group_full, group_thresholds=thr, has_text=has_text, meta=meta,
    )
