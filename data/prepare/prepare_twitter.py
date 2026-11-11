"""
One-off data preparation script.
Generates the format expected by the training code from the twitter dataset on
HuggingFace ZhuofengLi/TEG-Datasets:
  data/twitter/raw/graph.csv
  data/twitter/raw/node-data.csv
  data/twitter/raw/texts.csv

Note on data provenance:
This script directly unpacks the officially processed ``data/twitter/hf_src/twitter.pkl``.
The official notebook (``process_final_twitter.ipynb``) has a bug: it loads part2.npy twice
(``p1 = np.load("...part2.npy")`` should have been part1.npy), so the official pkl actually
just has the part2 data duplicated. The 60785 nodes / 74666 edges reported in the
TEG-Benchmark README are exactly the output of this "buggy" version.

To stay aligned with the paper / benchmark numbers, this script reuses the official pkl
as-is and **does not fix the bug**. If you want the full data from part1 + part2
(128864 nodes / 76551 edges), see the older version of this file in the git history.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

HF_SRC_DIR = "./data/twitter/hf_src"
OUTPUT_DIR = "./data/twitter/raw"
PKL_PATH = os.path.join(HF_SRC_DIR, "twitter.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading official processed pickle: {PKL_PATH}")
with open(PKL_PATH, "rb") as f:
    g = pickle.load(f)

edge_index: torch.Tensor = g.edge_index           # [2, num_edges]
text_nodes: list = list(g.text_nodes)             # node_id -> "tweet" / "user" / "mentioned user"
text_edges: list = list(g.text_edges)             # edge_id -> tweet content
node_labels: list = list(g.node_labels)           # tweet nodes hold event_id, user nodes hold -1

num_nodes = len(text_nodes)
num_edges = int(edge_index.shape[1])
num_tweet = sum(1 for t in text_nodes if t == "tweet")
num_user = num_nodes - num_tweet
print(f"  Nodes: {num_nodes}  (tweet={num_tweet}, user/mentioned={num_user})")
print(f"  Edges: {num_edges}")
print(f"  Unique events (label): "
      f"{len({str(x) for x in node_labels if str(x) != '-1'})}")

# ---------------------------------------------------------------------------
# 1. graph.csv  -- columns: source, target
# In the official edge_index, row 0 is the user node and row 1 is the tweet node
# (cell 4: edge_index = [user, tweet]). Consistent with pubmed/arxiv's graph.csv, so it's
# enough to write it as two columns, source/target.
# ---------------------------------------------------------------------------
print("\nGenerating graph.csv ...")
src = edge_index[0].cpu().numpy().astype(np.int64)
tgt = edge_index[1].cpu().numpy().astype(np.int64)
edge_df = pd.DataFrame({"source": src, "target": tgt})
edge_df.to_csv(os.path.join(OUTPUT_DIR, "graph.csv"), sep="\t", index=False)
print(f"  Saved {len(edge_df)} edges -> graph.csv")

# ---------------------------------------------------------------------------
# 2. node-data.csv  -- columns: node_id, node_type, label
# label: tweet -> event_id (int), user/mentioned user -> -1
# ---------------------------------------------------------------------------
print("\nGenerating node-data.csv ...")


def _label_to_int(x: object) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return -1


labels_int = [_label_to_int(x) for x in node_labels]
node_df = pd.DataFrame({
    "node_id": np.arange(num_nodes, dtype=np.int64),
    "node_type": text_nodes,
    "label": labels_int,
})
node_df.to_csv(os.path.join(OUTPUT_DIR, "node-data.csv"), sep="\t", index=False)
print(f"  Saved {len(node_df)} nodes -> node-data.csv")

# ---------------------------------------------------------------------------
# 3. texts.csv  -- columns: node_id, kind, order, content
# The official text_nodes field only marks three type labels ("tweet" / "user" /
# "mentioned user") and carries no actual text. The actual text lives on text_edges
# (the tweet content).
# Here we follow pubmed's "node-side text" convention:
#   - the text carried by each (user, tweet) edge is the content of that same tweet
#   - so each edge's text is attributed to the corresponding tweet node, with order
#     following the order of appearance
#   - user / mentioned user nodes get a placeholder ("user" / "mentioned user")
# ---------------------------------------------------------------------------
print("\nGenerating texts.csv ...")
node_id_col, kind_col, order_col, content_col = [], [], [], []

# For each tweet node, collect all edge texts pointing to it
tweet_texts: dict = {}
for e_idx in range(num_edges):
    t_node = int(tgt[e_idx])
    text = text_edges[e_idx]
    if text is None:
        continue
    tweet_texts.setdefault(t_node, []).append(str(text))

for nid in range(num_nodes):
    ntype = text_nodes[nid]
    if ntype == "tweet":
        # tweet node: use all the collected edge texts as this tweet's text (dedup, preserving order)
        seen = set()
        order = 0
        for txt in tweet_texts.get(nid, []):
            if txt in seen:
                continue
            seen.add(txt)
            node_id_col.append(nid)
            kind_col.append("text")
            order_col.append(order)
            content_col.append(txt)
            order += 1
        if order == 0:
            # tweet node with no associated edge text at all; write one placeholder row (so this node isn't missing)
            node_id_col.append(nid)
            kind_col.append("text")
            order_col.append(0)
            content_col.append("tweet")
    else:
        # user / mentioned user node
        node_id_col.append(nid)
        kind_col.append("text")
        order_col.append(0)
        content_col.append(ntype)

texts_df = pd.DataFrame({
    "node_id": node_id_col,
    "kind": kind_col,
    "order": order_col,
    "content": content_col,
})
# Strip newlines/tabs to avoid breaking the TSV structure
texts_df["content"] = (
    texts_df["content"].astype(str)
    .str.replace(r"[\r\n\t]+", " ", regex=True)
    .str.strip()
)
texts_df = texts_df[texts_df["content"].str.len() > 0].reset_index(drop=True)
texts_df.to_csv(os.path.join(OUTPUT_DIR, "texts.csv"), sep="\t", index=False)
print(f"  Saved {len(texts_df)} text rows -> texts.csv")

print("\nAll done! Files saved to:", OUTPUT_DIR)
print("\nBenchmark check: expect 60785 nodes / 74666 edges (matches TEG-Benchmark README)")
print(f"  Actual: {num_nodes} nodes / {num_edges} edges")
