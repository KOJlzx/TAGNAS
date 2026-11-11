"""
One-off data preparation script.
Converts the raw ogbn-arxiv format into the format expected by the training code:
  data/ogbn-arxiv/raw/graph.csv
  data/ogbn-arxiv/raw/node-data.csv
  data/ogbn-arxiv/raw/texts.csv
"""

import os
import numpy as np
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset

OGB_ROOT       = "./data/ogbn-arxiv"
TITLEABS_PATH  = "./data/ogbn-arxiv/ogbn_arxiv/titleabs.tsv.gz"
OUTPUT_DIR     = "./data/ogbn-arxiv/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading ogbn-arxiv via ogb...")
dataset   = NodePropPredDataset(name="ogbn-arxiv", root=OGB_ROOT)
graph, labels = dataset[0]
split_idx = dataset.get_idx_split()

num_nodes  = graph['num_nodes']
node_feats = graph['node_feat']
src, dst   = graph['edge_index']

print(f"  Nodes : {num_nodes}")
print(f"  Edges : {len(src)}")
print(f"  Features: {node_feats.shape}")

# 2. graph.csv
print("\nGenerating graph.csv ...")
edge_df = pd.DataFrame({'source': src, 'target': dst})
edge_df.to_csv(os.path.join(OUTPUT_DIR, 'graph.csv'), sep='\t', index=False)
print(f"  Saved {len(edge_df)} edges")

# 3. node-data.csv
print("\nGenerating node-data.csv ...")
node_df = pd.DataFrame({'node_id': np.arange(num_nodes), 'label': labels.flatten()})
feat_df = pd.DataFrame(node_feats, columns=[f'feat_{i}' for i in range(node_feats.shape[1])])

split_col = np.full(num_nodes, 'test', dtype=object)
split_col[split_idx['train']] = 'train'
split_col[split_idx['valid']] = 'valid'

node_df = pd.concat([node_df, feat_df], axis=1)
node_df['split'] = split_col

node_df.to_csv(os.path.join(OUTPUT_DIR, 'node-data.csv'), sep='\t', index=False)
print(f"  Saved {len(node_df)} nodes")
print(f"  Train: {(split_col=='train').sum()}  Valid: {(split_col=='valid').sum()}  Test: {(split_col=='test').sum()}")


# 4. texts.csv
print("\nGenerating texts.csv ...")

if not os.path.exists(TITLEABS_PATH):
    raise FileNotFoundError(f"Could not find {TITLEABS_PATH}")

# The first line of titleabs.tsv.gz is the filename "titleabs.tsv", so skiprows=1 is needed to skip it
import tarfile
import io

# titleabs.tsv.gz is actually in tar.gz format, so it needs to be extracted with tarfile
with tarfile.open(TITLEABS_PATH, 'r:gz') as tar:
    member = tar.getmember('titleabs.tsv')
    f = tar.extractfile(member)
    titleabs = pd.read_csv(
        io.TextIOWrapper(f, encoding='utf-8'),
        sep='\t',
        header=None,
        names=['node_id', 'title', 'abstract']
    )

# Read the node-index <-> original paper-ID mapping
mapping_path = "./data/ogbn-arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
mapping = pd.read_csv(mapping_path, compression='gzip')
mapping.columns = ['node_idx', 'paper_id']

# titleabs's node_id is the original paper ID; convert it to the graph's node index via the mapping
titleabs = titleabs.merge(mapping, left_on='node_id', right_on='paper_id', how='inner')
titleabs['node_id'] = titleabs['node_idx'].astype(np.int64)

titleabs['content'] = titleabs['title'].str.strip() + '. ' + titleabs['abstract'].str.strip()
titleabs['id'] = np.arange(len(titleabs), dtype=np.int64)

texts_df = titleabs[['id', 'node_id', 'content']]
texts_df.to_csv(os.path.join(OUTPUT_DIR, 'texts.csv'), sep='\t', index=False)
print(f"  Saved {len(texts_df)} text entries")

print("\nAll done! Files saved to:", OUTPUT_DIR)