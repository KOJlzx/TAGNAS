# Data

PubMed and T-REx inputs are included under `data/pubmed/raw/` and
`data/trex/raw/`. Download Citation, Amazon-Baby, and Reddit with:

```bash
bash data/download_data.sh
```

To supply an alternative download URL:

```bash
DATA_URL='https://example.org/datasets.tar.gz' bash data/download_data.sh
```

Twitter preparation uses `data/prepare/prepare_twitter.py` and requires
`data/twitter/hf_src/twitter.pkl` as input.

Each dataset uses three tab-separated files in `data/<dataset>/raw/`:

| File | Columns |
| --- | --- |
| `graph.csv` | `source`, `target`, `new`, `orig` |
| `node-data.csv` | `node_id`, `node_type`, `label` |
| `texts.csv` | `node_id`, `kind`, `order`, `content` |

Derived tensors are generated on first use. Dataset licenses remain with their
original providers.
