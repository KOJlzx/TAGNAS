# Configuration index

YAML files describe model and data settings. Shell runners supply execution and
experiment overrides; relative data paths are resolved from the repository root.

| Location | Purpose |
| --- | --- |
| [`clip-graph/`](clip-graph/) | Joint graph/text models, including supernet configurations |
| [`gnn-pretrain/`](gnn-pretrain/) | Graph-only pretraining |
| [`lm-pretrain/`](lm-pretrain/) | Language-model pretraining |
| [`eval-datasets/`](eval-datasets/) | Dataset and split settings for evaluation |
| [`deepspeed/`](deepspeed/) | ZeRO stage and CPU-offload settings |
| [`comparisons.yaml`](comparisons.yaml) | Checkpoint/embedding comparisons for the Lightning evaluation workflow |

Useful starting points:

- [`clip-graph/inductive-causal/pubmed/gassobase.yaml`](clip-graph/inductive-causal/pubmed/gassobase.yaml): graph supernet and language model.
- [`clip-graph/inductive-causal/pubmed/base.yaml`](clip-graph/inductive-causal/pubmed/base.yaml): fixed graph encoder and language model.
- [`deepspeed/ds_config_b4_offload.json`](deepspeed/ds_config_b4_offload.json): ZeRO-Offload baseline.
