# Citation CPU sensitivity

Citation + DistilGPT-2, three GPUs. Set the CPU affinity and thread budget for each run.

[Configuration](citation.yaml) · [Results](../../artifacts/cpu_sensitivity/README.md)

```bash
mkdir -p model/cpu-sensitivity
RUN_DIR="$PWD/model/cpu-sensitivity/t16-$(date +%Y%m%dT%H%M%S)"
mkdir "$RUN_DIR"
export CPU_SENSITIVITY_SOURCE_ROOT="$PWD"
export CPU_SENSITIVITY_ATTEMPT_DIR="$RUN_DIR"
export CPU_SENSITIVITY_CPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CPU_SENSITIVITY_THREADS=16 CPU_SENSITIVITY_KIND=search
export TAGNAS_CPU_SENSITIVITY_ATTEMPT="manual-citation-t16"
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

taskset -c "$CPU_SENSITIVITY_CPUS" uv run --no-sync python \
  experiments/cpu_sensitivity/native_entry.py \
  --config-path experiments/cpu_sensitivity/citation.yaml --dataset citation \
  --output-dir "$RUN_DIR/artifacts" --phase search --seed 42 \
  --num-layers 3 --graph-k-hop 1 --graph-batch-size 1024 --text-chunk-size 64 \
  --num-text-workers 3 --graph-cache-capacity 600000 --text-cache-capacity 600000 \
  --train-node-fraction 1.0 --graph-push-extra-max 8192 \
  --search-epochs 100000 --graph-epochs 100000 --text-epochs 100000 \
  --text-local-avg --text-sync-interval 16 --text-sync-warmup 30 \
  --text-shard-nodes --graph-cache-sampling --graph-cache-explore-every 0 \
  --val-eval-interval-seconds 50 --val-eval-min-updates 3 --val-patience 10 \
  --val-min-delta 0.005 --val-early-stop-min-text-updates 1853 \
  --val-negative-sampling auto --val-early-stop --use-adaptive-pruning \
  --pruning-threshold 0.2 --pruning-check-interval-updates 100 \
  --pruning-min-updates-before-start 100 --pruning-min-candidates-per-layer 2 \
  --pruning-threshold-mode relative --pruning-threshold-abs 0.0 \
  --pruning-groups op,norm,act --metrics-interval-seconds 10 \
  --checkpoint-interval-seconds 600 --max-runtime-seconds 10800 \
  --search-max-runtime-seconds 10800 --offline-models-only \
  --dist-port 29590 --rpc-port 29591
```
