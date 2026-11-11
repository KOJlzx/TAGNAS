# Experiments

Run scripts from the repository root. Set the dataset, GPU IDs, and training
parameters through the environment variables exposed by each script.

| Experiment | Script |
| --- | --- |
| End-to-end performance | [`01_end_to_end.sh`](01_end_to_end.sh) |
| Time to accuracy | [`02_time_to_accuracy.sh`](02_time_to_accuracy.sh) |
| Memory limit | [`03_memory_limit.sh`](03_memory_limit.sh) |
| Iteration breakdown | [`04_time_breakdown.sh`](04_time_breakdown.sh) |
| Multi-GPU scaling | [`05_multigpu_scalability.sh`](05_multigpu_scalability.sh) |
| Ablation | [`06_ablation.sh`](06_ablation.sh) |
| Citation CPU sensitivity | [Command and configuration](cpu_sensitivity/README.md) |
