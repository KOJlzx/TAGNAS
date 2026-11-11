"""Configuration objects for the async co-execution runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


# sync_push lock-step ablation can deadlock on large graphs. Use a wall-clock
# safety cap so impossible runs fail within hours instead of occupying resources
# for a full day. Explicit smaller max-runtime values from orchestration scripts
# still take precedence.
_SYNC_PUSH_RUNTIME_CAP_SECONDS = 14400.0  # 4h

# Default artifact root derived from this file's location (repo_root/model/distributed),
# so the packaged default carries no hardcoded personal absolute path. Overridable via
# --output-dir / output_dir. config.py lives at src/clip_graph/async_coexec/ -> parents[3] = repo root.
_DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[3] / "model" / "distributed")


@dataclass
class AsyncCoExecConfig:
    """Runtime settings for the demo and for future real adapters."""

    # These parameters describe system topology and communication/cache budgets,
    # not model hyperparameters.
    master_addr: str = "127.0.0.1"
    master_port: int = 29591
    graph_batch_size: int = 4000
    text_chunk_size: int = 4
    graph_k_hop: int = 3
    embedding_dim: int = 32
    num_nodes: int = 24
    num_rounds: int = 3
    rpc_timeout_seconds: float = 60.0
    graph_cache_capacity: int = 8192
    text_cache_capacity: int = 8192
    cache_dtype_name: str = "float32"
    seed: int = 42
    demo_consumer_nodes: int = 2
    num_text_workers: int = 2
    use_gpu_if_available: bool = True
    # Ablation switch: True degrades to bidirectional lock-step push. Graph and
    # text sides use the same batch and node order, and each step waits for both
    # sides before advancing. False is the full async drop-on-miss system.
    sync_push: bool = False
    # sync_push memory safety: lock-step may force text side to cover graph_batch_size
    # long sequences per step. Micro-batching plus per-chunk checkpointing keeps peak
    # attention activation memory near micro_batch scale while remaining numerically
    # equivalent for independent GPT-style sequences.
    sync_text_micro_batch: int = 64
    # Node-selection strategy (which seed nodes a batch covers; cache insertion,
    # drop-on-miss and the RPC fallback are unaffected by both switches).
    # text_shard_nodes: each text worker covers its own 1/W shard of the training
    # nodes instead of independently shuffling all of them, so W workers fill the
    # embedding caches W times faster.
    text_shard_nodes: bool = True
    # graph_cache_sampling: each graph pass visits every training seed exactly once.
    # Prefer remaining seeds with cached text embeddings, then fill from other
    # remaining seeds without waiting. The last graph batch may be partial.
    graph_cache_sampling: bool = True
    # Legacy compatibility argument. Coverage is mandatory on every pass; zero
    # no longer disables exploration. Nonnegative values select the same policy.
    graph_cache_explore_every: int = 0
    # graph_push_extra_max: besides its 1024 seeds, each graph batch also pushes up to this many
    # by-product embeddings -- training nodes of the same k-hop subgraph at least k hops inside
    # it, whose embedding is as accurate as a seed's (PubmedRuntimeStore.exact_train_nodes).
    # The forward computes them anyway: on amazon-baby / citation the 3-hop subgraph of 1024
    # seeds is 130K / 88K nodes, and without this 99% of every ~8 s CPU forward is discarded,
    # the graph cache fills at ~90 nodes/s and the text side (all-or-nothing chunks) starves for
    # ~1500 s. Nodes never pushed (then least recently pushed) go first. 0 disables.
    graph_push_extra_max: int = 0
    # text_grad_checkpointing: activation checkpointing in the text workers' LM (recompute each
    # block in backward). Trades ~1/3 more LM compute for a several-times larger text chunk.
    text_grad_checkpointing: bool = False
    # Negative sampling of the online validation monitor; "auto" = the final eval's and the
    # baselines' rule (type-aware on datasets dominated by cross-type edges).
    val_negative_sampling: str = "auto"
    # text_token_budget: > 0 replaces the fixed text_chunk_size with length-bucketed chunks bounded
    # by rows x padded length <= this many tokens (and <= text_max_chunk_rows rows). Text lengths are
    # very uneven and chunks are padded to their longest row, so fixed-row chunks mostly pay for
    # padding. 0 = off (fixed text_chunk_size rows).
    # val_eval_always: run the periodic validation even with val_early_stop off (it then only records the
    # AUC-over-time curve; nothing stops on it).
    val_eval_always: bool = False
    # Embedding-lag injection (lag.py): the coordinator holds a push back from the global cache and the
    # broadcast until its producer has processed lag_inject_steps more batches AND lag_inject_seconds have
    # passed since it was made. Direction: both | graph_to_text (text workers see older graph embeddings)
    # | text_to_graph. 0 / 0.0 = off (no change).
    lag_inject_steps: int = 0
    lag_inject_seconds: float = 0.0
    lag_inject_direction: str = "both"
    # staleness_stats: count-only instrumentation for the staleness study (no effect on training):
    # consumers report the cache lag of every peer entry they hit (lag_G in graph updates, lag_T in
    # each text replica's updates) and their peer cache's training-node coverage; lag_G is recorded
    # only once a replica's C_G covers staleness_cover_gate of the training nodes.
    staleness_stats: bool = False
    staleness_cover_gate: float = 0.98
    text_token_budget: int = 0
    text_max_chunk_rows: int = 1024
    # text_probe_worst_case: memory probe only -- the first text chunk of every worker carries the
    # training pool's longest text(s), so a short probe window measures the true worst case.
    text_probe_worst_case: bool = False
    # Validation early stopping must not fire before the text tower has performed a single
    # aligned update, or the run reports convergence of a model half of which never trained.
    # On a large graph the text side cannot update until the graph side has pushed an
    # embedding for essentially every training node -- its chunks are all-or-nothing -- which
    # takes one full coverage pass (amazon-baby: 130,753 nodes at ~90 nodes/s, so ~1450 s).
    # Cache-aware graph sampling made the graph side produce updates immediately, which let
    # validation start at once; with the text tower still frozen the AUC plateaued and the
    # run early-stopped at 628 s, before that pass could finish. 0 disables the guard.
    val_early_stop_min_text_updates: int = 1

    def __post_init__(self) -> None:
        # Hard constraint for sync mode: the barrier is graph-to-text two-party, so
        # exactly one text worker is required and both sides must use identical batches.
        if self.sync_push:
            if self.num_text_workers != 1:
                print(f"[sync_push] num_text_workers={self.num_text_workers} -> 1 (lock-step requires exactly 1 text worker)")
                self.num_text_workers = 1
            if self.text_chunk_size != self.graph_batch_size:
                print(f"[sync_push] text_chunk_size={self.text_chunk_size} -> {self.graph_batch_size} (both sides must use the same batch size)")
                self.text_chunk_size = self.graph_batch_size
            # Lock-step needs both sides to walk the identical node order, which neither
            # sharding (one worker anyway) nor cache-driven sampling can guarantee.
            if self.text_shard_nodes or self.graph_cache_sampling:
                print("[sync_push] text_shard_nodes / graph_cache_sampling -> False (lock-step requires identical node order on both sides)")
                self.text_shard_nodes = False
                self.graph_cache_sampling = False
            # Deadlock safety default. If max_runtime_seconds is unset, use this
            # cap. Explicit values are respected, including larger values for known
            # long-running but progressing lock-step ablations.
            if hasattr(self, "max_runtime_seconds"):
                _cur = getattr(self, "max_runtime_seconds", None)
                if _cur is None:
                    print(f"[sync_push] max_runtime_seconds=None -> {_SYNC_PUSH_RUNTIME_CAP_SECONDS}s "
                          f"(safety default cap to prevent lock-step deadlock)")
                    self.max_runtime_seconds = _SYNC_PUSH_RUNTIME_CAP_SECONDS
                elif float(_cur) > _SYNC_PUSH_RUNTIME_CAP_SECONDS:
                    print(f"[sync_push] max_runtime_seconds={_cur}s (explicitly set, > default cap "
                          f"{_SYNC_PUSH_RUNTIME_CAP_SECONDS}s, respecting it)")

    @property
    def coordinator_name(self) -> str:
        return "coordinator"

    @property
    def graph_worker_name(self) -> str:
        return "graph_worker"

    def text_worker_name(self, index: int) -> str:
        return f"text_worker_{index}"

    @property
    def world_size(self) -> int:
        # Always includes one coordinator, one graph worker, and N text workers.
        return 2 + self.num_text_workers

    def role_for_rank(self, rank: int) -> str:
        # Rank-to-role mapping must be globally consistent for RPC names to match process identity.
        if rank == 0:
            return self.coordinator_name
        if rank == 1:
            return self.graph_worker_name
        text_index = rank - 2
        return self.text_worker_name(text_index)

    def text_worker_index(self, role_name: str) -> int:
        prefix = "text_worker_"
        if not role_name.startswith(prefix):
            raise ValueError(f"Role {role_name!r} is not a text worker")
        return int(role_name[len(prefix):])

    def text_device(self, index: int) -> str:
        # Prefer GPUs for text workers; fall back to CPU when unavailable or disabled.
        if self.use_gpu_if_available and torch.cuda.is_available() and torch.cuda.device_count() > index:
            return f"cuda:{index}"
        return "cpu"

    @property
    def graph_scheduler_policy(self) -> str:
        return ("cache_aware_without_replacement_v1" if self.graph_cache_sampling
                else "streaming_without_replacement_v1")

    def validate_training_cache_capacity(self, train_nodes: int) -> None:
        """Reject asynchronous settings that cannot retain one complete pass.

        Lock-step exchanges only require the current batch and retain their
        existing capacity semantics. Async full-pool training needs both tables
        to retain every training node, including isolated seeds.
        """
        if self.sync_push:
            return
        for name in ("graph_cache_capacity", "text_cache_capacity"):
            capacity = getattr(self, name)
            if capacity < train_nodes:
                raise ValueError(
                    f"{name}={capacity} is smaller than the {train_nodes} training nodes; "
                    "asynchronous full-coverage training requires capacity >= training node count"
                )

    @property
    def cache_dtype(self) -> torch.dtype:
        try:
            return getattr(torch, self.cache_dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported cache dtype: {self.cache_dtype_name}") from exc


@dataclass
class AsyncTrainConfig(AsyncCoExecConfig):
    """Configuration for the real async PubMed GraphNAS + LM training runtime."""

    # Real-training fields start here: config file, stage, output path, learning rates, etc.
    config_path: str = "configs/clip-graph/inductive-causal/pubmed/gassobase.yaml"
    dataset: str = "pubmed"
    stage: Literal["search", "retrain"] = "search"
    output_dir: str = _DEFAULT_OUTPUT_DIR
    run_name: str | None = None
    genotype_path: str | None = None

    dist_port: int = 29590
    rpc_port: int = 29591

    search_epochs: int = 4
    retrain_epochs: int = 8
    graph_epochs: int | None = None
    text_epochs: int | None = None
    eval_every_epochs: int = 1
    save_every_epochs: int = 1

    graph_weight_lr: float = 1.0e-4
    graph_arch_lr: float = 1.0e-2
    # Architecture softmax temperature annealing from start to end. 1.0/1.0 disables
    # annealing. end < 1 sharpens alpha, encourages commitment, enables faithful
    # pruning, and stabilizes greedy genotype extraction.
    graph_arch_temp_start: float = 1.0
    graph_arch_temp_end: float = 1.0
    lm_lr: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    # Optional overrides for YAML model_args.weight_decay / eps. They align async
    # runs with DDP/DeepSpeed settings used to reduce contrastive embedding collapse
    # and stabilize search on large graphs.
    weight_decay_override: float | None = None
    adam_eps_override: float | None = None

    split_proportions: tuple[float, float, float] = (0.7, 0.1, 0.2)
    local_files_only: bool = False
    disable_text_sync: bool = False
    # Use true DDP instead of manual per-parameter all_reduce. Bucketed communication
    # overlaps with backward and improves multi-GPU scaling when async and same-device.
    text_ddp: bool = False
    # LM periodic weighted averaging (post-local-SGD): text workers train locally
    # and rendezvous every K batches to average parameters and Adam moments by update
    # counts. Priority: local_avg > ddp > manual gradients.
    text_local_avg: bool = False
    text_sync_interval: int = 16     # K: batches between weighted-average rendezvous.
    text_sync_warmup: int = 0        # Warmup sync count with K_eff=1.
    # --- Ablation switches ---
    # w/o heterogeneous placement: move graph tower from CPU to GPU while keeping the
    # rest unchanged, isolating the contribution of heterogeneous placement.
    graph_device: str = "cpu"
    # w/o drop-on-miss: block and poll for peer embeddings on cache misses instead
    # of skipping updates. A timeout preserves liveness during cold-start gaps.
    block_on_miss: bool = False
    block_on_miss_timeout_seconds: float = 30.0
    # test6 time breakdown switch.
    profile: bool = False
    profile_out: str | None = None
    # Use only this fraction of train nodes for alignment. The subset is sampled
    # once by seed and shared by graph/text sides. Validation/test still use full splits.
    train_node_fraction: float | None = None

    # Override YAML model.init_args.gnn_params.num_layers. This is distinct from
    # graph_k_hop, though the two are usually aligned to avoid wasted subgraph nodes.
    num_layers: int | None = None

    max_train_rounds: int | None = None
    max_eval_rounds: int | None = None
    max_runtime_seconds: float | None = None
    max_graph_steps: int | None = None
    max_text_steps: int | None = None
    metrics_interval_seconds: float = 10.0
    checkpoint_interval_seconds: float = 600.0
    checkpoint_every_successful_updates: int | None = None

    # --- Validation early stopping (Scheme B) ---
    val_eval_interval_seconds: float = 60.0     # Minimum seconds between validation evals.
    val_eval_min_updates: int = 20              # Minimum graph updates before first validation.
    val_patience: int = 5                       # Consecutive non-improvements before stopping.
    val_min_delta: float = 0.0                  # An evaluation only resets the patience if it beats the
                                                # patience reference by more than this (0 = any strict gain).
    val_early_stop: bool = True                 # Whether validation early stopping is enabled.
    # Monitoring split for periodic eval. val is strict but can have few edges in
    # inductive setup; test gives denser time-to-accuracy curves but peeks at test.
    val_eval_on_test: bool = False

    # --- Adaptive pruning for dynamic search-space reduction ---
    # Monitor GassoSpace alpha groups and permanently mask low-weight candidates.
    # Masked candidates are skipped in forward, have zero gradients, and are pushed
    # to a very negative alpha value.
    use_adaptive_pruning: bool = False
    pruning_threshold: float = 0.05
    pruning_check_interval_updates: int = 50
    pruning_min_updates_before_start: int = 100
    pruning_min_candidates_per_layer: int = 1
    # "absolute": prune when weight < pruning_threshold.
    # "relative": prune when weight < pruning_threshold * max(alive weights),
    # the fraction-of-group-maximum rule (adapts to how peaked the group is).
    pruning_threshold_mode: Literal["absolute", "relative"] = "absolute"
    # Absolute floor combined with the per-mode cutoff as max(floor, cutoff) --
    # the delta term of PruneOnce's max(delta, rho * p_max). 0.0 disables it.
    pruning_threshold_abs: float = 0.0
    # Groups allowed for pruning. Norm/act have few candidates and identity can
    # dominate, so pruning only ops may be safer in some experiments.
    pruning_groups: tuple[str, ...] = ("op", "norm", "act")

    # --- Retrain warm start from search checkpoints ---
    # When enabled, retrain loads search model_state for graph and text instead of
    # starting from scratch. Only affects stage="retrain".
    retrain_warm_start: bool = False
    # Graph warm-start checkpoint path. In both mode this is inferred from search artifacts.
    warm_start_graph_ckpt: str | None = None
    # Text warm-start checkpoint path.
    warm_start_text_ckpt: str | None = None

    @property
    def num_epochs(self) -> int:
        # search and retrain share the runtime but usually use different epoch counts.
        return self.search_epochs if self.stage == "search" else self.retrain_epochs

    @property
    def graph_num_epochs(self) -> int:
        """Graph-side epoch count, falling back to num_epochs."""
        return self.graph_epochs if self.graph_epochs is not None else self.num_epochs

    @property
    def text_num_epochs(self) -> int:
        """Text-side epoch count, falling back to num_epochs."""
        return self.text_epochs if self.text_epochs is not None else self.num_epochs

    @property
    def text_worker_ranks(self) -> list[int]:
        return [rank for rank in range(self.world_size) if self.role_for_rank(rank).startswith("text_worker_")]

    @property
    def train_output_dir(self) -> Path:
        output_root = Path(self.output_dir).expanduser().resolve()
        if self.run_name:
            return output_root / self.run_name / self.stage
        return output_root / self.stage

    @property
    def manifest_path(self) -> Path:
        return self.train_output_dir / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.train_output_dir / "metrics.jsonl"

    @property
    def graph_ckpt_path(self) -> Path:
        return self.train_output_dir / f"{self.stage}_graph.pt"

    @property
    def text_ckpt_path(self) -> Path:
        return self.train_output_dir / f"{self.stage}_text.pt"

    @property
    def genotype_file_path(self) -> Path:
        if self.genotype_path is not None:
            return Path(self.genotype_path).expanduser().resolve()
        return self.train_output_dir / "genotype.json"

    @property
    def graph_best_ckpt_path(self) -> Path:
        return self.train_output_dir / f"{self.stage}_graph_best.pt"

    @property
    def text_best_ckpt_path(self) -> Path:
        return self.train_output_dir / f"{self.stage}_text_best.pt"

    @property
    def genotype_best_file_path(self) -> Path:
        return self.train_output_dir / "genotype_best.json"

    @property
    def dist_init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.dist_port}"

    @property
    def rpc_init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.rpc_port}"

    @property
    def text_sync_enabled(self) -> bool:
        # Text parameter synchronization is enabled only with at least two text
        # workers on the same device kind.
        if self.num_text_workers <= 1 or self.disable_text_sync:
            return False
        device_kinds = {self.text_device(index).split(":")[0] for index in range(self.num_text_workers)}
        return len(device_kinds) == 1

    def text_sync_backend(self) -> str:
        if all(self.text_device(index).startswith("cuda") for index in range(self.num_text_workers)):
            return "nccl"
        return "gloo"
