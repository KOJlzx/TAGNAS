"""Configuration dataclass for the UniTG-style baseline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class UniTGConfig:
    # ---- what / where -------------------------------------------------------
    dataset: str
    config_path: str
    output_root: str = "model/runpytorch"
    run_name: Optional[str] = None
    run_dir: Optional[str] = None            # filled in by the launcher
    paper_v3_attempt: Optional[str] = field(default_factory=lambda: os.environ.get("PAPER_V3_ATTEMPT"))
    paper_v3_cell: Optional[str] = field(default_factory=lambda: os.environ.get("PAPER_V3_CELL"))
    world_size: int = 3                      # = number of pipeline stages = number of GPUs
    master_addr: str = "127.0.0.1"
    master_port: int = 30500
    seed: int = 42

    # ---- deep GNN (flow-parallel stages) -------------------------------------
    num_layers: int = 112                    # UniTG artifact default for RevGCN
    hidden_channels: int = 224
    num_groups: int = 2                      # GroupAddRev groups
    embed_dim: int = 768                     # link-prediction embedding size (matches LM dim)
    gnn_lr: float = 1e-3
    gnn_dropout: float = 0.2
    gnn_weight_decay: float = 0.0
    gnn_grad_clip: float = 1.0
    microbatches: int = 3                    # M microbatches per mini-batch (M <= P in the paper)
    max_nodes_per_microbatch: int = 100_000  # larger graphs are cut into more mini-batches
    halo_hops: int = 0                       # >0: microbatch = cluster seeds + their k-hop neighbourhood (TAGNAS's extraction)
    max_pos_edges_per_microbatch: int = 200_000

    # ---- learning objective ---------------------------------------------------
    # "contrastive": the objective every other system in this repo trains (graph-text InfoNCE,
    #   clip_graph.losses.square_contrastive_loss): the GNN reads the adjacency-SVD features
    #   and aligns its node embeddings with the LM text embeddings, the LM aligns with the GNN
    #   embeddings; link prediction is evaluation-only. This keeps the AUC column comparable.
    # "link": the original UniTG-style task-loss port (BCE on edges, LM table as GNN input).
    #   Not comparable with the other systems and it learns the "edge present in the
    #   message-passing graph" shortcut of the evaluation protocol; kept for reference only.
    objective: str = "contrastive"
    # How the LM is fine-tuned while the GNN aligns to its table:
    #   "link"        – on its own text-side edge InfoNCE (endpoint texts of train edges are positives,
    #                   in-batch others negatives). The LM is anchored by text, the flow is one-directional
    #                   (LM -> GNN) as in UniTG, and the GNN's alignment target cannot drift. Default.
    #   "contrastive" – aligned back to the GNN embedding table (bidirectional exchange). Without an
    #                   external anchor the two towers chase each other's stale tables and collapse in
    #                   practice (val AUC -> 0.5); kept for the record.
    lm_objective: str = "link"
    contrastive_nodes: int = 1024            # nodes per microbatch entering the GNN-side InfoNCE
    tau_init: float = 3.5                    # logit scale = exp(tau), clamped at exp(max_tau) (as B1)
    max_tau: float = 4.605
    lm_warmup_hold: bool = True              # LM only encodes (no updates) until the first GNN table
                                             # produced by a *trained* GNN arrives

    # ---- LM (data-parallel, interleaved in bubbles) --------------------------
    lm_lr: Optional[float] = None            # None -> YAML model.init_args.lr
    lm_adam_eps: Optional[float] = None      # None -> YAML eps
    lm_weight_decay: Optional[float] = None  # None -> YAML weight_decay
    lm_batch_size: int = 64                  # contrastive: nodes per LM step; link: edges per step
    lm_max_length: int = 128
    # Which text a node contributes. "joined" (the UniTG artifact's behaviour) concatenates up to
    # MAX_TEXT_ROWS_PER_NODE of its rows; "tagnas" takes, for every node, exactly the string
    # TAGNAS's text worker feeds -- built through TAGNAS's own PubmedRuntimeStore from
    # tagnas_text_config -- so the two systems' text steps see identical inputs.
    text_per_node: str = "joined"
    tagnas_text_config: str = "configs/clip-graph/inductive-causal/pubmed/gassobase.yaml"
    # Trim every LM batch to its longest real row instead of the fixed lm_max_length, i.e. the
    # pad-to-longest-in-batch that B1-B4 and TAGNAS get from the shared tokenizer call. Only
    # changes how much padding is computed on, never the tokens that are kept.
    lm_dynamic_padding: bool = False
    # Draw the LM's contrastive anchors from a seeded fraction of the train text nodes instead of
    # all of them -- the same rule TAGNAS applies to its seed nodes (train_node_fraction). Under
    # pad-to-longest a run's peak memory is set by the longest text it ever meets, and a 10x larger
    # pool meets longer outliers (citation: 1024 tokens over the full split vs ~838 in a 10% subset),
    # so matching the pool size is part of matching TAGNAS's text measurement. The GNN side and the
    # table encode still cover every node. None = the whole pool.
    lm_train_node_fraction: Optional[float] = None
    # Memory probes only. With pad-to-longest, a run's peak is set by the longest text it meets,
    # and B5's LM (trained in pipeline bubbles) takes tens of steps in a 500 s probe window where
    # TAGNAS's text worker takes ~1700 batches and meets every node ~10 times. Forcing the pool's
    # longest text into the first LM batch makes both systems face their worst case at least once.
    # Changes which nodes one batch draws, nothing else; leave off for training runs.
    lm_probe_worst_case: bool = False
    lm_grad_clip: float = 1.0
    lm_num_exits: int = 4                    # centrality groups / exit points
    lazy_tuner: bool = True                  # False = every node goes through the full LM
    # Which exit produces the synchronised table: "full" = every node is encoded at full depth
    # (one representation space, required for a consistent contrastive target; the early exits
    # still accelerate the fine-tuning steps); "group" = each node's own centrality exit (the
    # paper's literal formulation, fine for the 'link' objective where the table is an input).
    encode_exit: str = "full"
    lm_autocast: bool = True                 # bf16 autocast for the LM
    lm_encode_batch: int = 256               # batch size for the full-graph encode at a sync
    bubble_interleave: bool = True           # False = LM runs unrestricted (no pause in GNN compute)

    # ---- modality collaborator (Algorithm 1) ---------------------------------
    sync_schedule: Tuple[float, ...] = (0.3, 0.2, 0.1, 0.02)
    sync_tighten_epochs: int = 20
    sync_ema: float = 0.8
    sync_unit: str = "auto"                  # auto | seconds | epochs
    initial_sync: bool = True                # encode once with the pretrained LM before epoch 0

    # ---- budget / validation / early stopping --------------------------------
    num_epochs: int = 1000
    max_runtime_seconds: Optional[float] = 3000.0
    val_eval_interval_seconds: float = 60.0
    val_eval_min_epochs: int = 5
    eval_at_start: bool = True               # score the untrained model too (B1 scores from its first epoch)
    early_stopping_patience: int = 5
    min_lm_texts_before_stop: int = 0        # cumulative text budget: -1 = this train text pool's size, 0 = off
    val_eval_on_test: bool = False
    link_neg_sampling: str = "auto"
    post_eval: bool = True
    post_eval_split: str = "test"

    # ---- accepted for CLI parity with the other runners (recorded only) ------
    text_subsample_fraction: Optional[float] = None
    early_stop_metric: str = "auc"

    # ---- per-iteration telemetry (experiments/04_time_breakdown.sh) --------------
    # Same contract as the other runners' PROFILE / PROFILE_OUT: profile_out is the legacy
    # breakdown JSON path, whose directory is where derive_jsonl_path() puts telemetry/*.jsonl.
    profile: bool = False
    profile_out: Optional[str] = None

    # ---- misc -----------------------------------------------------------------
    log_every_epochs: int = 10
    save_features: bool = False              # LM table alongside best/final (input features under 'link')

    def __post_init__(self) -> None:
        if self.min_lm_texts_before_stop < -1:
            raise ValueError("min_lm_texts_before_stop must be -1 (auto) or nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def num_stages(self) -> int:
        return self.world_size

    def sync_budget(self) -> Tuple[str, float]:
        """Return (unit, budget) for the Δt schedule fractions."""
        unit = self.sync_unit
        if unit == "auto":
            unit = "seconds" if self.max_runtime_seconds else "epochs"
        if unit == "seconds":
            if not self.max_runtime_seconds:
                raise ValueError("sync_unit=seconds requires max_runtime_seconds")
            return unit, float(self.max_runtime_seconds)
        if unit == "epochs":
            return unit, float(self.num_epochs)
        raise ValueError(f"unknown sync_unit {unit!r}")
