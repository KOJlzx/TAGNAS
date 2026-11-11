"""In-process coordination between the GNN thread and the LM thread of one rank."""

from __future__ import annotations

import threading
from typing import Optional

import torch


class SyncChannel:
    """Sync requests (GNN -> LM), published embedding tables (both directions), stop/error flags.

    * ``table``      – LM text embeddings of every node, ``[N, D]`` fp16 (LM -> GNN).
    * ``gnn_table``  – GNN embeddings of every *train* node, ``[n_train, D]`` fp16 (GNN -> LM),
                       used as the LM's contrastive targets under the contrastive objective.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self.request_id = 0
        self.served_id = 0
        self.table: Optional[torch.Tensor] = None
        self.table_version = 0
        self.gnn_table: Optional[torch.Tensor] = None
        self.gnn_table_version = 0
        self.stop = False
        self.error: Optional[BaseException] = None
        self.lm_stats = {}

    # -- GNN side -----------------------------------------------------------
    def request_sync(self, rid: int) -> None:
        with self._cond:
            self.request_id = max(self.request_id, int(rid))
            self._cond.notify_all()

    def wait_served(self, rid: int, poll: float = 0.5) -> torch.Tensor:
        with self._cond:
            while self.served_id < rid:
                if self.error is not None:
                    raise RuntimeError("LM thread failed") from self.error
                self._cond.wait(timeout=poll)
            return self.table

    def request_stop(self) -> None:
        with self._cond:
            self.stop = True
            self._cond.notify_all()

    def current_table(self) -> Optional[torch.Tensor]:
        with self._cond:
            return self.table

    def publish_gnn(self, table: torch.Tensor, version: int) -> None:
        with self._cond:
            self.gnn_table = table
            self.gnn_table_version = int(version)
            self._cond.notify_all()

    # -- LM side ------------------------------------------------------------
    def snapshot_flags(self):
        with self._cond:
            return int(self.stop), int(self.request_id)

    def current_gnn_table(self):
        with self._cond:
            return self.gnn_table, self.gnn_table_version

    def publish(self, table: torch.Tensor, rid: int) -> None:
        with self._cond:
            self.table = table
            self.table_version += 1
            self.served_id = max(self.served_id, int(rid))
            self._cond.notify_all()

    def mark_error(self, exc: BaseException) -> None:
        with self._cond:
            self.error = exc
            self.stop = True
            self._cond.notify_all()
