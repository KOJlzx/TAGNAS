"""Modality Collaborator: dynamic synchronisation schedule (UniTG Algorithm 1).

The GNN is fed LM embeddings every Δt; Δt starts large (0.3·B) and is tightened
through the schedule S = {0.3B, 0.2B, 0.1B, 0.02B} whenever the EMA-smoothed
GNN loss (F = 0.8·F + 0.2·L, the paper's §4.3 formula) has decreased for T=20
consecutive epochs. ``B`` is the training budget: the number of epochs in the
paper (fixed-epoch protocol) or, under this repository's wall-clock protocol,
the runtime cap in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class SyncScheduler:
    fractions: Sequence[float] = (0.3, 0.2, 0.1, 0.02)
    budget: float = 1000.0
    unit: str = "epochs"                 # informational: "epochs" or "seconds"
    tighten_after: int = 20
    ema: float = 0.8

    idx: int = 0
    ema_loss: Optional[float] = None
    consecutive_decreases: int = 0
    last_sync_mark: float = 0.0
    num_syncs: int = 0
    tighten_history: List[dict] = field(default_factory=list)

    @property
    def intervals(self) -> List[float]:
        return [float(f) * float(self.budget) for f in self.fractions]

    @property
    def current_interval(self) -> float:
        return self.intervals[self.idx]

    def observe_epoch_loss(self, loss: float, epoch: int) -> Optional[dict]:
        """Update the EMA and, if warranted, tighten Δt. Returns a tighten event or None."""
        loss = float(loss)
        prev = self.ema_loss
        if prev is None:
            self.ema_loss = loss
            return None
        self.ema_loss = self.ema * prev + (1.0 - self.ema) * loss
        if self.ema_loss < prev:
            self.consecutive_decreases += 1
        else:
            self.consecutive_decreases = 0
        if self.consecutive_decreases >= self.tighten_after and self.idx < len(self.fractions) - 1:
            self.idx += 1
            self.consecutive_decreases = 0
            event = {
                "event": "sync_tighten",
                "epoch": int(epoch),
                "idx": self.idx,
                "interval": self.current_interval,
                "unit": self.unit,
                "ema_loss": self.ema_loss,
            }
            self.tighten_history.append(event)
            return event
        return None

    def due(self, mark: float) -> bool:
        """Is a sync due at position ``mark`` (epoch index or elapsed seconds)?"""
        return (float(mark) - self.last_sync_mark) >= self.current_interval

    def mark_synced(self, mark: float) -> None:
        self.last_sync_mark = float(mark)
        self.num_syncs += 1

    def state(self) -> dict:
        return {
            "idx": self.idx,
            "interval": self.current_interval,
            "unit": self.unit,
            "ema_loss": self.ema_loss,
            "consecutive_decreases": self.consecutive_decreases,
            "num_syncs": self.num_syncs,
            "last_sync_mark": self.last_sync_mark,
        }
