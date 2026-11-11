"""Core types for the async graph/LLM co-execution skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import time

import torch

Modality = Literal["graph", "text"]


@dataclass(frozen=True, order=True)
class CacheKey:
    """Stable key used by all caches and single-flight lookups."""

    # Keep only the two dimensions required for cross-modal cache identity:
    # 1. modality distinguishes graph and text records.
    # 2. node_id is the shared cross-modal primary key.
    modality: Modality
    node_id: int


@dataclass
class EmbeddingRecord:
    """A versioned embedding payload stored and transferred across workers."""

    # Version is critical in the async system: stale late-arriving embeddings must
    # not overwrite newer values.
    key: CacheKey
    tensor: torch.Tensor
    version: int
    producer: str
    created_at: float = field(default_factory=time.time)
    # test6 staleness: global successful update step when this embedding was
    # produced. Consumers estimate async lag as consumer_step - producer_step.
    # -1 means unknown for backward compatibility. This is decoupled from version:
    # version controls cache freshness, producer_step is telemetry only.
    producer_step: int = -1
    # Batches the producer had processed (drops included) when this embedding was made; the lag
    # measurement and injection count in producer batches too (see lag.py). -1 = unknown.
    producer_batch: int = -1

    def clone_to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> "EmbeddingRecord":
        # Clone across caches/workers to avoid sharing underlying storage.
        target = self.tensor.detach()
        if dtype is not None:
            target = target.to(dtype=dtype)
        target = target.to(device=device).clone()
        return EmbeddingRecord(
            key=self.key,
            tensor=target,
            version=self.version,
            producer=self.producer,
            created_at=self.created_at,
            producer_step=self.producer_step,
            producer_batch=self.producer_batch,
        )
