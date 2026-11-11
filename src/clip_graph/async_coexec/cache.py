"""Local cache implementation used by the coordinator and the workers."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Iterable

import torch

from .types import CacheKey, EmbeddingRecord


_COPY_CHUNK_BYTES = 4 * 1024 * 1024
_COPY_CHUNK_ROWS = 1024


def clone_embedding_records(
    records: Iterable[EmbeddingRecord],
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> list[EmbeddingRecord]:
    """Copy embedding batches without retaining a large slab behind each row.

    Homogeneous dense rows share one staging allocation and device transfer per
    bounded chunk. ``foreach_copy_`` then copies the rows into independent owning
    allocations on the destination device. Replacing or evicting one cache key
    therefore never keeps an otherwise unused batch allocation alive.

    CPU results are complete on return, including CUDA-to-CPU transfers. CUDA
    results follow ordinary PyTorch current-stream ordering; callers publishing
    them to another thread/stream must establish an event dependency, as
    ``AsyncEmbeddingCache`` does below.
    """
    source = list(records)
    if not source:
        return []

    target = torch.device(device)
    normalized: list[EmbeddingRecord | None] = [None] * len(source)
    groups: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], list[int]] = {}
    for index, record in enumerate(source):
        tensor = record.tensor
        if tensor.layout != torch.strided or tensor.is_quantized:
            # Embeddings are normally dense. Preserve the prior API for other
            # layouts rather than trying to unbind a sparse staging tensor.
            copied = tensor.detach().to(device=target, dtype=dtype or tensor.dtype).clone()
            normalized[index] = _record_with_tensor(record, copied)
            continue
        signature = (tensor.device, tensor.dtype, tuple(tensor.shape))
        groups.setdefault(signature, []).append(index)

    for indices in groups.values():
        first = source[indices[0]].tensor
        target_element_size = torch.empty((), dtype=dtype or first.dtype).element_size()
        row_bytes = max(1, first.numel() * max(first.element_size(), target_element_size))
        rows_per_chunk = max(1, min(_COPY_CHUNK_ROWS, _COPY_CHUNK_BYTES // row_bytes))
        for offset in range(0, len(indices), rows_per_chunk):
            chunk = indices[offset:offset + rows_per_chunk]
            staged = torch.stack([source[index].tensor.detach() for index in chunk])
            # Blocking host transfers ensure a source staging allocation cannot
            # disappear while a DMA is reading it. There is one transfer per
            # chunk, rather than one blocking transfer per embedding.
            staged = staged.to(device=target, dtype=dtype or staged.dtype, non_blocking=False)
            rows = list(staged.unbind(0))
            owned = [torch.empty_like(row) for row in rows]
            foreach_copy = getattr(torch, "_foreach_copy_", None)
            if foreach_copy is None:
                # Compatibility fallback still transfers across devices only
                # once; these copies are all on the destination device.
                for destination, row in zip(owned, rows):
                    destination.copy_(row)
            else:
                foreach_copy(owned, rows)
            for index, tensor in zip(chunk, owned):
                normalized[index] = _record_with_tensor(source[index], tensor)

    return [record for record in normalized if record is not None]


def _record_with_tensor(record: EmbeddingRecord, tensor: torch.Tensor) -> EmbeddingRecord:
    return EmbeddingRecord(
        key=record.key,
        tensor=tensor,
        version=record.version,
        producer=record.producer,
        created_at=record.created_at,
        producer_step=record.producer_step,
        producer_batch=record.producer_batch,
    )


class AsyncEmbeddingCache:
    """Version-aware LRU cache for graph/text embeddings."""

    def __init__(
        self,
        device: str | torch.device,
        capacity: int,
        dtype: torch.dtype = torch.float32,
        name: str = "cache",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.device = torch.device(device)
        self.capacity = capacity
        self.dtype = dtype
        self.name = name

        self._store: OrderedDict[CacheKey, EmbeddingRecord] = OrderedDict()
        # Events are shared by all rows in a put. They protect RPC-thread writes
        # consumed on a training thread's CUDA stream without a device-wide sync.
        self._ready_events: dict[CacheKey, torch.cuda.Event] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def contains(self, keys: Iterable[CacheKey]) -> list[bool]:
        with self._lock:
            return [key in self._store for key in keys]

    def get_many(self, keys: Iterable[CacheKey]) -> dict[CacheKey, EmbeddingRecord]:
        with self._lock:
            hits: dict[CacheKey, EmbeddingRecord] = {}
            ready_events: dict[int, torch.cuda.Event] = {}
            for key in keys:
                record = self._store.get(key)
                if record is None:
                    continue
                # Mark cache hits as recently used, matching standard LRU behavior.
                self._store.move_to_end(key)
                hits[key] = record
                event = self._ready_events.get(key)
                if event is not None:
                    ready_events[id(event)] = event

        # CUDA calls and stream dependencies stay outside the dictionary lock.
        # record_stream also prevents eviction on another thread from allowing
        # allocator reuse before the consumer stream has finished with a row.
        if ready_events:
            stream = torch.cuda.current_stream(self.device)
            for event in ready_events.values():
                stream.wait_event(event)
            for record in hits.values():
                record.tensor.record_stream(stream)
        return hits

    def missing_keys(self, keys: Iterable[CacheKey]) -> list[CacheKey]:
        with self._lock:
            return [key for key in keys if key not in self._store]

    def node_ids(self, modality: str | None = None) -> list[int]:
        """Node ids currently held, optionally restricted to one modality.

        Used by the graph side to sample seed nodes whose peer embeddings are already
        cached. Unlike ``get_many`` this does not touch the LRU order, so polling it
        every batch cannot change which entries get evicted.
        """
        with self._lock:
            return [
                key.node_id
                for key in self._store
                if modality is None or key.modality == modality
            ]

    def put_many(self, records: Iterable[EmbeddingRecord]) -> tuple[list[CacheKey], list[CacheKey]]:
        # Materialize before any mutation: a failing generator or failed device
        # transfer must not leave a partly committed cache update.
        incoming = list(records)
        if not incoming:
            return [], []

        candidates: list[EmbeddingRecord] = []
        candidate_indices: list[int] = []
        with self._lock:
            observed_versions: dict[CacheKey, int] = {}
            for index, record in enumerate(incoming):
                current = self._store.get(record.key)
                observed = observed_versions.get(record.key)
                if observed is None and current is not None:
                    observed = current.version
                if observed is not None and observed >= record.version:
                    continue
                observed_versions[record.key] = record.version
                candidates.append(record)
                candidate_indices.append(index)

        # All tensor allocation, batching and transfers happen without the LRU
        # lock, so readers can keep training while a push is being prepared.
        prepared = clone_embedding_records(candidates, self.device, self.dtype)
        by_index = dict(zip(candidate_indices, prepared))
        ready_event = None
        if prepared and self.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(self.device))

        inserted: list[CacheKey] = []
        skipped: list[CacheKey] = []
        with self._lock:
            for index, record in enumerate(incoming):
                normalized = by_index.get(index)
                current = self._store.get(record.key)
                if normalized is None or (current is not None and current.version >= record.version):
                    # Recheck after the out-of-lock transfer: another RPC thread
                    # may have installed a newer version while this put copied.
                    skipped.append(record.key)
                    continue
                self._store[record.key] = normalized
                if ready_event is not None:
                    self._ready_events[record.key] = ready_event
                self._store.move_to_end(record.key)
                inserted.append(record.key)

            self._evict_if_needed()

        return inserted, skipped

    def snapshot_versions(self) -> dict[CacheKey, int]:
        with self._lock:
            return {key: record.version for key, record in self._store.items()}

    def _evict_if_needed(self) -> None:
        # last=False pops the oldest OrderedDict entry, i.e. the least-recently used item.
        while len(self._store) > self.capacity:
            key, _ = self._store.popitem(last=False)
            self._ready_events.pop(key, None)
