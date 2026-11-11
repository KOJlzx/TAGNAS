"""Opt-in CPU slab transport for the existing embedding RPC protocol.

The training services, synchronous push acknowledgement, fresh-version filter,
fanout scheduling, lag injection and cache ownership remain in train_runtime.
Only the wire representation changes: thousands of tiny TensorPipe tensor-table
entries become bounded slabs. This module must be installed in every process of
a separately identified run, after any retained runtime remapping.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import functools
from typing import Any, Iterable

import torch

from .cache import clone_embedding_records
from .types import CacheKey, EmbeddingRecord

REVISION = "tagnas-compact-cpu-wire-v1"
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_CHUNK_ROWS = 1024
DEFAULT_MIN_RECORDS = 128

# No lossy integer/float conversion or compression: all record fields survive.
Metadata = tuple[str, int, int, str, float, int, int]


@dataclass(frozen=True)
class PackedEmbeddingRecords:
    """Transport-only slabs; views never become persistent cache entries."""

    revision: str
    metadata: tuple[Metadata, ...]
    slabs: tuple[torch.Tensor, ...]


def can_pack_records(records: Any, min_records: int = 0) -> bool:
    """Retain the native protocol for unsupported layouts or non-CPU inputs."""
    return isinstance(records, (list, tuple)) and len(records) >= min_records and all(
        isinstance(record, EmbeddingRecord) and record.tensor.device.type == "cpu"
        and record.tensor.layout == torch.strided and not record.tensor.is_quantized
        for record in records
    )


def pack_records(
    records: Iterable[EmbeddingRecord], *, max_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> PackedEmbeddingRecords:
    """Snapshot CPU rows without changing order, values, duplicates or metadata.

    A chunk is bounded by both limits, except that one oversized row necessarily
    occupies its own chunk. Mixed shapes/dtypes start new contiguous chunks.
    Packing owns a snapshot; mutating or evicting source rows cannot change it.
    """
    if max_chunk_bytes < 1 or max_chunk_rows < 1:
        raise ValueError("Chunk bounds must be positive")
    source = list(records)
    if not can_pack_records(source):
        raise ValueError("Compact wire accepts dense, non-quantized CPU records only")
    return _pack_validated_records(source, max_chunk_bytes, max_chunk_rows)


def _pack_validated_records(source, max_chunk_bytes, max_chunk_rows):
    """Internal fast path after the caller has validated every source row once."""
    metadata = tuple((r.key.modality, r.key.node_id, r.version, r.producer, r.created_at,
                      r.producer_step, r.producer_batch) for r in source)
    slabs = []
    begin = 0
    while begin < len(source):
        first = source[begin].tensor
        signature = (first.dtype, tuple(first.shape))
        row_bytes = max(1, first.numel() * first.element_size())
        bound = min(max_chunk_rows, max(1, max_chunk_bytes // row_bytes))
        end = begin + 1
        while end < min(len(source), begin + bound):
            candidate = source[end].tensor
            if (candidate.dtype, tuple(candidate.shape)) != signature:
                break
            end += 1
        # One no-grad context avoids thousands of individual Python/C++ detach
        # calls while still producing a constant, independently owned snapshot.
        with torch.no_grad():
            slabs.append(torch.stack([r.tensor for r in source[begin:end]]))
        begin = end
    return PackedEmbeddingRecords(REVISION, metadata, tuple(slabs))


def unpack_records(payload: PackedEmbeddingRecords, *, independent: bool = True) -> list[EmbeddingRecord]:
    """Restore records; callers normally receive independently owned row tensors.

    The two push handlers use ephemeral views only while the native handler
    immediately clones into its original independent-owning cache/delay records.
    This avoids an extra allocation/copy and cannot retain a slab in a cache.
    """
    if not isinstance(payload, PackedEmbeddingRecords) or payload.revision != REVISION:
        raise ValueError("Unrecognized compact embedding payload")
    if any(s.device.type != "cpu" or s.layout != torch.strided or s.is_quantized or s.ndim < 1
           for s in payload.slabs):
        raise ValueError("Compact slabs must be dense CPU tensors with a row dimension")
    if sum(int(s.shape[0]) for s in payload.slabs) != len(payload.metadata):
        raise ValueError("Compact row/metadata count mismatch")
    result = []
    offset = 0
    for slab in payload.slabs:
        for row in slab.unbind(0):
            modality, node_id, version, producer, created_at, step, batch = payload.metadata[offset]
            result.append(EmbeddingRecord(CacheKey(modality, node_id), row, version, producer,
                                          created_at, step, batch))
            offset += 1
    return clone_embedding_records(result, "cpu") if independent else result


# Importable top-level handlers are essential: TensorPipe serializes functions by
# module and qualified name. Do not replace runtime's original named handlers.
def receive_compact_push(modality, records, source_name, telemetry=None):
    from . import train_runtime

    return train_runtime._rpc_receive_push(
        modality, unpack_records(records, independent=False), source_name, telemetry)


def ingest_compact_push(modality, records, source_name):
    from . import train_runtime

    return train_runtime._rpc_ingest_remote_push(
        modality, unpack_records(records, independent=False), source_name)


def lookup_compact_cache(modality, node_ids, min_records, max_chunk_bytes, max_chunk_rows):
    from . import train_runtime

    records = train_runtime._rpc_lookup_cache(modality, node_ids)
    if not can_pack_records(records, min_records):
        return records
    return _pack_validated_records(records, max_chunk_bytes, max_chunk_rows)


@dataclass
class Installation:
    """Explicit reversible process-local installation; never edits source files."""

    rpc: Any
    original_sync: Any
    original_async: Any
    wrapped_sync: Any
    wrapped_async: Any

    def restore(self):
        if self.rpc.rpc_sync is not self.wrapped_sync or self.rpc.rpc_async is not self.wrapped_async:
            raise RuntimeError("Cannot restore after another wrapper has replaced compact RPC")
        self.rpc.rpc_sync, self.rpc.rpc_async = self.original_sync, self.original_async


def install(runtime, *, min_records=DEFAULT_MIN_RECORDS, max_chunk_bytes=DEFAULT_CHUNK_BYTES,
            max_chunk_rows=DEFAULT_CHUNK_ROWS, observer=None) -> Installation:
    """Wrap only the exact embedding RPC calls; all other RPC calls pass through.

    Native sync/async status, exception behavior, timeout and per-call arguments
    stay intact. A small/unsupported push uses the unchanged original handler.
    ``observer`` is optional and uses host-only spans, adding no CUDA barriers.
    """
    if min_records < 1 or max_chunk_bytes < 1 or max_chunk_rows < 1:
        raise ValueError("Positive transport bounds required")
    rpc = runtime.rpc
    if getattr(rpc.rpc_sync, "__compact_transport_revision__", None):
        raise RuntimeError("Compact wire is already installed in this process")
    originals = rpc.rpc_sync, rpc.rpc_async
    push_handlers = {runtime._rpc_receive_push: receive_compact_push,
                     runtime._rpc_ingest_remote_push: ingest_compact_push}
    lookup_handler = runtime._rpc_lookup_cache

    def span(name):
        return observer.span(f"transport.{name}") if observer is not None else nullcontext()

    def wrap(original, asynchronous):
        @functools.wraps(original)
        def call(to, func, *positional, **options):
            # args/kwargs here are the remote function's, not RPC timeout options.
            remote_args = positional[0] if positional else options.get("args", ())
            remote_kwargs = positional[1] if len(positional) > 1 else options.get("kwargs")
            changed = False
            if func in push_handlers:
                args = list(remote_args or ())
                kwargs = dict(remote_kwargs or {})
                records = args[1] if len(args) > 1 else kwargs.get("records")
                if can_pack_records(records, min_records):
                    with span("pack"):
                        packed = _pack_validated_records(records, max_chunk_bytes, max_chunk_rows)
                    if len(args) > 1:
                        args[1] = packed
                    else:
                        kwargs["records"] = packed
                    remote_args, remote_kwargs = tuple(args), kwargs
                    func = push_handlers[func]
                    changed = True
            elif func is lookup_handler and not asynchronous:
                remote_args = tuple(remote_args or ())
                remote_kwargs = dict(remote_kwargs or {}, min_records=min_records,
                                     max_chunk_bytes=max_chunk_bytes, max_chunk_rows=max_chunk_rows)
                func, changed = lookup_compact_cache, True
            if changed:
                positional = list(positional)
                if positional:
                    positional[0] = remote_args
                else:
                    options["args"] = remote_args
                if len(positional) > 1:
                    positional[1] = remote_kwargs
                else:
                    options["kwargs"] = remote_kwargs
            result = original(to, func, *positional, **options)
            if isinstance(result, PackedEmbeddingRecords):
                with span("unpack_lookup"):
                    return unpack_records(result)
            return result

        call.__compact_transport_revision__ = REVISION
        return call

    wrapped_sync, wrapped_async = wrap(originals[0], False), wrap(originals[1], True)
    rpc.rpc_sync, rpc.rpc_async = wrapped_sync, wrapped_async
    return Installation(rpc, *originals, wrapped_sync, wrapped_async)
