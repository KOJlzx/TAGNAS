"""Node-selection strategies for the graph and text training loops.

Two selection policies live here, both of which only decide *which* seed nodes a
batch covers; cache insertion, drop-on-miss and the RPC fallback are unchanged.

* :class:`StreamingNodeScheduler` walks a shuffled node list and reshuffles on
  wrap-around. Text workers use it over their own shard (see
  :func:`shard_node_ids`), so ``W`` workers cover the training set ``W`` times
  faster than when each of them independently shuffles the whole set.
* :class:`CacheAwareNodeScheduler` traverses every training node once per graph
  epoch. Within the remaining nodes of that epoch it prefers nodes whose text
  embeddings are cached, then fills the batch with uncached nodes. This keeps
  graph updates aligned where possible while every node, including isolated
  nodes, gets its own forward and embedding push within a bounded traversal.
  Cache misses never block traversal. The last graph batch can be partial so an
  epoch boundary does not mix repeated nodes into that batch.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence


def _unique_ints(node_ids: Iterable[int]) -> list[int]:
    # dict.fromkeys deduplicates while preserving order, avoiding repeated node scheduling.
    return [int(node_id) for node_id in dict.fromkeys(node_ids)]


def shard_node_ids(
    node_ids: Sequence[int],
    num_shards: int,
    shard_index: int,
    seed: int = 42,
) -> list[int]:
    """Deterministic contiguous shard of a shuffled copy of ``node_ids``.

    Every rank computes the same shuffle from ``seed``, so the shards partition the
    node set exactly: they are disjoint, their union is the deduplicated input, and
    their sizes differ by at most one. Shuffling before slicing avoids handing one
    worker a contiguous id range, which would correlate its shard with graph locality.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for {num_shards} shards")

    nodes = _unique_ints(node_ids)
    if not nodes:
        raise ValueError("node_ids must be non-empty")
    random.Random(seed).shuffle(nodes)

    base, extra = divmod(len(nodes), num_shards)
    # The first `extra` shards take one node more than the rest.
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return nodes[start:start + size]


class StreamingNodeScheduler:
    """Infinite per-role sampler that reshuffles when it wraps around."""

    def __init__(
        self,
        node_ids: Sequence[int],
        batch_size: int,
        seed: int = 42,
    ) -> None:
        unique_nodes = _unique_ints(node_ids)
        if not unique_nodes:
            raise ValueError("node_ids must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._rng = random.Random(seed)
        self.node_sequence = unique_nodes[:]
        self._rng.shuffle(self.node_sequence)
        self.batch_size = batch_size
        self.cursor = 0
        self.completed_cycles = 0

    def next_batch(self) -> list[int]:
        batch: list[int] = []
        total = len(self.node_sequence)
        while len(batch) < self.batch_size:
            remaining = total - self.cursor
            if remaining == 0:
                self._reshuffle()
                continue

            take = min(self.batch_size - len(batch), remaining)
            batch.extend(self.node_sequence[self.cursor:(self.cursor + take)])
            self.cursor += take
            if self.cursor >= total:
                self.completed_cycles += 1
        return batch

    def _reshuffle(self) -> None:
        self._rng.shuffle(self.node_sequence)
        self.cursor = 0


class CacheAwareNodeScheduler:
    """Cover each training node once per epoch, preferring remaining cached nodes.

    ``cached_node_ids`` is read once per batch; its order cannot affect selection.
    A seeded permutation of the training nodes resolves ties. Cached nodes that
    were already emitted this epoch are ineligible. When fewer than one batch of
    eligible cached nodes remain, uncached nodes fill it immediately, preserving
    asynchronous forward/push/drop-on-miss behavior even if the cache is empty,
    permanently incomplete, or unavailable.

    Every epoch has exactly ``ceil(num_nodes / batch_size)`` nonempty batches.
    Its final batch may be partial. ``completed_cycles`` advances only after all
    training nodes have been emitted once; ``cursor`` counts nodes in the current
    epoch and returns to zero on completion.

    ``explore_every`` is a deprecated compatibility argument. Nonnegative values,
    including zero, all use the same mandatory coverage policy. ``explore_batches``
    now counts batches containing at least one node absent from that batch's cache
    snapshot; ``last_batch_from_cache`` is true only if every emitted node was cached.
    """

    def __init__(
        self,
        node_ids: Sequence[int],
        batch_size: int,
        cached_node_ids: Callable[[], Iterable[int]],
        seed: int = 42,
        explore_every: int = 4,
    ) -> None:
        unique_nodes = _unique_ints(node_ids)
        if not unique_nodes:
            raise ValueError("node_ids must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if explore_every < 0:
            raise ValueError("explore_every must be non-negative")

        self._rng = random.Random(seed)
        self._cached_node_ids = cached_node_ids
        self.node_sequence = unique_nodes[:]
        self._rng.shuffle(self.node_sequence)
        self._train_set = set(self.node_sequence)
        self.batch_size = batch_size
        self.explore_every = explore_every
        self.cursor = 0
        self.completed_cycles = 0
        self.batches_emitted = 0
        self.explore_batches = 0
        self._remaining = self.node_sequence[:]
        self.last_batch_from_cache = False

    def next_batch(self) -> list[int]:
        if not self._remaining:
            self._rng.shuffle(self.node_sequence)
            self._remaining = self.node_sequence[:]

        try:
            cached = {node_id for node_id in self._cached_node_ids() if node_id in self._train_set}
        except Exception:  # noqa: BLE001 - a cache read must never kill the training loop
            cached = set()

        take = min(self.batch_size, len(self._remaining))
        batch = [node_id for node_id in self._remaining if node_id in cached][:take]
        if len(batch) < take:
            for node_id in self._remaining:
                if node_id not in cached:
                    batch.append(node_id)
                    if len(batch) == take:
                        break

        selected = set(batch)
        self._remaining = [node_id for node_id in self._remaining if node_id not in selected]
        self.last_batch_from_cache = all(node_id in cached for node_id in batch)
        self.explore_batches += int(not self.last_batch_from_cache)
        self.batches_emitted += 1
        self.cursor += len(batch)
        if not self._remaining:
            self.completed_cycles += 1
            self.cursor = 0
        return batch


class TokenBudgetNodeScheduler:
    """Length-bucketed text chunks bounded by a padded-token budget instead of a row count.

    Text chunks are padded to their longest row, and node texts are very unevenly long (citation:
    mean 98 tokens, median 50, p99 406, max 1024), so a fixed-row chunk usually pays for one long
    outlier in every row. Memory of a text step grows with rows x padded length (measured linear in
    both on distilgpt2 with SDPA), so bounding that product bounds memory the way a fixed row count
    bounded it for the worst case, while short-text chunks carry many more rows.

    Each pass shuffles the node set, cuts it into windows of ``window`` nodes, sorts each window by
    length and slices it greedily into chunks with ``rows * max_len <= token_budget`` (and at most
    ``max_rows`` rows), then shuffles the chunk order -- so chunks are length-homogeneous but their
    contents and order stay random from pass to pass. A single row longer than the budget still
    forms a chunk of one. ``completed_cycles`` counts full passes, like StreamingNodeScheduler.
    """

    def __init__(
        self,
        node_ids: Sequence[int],
        lengths: dict[int, int] | Callable[[int], int],
        token_budget: int,
        max_rows: int = 1024,
        seed: int = 42,
        window: int = 8192,
    ) -> None:
        unique_nodes = _unique_ints(node_ids)
        if not unique_nodes:
            raise ValueError("node_ids must be non-empty")
        if token_budget <= 0 or max_rows <= 0 or window <= 0:
            raise ValueError("token_budget, max_rows and window must be positive")
        get = lengths.__getitem__ if isinstance(lengths, dict) else lengths
        self._len = {n: max(1, int(get(n))) for n in unique_nodes}
        self._rng = random.Random(seed)
        self.node_sequence = unique_nodes[:]
        self.token_budget = int(token_budget)
        self.max_rows = int(max_rows)
        self.window = int(window)
        self.completed_cycles = 0
        self.batches_emitted = 0
        self._chunks: list[list[int]] = []
        self._plan()

    def _plan(self) -> None:
        self._rng.shuffle(self.node_sequence)
        chunks: list[list[int]] = []
        for start in range(0, len(self.node_sequence), self.window):
            win = sorted(self.node_sequence[start:start + self.window], key=self._len.__getitem__)
            cur: list[int] = []
            cur_max = 0
            for n in win:
                new_max = max(cur_max, self._len[n])
                if cur and ((len(cur) + 1) * new_max > self.token_budget or len(cur) >= self.max_rows):
                    chunks.append(cur)
                    cur, new_max = [], self._len[n]
                cur.append(n)
                cur_max = new_max
            if cur:
                chunks.append(cur)
        self._rng.shuffle(chunks)
        self._chunks = chunks

    def next_batch(self) -> list[int]:
        if not self._chunks:
            self.completed_cycles += 1
            self._plan()
        self.batches_emitted += 1
        return self._chunks.pop()
