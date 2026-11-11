"""Embedding lag: measurement and controlled injection (asynchronous push, drop-on-miss).

A worker trains on the peer's embeddings as they sit in its cache; they may have been produced several
of the peer's steps ago. Three lags are measured for every embedding that is actually used in an update:

  * age      -- seconds since the producer created it (all processes share one host clock);
  * steps    -- how many successful updates the producer has made since producing it;
  * batches  -- how many batches the producer has processed since (dropped batches included).

The producer's *current* step/batch is known exactly by the coordinator, which sees every push the
moment it arrives (before any injected delay), so consumers only report what they used
(:func:`used_lag_payload`) and the coordinator subtracts (:class:`LagTracker`).

Injection (:class:`DelayBuffer`): the coordinator holds a push back from the global cache and the
broadcast until its producer has processed ``steps`` more batches and ``seconds`` have passed since it
was created. Batches, not successful updates, gate the release: with both directions delayed by
updates, each side would wait for the other's first update and neither would ever train.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any, Iterable

DIRECTIONS = ("both", "graph_to_text", "text_to_graph")


def delayed_modality(modality: str, direction: str) -> bool:
    """True when pushes of ``modality`` ("graph"/"text") are delayed under ``direction``."""
    if direction not in DIRECTIONS:
        raise ValueError(f"lag_inject_direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction == "both" or (direction == "graph_to_text" and modality == "graph") or \
        (direction == "text_to_graph" and modality == "text")


def used_lag_payload(records: Iterable[Any], now: float | None = None) -> dict[str, Any]:
    """Compact description of the peer embeddings a batch trained on (consumer side)."""
    now = time.time() if now is None else now
    by_producer: dict[str, dict[str, list[int]]] = {}
    ages: list[float] = []
    for r in records:
        d = by_producer.setdefault(str(r.producer), {"steps": [], "batches": []})
        d["steps"].append(int(getattr(r, "producer_step", -1)))
        d["batches"].append(int(getattr(r, "producer_batch", -1)))
        ages.append(round(max(0.0, now - float(r.created_at)), 1))
    return {"by_producer": by_producer, "ages": ages}


def _quantile(counter: Counter, q: float) -> float | None:
    n = sum(counter.values())
    if n == 0:
        return None
    target, acc = q * (n - 1), 0
    for value in sorted(counter):
        acc += counter[value]
        if acc > target:
            return float(value)
    return float(max(counter))


class LagTracker:
    """Coordinator side: producers' latest step/batch, and exact lag distributions per direction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest: dict[str, tuple[int, int]] = {}          # producer -> (step, batch)
        self._dist: dict[str, dict[str, Counter]] = {}        # direction -> {"steps"|"batches"|"age": Counter}

    def update_from_records(self, records: Iterable[Any]) -> None:
        with self._lock:
            for r in records:
                p = str(r.producer)
                s0, b0 = self.latest.get(p, (-1, -1))
                self.latest[p] = (max(s0, int(getattr(r, "producer_step", -1))),
                                  max(b0, int(getattr(r, "producer_batch", -1))))

    def latest_batch(self, producer: str) -> int:
        with self._lock:
            return self.latest.get(producer, (-1, -1))[1]

    def record_usage(self, direction: str, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        with self._lock:
            d = self._dist.setdefault(direction, {"steps": Counter(), "batches": Counter(), "age": Counter()})
            for producer, vals in payload.get("by_producer", {}).items():
                ls, lb = self.latest.get(producer, (-1, -1))
                for s in vals.get("steps", []):
                    if s >= 0 and ls >= 0:
                        d["steps"][max(0, ls - s)] += 1
                for b in vals.get("batches", []):
                    if b >= 0 and lb >= 0:
                        d["batches"][max(0, lb - b)] += 1
            for a in payload.get("ages", []):
                d["age"][float(a)] += 1

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for direction, d in self._dist.items():
                row: dict[str, Any] = {}
                for name, c in d.items():
                    n = sum(c.values())
                    row[f"{name}_n"] = n
                    row[f"{name}_mean"] = (sum(k * v for k, v in c.items()) / n) if n else None
                    row[f"{name}_p50"] = _quantile(c, 0.5)
                    row[f"{name}_p90"] = _quantile(c, 0.9)
                    row[f"{name}_max"] = float(max(c)) if n else None
                out[direction] = row
        return out


class StalenessStats:
    """Coordinator side of the staleness study: exact histograms of the cache lag seen at lookup.

    Consumers send the lags themselves (they know the producer's current update count from the
    pushes they ingest): ``lag_G`` = graph worker's successful updates now minus the version
    (graph updates) of each C_G entry an LM replica hit, recorded only once that replica's C_G
    covers ``cover_gate`` of the training nodes; ``lag_T`` = a text replica's own successful updates
    now minus the version of each C_T entry the graph worker hit, in that replica's units.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hist: dict[str, Counter] = {}
        self.coverage: dict[str, float] = {}           # consumer -> latest train-node coverage of its peer cache
        self.gate_open_at: dict[str, float] = {}       # LM replica -> uptime its C_G coverage passed the gate

    def add(self, key: str, values: Iterable[int]) -> None:
        with self._lock:
            c = self._hist.setdefault(key, Counter())
            for v in values:
                c[int(v)] += 1

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for key, c in sorted(self._hist.items()):
                n = sum(c.values())
                out[key] = {"n": n, "mean": (sum(k * v for k, v in c.items()) / n) if n else None,
                            "p50": _quantile(c, 0.5), "p90": _quantile(c, 0.9),
                            "max": float(max(c)) if n else None}
        return out

    def histograms(self) -> dict[str, dict[int, int]]:
        with self._lock:
            return {key: dict(sorted(c.items())) for key, c in self._hist.items()}


class DelayBuffer:
    """FIFO of held pushes; an item is due once its producer has processed ``steps`` more batches and
    ``seconds`` have passed since the item was created."""

    def __init__(self, steps: int = 0, seconds: float = 0.0) -> None:
        self.steps, self.seconds = int(steps), float(seconds)
        self._lock = threading.Lock()
        self._q: deque = deque()

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def add(self, producer: str, batch: int, created_at: float, item: Any) -> None:
        with self._lock:
            self._q.append((str(producer), int(batch), float(created_at), item))

    def pop_due(self, latest_batch_of, now: float | None = None) -> list[Any]:
        """Remove and return every due item (in arrival order); ``latest_batch_of(producer) -> int``."""
        now = time.time() if now is None else now
        due, keep = [], deque()
        with self._lock:
            for entry in self._q:
                producer, batch, created, item = entry
                if latest_batch_of(producer) - batch >= self.steps and now - created >= self.seconds:
                    due.append(item)
                else:
                    keep.append(entry)
            self._q = keep
        return due
