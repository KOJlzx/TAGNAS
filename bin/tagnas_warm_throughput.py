#!/usr/bin/env python3
"""Compute post-warmup throughput from TAGNAS's search/metrics.jsonl (excluding cold start).

TAGNAS is an asynchronous drop-on-miss system: after startup there is a cold-start
period -- the text workers first push embeddings into the cache, and only once the
cache has filled enough peer embeddings to assemble a graph batch does an "aligned
update" (a real training step) occur. This cold-start period drags down the average
rate `text_pushes_total / total_seconds`.

This tool moves the start time from 0 to "the moment the cache warmup finished" =
the uptime_seconds of the first record in metrics.jsonl where `--signal` (default
graph_updates_total) > 0, then computes:

    warm_throughput = (final[numerator] - warm[numerator])
                      / (final[uptime_seconds] - warm[uptime_seconds])

It only prints a single float; for any case where the value can't be obtained
(missing file / not enough records / never warmed up / dt<=0), it prints "NA",
so bash `$(...)` can consume it as a number or treat it as N/A. It does not touch
the training runtime -- it's purely a post-hoc computation from existing metrics.
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_records(path: str) -> list[dict]:
    recs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def warm_throughput(path: str, signal: str, numerator: str) -> float | None:
    try:
        recs = _load_records(path)
    except OSError:
        return None
    if len(recs) < 2:
        return None
    final = recs[-1]
    # Warm start point: the first record where signal > 0 (= the cache has warmed up
    # enough to assemble a batch, i.e. the first aligned update).
    warm = None
    for r in recs:
        try:
            if float(r.get(signal, 0)) > 0:
                warm = r
                break
        except (TypeError, ValueError):
            continue
    if warm is None:
        return None  # Never warmed up (signal is all 0, usually a bad run that OOM'd or hung).
    try:
        num = float(final.get(numerator, 0)) - float(warm.get(numerator, 0))
        dt = float(final.get("uptime_seconds", 0)) - float(warm.get("uptime_seconds", 0))
    except (TypeError, ValueError):
        return None
    if dt <= 0:
        return None
    return num / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", help="Path to a TAGNAS run's search/metrics.jsonl")
    ap.add_argument("--signal", default="graph_updates_total",
                    help="Cumulative counter field used to determine \"warmup complete\"; the start point is the first time it is >0 (default graph_updates_total)")
    ap.add_argument("--numerator", default="text_pushes_total",
                    help="Cumulative counter field used as the throughput numerator (default text_pushes_total = text-side GPU throughput)")
    args = ap.parse_args()

    thr = warm_throughput(args.metrics, args.signal, args.numerator)
    print("NA" if thr is None else f"{thr:.4f}")


if __name__ == "__main__":
    sys.exit(main())
