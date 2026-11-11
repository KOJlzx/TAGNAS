#!/usr/bin/env python3
"""test6 dual-lane overlap timeline plot: draws the graph / text lanes using the absolute
t_start/t_end from unified telemetry.

Visually demonstrates that TAGNAS's "effective iteration ~= max of the two lanes'
wall-clock time" rather than their sum -- the graph worker (CPU) and the text worker (GPU)
**run concurrently and overlap** on the same wall-clock axis, so summing each lane's
buckets would overestimate the true wall-clock time.

Reads ``<in_dir>/telemetry/tagnas_<dataset>_*_*.jsonl``, splits into graph/text lanes by the
worker field, takes the first N iterations of each lane, and colors them by phase using
broken_barh on a shared x-axis (wall-clock seconds, zeroed at min t_start).
Outputs ``results/test6/overlap_<dataset>.png``.

Depends on matplotlib (Agg backend); follows the plotting conventions of
make_time_to_accuracy.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import Patch      # noqa: E402

# Phase order + color scheme (consistent with the breakdown table)
PHASE_KEYS = ["build_batch", "graph_fwd", "lm_fwd", "xfer", "backward", "sync_wait", "update"]
PHASE_COLOR = {
    "build_batch": "#9e9e9e",
    "graph_fwd":   "#1f77b4",
    "lm_fwd":      "#ff7f0e",
    "xfer":        "#2ca02c",
    "backward":    "#d62728",
    "sync_wait":   "#9467bd",
    "update":      "#8c564b",
}
RESIDUAL_COLOR = "#e8e8e8"
LANE_Y = {"graph": 1.0, "text": 0.0}
LANE_LABEL = {"graph": "graph worker (CPU)", "text": "text worker (GPU)"}


def load_lane_iters(tel_dir: str, dataset: str, seed: str | None) -> Dict[str, List[Dict[str, Any]]]:
    """Read TAGNAS's per-iteration JSONL and group by worker lane (each record contains t_start/t_end/phases_s)."""
    lanes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(tel_dir, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("system") != "tagnas" or str(d.get("dataset")) != dataset:
                    continue
                if seed is not None and str(d.get("seed")) != str(seed):
                    continue
                lane = str(d.get("worker", ""))
                if lane in LANE_Y:
                    lanes[lane].append(d)
    for lane in lanes:
        lanes[lane].sort(key=lambda r: float(r.get("t_start", 0.0)))
    return lanes


def make_figure(lanes: Dict[str, List[Dict[str, Any]]], dataset: str, n_iters: int,
                out_path: str) -> None:
    iters = [r for lane in lanes.values() for r in lane[:n_iters]]
    if not iters:
        print(f"[overlap] {dataset}: no usable iterations, skipping")
        return
    t0 = min(float(r["t_start"]) for r in iters)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    height = 0.8
    for lane, rows in lanes.items():
        y = LANE_Y[lane] - height / 2.0
        for r in rows[:n_iters]:
            seg_start = float(r["t_start"]) - t0
            phases = r.get("phases_s", {}) or {}
            for k in PHASE_KEYS:
                w = float(phases.get(k, 0.0))
                if w <= 0:
                    continue
                ax.broken_barh([(seg_start, w)], (y, height), facecolors=PHASE_COLOR[k],
                               edgecolors="none")
                seg_start += w
            # residual (contrastive loss forward pass + framework overhead) fills to t_end
            t_end_rel = float(r["t_end"]) - t0
            if t_end_rel > seg_start:
                ax.broken_barh([(seg_start, t_end_rel - seg_start)], (y, height),
                               facecolors=RESIDUAL_COLOR, edgecolors="none")

    ax.set_yticks([LANE_Y["text"], LANE_Y["graph"]])
    ax.set_yticklabels([LANE_LABEL["text"], LANE_LABEL["graph"]])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel("wall-clock time (s, zeroed at first iteration)")
    ax.set_title(f"TAGNAS dual-lane overlap — {dataset}\n"
                 "(graph/text run concurrently ⇒ effective iter ≈ max, not sum)")
    handles = [Patch(facecolor=PHASE_COLOR[k], label=k) for k in PHASE_KEYS]
    handles.append(Patch(facecolor=RESIDUAL_COLOR, label="residual"))
    ax.legend(handles=handles, ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="test6 dual-lane overlap timeline plot")
    ap.add_argument("--in-dir", default="results/test6")
    ap.add_argument("--telemetry-dir", default=None, help="Default: <in-dir>/telemetry")
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated; default auto-discovers all TAGNAS datasets present in telemetry")
    ap.add_argument("--seed", default=None, help="Only plot the specified seed (default: mix all seeds)")
    ap.add_argument("--n-iters", type=int, default=30, help="Plot the first N iterations of each lane")
    ap.add_argument("--out-dir", default=None, help="Default: <in-dir>")
    args = ap.parse_args()

    tel_dir = args.telemetry_dir or os.path.join(args.in_dir, "telemetry")
    out_dir = args.out_dir or args.in_dir
    if not os.path.isdir(tel_dir):
        print(f"[overlap] telemetry directory does not exist: {tel_dir}")
        return

    if args.datasets:
        datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        datasets = set()
        for fp in glob.glob(os.path.join(tel_dir, "*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("system") == "tagnas":
                        datasets.add(str(d.get("dataset")))
                    break
        datasets = sorted(datasets)
    if not datasets:
        print("[overlap] no TAGNAS datasets found")
        return

    for ds in datasets:
        lanes = load_lane_iters(tel_dir, ds, args.seed)
        make_figure(lanes, ds, args.n_iters, os.path.join(out_dir, f"overlap_{ds}.png"))


if __name__ == "__main__":
    main()
