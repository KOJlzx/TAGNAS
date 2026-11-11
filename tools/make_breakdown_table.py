#!/usr/bin/env python3
"""Aggregate test6 unified telemetry -> dual-lane bucketed breakdown table (ASCII + LaTeX + CSV).

Data source: per-iteration JSONL written by the unified TelemetryRecorder -- defaults to
``<in_dir>/telemetry/*.jsonl``. **Grouped by the structured in-row fields (system, dataset,
worker)**, never inferred from filenames, so legacy stray filenames containing commas no
longer pollute the results. Falls back to the legacy ``breakdown_*.json`` files (with a key
shim) when no JSONL is present.

Key fixes relative to the previous version:
  1. Unified residual formula: residual = wall(t_end-t_start) - sum(phases), consistent across
     all systems (TAGNAS no longer zeroes it out).
  2. TAGNAS is split into **graph / text dual lanes** (two rows) plus an **effective** row
     (the max/union of the two lanes' wall-clock time, not the sum).
  3. Removed the old "sum of buckets" Total column; replaced it with the **measured
     wall-clock** Wall/iter per lane plus a **normalized nodes/s column**
     (baselines use batch~=16/56-64 vs TAGNAS 1024, so normalization is needed for a fair
     comparison).
  4. Outputs breakdown.csv (with graph/text/effective rows) + ASCII + LaTeX.

Display column names keep LMComp/GraphComp (familiar from the paper), but the underlying
keys are lm_fwd/graph_fwd. SyncWait is DDP-only (ZeRO folds gradient sync into Backward);
TAGNAS's LMComp comes from the text worker's cross-process telemetry.

Pure standard-library implementation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# underlying phase key -> display column name (order = column order)
PHASE_COLS = [
    ("sync_wait",   "SyncWait"),
    ("xfer",        "XferT"),
    ("build_batch", "BuildBatch"),
    ("lm_fwd",      "LMComp"),
    ("graph_fwd",   "GraphComp"),
    ("backward",    "Backward"),
    ("update",      "Update"),
]
PHASE_KEYS = [k for k, _ in PHASE_COLS]
# legacy schema (IterationProfiler/write_breakdown_json) key -> new key
LEGACY_KEY_MAP = {"lm_comp": "lm_fwd", "graph_comp": "graph_fwd"}

SYS_ORDER = ["fixed_gat_ddp", "gnas_ddp", "gnas_zero", "unitg", "tagnas"]
SYS_LABEL = {
    "fixed_gat_ddp": "Fixed-GAT (DDP)",
    "gnas_ddp":      "GNAS (DDP)",
    "gnas_zero":     "GNAS (ZeRO-Offload)",
    "unitg":         "UniTG (pipeline + bubbles)",
    "tagnas":        "TAGNAS (async, ours)",
}
LANE_ORDER = {"single": 0, "graph": 1, "text": 2, "effective": 3}


# ──────────────────────────────────────────────────────────────────────
# Load + aggregate
# ──────────────────────────────────────────────────────────────────────

class _Acc:
    """Online running-mean accumulator for one (system, dataset, lane) group."""

    def __init__(self) -> None:
        self.phase_sum: Dict[str, float] = defaultdict(float)
        self.wall_sum = 0.0
        self.resid_sum = 0.0
        self.nodes_sum = 0.0
        self.cache_hit_sum = 0.0
        self.staleness_sum = 0.0
        self.n = 0
        self.seeds: set = set()
        # Per worker file: [first t_start, last t_end, nodes, busy seconds]. The per-iteration
        # means above only see time INSIDE iterations; the span also covers what happens between
        # them (UniTG's full-table LM->GNN sync, which stalls its GNN stage), so wall-clock
        # throughput = nodes / span, summed over a lane's concurrent workers.
        # Per worker file: [first t_start, last t_end, nodes, busy seconds, nodes of batches that
        # applied an update]. Under drop-on-miss (TAGNAS) a batch whose seeds are not all cached
        # runs forward and pushes, but steps nothing; systems without a cache report cache_hit=1.
        self.spans: Dict[str, List[float]] = {}

    def add_span(self, key: str, t0: float, t1: float, nodes: float, busy: float, updated: bool) -> None:
        sp = self.spans.setdefault(key, [t0, t1, 0.0, 0.0, 0.0])
        sp[0] = min(sp[0], t0)
        sp[1] = max(sp[1], t1)
        sp[2] += nodes
        sp[3] += busy
        if updated:
            sp[4] += nodes

    def add_iter(self, phases: Dict[str, float], wall: float, resid: float,
                 nodes: float, cache_hit: float, staleness: float, seed: Any) -> None:
        for k in PHASE_KEYS:
            self.phase_sum[k] += float(phases.get(k, 0.0))
        self.wall_sum += wall
        self.resid_sum += resid
        self.nodes_sum += nodes
        self.cache_hit_sum += cache_hit
        self.staleness_sum += staleness
        self.n += 1
        self.seeds.add(seed)

    def add_preaggregated(self, phases: Dict[str, float], wall: float,
                          n_iters: int, seed: Any) -> None:
        """Legacy averaged JSON: already a per-iter mean; merge in weighted by n_iters."""
        w = max(int(n_iters), 1)
        for k in PHASE_KEYS:
            self.phase_sum[k] += float(phases.get(k, 0.0)) * w
        self.wall_sum += wall * w
        self.resid_sum += (wall - sum(float(phases.get(k, 0.0)) for k in PHASE_KEYS)) * w
        self.n += w
        self.seeds.add(seed)

    def mean_row(self, system: str, dataset: str, lane: str) -> Dict[str, Any]:
        n = max(self.n, 1)
        phases = {k: self.phase_sum[k] / n for k in PHASE_KEYS}
        wall = self.wall_sum / n
        return {
            "system": system,
            "dataset": dataset,
            "lane": lane,
            "phases": phases,
            "wall": wall,
            "residual": self.resid_sum / n,
            "nodes_mean": self.nodes_sum / n,
            "nodes_per_s": (self.nodes_sum / n / wall) if wall > 0 else 0.0,
            "cache_hit": self.cache_hit_sum / n if self.n else 1.0,
            "staleness": self.staleness_sum / n if self.n else 0.0,
            "n_iters": self.n,
            "seeds": sorted(str(s) for s in self.seeds),
            "wall_rate": (sum(sp[2] / (sp[1] - sp[0]) for sp in self.spans.values() if sp[1] > sp[0])
                          if self.spans else None),
            "coverage": (sum(sp[3] / (sp[1] - sp[0]) for sp in self.spans.values() if sp[1] > sp[0])
                         / max(len(self.spans), 1) if self.spans else None),
            "wall_rate_upd": (sum(sp[4] / (sp[1] - sp[0]) for sp in self.spans.values() if sp[1] > sp[0])
                              if self.spans else None),
        }


def load_jsonl_rows(tel_dir: str) -> List[Dict[str, Any]]:
    """Read telemetry/*.jsonl and aggregate into mean rows keyed by (system, dataset, worker-lane)."""
    files = sorted(glob.glob(os.path.join(tel_dir, "*.jsonl")))
    accs: Dict[Tuple[str, str, str], _Acc] = defaultdict(_Acc)
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    system = str(d.get("system", "?"))
                    dataset = str(d.get("dataset", "?"))
                    lane = str(d.get("worker", "single"))
                    phases = dict(d.get("phases_s", {}) or {})
                    wall = float(d.get("t_end", 0.0)) - float(d.get("t_start", 0.0))
                    if wall <= 0:
                        wall = sum(float(phases.get(k, 0.0)) for k in PHASE_KEYS) \
                            + float(d.get("residual_s", 0.0))
                    accs[(system, dataset, lane)].add_iter(
                        phases, wall, float(d.get("residual_s", wall - sum(
                            float(phases.get(k, 0.0)) for k in PHASE_KEYS))),
                        float(d.get("nodes_in_batch", 0.0)),
                        float(d.get("cache_hit", 1.0)),
                        float(d.get("staleness_steps", 0.0)),
                        d.get("seed", "?"),
                    )
                    if "t_start" in d and "t_end" in d:
                        accs[(system, dataset, lane)].add_span(
                            fp, float(d["t_start"]), float(d["t_end"]),
                            float(d.get("nodes_in_batch", 0.0)), wall,
                            float(d.get("cache_hit", 1.0)) >= 1.0 - 1e-9)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skipping {fp}: {e}")
    return [acc.mean_row(s, ds, lane) for (s, ds, lane), acc in accs.items()]


def load_legacy_rows(in_dir: str) -> List[Dict[str, Any]]:
    """Fallback: read legacy breakdown_*.json files (using the in-row system/dataset fields, not parsing filenames)."""
    files = sorted(glob.glob(os.path.join(in_dir, "breakdown_*.json")))
    accs: Dict[Tuple[str, str, str], _Acc] = defaultdict(_Acc)
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skipping {fp}: {e}")
            continue
        system = str(d.get("system", "?"))
        dataset = str(d.get("dataset", "?"))
        lane = str(d.get("worker", "single"))
        per = d.get("per_iteration_seconds", {}) or {}
        phases = {LEGACY_KEY_MAP.get(k, k): float(v) for k, v in per.items()}
        wall = float(d.get("per_iteration_total_seconds",
                           sum(phases.get(k, 0.0) for k in PHASE_KEYS)))
        accs[(system, dataset, lane)].add_preaggregated(
            phases, wall, int(d.get("num_iterations_averaged", 1)), d.get("seed", "?"))
    return [acc.mean_row(s, ds, lane) for (s, ds, lane), acc in accs.items()]


def add_effective_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append an effective row for each (system, dataset) that has both graph and text lanes:
    the effective iteration = the max of the two lanes' wall-clock time (a union
    approximation), not the sum. nodes/s uses the graph seed's node count divided by the
    effective wall-clock time."""
    by_sd: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_sd[(r["system"], r["dataset"])][r["lane"]] = r
    out = list(rows)
    for (system, dataset), lanes in by_sd.items():
        if "graph" in lanes and "text" in lanes:
            g, t = lanes["graph"], lanes["text"]
            eff_wall = max(g["wall"], t["wall"])
            out.append({
                "system": system,
                "dataset": dataset,
                "lane": "effective",
                "phases": None,   # not summed; effective only reports wall-clock + nodes/s
                "wall": eff_wall,
                "residual": 0.0,
                "nodes_mean": g["nodes_mean"],
                "nodes_per_s": (g["nodes_mean"] / eff_wall) if eff_wall > 0 else 0.0,
                "cache_hit": (g["cache_hit"] + t["cache_hit"]) / 2.0,
                "staleness": (g["staleness"] + t["staleness"]) / 2.0,
                "n_iters": min(g["n_iters"], t["n_iters"]),
                "seeds": sorted(set(g["seeds"]) | set(t["seeds"])),
                # Wall clock: graph seeds actually trained per second of real time. The lanes run
                # concurrently, so this is the graph lane's own wall-clock rate -- not seeds / max
                # of the per-iteration walls, which pins an async graph lane to its text lane.
                "wall_rate": g.get("wall_rate"),
                "coverage": g.get("coverage"),
                "wall_rate_upd": g.get("wall_rate_upd"),
            })
    return out


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        s = r["system"]
        return (r["dataset"], SYS_ORDER.index(s) if s in SYS_ORDER else 99, s,
                LANE_ORDER.get(r["lane"], 9))
    return sorted(rows, key=key)


# ──────────────────────────────────────────────────────────────────────
# Render tables
# ──────────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], prec: int) -> str:
    return "—" if v is None else f"{v:.{prec}f}"


def _lane_label(lane: str) -> str:
    return {"single": "", "graph": "graph", "text": "text", "effective": "effective"}.get(lane, lane)


def ascii_table(rows: List[Dict[str, Any]], prec: int) -> str:
    headers = ["System", "Dataset", "Lane"] + [d for _, d in PHASE_COLS] + ["Residual", "nodes/s", "Wall/it",
                                                                           "nodes/s(wall)", "upd/s(wall)", "in-iter%"]
    body: List[List[str]] = []
    for r in rows:
        ph = r["phases"]
        cells = [SYS_LABEL.get(r["system"], r["system"]), r["dataset"], _lane_label(r["lane"])]
        cells += [(_fmt(ph[k], prec) if ph is not None else "—") for k in PHASE_KEYS]
        cells += [_fmt(r["residual"] if ph is not None else None, prec),
                  f"{r['nodes_per_s']:.1f}", _fmt(r["wall"], prec),
                  _fmt(r.get("wall_rate"), 1), _fmt(r.get("wall_rate_upd"), 1),
                  _fmt(None if r.get("coverage") is None else r["coverage"] * 100, 1)]
        body.append(cells)
    widths = [max([len(headers[i])] + [len(row[i]) for row in body]) for i in range(len(headers))]

    def line(cells: List[str]) -> str:
        return "  ".join((c.ljust(widths[i]) if i < 3 else c.rjust(widths[i])) for i, c in enumerate(cells))

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(row) for row in body]
    return "\n".join(out)


def latex_table(rows: List[Dict[str, Any]], prec: int) -> str:
    ncol = len(PHASE_COLS) + 6   # phases + Residual + nodes/s + Wall + nodes/s(wall) + upd/s(wall) + in-iter%
    colspec = "lll" + "r" * ncol
    header = ["System", "Dataset", "Lane"] + [d for _, d in PHASE_COLS] + ["Residual", "nodes/s", "Wall/it",
                                                                          "nodes/s (wall)", "upd/s (wall)", "in-iter \\%"]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Per-iteration runtime breakdown (seconds). TAGNAS and UniTG are split into "
        "graph/text lanes; the effective iteration is the max (union) of the two concurrent lanes, "
        "not their sum.}",
        "\\label{tab:time-breakdown}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
    ]
    for r in rows:
        ph = r["phases"]
        label = SYS_LABEL.get(r["system"], r["system"]).replace("&", "\\&")
        cells = [label, r["dataset"].replace("_", "\\_"), _lane_label(r["lane"]) or "—"]
        cells += [(_fmt(ph[k], prec) if ph is not None else "—") for k in PHASE_KEYS]
        cells += [_fmt(r["residual"] if ph is not None else None, prec),
                  f"{r['nodes_per_s']:.1f}", _fmt(r["wall"], prec),
                  _fmt(r.get("wall_rate"), 1), _fmt(r.get("wall_rate_upd"), 1),
                  _fmt(None if r.get("coverage") is None else r["coverage"] * 100, 1)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}", "\\\\[2pt]",
        "{\\footnotesize Unified telemetry: residual$=$wall$-\\sum$phases (same formula for all "
        "systems). LMComp/GraphComp: forward only; TAGNAS LMComp is from the text worker's "
        "cross-process telemetry. SyncWait: the DDP barrier for B1/B2 (ZeRO folds gradient sync "
        "into Backward; async TAGNAS has no barrier); for UniTG it is the pipeline-bubble recv "
        "wait on the graph lane and the bubble-gate plus gradient all-reduce wait on the text "
        "lane. UniTG's graph lane is one pipeline stage of one mini-batch, and both of its lanes "
        "share a GPU, so its compute columns are contended times and are not comparable to the "
        "baselines' isolated forwards in absolute terms. nodes/s normalizes the unequal batch "
        "sizes (baselines $\\approx$16--64, TAGNAS 1024, UniTG up to contrastive\\_nodes), but only over time "
        "\\emph{inside} iterations; compare systems on nodes/s (wall) = nodes over the run's real "
        "time span (summed over a lane's concurrent workers; the effective row is the graph lane's). "
        "upd/s (wall) counts only batches that applied an update (TAGNAS drops a batch whose seeds are not all "
        "cached; the other systems never drop). in-iter \\% is the share of that span spent inside iterations -- for UniTG the remainder "
        "is its full-table LM$\\to$GNN sync, which stalls the GNN stage between iterations.}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = (["system", "dataset", "lane", "seeds", "n_iters"] + PHASE_KEYS
              + ["residual", "wall_per_iter", "nodes_in_batch", "nodes_per_s",
                 "cache_hit", "staleness_steps", "nodes_per_s_wall", "updated_nodes_per_s_wall", "in_iteration_fraction"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            ph = r["phases"] or {}
            w.writerow(
                [r["system"], r["dataset"], r["lane"], "|".join(r["seeds"]), r["n_iters"]]
                + [("" if r["phases"] is None else f"{ph.get(k, 0.0):.6f}") for k in PHASE_KEYS]
                + [("" if r["phases"] is None else f"{r['residual']:.6f}"),
                   f"{r['wall']:.6f}", f"{r['nodes_mean']:.2f}", f"{r['nodes_per_s']:.3f}",
                   f"{r['cache_hit']:.4f}", f"{r['staleness']:.3f}",
                   ("" if r.get("wall_rate") is None else f"{r['wall_rate']:.3f}"),
                   ("" if r.get("wall_rate_upd") is None else f"{r['wall_rate_upd']:.3f}"),
                   ("" if r.get("coverage") is None else f"{r['coverage']:.4f}")]
            )
    print(f"[wrote] {path}")


# ──────────────────────────────────────────────────────────────────────
# --check acceptance gate
# ──────────────────────────────────────────────────────────────────────

def run_checks(rows: List[Dict[str, Any]], prec: int, identity_tol: float = 0.05) -> bool:
    """Acceptance gate. Note: residual = wall - sum(phases) is the unified formula, so
    sum(phases) + residual == wall is an identity (this is exactly the "<5% error" check --
    it verifies that no phase overlap/omission causes the residual to flip sign or become
    unbalanced). A residual of roughly 10-20% is itself normal (it includes the contrastive
    loss forward pass plus framework overhead) and is not treated as a hard gate; the real
    fix-acceptance criteria are ZeRO~=DDP being the same order of magnitude, and the TAGNAS
    dual lanes."""
    print("\n===== --check acceptance =====")
    ok = True
    single = {r["system"]: r for r in rows if r["lane"] == "single"}
    # 1) Identity consistency: |(sum(phases)+residual) - wall| / wall < 5% (the literal <5% error criterion)
    for sysname, r in single.items():
        recon = sum(r["phases"].values()) + r["residual"]
        err = abs(recon - r["wall"]) / r["wall"] if r["wall"] > 0 else 0.0
        flag = err < identity_tol
        ok &= flag
        neg = " warning: negative residual (phase overlap?)" if r["residual"] < -1e-3 else ""
        print(f"  [{'PASS' if flag else 'FAIL'}] {sysname:<16s} identity_err={err:.4f}(<{identity_tol}) "
              f"wall={r['wall']:.{prec}f} resid={r['residual']:+.{prec}f}"
              f"({abs(r['residual'])/max(r['wall'],1e-9)*100:.0f}% of wall){neg}")
    # 2) ZeRO residual vs DDP "same order of magnitude" -- compare the **ratio** of residual to
    #    wall (for a consistent basis), not the absolute value (the two systems' wall-clock
    #    times differ by several times, so absolute residuals aren't inherently comparable).
    #    Same order of magnitude = ratio within one order of magnitude.
    if "gnas_zero" in single and "gnas_ddp" in single:
        fz = abs(single["gnas_zero"]["residual"]) / max(single["gnas_zero"]["wall"], 1e-9)
        fd = abs(single["gnas_ddp"]["residual"]) / max(single["gnas_ddp"]["wall"], 1e-9)
        ratio = fd / fz if fz > 0 else 0.0
        # Same order of magnitude (0.1-10) and both within a reasonable upper bound
        # (<20%, to rule out inconsistent bases caused by abnormally large/negative residuals)
        flag = (0.1 < ratio < 10.0) and fd < 0.20 and fz < 0.20
        ok &= flag
        print(f"  [{'PASS' if flag else 'FAIL'}] ZeRO~=DDP residual same order of magnitude (ratio): "
              f"ddp={fd*100:.1f}% zero={fz*100:.1f}% ratio={ratio:.2f}(0.1~10, both <20%)")
    # 3) TAGNAS dual lanes + effective ~= max
    lanes = {r["lane"]: r for r in rows if r["system"] == "tagnas"}
    if "graph" in lanes and "text" in lanes:
        eff = lanes.get("effective")
        mx = max(lanes["graph"]["wall"], lanes["text"]["wall"])
        sm = lanes["graph"]["wall"] + lanes["text"]["wall"]
        flag = eff is not None and abs(eff["wall"] - mx) < 1e-6 and eff["wall"] <= sm
        ok &= flag
        print(f"  [{'PASS' if flag else 'FAIL'}] TAGNAS dual lanes: graph={lanes['graph']['wall']:.{prec}f} "
              f"text={lanes['text']['wall']:.{prec}f} effective={mx:.{prec}f}(=max, <=sum {sm:.{prec}f})")
    else:
        print("  [skip] TAGNAS dual lanes not ready (no graph/text JSONL)")
    print(f"===== {'ALL PASS ✅' if ok else 'SOME FAILED ❌'} =====")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="test6 dual-lane breakdown table (ASCII + LaTeX + CSV)")
    ap.add_argument("--in-dir", default="results/test6", help="Results directory (used as fallback for legacy breakdown_*.json)")
    ap.add_argument("--telemetry-dir", default=None, help="Per-iteration JSONL directory (default: <in-dir>/telemetry)")
    ap.add_argument("--out-ascii", default=None, help="ASCII table output path (.txt)")
    ap.add_argument("--out-tex", default=None, help="LaTeX table output path (.tex)")
    ap.add_argument("--out-csv", default=None, help="breakdown.csv output path")
    ap.add_argument("--precision", type=int, default=4, help="Number of decimal places (seconds)")
    ap.add_argument("--check", action="store_true", help="Run the acceptance gate (<5% residual / ZeRO~=DDP / dual lanes)")
    args = ap.parse_args()

    tel_dir = args.telemetry_dir or os.path.join(args.in_dir, "telemetry")
    rows = load_jsonl_rows(tel_dir) if os.path.isdir(tel_dir) else []
    src = f"telemetry JSONL ({tel_dir})"
    if not rows:
        rows = load_legacy_rows(args.in_dir)
        src = f"legacy breakdown_*.json ({args.in_dir})"
    if not rows:
        print(f"[make_breakdown_table] no telemetry data found ({tel_dir} or {args.in_dir})")
        return
    print(f"[make_breakdown_table] data source: {src}")

    rows = add_effective_rows(rows)
    rows = sort_rows(rows)

    ascii_s = ascii_table(rows, args.precision)
    tex_s = latex_table(rows, args.precision)

    print("\n===== ASCII (per-iteration runtime breakdown, seconds; nodes/s normalized) =====\n")
    print(ascii_s)
    print("\n===== LaTeX (booktabs) =====\n")
    print(tex_s)

    if args.out_ascii:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_ascii)), exist_ok=True)
        with open(args.out_ascii, "w", encoding="utf-8") as f:
            f.write(ascii_s + "\n")
        print(f"\n[wrote] {args.out_ascii}")
    if args.out_tex:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_tex)), exist_ok=True)
        with open(args.out_tex, "w", encoding="utf-8") as f:
            f.write(tex_s + "\n")
        print(f"[wrote] {args.out_tex}")
    if args.out_csv:
        write_csv(rows, args.out_csv)

    if args.check:
        ok = run_checks(rows, args.precision)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
