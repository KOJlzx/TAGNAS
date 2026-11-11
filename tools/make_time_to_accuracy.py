#!/usr/bin/env python3
"""test1-2: honest speedup -- iso-quality (time-to-accuracy) + iso-time.

Replaces the "total wall-clock ratio under different stopping criteria" style of comparison
(e.g. "TAGNAS 17.7min vs B2 12h = 42x") with two apples-to-apples measures, reconstructed
from existing training logs (no retraining needed) into figures + tables:

  1. time-to-accuracy curve: x=wall-clock (log minutes), y=val-AUC best-so-far, one line per
     system (four systems total) -- a single figure lets you read off both iso-quality
     (horizontal slice) and iso-time (vertical slice).
  2. iso-quality table: wall-clock needed to reach the same AUC target; speedup = t_baseline / t_TAGNAS.
  3. iso-time table: best AUC achieved by each system within the same wall-clock budget.

Data sources (read-only):
  - Synchronous systems B1/B2/B3: <run>/metrics.csv, with columns val_auc + epoch_seconds
    (per epoch); wall-clock = cumsum(epoch_seconds).
  - Asynchronous TAGNAS: val_history = [{time, auc, best_auc}, ...] from
    <run>/search/manifest.json, where time is already elapsed seconds from the start of
    training (falls back to reading val_eval from metrics.jsonl).

Pure standard-library parsing + matplotlib (Agg) plotting, no new dependencies. Styled after
make_breakdown_table.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -- System order / labels / colors (shared by plots and tables) --------------------
SYS_ORDER = ["B1", "B2", "B3", "B4", "B5", "TAGNAS"]
SYS_LABEL = {
    "B1": "B1: DDP-Fixed (GAT+LM)",
    "B2": "B2: DDP-Sync (GNAS+LM)",
    "B3": "B3: ZeRO-Offload (GNAS+LM)",
    "B4": "B4: ZeRO-3 Offload (GNAS+LM)",
    "B5": "B5: UniTG (deep RevGNN pipeline + LM)",
    "TAGNAS": "TAGNAS (async, ours)",
}
SYS_STYLE = {  # (color, marker, linestyle)
    "B1": ("#888888", "s", "--"),
    "B2": ("#1f77b4", "o", "-"),
    "B3": ("#2ca02c", "^", "-."),
    "B4": ("#8c564b", "P", (0, (5, 1, 1, 1))),
    "B5": ("#9467bd", "v", ":"),
    "TAGNAS": ("#d62728", "D", "-"),
}
# Synchronous baselines used to compute speedup (same GNAS task; B1 and B5 have fixed
# architectures so they aren't NAS baselines -> no speedup column for them)
SPEEDUP_BASELINES = ["B2", "B3"]
# --exclude-eval-time: put each sync system's samples on a training-time axis = wall clock minus the time
# its training spent paused for the periodic evaluations so far (val_timeline.csv's eval_seconds column).
# TAGNAS is not adjusted: its evaluation runs on the coordinator / graph-worker RPC thread while the text
# workers keep training, so nothing is paused.
EXCLUDE_EVAL = False

# -- amazon-baby: 4 default representative runs (verified to exist and contain a val_auc trajectory) --------
DEFAULT_RUNS: Dict[str, Dict[str, str]] = {
    "B1": {"kind": "sync",
           "path": "model/runpytorch/amazon-baby_GAT_ddp_20260621_220143/metrics.csv"},
    "B2": {"kind": "sync",
           "path": "model/runpytorch/gasso_amazon-baby_ddp_20260620_061356/metrics.csv"},
    "B3": {"kind": "sync",
           "path": "model/runpytorch/gasso_amazon-baby_b4_zero_20260620_183933/metrics.csv"},
    "TAGNAS": {"kind": "tagnas",
               "path": "model/distributed/run-20260621-032748-pid1877264/search/manifest.json"},
}


# ─────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────
def _f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _running_max(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(t, auc) sequence -> (t, best-so-far) monotonically non-decreasing sequence."""
    out: List[Tuple[float, float]] = []
    m = float("-inf")
    for t, a in points:
        m = max(m, a)
        out.append((t, m))
    return out


def _load_run_args(metrics_csv: str) -> Dict[str, Any]:
    ra = os.path.join(os.path.dirname(metrics_csv), "run_args.json")
    if os.path.exists(ra):
        try:
            return json.load(open(ra, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def load_sync(path: str) -> Dict[str, Any]:
    """Read the synchronous system's trajectory + metadata.

    Prefers the densely-sampled (elapsed_seconds, val_auc) sequence in val_timeline.csv in
    the same directory (produced by --val-eval-interval-seconds); falls back to the (cumsum
    epoch_seconds, val_auc) sequence in metrics.csv (per epoch) if unavailable. total_s
    always uses the sum of epoch_seconds from metrics.csv (the full training duration).
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cum_total = sum(_f(r.get("epoch_seconds")) or 0.0 for r in rows)

    run_dir = os.path.dirname(path)
    timeline = os.path.join(run_dir, "val_timeline.csv")
    points: List[Tuple[float, float]] = []
    source = "metrics.csv"
    eval_total = 0.0
    if os.path.exists(timeline):
        for r in csv.DictReader(open(timeline, encoding="utf-8")):
            t, a = _f(r.get("elapsed_seconds")), _f(r.get("val_auc"))
            eval_total += _f(r.get("eval_seconds")) or 0.0
            if t is not None and a is not None:
                points.append((t - eval_total if EXCLUDE_EVAL else t, a))
        if points:
            source = "val_timeline.csv"
    if source == "metrics.csv":
        cum = 0.0
        for r in rows:
            cum += _f(r.get("epoch_seconds")) or 0.0
            a = _f(r.get("val_auc"))
            if a is not None:
                points.append((cum, a))
    points.sort(key=lambda x: x[0])

    cfg = _load_run_args(path)
    keep = ("world_size", "batch_size", "num_layers", "adam_eps",
            "weight_decay", "num_epochs", "early_stop_metric", "val_eval_interval_seconds")
    if EXCLUDE_EVAL:
        cum_total = max(0.0, cum_total - eval_total)
    return {
        "points": points,
        "eval_seconds_total": eval_total,
        "total_s": max(cum_total, points[-1][0] if points else 0.0),
        "n_evals": len(points),
        "config": {k: cfg[k] for k in keep if k in cfg},
        "run_dir": run_dir,
        "source": source,
    }


def _resolve_tagnas_manifest(path: str) -> Optional[str]:
    if path.endswith(".json"):
        return path if os.path.exists(path) else None
    if os.path.isdir(path):
        for cand in (os.path.join(path, "manifest.json"),
                     os.path.join(path, "search", "manifest.json")):
            if os.path.exists(cand):
                return cand
    return None


def load_tagnas(path: str) -> Dict[str, Any]:
    """Read TAGNAS. Prefers val_history from manifest.json; falls back to val_eval from metrics.jsonl."""
    points: List[Tuple[float, float]] = []
    total_s: Optional[float] = None
    best_auc: Optional[float] = None
    cfg: Dict[str, Any] = {}
    run_dir = path

    mf = _resolve_tagnas_manifest(path)
    if mf is not None:
        run_dir = os.path.dirname(mf)
        d = json.load(open(mf, encoding="utf-8"))
        for e in d.get("val_history", []) or []:
            t, a = _f(e.get("time")), _f(e.get("auc"))
            if t is not None and a is not None:
                points.append((t, a))
        total_s = _f(d.get("total_training_seconds"))
        best_auc = _f(d.get("best_val_auc"))
    else:
        # Fallback: treat path as metrics.jsonl (or its containing directory)
        jl = path if path.endswith(".jsonl") else os.path.join(path, "metrics.jsonl")
        run_dir = os.path.dirname(jl)
        with open(jl, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ve = obj.get("val_eval")
                if ve:
                    t, a = _f(ve.get("time")), _f(ve.get("auc"))
                    if t is not None and a is not None:
                        points.append((t, a))

    # Config: cli_args.json in the same directory
    ca = os.path.join(run_dir, "cli_args.json")
    if os.path.exists(ca):
        try:
            raw = json.load(open(ca, encoding="utf-8"))
            keep = ("dataset", "val_eval_interval_seconds", "val_eval_min_updates",
                    "val_patience", "graph_batch_size", "text_chunk_size", "num_layers")
            cfg = {k: raw[k] for k in keep if k in raw}
        except (OSError, json.JSONDecodeError):
            pass

    if total_s is None and points:
        total_s = points[-1][0]
    if best_auc is None and points:
        best_auc = max(a for _, a in points)
    return {
        "points": points,
        "total_s": total_s or 0.0,
        "n_evals": len(points),
        "config": cfg,
        "run_dir": run_dir,
        "source": "manifest.json" if mf is not None else "metrics.jsonl",
    }


def load_system(name: str, spec: Dict[str, str], repo_root: str) -> Dict[str, Any]:
    kind = spec.get("kind", "sync")
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    info = load_sync(path) if kind == "sync" else load_tagnas(path)
    pts = info["points"]
    info.update({
        "name": name,
        "label": SYS_LABEL.get(name, name),
        "kind": kind,
        "best_so_far": _running_max(pts),
        "best_auc": (max(a for _, a in pts) if pts else None),
    })
    return info


# ─────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────
def time_to_target(best: List[Tuple[float, float]], target: float) -> Optional[float]:
    """Wall-clock time (seconds) at which best-so-far first reaches >= target; None if never reached."""
    for t, m in best:
        if m >= target:
            return t
    return None


def auc_at_budget(best: List[Tuple[float, float]], budget_s: float) -> Optional[float]:
    """Best AUC achieved within wall-clock <= budget; None if no validation occurred within the budget."""
    v: Optional[float] = None
    for t, m in best:
        if t <= budget_s:
            v = m
    return v


# ─────────────────────────────────────────────────────────────────────────
# Rendering: ASCII / Markdown / LaTeX (booktabs)
# ─────────────────────────────────────────────────────────────────────────
def fmt_dur(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    m = sec / 60.0
    return f"{m:.1f}m" if m < 90 else f"{m / 60.0:.1f}h"


def fmt_speedup(ratio: Optional[float]) -> str:
    if ratio is None:
        return "—"
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def fmt_auc(a: Optional[float]) -> str:
    return "—" if a is None else f"{a:.3f}"


def render_ascii(headers: List[str], rows: List[List[str]], left_cols: int = 1) -> str:
    cols = len(headers)
    widths = [max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(cols)]

    def line(cells: List[str]) -> str:
        return "  ".join(
            (c.ljust(widths[i]) if i < left_cols else c.rjust(widths[i]))
            for i, c in enumerate(cells)
        )

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def render_md(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _tex_escape(s: str) -> str:
    return (s.replace("×", r"$\times$").replace("—", "--")
             .replace("%", r"\%").replace("_", r"\_").replace("≥", r"$\geq$"))


def render_tex(headers: List[str], rows: List[List[str]], colspec: str,
               caption: str, label: str, footnote: str = "") -> str:
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(_tex_escape(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if footnote:
        lines += ["\\\\[2pt]", "{\\footnotesize " + footnote + "}"]
    lines += ["\\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Three output artifacts
# ─────────────────────────────────────────────────────────────────────────
def build_iso_quality(systems: Dict[str, Dict[str, Any]], targets: List[float]):
    present = [s for s in SYS_ORDER if s in systems]
    spd_bases = [b for b in SPEEDUP_BASELINES if b in systems] if "TAGNAS" in systems else []
    headers = ["Target AUC"] + present + [f"{b}/TAG" for b in spd_bases]
    rows: List[List[str]] = []
    for A in targets:
        t = {s: time_to_target(systems[s]["best_so_far"], A) for s in present}
        cells = [f"{A:.2f}"] + [fmt_dur(t[s]) for s in present]
        for b in spd_bases:
            tb, tt = t.get(b), t.get("TAGNAS")
            cells.append(fmt_speedup(tb / tt if (tb and tt) else None))
        rows.append(cells)
    caption = ("Iso-quality (time-to-accuracy): wall-clock to first reach a "
               "validation-AUC target. Speedup $=t_{\\text{baseline}}/t_{\\text{TAGNAS}}$.")
    foot = ("Times are wall-clock (m = minutes, h = hours). ``--'' = target never "
            "reached within the run. TAGNAS times are conservative upper bounds: its "
            "first validation is gated by \\texttt{val\\_eval\\_min\\_updates}.")
    colspec = "l" + "r" * (len(headers) - 1)
    return headers, rows, caption, foot, colspec


def build_iso_time(systems: Dict[str, Dict[str, Any]], budgets_min: List[float]):
    present = [s for s in SYS_ORDER if s in systems]
    headers = ["Budget"] + present
    rows: List[List[str]] = []
    for bm in budgets_min:
        bs = bm * 60.0
        cells = [fmt_dur(bs)] + [fmt_auc(auc_at_budget(systems[s]["best_so_far"], bs))
                                 for s in present]
        rows.append(cells)
    caption = ("Iso-time: best validation-AUC reached within a fixed wall-clock budget.")
    foot = ("``--'' = no validation completed within the budget (e.g. a synchronous "
            "baseline has not finished its first epoch).")
    colspec = "l" + "r" * (len(headers) - 1)
    return headers, rows, caption, foot, colspec


def make_figure(systems: Dict[str, Dict[str, Any]], targets: List[float],
                out_png: str, out_pdf: str, dataset: str) -> None:
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        PchipInterpolator = None

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    _SHORT = {"B1": "B1: DDP-Fixed", "B2": "B2: DDP-Sync", "B3": "B3: ZeRO-Offload", "B4": "B4: ZeRO-3",
              "B5": "B5: UniTG", "TAGNAS": "TAGNAS (ours)"}  # simplified curve labels
    for name in SYS_ORDER:
        if name not in systems:
            continue
        best = systems[name]["best_so_far"]
        if not best:
            continue
        xs = [t / 60.0 for t, _ in best]
        ys = [m for _, m in best]
        color, marker, ls = SYS_STYLE[name]
        label = _SHORT.get(name, systems[name]["label"])
        if os.environ.get("TTA_SMOOTH") and PchipInterpolator is not None and len(xs) >= 3:
            # (opt-in, TTA_SMOOTH=1) monotone PCHIP interpolation in log-x space
            lx = np.log10(xs)
            grid = np.linspace(lx[0], lx[-1], 240)
            ax.plot(np.power(10.0, grid), PchipInterpolator(lx, ys)(grid), color=color, linestyle=ls,
                    linewidth=2.0, label=label, zorder=3)
        else:
            # Best-so-far is a step function: it only changes at a measured evaluation, so it is drawn
            # as steps (no interpolation between checks suggesting measurements that do not exist).
            ax.step(xs, ys, where="post", color=color, linestyle=ls, linewidth=2.0, label=label, zorder=3)
        if len(xs) > 16:                     # marker thinning: ~14 real measured points, evenly in log-x
            _lx = np.log10(xs)
            _tg = np.linspace(_lx[0], _lx[-1], 14)
            mk_idx = sorted({int(np.argmin(np.abs(_lx - t))) for t in _tg})
        else:
            mk_idx = list(range(len(xs)))
        ax.plot([xs[i] for i in mk_idx], [ys[i] for i in mk_idx], linestyle="none", marker=marker, color=color,
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    # Horizontal dashed lines for target AUC
    for A in targets:
        ax.axhline(A, color="0.7", linewidth=0.6, linestyle=":", zorder=0)
    # (The separate TAGNAS early-stop vertical line has been removed; per request, per-curve
    # cutoff-time annotations are also no longer drawn.)

    ax.set_xscale("log")
    # Add more x-axis tick marks (1-2-5 sequence): the default log ticks (decades only) are too sparse.
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    _xlo, _xhi = ax.get_xlim()
    _cand = [m * 10.0 ** d for d in range(-3, 5) for m in (1, 2, 5)]
    _ticks = [t for t in _cand if _xlo <= t <= _xhi]
    if _ticks:
        ax.xaxis.set_major_locator(FixedLocator(_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v:.0f}" if v >= 1 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Wall-clock time (minutes, log scale)")
    ax.set_ylabel(os.environ.get("TTA_YLABEL", "Validation AUC (best-so-far)"))
    ax.set_title(f"Time-to-accuracy on {dataset}")
    ax.grid(True, which="both", linestyle="-", linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for path in (out_png, out_pdf):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)


def write_series_csv(systems: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "t_seconds", "t_minutes", "val_auc", "best_so_far"])
        for name in SYS_ORDER:
            if name not in systems:
                continue
            pts = systems[name]["points"]
            best = systems[name]["best_so_far"]
            for (t, a), (_, m) in zip(pts, best):
                w.writerow([name, f"{t:.3f}", f"{t / 60.0:.4f}", f"{a:.6f}", f"{m:.6f}"])


def write_runs_used(systems: Dict[str, Dict[str, Any]], dataset: str, path: str) -> None:
    out = {"dataset": dataset, "systems": {}}
    for name in SYS_ORDER:
        if name not in systems:
            continue
        s = systems[name]
        out["systems"][name] = {
            "label": s["label"], "kind": s["kind"], "run_dir": s["run_dir"],
            "source": s.get("source"),
            "best_val_auc": s["best_auc"], "total_seconds": s["total_s"],
            "total_hours": round(s["total_s"] / 3600.0, 3) if s["total_s"] else None,
            "n_evals": s["n_evals"], "config": s["config"],
        }
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _sanitize(s: str) -> str:
    """Sanitize for use as a filename: replace anything that isn't alphanumeric or -._ with an underscore."""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(s))


def _interval_token(s: Dict[str, Any]) -> Optional[float]:
    iv = s.get("config", {}).get("val_eval_interval_seconds")
    try:
        return float(iv) if iv is not None else None
    except (TypeError, ValueError):
        return None


def auto_tag(systems: Dict[str, Dict[str, Any]]) -> str:
    """Derive the output subdirectory signature from each run's sampling method: sync_<epoch|Ns|dense|mixed>__tagnas_<Ns|async>."""
    sync_sigs = set()
    for name, s in systems.items():
        if name == "TAGNAS":
            continue
        if s.get("source") == "val_timeline.csv":
            iv = _interval_token(s)
            sync_sigs.add(f"{int(iv)}s" if iv else "dense")
        else:
            sync_sigs.add("epoch")
    if not sync_sigs:
        sync_sig = "none"
    elif len(sync_sigs) == 1:
        sync_sig = next(iter(sync_sigs))
    else:
        sync_sig = "mixed"
    t = systems.get("TAGNAS")
    if t is None:
        tag_t = "none"
    else:
        iv = _interval_token(t)
        tag_t = f"{int(iv)}s" if iv else "async"
    return f"sync_{sync_sig}__tagnas_{tag_t}"


def _append_index(base_out: str, dataset: str, tag: str, out_dir: str,
                  systems: Dict[str, Dict[str, Any]]) -> None:
    """Append a row to base_out/index.tsv (never overwritten), to make it easy to look up all (dataset, sampling) combinations."""
    os.makedirs(base_out, exist_ok=True)
    idx = os.path.join(base_out, "index.tsv")
    is_new = not os.path.exists(idx)
    rel = os.path.relpath(out_dir, base_out)
    bests = ";".join(
        f"{n}={systems[n]['best_auc']:.3f}"
        for n in SYS_ORDER if n in systems and systems[n]["best_auc"] is not None
    )
    srcs = ";".join(f"{n}:{systems[n].get('source', '?')}"
                    for n in SYS_ORDER if n in systems)
    with open(idx, "a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp\tdataset\ttag\tout_subdir\tbest_val_auc\tsources\n")
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{dataset}\t{tag}\t"
                f"{rel}\t{bests}\t{srcs}\n")


# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="test1-2 honest speedup: iso-quality / iso-time / time-to-accuracy curves")
    ap.add_argument("--out", default="results/test1-2", help="Output directory for artifacts")
    ap.add_argument("--dataset", default="amazon-baby", help="Dataset name (used only for the title/provenance)")
    ap.add_argument("--runs", default=None,
                    help="JSON file overriding the default runs: {\"B2\":{\"kind\":\"sync\",\"path\":...}}")
    ap.add_argument("--baseline", default="B2", help="Primary speedup baseline (default: B2)")
    ap.add_argument("--targets", default="0.55 0.60 0.65 0.70 0.72 0.74",
                    help="AUC targets for iso-quality (space-separated)")
    ap.add_argument("--budgets", default="17.7 60 180 360 540 720",
                    help="Wall-clock budgets for iso-time (minutes, space-separated)")
    ap.add_argument("--exclude-eval-time", action="store_true",
                    help="Time axis = wall clock minus the time training was paused by periodic evaluation "
                         "(sync systems' val_timeline eval_seconds); TAGNAS is not adjusted (non-blocking eval)")
    ap.add_argument("--tag", default=None,
                    help="Output subdirectory name (auto-generated from each run's sampling method if not given, e.g. sync_epoch__tagnas_100s)")
    args = ap.parse_args()
    global EXCLUDE_EVAL
    EXCLUDE_EVAL = bool(args.exclude_eval_time)

    # This script lives in tools/, so the repo root is its PARENT. It used to sit at the root,
    # and the move (commit "Tidy the repository root") left this resolving every relative run
    # path to tools/model/... -- which made a completed sweep report
    # "no system has a usable trajectory" and produce nothing.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)

    runs = dict(DEFAULT_RUNS)
    if args.runs:
        override = json.load(open(args.runs, encoding="utf-8"))
        for k, v in override.items():
            runs[k] = v

    targets = [float(x) for x in args.targets.split()]
    budgets = [float(x) for x in args.budgets.split()]

    systems: Dict[str, Dict[str, Any]] = {}
    for name in SYS_ORDER:
        if name not in runs:
            continue
        try:
            info = load_system(name, runs[name], repo_root)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"[warn] skipping {name}: {e}")
            continue
        if not info["points"]:
            print(f"[warn] {name} has no val_auc trajectory (run_dir={info['run_dir']}), skipping")
            continue
        systems[name] = info

    if not systems:
        print("[make_time_to_accuracy] no system has a usable trajectory, exiting")
        return

    # Output subdirectory: bucketed by <dataset>/<sampling signature>, to avoid results from
    # different datasets/sampling frequencies overwriting each other.
    # The sampling signature is derived automatically from each run (overridable via --tag);
    # re-running the same (dataset, sampling) combination updates the same subdirectory.
    tag = _sanitize(args.tag) if args.tag else auto_tag(systems)
    if EXCLUDE_EVAL:
        tag += "__evalexcluded"
    out_dir = os.path.join(base_out, _sanitize(args.dataset), tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[out-dir] {out_dir}")

    # Trajectory overview
    print("\n===== best-so-far trajectories (t_min, best_auc) =====")
    for name in SYS_ORDER:
        if name in systems:
            best = systems[name]["best_so_far"]
            print(f"{name:7s} " + "  ".join(f"({t/60:.1f}m,{m:.3f})" for t, m in best))

    # Three output artifacts
    q_h, q_rows, q_cap, q_foot, q_spec = build_iso_quality(systems, targets)
    t_h, t_rows, t_cap, t_foot, t_spec = build_iso_time(systems, budgets)

    iso_q_ascii = render_ascii(q_h, q_rows)
    iso_t_ascii = render_ascii(t_h, t_rows)

    print("\n===== ISO-QUALITY (time-to-accuracy) =====\n")
    print(iso_q_ascii)
    print("\n===== ISO-TIME (best AUC within budget) =====\n")
    print(iso_t_ascii)

    make_figure(systems, targets,
                os.path.join(out_dir, "time_to_accuracy.png"),
                os.path.join(out_dir, "time_to_accuracy.pdf"),
                args.dataset)

    # Write files
    def _w(name: str, text: str) -> None:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(text + "\n")

    _w("iso_quality.md", render_md(q_h, q_rows))
    _w("iso_quality.tex", render_tex(q_h, q_rows, q_spec, q_cap,
                                     "tab:iso-quality", q_foot))
    _w("iso_time.md", render_md(t_h, t_rows))
    _w("iso_time.tex", render_tex(t_h, t_rows, t_spec, t_cap, "tab:iso-time", t_foot))

    summary = (
        f"# test1-2 honest speedup ({args.dataset})\n\n"
        f"baseline={args.baseline}  targets={targets}  budgets(min)={budgets}\n\n"
        "## iso-quality (time-to-accuracy)\n\n" + iso_q_ascii +
        "\n\n## iso-time (best AUC within budget)\n\n" + iso_t_ascii + "\n"
    )
    _w("summary.txt", summary)

    write_series_csv(systems, os.path.join(out_dir, "series.csv"))
    write_runs_used(systems, args.dataset, os.path.join(out_dir, "runs_used.json"))

    print(f"\n[wrote] {out_dir}/")
    for fn in ("time_to_accuracy.png", "time_to_accuracy.pdf", "iso_quality.md",
               "iso_quality.tex", "iso_time.md", "iso_time.tex", "series.csv",
               "runs_used.json", "summary.txt"):
        print(f"        {fn}")

    _append_index(base_out, args.dataset, tag, out_dir, systems)
    print(f"[index] appended → {os.path.join(base_out, 'index.tsv')}")


if __name__ == "__main__":
    main()
