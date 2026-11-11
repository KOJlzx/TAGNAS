#!/usr/bin/env python3
"""Scan the GASSO alphas across all ckpts under a DDP run-dir and compare divergence.

Usage:
    python bin/inspect_alphas.py --run-dir model/runpytorch/gasso_arxiv_ddp_20260507_224955
    python bin/inspect_alphas.py --run-dir <path> --include-best --include-final
    python bin/inspect_alphas.py --run-dir <path> --layer 0 --group op   # only look at layer 0's op
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch

OP_NAMES = ["gat", "gat_2h", "gat_4h", "gcn", "gin", "sage", "linear"]
NORM_NAMES = ["layernorm", "batchnorm", "graphnorm", "identity"]
ACT_NAMES = ["relu", "gelu", "elu", "identity"]

GROUP_SPECS = [
    ("op",   "alphas_normal", "mask_op",   OP_NAMES),
    ("norm", "alphas_norm",   "mask_norm", NORM_NAMES),
    ("act",  "alphas_act",    "mask_act",  ACT_NAMES),
]


def collect_ckpts(run_dir: Path, include_best: bool, include_final: bool) -> list[tuple[str, Path]]:
    out = []
    ckpt_dir = run_dir / "checkpoints"
    epoch_pat = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    epoch_files = []
    if ckpt_dir.is_dir():
        for f in ckpt_dir.iterdir():
            m = epoch_pat.match(f.name)
            if m:
                epoch_files.append((int(m.group(1)), f))
    epoch_files.sort()
    for n, f in epoch_files:
        out.append((f"epoch_{n}", f))

    def _add(tag: str, name: str) -> None:
        for cand in (run_dir / name, ckpt_dir / name):
            if cand.exists():
                out.append((tag, cand))
                return

    if include_best:
        _add("best", "best_model.pth")
    if include_final:
        _add("final", "final_model.pth")
    return out


def softmax_with_mask(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = alpha.clone()
    a[~mask] = float("-inf")
    return torch.softmax(a, dim=-1)


def stats_for(p: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """p: softmax probabilities [N], mask: [N] bool. Computed only over unpruned ops."""
    p_kept = p[mask]
    n_kept = int(mask.sum().item())
    max_p = float(p_kept.max())
    min_p = float(p_kept.min())
    sorted_p = torch.sort(p_kept, descending=True).values
    second_p = float(sorted_p[1]) if n_kept > 1 else float("nan")
    margin = max_p - second_p if n_kept > 1 else float("nan")
    # entropy
    ent = float(-(p_kept * torch.clamp(p_kept, min=1e-30).log()).sum())
    ent_uniform = math.log(n_kept) if n_kept > 0 else 0.0
    ent_norm = ent / ent_uniform if ent_uniform > 0 else float("nan")
    argmax_idx = int(p.argmax().item())  # In the original space (includes pruned ops, but their p≈0)
    return dict(
        n_kept=n_kept,
        max=max_p,
        min=min_p,
        second=second_p,
        margin=margin,
        entropy=ent,
        entropy_norm=ent_norm,
        argmax=argmax_idx,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--include-best",  action="store_true", default=True)
    ap.add_argument("--include-final", action="store_true", default=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="Only look at a specific layer; omit for all layers")
    ap.add_argument("--group", choices=["op", "norm", "act"], default=None,
                    help="Only look at one of op/norm/act; omit to print all three")
    args = ap.parse_args()

    ckpts = collect_ckpts(args.run_dir, args.include_best, args.include_final)
    if not ckpts:
        raise SystemExit(f"no ckpts found under {args.run_dir}")

    print(f"[plan] inspecting {len(ckpts)} ckpts under {args.run_dir}")
    for tag, p in ckpts:
        print(f"  - {tag:<20} -> {p}")
    print()

    # First pass: detect the number of layers.
    sd0 = torch.load(ckpts[0][1], weights_only=False, map_location="cpu")["model_state_dict"]
    num_layers = len([k for k in sd0 if k.startswith("model.gnn.alphas_normal.")])
    print(f"[meta] num_layers={num_layers}\n")

    layers_to_show = [args.layer] if args.layer is not None else list(range(num_layers))
    groups_to_show = [g for g in GROUP_SPECS if (args.group is None or g[0] == args.group)]

    for layer in layers_to_show:
        for group_name, alpha_prefix, mask_prefix, names in groups_to_show:
            print("=" * 100)
            print(f"layer {layer}  group={group_name}  ({len(names)} candidates: {names})")
            print("=" * 100)
            # Header row
            print(f"{'ckpt':<22} {'argmax':<12} {'max_p':>7} {'2nd_p':>7} {'margin':>8} "
                  f"{'min_p':>7} {'H':>6} {'H/Hmax':>7}  full_distribution")
            print("-" * 130)

            for tag, ckpt_path in ckpts:
                sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model_state_dict"]
                alpha = sd[f"model.gnn.{alpha_prefix}.{layer}"]
                mask = sd[f"model.gnn.{mask_prefix}_{layer}"]
                p = softmax_with_mask(alpha, mask)
                s = stats_for(p, mask)
                argmax_name = names[s["argmax"]]
                # Format the whole layer's probabilities into a short list.
                dist_str = ", ".join(
                    f"{names[i]}:{p[i].item():.3f}" + ("*" if not mask[i] else "")
                    for i in range(len(names))
                )
                print(
                    f"{tag:<22} {argmax_name:<12} "
                    f"{s['max']:>7.4f} {s['second']:>7.4f} {s['margin']:>+8.4f} "
                    f"{s['min']:>7.4f} {s['entropy']:>6.3f} {s['entropy_norm']:>7.3f}  "
                    f"{dist_str}"
                )
            print()

    # Summary: trend of average margin / average H/Hmax.
    print("=" * 100)
    print("summary: avg max_p / avg margin / avg H_norm across all (layer, group) pairs")
    print("=" * 100)
    print(f"{'ckpt':<22} {'avg_max_p':>11} {'avg_margin':>12} {'avg_H/Hmax':>12}  "
          f"{'(H/Hmax≈1 -> fully uniform; H/Hmax->0 -> converged to a single op)'}")
    print("-" * 130)
    for tag, ckpt_path in ckpts:
        sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model_state_dict"]
        all_max, all_margin, all_hn = [], [], []
        for layer in range(num_layers):
            for group_name, alpha_prefix, mask_prefix, names in GROUP_SPECS:
                alpha = sd[f"model.gnn.{alpha_prefix}.{layer}"]
                mask = sd[f"model.gnn.{mask_prefix}_{layer}"]
                p = softmax_with_mask(alpha, mask)
                s = stats_for(p, mask)
                all_max.append(s["max"])
                if not math.isnan(s["margin"]):
                    all_margin.append(s["margin"])
                if not math.isnan(s["entropy_norm"]):
                    all_hn.append(s["entropy_norm"])
        avg_max = sum(all_max) / len(all_max)
        avg_margin = sum(all_margin) / len(all_margin) if all_margin else float("nan")
        avg_hn = sum(all_hn) / len(all_hn) if all_hn else float("nan")
        print(f"{tag:<22} {avg_max:>11.4f} {avg_margin:>+12.4f} {avg_hn:>12.4f}")


if __name__ == "__main__":
    main()
