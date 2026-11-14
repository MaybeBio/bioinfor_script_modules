#!/usr/bin/env python3
"""
Diagnostic: which sliding-window stitching strategy best reconstructs a
full-context ESM-2 embedding?

A protein short enough for one forward pass (<= --gt_max_len residues) is the
ground truth: embed it once, untouched. Then chunk the SAME protein with a
smaller window (--win_len) and stitch it back under five strategies:

  select     keep the row from the window that truncates the least context on
             either side (identical to script 5's max-centered rule except at
             true termini, where only one window covers the residue anyway)
  leftmost   take the row from the earliest window covering the residue
  rightmost  take the row from the latest window covering the residue
  mean       arithmetic mean over all covering windows
  weighted   mean weighted by (distance to nearest edge + 1)

Per-residue cosine similarity to the one-shot embedding says which strategy
reconstructs full-context rows most faithfully. Residues are also bucketed by
their best available quality q = min(left context lost, right context lost);
if cosine drops as q drops, the q premise behind `select` is validated.

Uses script 5's own _window_starts / _forward_esm2 so windowing and tokenization
match production exactly; only the stitch rule varies.

Example
-------
python data/scripts/compare_stitching.py \
  --input_dir data/processed/script_03/max200_min20_dp0.2_umax10000_pdbs500/max200_ct8_ib1_dg0_mask1_s2grow \
  --n_proteins 20 --win_len 500 --overlap 250

Run again with --overlap 100 (production's 10% ratio) to compare ratios.
"""

import argparse
import importlib.util
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT5 = Path(__file__).resolve().parent / "5_extract_embeddings.py"
LOGS = PROJECT_ROOT / "logs"

spec = importlib.util.spec_from_file_location("script5", SCRIPT5)
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)

logger = logging.getLogger("compare_stitching")


def _quality(start: int, window_len: int, length: int) -> torch.Tensor:
    """Context lost at each residue = distance to the nearest *truncation* edge.

    A window edge that coincides with the true sequence terminus truncates
    nothing, so it is not penalized (the one-shot embedding has the same
    boundary there). Only interior window edges lose context.
    """
    idx = torch.arange(window_len, dtype=torch.float32)
    far = float(length)
    left = idx if start > 0 else torch.full((window_len,), far)
    right = (window_len - 1 - idx) if start + window_len < length else torch.full((window_len,), far)
    return torch.minimum(left, right)


def stitch(length: int, dim: int, windows) -> dict:
    """windows: list of (start, rows[w, dim] float32); return strategy -> [length, dim]."""
    sentinel = float("nan")

    sel = torch.full((length, dim), sentinel)
    sel_q = torch.full((length,), -1.0)

    left = torch.full((length, dim), sentinel)
    left_done = torch.zeros(length, dtype=torch.bool)

    right = torch.full((length, dim), sentinel)

    mean_acc = torch.zeros((length, dim))
    mean_cnt = torch.zeros(length)

    w_acc = torch.zeros((length, dim))
    w_sum = torch.zeros(length)

    best_q = torch.full((length,), -1.0)

    for start, rows in windows:
        w = rows.shape[0]
        g = start + torch.arange(w)
        q = _quality(start, w, length)

        better = q > sel_q[g]
        sel[g[better]] = rows[better]
        sel_q[g[better]] = q[better]

        fresh = ~left_done[g]
        left[g[fresh]] = rows[fresh]
        left_done[g[fresh]] = True

        right[g] = rows

        mean_acc[g] += rows
        mean_cnt[g] += 1

        weight = q + 1.0                      # avoid a zero weight at true termini
        w_acc[g] += weight[:, None] * rows
        w_sum[g] += weight

        best_q[g] = torch.maximum(best_q[g], q)

    if torch.isnan(sel).any() or torch.isnan(left).any() or torch.isnan(right).any():
        raise ValueError("a residue is not covered by any window")
    if (mean_cnt == 0).any() or (w_sum == 0).any():
        raise ValueError("a residue has no covering window for mean/weighted")

    return {
        "select": sel,
        "leftmost": left,
        "rightmost": right,
        "mean": mean_acc / mean_cnt[:, None],
        "weighted": w_acc / w_sum[:, None],
        "best_q": best_q,
    }


def run_one(embedder, model, alphabet, layer, seq: str, win_len: int, overlap: int) -> dict:
    L = len(seq)
    gt = embedder._forward_esm2(model, alphabet, layer, [seq])[0].float()
    if gt.shape != (L, gt.shape[1]):
        raise ValueError(f"full-length embedding shape {tuple(gt.shape)} != ({L}, C)")

    starts = s5._window_starts(L, win_len, overlap)
    rows = embedder._forward_esm2(model, alphabet, layer, [seq[s:s + win_len] for s in starts])
    windows = [(s, r.float()) for s, r in zip(starts, rows)]

    for start, r in windows:
        if r.shape[0] != min(win_len, L - start):
            raise ValueError(f"window at {start} has {r.shape[0]} rows, expected {min(win_len, L - start)}")

    stitched = stitch(L, gt.shape[1], windows)
    best_q = stitched.pop("best_q")

    out = {"uniprot": None, "length": L, "n_windows": len(starts)}
    for name, emb in stitched.items():
        out[name] = F.cosine_similarity(emb, gt, dim=1).tolist()
    out["best_q"] = best_q.tolist()
    return out


def summarize(rows: list, strategies: list, win_len: int) -> dict:
    all_q = torch.tensor([q for r in rows for q in r["best_q"]], dtype=torch.float32)
    max_q = (win_len - 1) / 2
    cuts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    stats = {"n_proteins": len(rows), "n_residues": int(all_q.numel()), "max_q": max_q}

    for name in strategies:
        cos = torch.tensor([c for r in rows for c in r[name]], dtype=torch.float32)
        entry = {
            "mean": cos.mean().item(),
            "median": cos.median().item(),
            "p05": cos.quantile(0.05).item(),
            "min": cos.min().item(),
        }
        # cap at the best two-sided context any interior residue can get
        ratio = torch.clamp(all_q, max=max_q) / max_q
        entry["by_q"] = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mask = (ratio >= lo) & (ratio < hi)
            entry["by_q"].append(
                {"lo": lo, "hi": hi if hi <= 1 else 1.0,
                 "n": int(mask.sum()), "mean": cos[mask].mean().item() if mask.any() else None}
            )
        stats[name] = entry
    return stats


def print_stats(stats: dict, strategies: list) -> None:
    print(f"\n=== {stats['n_proteins']} proteins, {stats['n_residues']} residues, "
          f"max q = {stats['max_q']:.0f} ===")
    head = f"{'strategy':<10}{'mean':>8}{'median':>8}{'p05':>8}{'min':>8}"
    for b in stats[strategies[0]]["by_q"]:
        head += f"{'q%.0f-%.0f%%' % (b['lo'] * 100, b['hi'] * 100):>11}"
    print(head)
    for name in strategies:
        e = stats[name]
        line = f"{name:<10}{e['mean']:>8.4f}{e['median']:>8.4f}{e['p05']:>8.4f}{e['min']:>8.4f}"
        for b in e["by_q"]:
            line += f"{b['mean']:>11.4f}" if b["mean"] is not None else f"{'-':>11}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="script_03 run directory (contains stage3_merged/)")
    parser.add_argument("--seq_json", default=None, help="full UniProt sequence json (default derived as in script 5)")
    parser.add_argument("--model_name", default=None,
                        help="fair-esm model name (default esm2_t33_650M_UR50D; use esm2_t12_35M_UR50D for a CPU smoke test)")
    parser.add_argument("--n_proteins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--win_len", type=int, default=500, help="simulated window size (must be < ground-truth length)")
    parser.add_argument("--overlap", type=int, default=250)
    parser.add_argument("--gt_max_len", type=int, default=1000,
                        help="max length for a one-shot ground-truth embedding (ESM-2 ctx is 1022)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="JSON output path (default logs/stitching_comparison_<ts>.json)")
    parser.add_argument("--verbose", action="store_true", help="print per-protein results")
    args = parser.parse_args()

    if not 0 < args.overlap < args.win_len:
        raise ValueError("need 0 < overlap < win_len")
    if args.win_len >= args.gt_max_len:
        raise ValueError("win_len must be smaller than gt_max_len, otherwise nothing is chunked")

    embedder = s5.Embedder.__new__(s5.Embedder)
    embedder.cfg = s5.EmbedConfig(
        input_dir=Path(args.input_dir), seq_json=Path(args.seq_json) if args.seq_json else None,
        backend="esm2", model_name=args.model_name, dtype="float32", device=args.device,
    )
    embedder.merged_dir = Path(args.input_dir) / "stage3_merged"
    embedder.seq_json = embedder.cfg.seq_json or embedder._default_seq_json()

    ids = embedder._unique_uniprots()
    seqs = embedder._load_sequences(ids)
    eligible = [u for u in ids if u in seqs and args.win_len < len(seqs[u]) <= args.gt_max_len]
    if not eligible:
        raise ValueError(f"no protein with {args.win_len} < length <= {args.gt_max_len}")
    random.Random(args.seed).shuffle(eligible)
    chosen = eligible[: args.n_proteins]
    logger.info(f"{len(eligible)} eligible proteins, using {len(chosen)} "
                f"(win_len={args.win_len}, overlap={args.overlap})")

    model, alphabet, layer = embedder._load_model_esm2()
    strategies = ["select", "leftmost", "rightmost", "mean", "weighted"]

    t0 = time.time()
    rows = []
    for i, u in enumerate(chosen, 1):
        r = run_one(embedder, model, alphabet, layer, seqs[u], args.win_len, args.overlap)
        r["uniprot"] = u
        rows.append(r)
        logger.info(f"[{i}/{len(chosen)}] {u} L={r['length']} windows={r['n_windows']} "
                    f"({time.time() - t0:.1f}s)")

    stats = summarize(rows, strategies, args.win_len)
    stats.update({"win_len": args.win_len, "overlap": args.overlap, "layer": layer,
                  "model_name": embedder.cfg.model_name, "proteins": chosen})
    print_stats(stats, strategies)

    out = Path(args.out) if args.out else LOGS / f"stitching_comparison_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": stats, "per_protein": rows if args.verbose else None,
                   "strategies": strategies}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
