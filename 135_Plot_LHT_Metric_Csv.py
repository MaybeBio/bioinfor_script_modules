# plot lht template metrics file set by csv logger

"""Plot training curves (val AUROC + val AUPRC vs epoch) from a Lightning ``metrics.csv``.

    python src/plot_metrics.py                          # auto-find the latest metrics.csv
    python src/plot_metrics.py --input path/metrics.csv --output curves.png
    python src/plot_metrics.py --show-best --baseline-auroc 0.594 --baseline-auprc 0.0044

Reads the LHT CSV (``epoch,step,train/*,val/*``) where each epoch has a val row and a
train row interleaved; keeps the val rows and plots the two ranking metrics on a shared
x-axis with twin y-axes (they live on very different scales: AUROC ~0.5-0.8, AUPRC ~1e-3-2e-2).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def find_latest_metrics(search_dir: Path | None = None) -> Path:
    """Return the most recently modified ``metrics.csv`` under ``logs/train/runs``."""
    base = search_dir or (ROOT / "logs" / "train" / "runs")
    candidates = sorted(base.glob("*/csv/version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no metrics.csv found under {base}; pass --input explicitly")
    return candidates[-1]


def load_val_rows(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV and keep only the per-epoch val rows (drop the train rows).

    The LHT CSV interleaves a val row and a train row per epoch; the train row has empty
    ``val/*`` cells, so dropping rows with a missing ``val/auroc`` isolates the val curve.
    """
    df = pd.read_csv(csv_path)
    if "val/auroc" not in df.columns:
        raise ValueError(f"{csv_path}: no 'val/auroc' column (expected LHT metrics.csv)")
    val = df.dropna(subset=["val/auroc"]).reset_index(drop=True)
    if val.empty:
        raise ValueError(f"{csv_path}: no val rows found (val/auroc is all NaN)")
    return val


def plot_metrics(
    val: pd.DataFrame,
    out_path: str | Path,
    show_best: bool = False,
    baseline_auroc: float | None = None,
    baseline_auprc: float | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Render the dual-axis AUROC/AUPRC training curves to ``out_path`` (PNG or PDF)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch = val["epoch"].to_numpy()
    auroc = val["val/auroc"].to_numpy()
    auprc = val["val/auprc"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax2 = ax1.twinx()

    # AUROC on the left axis (0.5-0.8 range of interest)
    (l1,) = ax1.plot(epoch, auroc, color="#1f77b4", lw=1.8, label="val AUROC")
    if show_best and "val/auroc/best" in val.columns:
        best = val["val/auroc/best"].to_numpy()
        ax1.step(epoch, best, color="#1f77b4", lw=0.9, ls="--", alpha=0.45, label="AUROC (running best)")

    # AUPRC on the right axis (tiny values, ~1e-3)
    (l2,) = ax2.plot(epoch, auprc, color="#d62728", lw=1.8, label="val AUPRC")
    if show_best and "val/auprc/best" in val.columns:
        best = val["val/auprc/best"].to_numpy()
        ax2.step(epoch, best, color="#d62728", lw=0.9, ls="--", alpha=0.45, label="AUPRC (running best)")

    # reference lines: random AUROC = 0.5 always; optional baseline method
    ax1.axhline(0.5, color="grey", lw=0.9, ls=":", alpha=0.8)
    ax1.text(epoch[0], 0.505, "random", color="grey", fontsize=8, va="bottom")
    if baseline_auroc is not None:
        ax1.axhline(baseline_auroc, color="#2ca02c", lw=0.9, ls=":", alpha=0.8)
        ax1.text(epoch[0], baseline_auroc + 0.003, "baseline", color="#2ca02c", fontsize=8, va="bottom")
    if baseline_auprc is not None:
        ax2.axhline(baseline_auprc, color="#2ca02c", lw=0.9, ls=":", alpha=0.8)

    # axis cosmetics
    ax1.set_xlabel("epoch", fontsize=11)
    ax1.set_ylabel("val AUROC", color="#1f77b4", fontsize=11)
    ax2.set_ylabel("val AUPRC", color="#d62728", fontsize=11)
    ax1.tick_params(labelsize=9, colors="#1f77b4")
    ax2.tick_params(labelsize=9, colors="#d62728")
    ax1.set_xlim(epoch[0], epoch[-1])
    ax1.grid(True, alpha=0.25, lw=0.5)

    lines = [l1, l2]
    if show_best:
        lines += [ax1.lines[-1]] if "val/auroc/best" in val.columns else []
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="lower right")

    if title is not None:
        ax1.set_title(title, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot val AUROC + AUPRC training curves from metrics.csv.")
    ap.add_argument("--input", default=None,
                    help="path to metrics.csv (default: latest under logs/train/runs/)")
    ap.add_argument("--output", default=None,
                    help="output image (default: <input dir>/training_curves.png)")
    ap.add_argument("--show-best", action="store_true",
                    help="overlay the running-best staircase for each metric")
    ap.add_argument("--baseline-auroc", type=float, default=None,
                    help="draw a reference line at this AUROC (e.g. 0.594 for ESM-cosine)")
    ap.add_argument("--baseline-auprc", type=float, default=None,
                    help="draw a reference line at this AUPRC (e.g. 0.0044 for ESM-cosine)")
    ap.add_argument("--title", default=None, help="optional plot title")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    csv_path = Path(args.input) if args.input else find_latest_metrics()
    val = load_val_rows(csv_path)

    out = Path(args.output) if args.output else csv_path.parent / "training_curves.png"
    path = plot_metrics(
        val, out,
        show_best=args.show_best,
        baseline_auroc=args.baseline_auroc,
        baseline_auprc=args.baseline_auprc,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"saved {path}  ({len(val)} epochs)")


if __name__ == "__main__":
    main()
