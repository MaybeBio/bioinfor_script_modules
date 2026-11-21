# plot lht template metrics file set by csv logger

"""Plot training curves (loss / AUROC / AUPRC, each train vs val) from a Lightning ``metrics.csv``.

    python src/plot_metrics.py                          # auto-find the latest metrics.csv
    python src/plot_metrics.py --input path/metrics.csv --output curves.png
    python src/plot_metrics.py --mark-best --baseline-auroc 0.594 --baseline-auprc 0.0044

Three side-by-side panels — ``Loss (BCE)``, ``AUROC``, ``AUPRC`` — each overlaying the
train (blue) and val (orange) curve, so under-/over-fitting is visible at a glance.

The LHT CSV interleaves two rows per epoch: a val row (``val/*`` filled, ``train/*``
empty) and a train row (``train/*`` filled, ``val/*`` empty). The two are split here and
re-indexed by epoch before plotting.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"
REF_COLOR = "#7f7f7f"
BASE_COLOR = "#2ca02c"


def find_latest_metrics(search_dir: Path | None = None) -> Path:
    """Return the most recently modified ``metrics.csv`` under ``logs/train/runs``."""
    base = search_dir or (ROOT / "logs" / "train" / "runs")
    candidates = sorted(base.glob("*/csv/version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no metrics.csv found under {base}; pass --input explicitly")
    return candidates[-1]


def load_curves(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the interleaved CSV into ``(train, val)`` frames indexed by epoch.

    The val row and the train row of one epoch carry disjoint columns, so each frame keeps
    only the rows where its own metrics are present. Duplicate rows for the same epoch
    (e.g. Lightning's pre-training sanity-check pass) are collapsed by keeping the last.
    """
    df = pd.read_csv(csv_path)
    needed = {"train/loss", "val/loss"}
    if not needed <= set(df.columns):
        raise ValueError(f"{csv_path}: missing columns {sorted(needed - set(df.columns))}")

    train = df.dropna(subset=["train/loss"]).groupby("epoch", sort=True).last()
    val = df.dropna(subset=["val/loss"]).groupby("epoch", sort=True).last()
    if val.empty:
        raise ValueError(f"{csv_path}: no val rows found")
    if train.empty:
        raise ValueError(f"{csv_path}: no train rows found")
    return train, val


def load_val_rows(csv_path: str | Path) -> pd.DataFrame:
    """Return only the per-epoch val rows (kept for callers that only need val)."""
    df = pd.read_csv(csv_path)
    if "val/auroc" not in df.columns:
        raise ValueError(f"{csv_path}: no 'val/auroc' column (expected LHT metrics.csv)")
    val = df.dropna(subset=["val/auroc"]).reset_index(drop=True)
    if val.empty:
        raise ValueError(f"{csv_path}: no val rows found (val/auroc is all NaN)")
    return val


def _panel(ax, train: pd.DataFrame, val: pd.DataFrame, trk: str, vk: str, title: str,
           show_best: bool = False, ref_line: float | None = None,
           ref_label: str = "random", baseline: float | None = None) -> None:
    """Draw one train-vs-val panel."""
    if trk in train.columns:
        ax.plot(train.index, train[trk], color=TRAIN_COLOR, lw=1.6, marker="o", ms=3,
                label="train")
    if vk in val.columns:
        ax.plot(val.index, val[vk], color=VAL_COLOR, lw=1.6, marker="o", ms=3,
                label="val")
    if show_best:
        bk = f"{vk}/best"
        if bk in val.columns:
            ax.step(val.index, val[bk], color=VAL_COLOR, lw=0.9, ls="--", alpha=0.45,
                    label="val (running best)")
    if ref_line is not None:
        ax.axhline(ref_line, color=REF_COLOR, lw=0.9, ls=":", alpha=0.9)
        ax.text(0.99, ref_line, ref_label, color=REF_COLOR, fontsize=8,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
    if baseline is not None:
        ax.axhline(baseline, color=BASE_COLOR, lw=0.9, ls=":", alpha=0.9)
        ax.text(0.99, baseline, "baseline", color=BASE_COLOR, fontsize=8,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)


def plot_metrics(
    train: pd.DataFrame,
    val: pd.DataFrame,
    out_path: str | Path,
    show_best: bool = False,
    baseline_auroc: float | None = None,
    baseline_auprc: float | None = None,
    mark_best: bool = False,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Render three side-by-side panels (Loss / AUROC / AUPRC), each train vs val."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    _panel(axes[0], train, val, "train/loss", "val/loss", "Loss (BCE)", show_best=show_best)
    _panel(axes[1], train, val, "train/auroc", "val/auroc", "AUROC", show_best=show_best,
           ref_line=0.5, ref_label="random", baseline=baseline_auroc)
    _panel(axes[2], train, val, "train/auprc", "val/auprc", "AUPRC", show_best=show_best,
           baseline=baseline_auprc)

    if mark_best and "val/loss" in val.columns:
        best_epoch = int(val["val/loss"].idxmin())
        for ax in axes:
            ax.axvline(best_epoch, color="#444444", lw=0.9, ls="--", alpha=0.7)
        axes[0].text(0.98, 0.02, f"dashed: selected ckpt (epoch {best_epoch})",
                     fontsize=8, color="#444444", ha="right", va="bottom",
                     transform=axes[0].transAxes)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot loss / AUROC / AUPRC training curves (train vs val) from metrics.csv.")
    ap.add_argument("--input", default=None,
                    help="path to metrics.csv (default: latest under logs/train/runs/)")
    ap.add_argument("--output", default=None,
                    help="output image (default: <input dir>/training_curves.png)")
    ap.add_argument("--show-best", action="store_true",
                    help="overlay the running-best staircase on the val curves")
    ap.add_argument("--mark-best", action="store_true",
                    help="mark the epoch with the lowest val/loss (the selected checkpoint)")
    ap.add_argument("--baseline-auroc", type=float, default=None,
                    help="reference line on the AUROC panel (e.g. 0.594 for ESM-cosine)")
    ap.add_argument("--baseline-auprc", type=float, default=None,
                    help="reference line on the AUPRC panel (e.g. 0.0044 for ESM-cosine)")
    ap.add_argument("--title", default=None, help="optional figure title")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    csv_path = Path(args.input) if args.input else find_latest_metrics()
    train, val = load_curves(csv_path)

    out = Path(args.output) if args.output else csv_path.parent / "training_curves.png"
    path = plot_metrics(
        train, val, out,
        show_best=args.show_best,
        baseline_auroc=args.baseline_auroc,
        baseline_auprc=args.baseline_auprc,
        mark_best=args.mark_best,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"saved {path}  ({len(val)} val epochs, {len(train)} train epochs)")


if __name__ == "__main__":
    main()
