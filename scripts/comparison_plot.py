
"""
    This file compares authorship metrics across models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import your fixed dirs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import PIC_DIR as _PIC_DIR, EVOLVE_DIR as _EVOLVE_DIR

EVOLVE_DIR = Path(_EVOLVE_DIR)
PIC_DIR = Path(_PIC_DIR)


# ---------- IO ----------
def load_metrics(metrics_dir: Path) -> List[Dict[str, Any]]:
    if not metrics_dir.exists():
        raise SystemExit(f"[comparison_plot] EVOLVE_DIR does not exist: {metrics_dir.resolve()}")
    files = sorted(metrics_dir.glob("*-metrics.json"))
    if not files:
        raise SystemExit(f"[comparison_plot] No *-metrics.json found under {metrics_dir.resolve()}")
    out: List[Dict[str, Any]] = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if not data:
                continue
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON structure in {p}")
        out.append(data)
    return out


def build_frames(metrics: List[Dict[str, Any]]):
    rows_overall, rows_author, rows_bucket = [], [], []
    for m in metrics:
        model = m["model"]
        o = m.get("overall", {})
        rows_overall.append({"model": model, "top1_acc": o.get("top1_acc", np.nan),
                             "top2_acc": o.get("top2_acc", np.nan), "n": o.get("n", np.nan)})
        for author, d in m.get("by_author", {}).items():
            rows_author.append({"model": model, "author": author,
                                "top1_acc": d.get("top1_acc", np.nan),
                                "top2_acc": d.get("top2_acc", np.nan),
                                "n": d.get("n", np.nan)})
        for bucket, d in m.get("by_bucket", {}).items():
            rows_bucket.append({"model": model, "bucket": bucket,
                                "top1_acc": d.get("top1_acc", np.nan),
                                "top2_acc": d.get("top2_acc", np.nan),
                                "n": d.get("n", np.nan)})
    df_overall = pd.DataFrame(rows_overall).sort_values("model").reset_index(drop=True)
    df_author  = pd.DataFrame(rows_author)
    df_bucket  = pd.DataFrame(rows_bucket)
    return df_overall, df_author, df_bucket


def save_csvs(df_overall: pd.DataFrame, df_author: pd.DataFrame, df_bucket: pd.DataFrame):
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)

    if not df_author.empty:
        best_by_author_idx = df_author.groupby("author")["top1_acc"].idxmax()
        best_by_author = df_author.loc[best_by_author_idx].sort_values("author").reset_index(drop=True)
    else:
        best_by_author = pd.DataFrame(columns=["author", "model", "top1_acc"])

    pivot_bucket = (df_bucket.pivot(index="model", columns="bucket", values="top1_acc").sort_index()
                    if not df_bucket.empty else pd.DataFrame())

    df_overall.to_csv(EVOLVE_DIR / "overall_summary.csv", index=False)
    best_by_author.to_csv(EVOLVE_DIR / "best_by_author.csv", index=False)
    pivot_bucket.to_csv(EVOLVE_DIR / "bucket_summary.csv")


# ---------- Plot helpers ----------
def _bar_label(ax, container, labels=None, fmt="{:.3f}", pad=3, fontsize=9):
    """Use ax.bar_label if available; else manual fallback."""
    try:
        ax.bar_label(container, labels=labels, fmt=fmt, padding=pad, fontsize=fontsize)
    except AttributeError:
        # fallback for very old matplotlib
        for rect, lab in zip(container.patches, labels or []):
            h = rect.get_height()
            if not np.isfinite(h):
                continue
            x = rect.get_x() + rect.get_width() / 2
            ax.annotate(lab or fmt.format(h), (x, h), ha="center", va="bottom", fontsize=fontsize,
                        xytext=(0, 3), textcoords="offset points")


# ---------- Plots ----------
def plot_overall(df_overall: pd.DataFrame):
    # numeric x to avoid categorical glitches
    x = np.arange(len(df_overall))
    models = df_overall["model"].tolist()

    # Top-1
    fig, ax = plt.subplots()
    bars = ax.bar(x, df_overall["top1_acc"].values)
    ax.set_title("Overall Top-1 Accuracy by Model")
    ax.set_xlabel("Model"); ax.set_ylabel("Top-1 Accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(x, models)
    fig.tight_layout()
    labels = [f"{v:.3f}" if np.isfinite(v) else "" for v in df_overall["top1_acc"].values]
    _bar_label(ax, bars, labels=labels)
    fig.savefig(PIC_DIR / "overall_top1_by_model.png", dpi=200); plt.close(fig)

    # Top-2
    fig, ax = plt.subplots()
    bars = ax.bar(x, df_overall["top2_acc"].values)
    ax.set_title("Overall Top-2 Accuracy by Model")
    ax.set_xlabel("Model"); ax.set_ylabel("Top-2 Accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(x, models)
    fig.tight_layout()
    labels = [f"{v:.3f}" if np.isfinite(v) else "" for v in df_overall["top2_acc"].values]
    _bar_label(ax, bars, labels=labels)
    fig.savefig(PIC_DIR / "overall_top2_by_model.png", dpi=200); plt.close(fig)


def plot_per_author(df_author: pd.DataFrame):
    if df_author.empty:
        return
    pivot_author = df_author.pivot(index="author", columns="model", values="top1_acc")
    pivot_author = pivot_author.reindex(sorted(df_author["author"].unique()))
    authors = list(pivot_author.index)
    models  = list(pivot_author.columns)

    x = np.arange(len(authors))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mname in enumerate(models):
        vals = pivot_author[mname].values.astype(float)
        bars = ax.bar(x + i*width - (len(models)-1)*width/2, np.nan_to_num(vals, nan=0.0), width, label=mname)
        labels = ["" if not np.isfinite(v) else f"{v:.3f}" for v in vals]
        _bar_label(ax, bars, labels=labels, fontsize=8)
    ax.set_title("Per-Author Top-1 Accuracy by Model")
    ax.set_xlabel("Author"); ax.set_ylabel("Top-1 Accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(x, authors, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PIC_DIR / "per_author_top1_by_model.png", dpi=200); plt.close(fig)


def plot_long_vs_short(df_bucket: pd.DataFrame):
    if df_bucket.empty:
        return
    pivot_bucket = df_bucket.pivot(index="model", columns="bucket", values="top1_acc").sort_index()
    models  = list(pivot_bucket.index)
    buckets = [c for c in ["long", "short"] if c in pivot_bucket.columns]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots()
    for j, b in enumerate(buckets):
        vals = pivot_bucket[b].values.astype(float)
        bars = ax.bar(x + (j-0.5)*width, np.nan_to_num(vals, nan=0.0), width, label=b)
        labels = ["" if not np.isfinite(v) else f"{v:.3f}" for v in vals]
        _bar_label(ax, bars, labels=labels)
    ax.set_title("Long vs Short Top-1 Accuracy by Model")
    ax.set_xlabel("Model"); ax.set_ylabel("Top-1 Accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(x, models); ax.legend()
    fig.tight_layout()
    fig.savefig(PIC_DIR / "long_vs_short_by_model.png", dpi=200); plt.close(fig)


# ---------- console ----------
def print_summaries(df_overall: pd.DataFrame, df_author: pd.DataFrame):
    print("=== Overall Top-1 by Model ===")
    print(df_overall[["model", "top1_acc"]].to_string(index=False) if not df_overall.empty else "(no data)")
    print("\n=== Best Model per Author (Top-1) ===")
    if not df_author.empty:
        best_idx = df_author.groupby("author")["top1_acc"].idxmax()
        best = df_author.loc[best_idx][["author","model","top1_acc"]]
        print(best.sort_values("author").to_string(index=False))
        row = df_author.loc[df_author["top1_acc"].idxmax()]
        print(f"\n=== Best (Author, Model) Pair ===\nauthor={row['author']}  model={row['model']}  top1_acc={row['top1_acc']:.4f}")
    else:
        print("(no data)")


# ---------- main ----------
if __name__ == "__main__":
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(EVOLVE_DIR)
    df_overall, df_author, df_bucket = build_frames(metrics)
    save_csvs(df_overall, df_author, df_bucket)
    plot_overall(df_overall)
    plot_per_author(df_author)
    plot_long_vs_short(df_bucket)
    print_summaries(df_overall, df_author)