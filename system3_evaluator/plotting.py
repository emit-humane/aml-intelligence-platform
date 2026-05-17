"""
E — Plotting utilities for evaluation reports.

Generates:
  - PR curve (precision-recall)
  - ROC curve
  - Pattern detection heatmap
  - Confusion matrix

Outputs PNG files to data/evaluation/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path | str,
    title: str = "Precision-Recall Curve",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec, prec, color="#3b82f6", linewidth=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[plotting] PR curve -> {out_path}")
    except Exception as exc:
        print(f"[plotting] PR curve failed: {exc}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path | str,
    title: str = "ROC Curve",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#ef4444", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR", fontsize=12)
        ax.set_ylabel("TPR", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[plotting] ROC curve -> {out_path}")
    except Exception as exc:
        print(f"[plotting] ROC curve failed: {exc}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | str,
    title: str = "Confusion Matrix",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        cm = np.array([[tn, fp], [fn, tp]])
        labels = [["TN", "FP"], ["FN", "TP"]]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Normal", "Predicted Fraud"])
        ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        ax.set_title(title, fontsize=13)
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{labels[i][j]}\n{cm[i, j]:,}",
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[plotting] Confusion matrix -> {out_path}")
    except Exception as exc:
        print(f"[plotting] Confusion matrix failed: {exc}")


def plot_pattern_coverage(
    coverage: dict[str, dict],
    out_path: Path | str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        patterns = list(coverage.keys())
        rates = [coverage[p]["detection_rate"] for p in patterns]
        colors = ["#ef4444" if r < 0.5 else "#f97316" if r < 0.75 else "#22c55e" for r in rates]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(patterns, rates, color=colors)
        ax.set_xlabel("Detection Rate", fontsize=12)
        ax.set_title("Per-Typology Detection Coverage", fontsize=14)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1%}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[plotting] Pattern coverage -> {out_path}")
    except Exception as exc:
        print(f"[plotting] Pattern coverage failed: {exc}")
