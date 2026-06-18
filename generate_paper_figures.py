"""
Generate paper figures from the final-holdout artifacts.

Inputs:
  logs/eval_report_final_holdout_2026-06-16.json
  logs/ablation_results_final_holdout.json
  logs/baseline_comparison.json
  logs/commercial_comparison.json
  logs/curves/roc_final_holdout.png
  logs/curves/pr_final_holdout.png

Outputs:
  figures/fig2_roc_pr.{png,pdf}
  figures/fig3_confusion_attribution.{png,pdf}
  figures/fig4_latency.{png,pdf}
  figures/fig5_method_comparison.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("figures")
REPORT_PATH = Path("logs/eval_report_final_holdout_2026-06-16.json")
ABLATION_PATH = Path("logs/ablation_results_final_holdout.json")
BASELINE_PATH = Path("logs/baseline_comparison.json")
COMMERCIAL_PATH = Path("logs/commercial_comparison.json")
ROC_PATH = Path("logs/curves/roc_final_holdout.png")
PR_PATH = Path("logs/curves/pr_final_holdout.png")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"saved figures/{stem}.png and .pdf")


def make_fig2_roc_pr(report: dict) -> None:
    """Use the final-holdout ROC/PR plots produced by eval_suite.py."""
    if not ROC_PATH.exists() or not PR_PATH.exists():
        raise FileNotFoundError(
            "Missing final-holdout ROC/PR curves in logs/curves. "
            "Run the final evaluation first or restore the curve artifacts."
        )

    roc_img = mpimg.imread(ROC_PATH)
    pr_img = mpimg.imread(PR_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
    for ax, img, title in zip(
        axes,
        (roc_img, pr_img),
        (
            f"ROC Curve (AUC = {report['AUC_ROC_curve']:.4f})",
            f"Precision-Recall Curve (AUC = {report['AUC_PR_curve']:.4f})",
        ),
    ):
        ax.imshow(img)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Final Holdout Discrimination Curves (n = 867)",
        y=1.02,
        fontsize=10,
        fontweight="bold",
    )
    _save(fig, "fig2_roc_pr")


def _draw_confusion(ax: plt.Axes, report: dict, policy: str) -> None:
    if policy == "prevention":
        tp, fp, tn, fn = (
            report["TP_prevention"],
            report["FP_prevention"],
            report["TN_prevention"],
            report["FN_prevention"],
        )
        title = "Prevention Policy"
        right_label = "Predicted Block"
    else:
        tp, fp, tn, fn = (
            report["TP_detection"],
            report["FP_detection"],
            report["TN_detection"],
            report["FN_detection"],
        )
        title = "Detection Policy"
        right_label = "Predicted Block/Monitor"

    matrix = np.array([[tn, fp], [fn, tp]])
    vmax = max(1, int(matrix.max()))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)

    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    row_totals = matrix.sum(axis=1)
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            pct = value / row_totals[i] * 100 if row_totals[i] else 0
            color = "white" if value > vmax * 0.55 else "#1f2933"
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{value}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted Allow", right_label], rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Benign", "Attack/Evasion"])
    ax.set_title(title, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_fig3_confusion_attribution(report: dict) -> None:
    fig = plt.figure(figsize=(7.1, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.55, wspace=0.35)

    ax_prev = fig.add_subplot(gs[0, 0])
    ax_det = fig.add_subplot(gs[0, 1])
    _draw_confusion(ax_prev, report, "prevention")
    _draw_confusion(ax_det, report, "detection")

    ax_attr = fig.add_subplot(gs[1, 0])
    attribution = report["layer_attribution"]
    names = ["Layer 2\nIntent", "Layer 1\nAnomaly"]
    values = [
        attribution.get("Layer 2 - Intent Classifier", 0),
        attribution.get("Layer 1 - Anomaly Detection", 0),
    ]
    colors = ["#d97706", "#2563eb"]
    bars = ax_attr.bar(names, values, color=colors, width=0.55)
    ax_attr.set_title("True-Positive Blocking Attribution", fontweight="bold")
    ax_attr.set_ylabel("Count")
    ax_attr.set_ylim(0, max(values) * 1.25)
    ax_attr.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        pct = value / report["TP_prevention"] * 100
        ax_attr.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.035,
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.axis("off")
    summary_rows = [
        ("ADR prevention", f"{report['ADR_prevention']*100:.2f}%"),
        ("ADR detection", f"{report['ADR_detection']*100:.2f}%"),
        ("FPR", f"{report['FPR_prevention']*100:.2f}%"),
        ("Precision", f"{report['precision_prevention']*100:.2f}%"),
        ("F1 prevention", f"{report['F1_prevention']:.4f}"),
    ]
    table = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.22)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")

    fig.suptitle(
        "Final Holdout Confusion Matrices and Layer Attribution",
        y=0.995,
        fontsize=10,
        fontweight="bold",
    )
    _save(fig, "fig3_confusion_attribution")


def make_fig4_latency(report: dict) -> None:
    latency = report["latency_breakdown"]
    labels = [
        "Chunking",
        "L1 anomaly",
        "L2 intent",
        "L1+L2 wall",
        "L3 monitor",
        "Meta",
        "Total",
    ]
    means = [
        latency["chunking_ms"]["mean_ms"],
        latency["l1_ms"]["mean_ms"],
        latency["l2_ms"]["mean_ms"],
        latency["l1_l2_wall_ms"]["mean_ms"],
        latency["l3_ms"]["mean_ms"],
        latency["meta_ms"]["mean_ms"],
        latency["total_ms"]["mean_ms"],
    ]
    p95s = [
        latency["chunking_ms"]["p95_ms"],
        latency["l1_ms"]["p95_ms"],
        latency["l2_ms"]["p95_ms"],
        latency["l1_l2_wall_ms"]["p95_ms"],
        latency["l3_ms"]["p95_ms"],
        latency["meta_ms"]["p95_ms"],
        latency["total_ms"]["p95_ms"],
    ]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    ax.barh(y + 0.18, means, height=0.34, label="Mean", color="#2563eb")
    ax.barh(y - 0.18, p95s, height=0.34, label="P95", color="#93c5fd")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Per-Component Latency on Final Holdout", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 8000)
    ax.grid(axis="x", which="both", alpha=0.18)
    ax.legend(loc="upper right")
    for yi, mean, p95 in zip(y, means, p95s):
        ax.text(mean * 1.12, yi + 0.18, f"{mean:.2g}", va="center", fontsize=7)
        ax.text(p95 * 1.12, yi - 0.18, f"{p95:.2g}", va="center", fontsize=7)
    ax.text(
        0.01,
        -0.18,
        "Log scale",
        transform=ax.transAxes,
        fontsize=7,
        color="#4b5563",
    )
    _save(fig, "fig4_latency")


def make_fig5_method_comparison() -> None:
    baseline = _load_json(BASELINE_PATH)
    commercial = _load_json(COMMERCIAL_PATH)

    names = [
        "Keyword\nBlocklist",
        "DeBERTa\nL2 only",
        "RAG-Shield\nbaseline set",
        "Prompt\nGuard 2",
        "Llama-3.1\nGuardrail",
        "NeMo\nRail",
        "RAG-Shield\nlive set",
    ]
    adr = [
        baseline["Keyword Blocklist"]["metrics"]["ADR"] * 100,
        baseline["DeBERTa (L2 only)"]["metrics"]["ADR"] * 100,
        baseline["RAG-Shield (full)"]["metrics"]["ADR"] * 100,
        commercial["Llama Prompt Guard 2"]["metrics"]["ADR"] * 100,
        commercial["Llama-3.1-8b Guardrail"]["metrics"]["ADR"] * 100,
        commercial["NeMo Rail (Llama-3.1)"]["metrics"]["ADR"] * 100,
        commercial["RAG-Shield (Ours)"]["metrics"]["ADR"] * 100,
    ]
    fpr = [
        baseline["Keyword Blocklist"]["metrics"]["FPR"] * 100,
        baseline["DeBERTa (L2 only)"]["metrics"]["FPR"] * 100,
        baseline["RAG-Shield (full)"]["metrics"]["FPR"] * 100,
        commercial["Llama Prompt Guard 2"]["metrics"]["FPR"] * 100,
        commercial["Llama-3.1-8b Guardrail"]["metrics"]["FPR"] * 100,
        commercial["NeMo Rail (Llama-3.1)"]["metrics"]["FPR"] * 100,
        commercial["RAG-Shield (Ours)"]["metrics"]["FPR"] * 100,
    ]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.bar(x - width / 2, adr, width, label="ADR", color="#2563eb")
    ax.bar(x + width / 2, fpr, width, label="FPR", color="#dc2626")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Baseline and Commercial Comparison (Separate Evaluation Sets)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left", ncol=2)
    ax.text(
        0.5,
        -0.24,
        "Classical baseline set: N=848. Live commercial set: N=856. Do not compare latency or FPR across sets as a single pooled benchmark.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#4b5563",
    )
    _save(fig, "fig5_method_comparison")


def main() -> None:
    report = _load_json(REPORT_PATH)
    _load_json(ABLATION_PATH)
    _load_json(BASELINE_PATH)
    _load_json(COMMERCIAL_PATH)

    print("Generating updated paper figures from final artifacts")
    make_fig2_roc_pr(report)
    make_fig3_confusion_attribution(report)
    make_fig4_latency(report)
    make_fig5_method_comparison()


if __name__ == "__main__":
    main()
