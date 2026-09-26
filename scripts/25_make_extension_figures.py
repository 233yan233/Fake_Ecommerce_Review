from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2F5D7C", "#B85C3A", "#4C7A56", "#8A6A3D"]


def external_f1_figure(external_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(external_dir / "external_validation_metrics.csv")
    selected = metrics[
        (
            metrics["experiment"].isin(
                ["main_hybrid_model_zero_shot", "derev_text_model_in_domain_oof"]
            )
        )
        & metrics["dataset"].eq("published")
    ].copy()
    selected["label"] = selected["experiment"].map(
        {
            "main_hybrid_model_zero_shot": "Main model\nzero-shot",
            "derev_text_model_in_domain_oof": "DeRev model\nin-domain OOF",
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(selected["label"], selected["f1_fake"], color=COLORS[: len(selected)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fake-class F1")
    ax.set_title("External Validation on Published DeRev Reviews")
    for bar, value in zip(bars, selected["f1_fake"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "external_validation_f1.png", dpi=180)
    plt.close(fig)


def normalized_confusion_figure(csv_path: Path, title: str, output_path: Path) -> None:
    matrix = pd.read_csv(csv_path, index_col=0)
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row proportion"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def score_distribution_figure(cross_platform_dir: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(cross_platform_dir / "cross_platform_predictions.csv")
    order = ["Amazon", "Alibaba", "AliExpress"]
    data = [
        predictions.loc[predictions["platform"].eq(platform), "prediction_score"].to_numpy()
        for platform in order
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    plot = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(plot["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_ylabel("Linear SVM decision score")
    ax.set_title("Decision-score Shift Across Unlabelled Sources")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_platform_score_distribution.png", dpi=180)
    plt.close(fig)


def vocabulary_shift_figure(cross_platform_dir: Path, output_dir: Path) -> None:
    shift = pd.read_csv(cross_platform_dir / "pairwise_vocabulary_shift.csv")
    shift["pair"] = shift["platform_a"] + " / " + shift["platform_b"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(shift["pair"], shift["jensen_shannon_distance"], color=COLORS[: len(shift)])
    ax.set_ylim(0, max(0.6, float(shift["jensen_shannon_distance"].max()) + 0.08))
    ax.set_ylabel("Jensen-Shannon distance")
    ax.set_title("Pairwise Vocabulary Shift")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, shift["jensen_shannon_distance"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_platform_vocabulary_shift.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=ROOT / "results" / "external_validation_derev2018",
    )
    parser.add_argument(
        "--cross-platform-dir",
        type=Path,
        default=ROOT / "results" / "cross_platform_domain_shift",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "domain_shift_figures",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Create only the DeRev external-validation figures.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    external_f1_figure(args.external_dir, args.output_dir)
    normalized_confusion_figure(
        args.external_dir / "main_hybrid_model_zero_shot_published_confusion_matrix.csv",
        "Main Model on Published DeRev Reviews",
        args.output_dir / "external_validation_zero_shot_confusion.png",
    )
    if args.external_only:
        print(f"External-validation figures saved to {args.output_dir}")
        return

    normalized_confusion_figure(
        args.cross_platform_dir / "product_source_classifier_confusion_matrix.csv",
        "Grouped Product-source Classification",
        args.output_dir / "product_source_classifier_confusion.png",
    )
    score_distribution_figure(args.cross_platform_dir, args.output_dir)
    vocabulary_shift_figure(args.cross_platform_dir, args.output_dir)
    print(f"Extension figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
