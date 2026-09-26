from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Amazon": "#2F5D7C", "AliExpress": "#B85C3A"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "matched_product_platform_comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "matched_product_figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.results_dir / "matched_product_predictions.csv")
    aspects = pd.read_csv(args.results_dir / "aspect_comparison.csv")
    terms = pd.read_csv(args.results_dir / "distinctive_terms.csv")
    matrix = pd.read_csv(args.results_dir / "source_classifier_confusion_matrix.csv", index_col=0)
    classifier = pd.read_csv(args.results_dir / "source_classifier_metrics.csv").iloc[0]

    order = ["Amazon", "AliExpress"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.boxplot(
        data=predictions,
        x="platform",
        y="word_count",
        order=order,
        hue="platform",
        palette=COLORS,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Words per review")
    ax.set_title("Same Product and Rating: Review Length")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_review_length.png", dpi=180)
    plt.close(fig)

    aspect_plot = aspects.melt(
        id_vars="aspect",
        value_vars=["amazon_share", "aliexpress_share"],
        var_name="platform",
        value_name="share",
    )
    aspect_plot["platform"] = aspect_plot["platform"].map(
        {"amazon_share": "Amazon", "aliexpress_share": "AliExpress"}
    )
    aspect_plot["aspect"] = aspect_plot["aspect"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.barplot(
        data=aspect_plot,
        x="share",
        y="aspect",
        hue="platform",
        hue_order=order,
        palette=COLORS,
        ax=ax,
    )
    ax.set_xlabel("Share of reviews mentioning aspect")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Aspect Mentions for Baseus Bowie MA10")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_aspect_mentions.png", dpi=180)
    plt.close(fig)

    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    normalized.index = [label.replace("actual_", "") for label in normalized.index]
    normalized.columns = [label.replace("pred_", "") for label in normalized.columns]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row proportion"},
        ax=ax,
    )
    ax.set_xlabel("Predicted platform")
    ax.set_ylabel("Actual platform")
    ax.set_title("Text-only Platform Classification")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_source_confusion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Full review", "First 15 words"]
    values = [classifier["macro_f1"], classifier["fifteen_word_macro_f1"]]
    bars = ax.bar(labels, values, color=[COLORS["Amazon"], COLORS["AliExpress"]], width=0.55)
    ax.axhline(0.5, color="#444444", linewidth=1, linestyle="--", label="Chance benchmark")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Five-fold macro F1")
    ax.set_title("Platform Classification Sensitivity Test")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_classifier_sensitivity.png", dpi=180)
    plt.close(fig)

    selected_terms = pd.concat(
        [
            group.sort_values("rank").head(10)
            for _, group in terms.groupby("platform", sort=False)
        ],
        ignore_index=True,
    )
    selected_terms["signed_score"] = selected_terms[
        "z_score_amazon_minus_aliexpress"
    ]
    selected_terms = selected_terms.sort_values("signed_score")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = selected_terms["platform"].map(COLORS)
    ax.barh(selected_terms["term"], selected_terms["signed_score"], color=colors)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Smoothed log-odds z-score (Amazon minus AliExpress)")
    ax.set_ylabel("")
    ax.set_title("Distinctive Terms in Five-star Reviews")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_distinctive_terms.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.boxplot(
        data=predictions,
        x="platform",
        y="prediction_score",
        order=order,
        hue="platform",
        palette=COLORS,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Main-model decision score")
    ax.set_title("Model-score Shift for the Same Product")
    fig.tight_layout()
    fig.savefig(args.output_dir / "matched_product_model_score.png", dpi=180)
    plt.close(fig)

    print(f"Matched-product figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
