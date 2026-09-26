from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PLATFORM_COLORS = {"Amazon": "#2F5D7C", "AliExpress": "#B85C3A"}
ACCENT_COLORS = ["#2F5D7C", "#B85C3A", "#4C7A56", "#8A6A3D"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "two_matched_product_replication",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "two_matched_product_figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.results_dir / "two_product_predictions.csv")
    within = pd.read_csv(args.results_dir / "within_product_classifier_metrics.csv")
    transfer = pd.read_csv(args.results_dir / "cross_product_transfer_metrics.csv")
    lexical = pd.read_csv(args.results_dir / "two_product_lexical_shift.csv")
    aspects = pd.read_csv(args.results_dir / "kospet_aspect_comparison.csv")

    product_order = ["Baseus Bowie MA10", "KOSPET TANK T3 Ultra"]
    platform_order = ["Amazon", "AliExpress"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(
        data=predictions,
        x="product",
        y="word_count",
        order=product_order,
        hue="platform",
        hue_order=platform_order,
        palette=PLATFORM_COLORS,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Words per review")
    ax.set_title("Review Length for Two Matched Products")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(args.output_dir / "two_product_review_length.png", dpi=180)
    plt.close(fig)

    within_plot = within[["product", "macro_f1", "first15_macro_f1"]].melt(
        id_vars="product", var_name="text_scope", value_name="score"
    )
    within_plot["text_scope"] = within_plot["text_scope"].map(
        {"macro_f1": "Full review", "first15_macro_f1": "First 15 words"}
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.barplot(
        data=within_plot,
        x="product",
        y="score",
        order=product_order,
        hue="text_scope",
        hue_order=["Full review", "First 15 words"],
        palette=ACCENT_COLORS[:2],
        ax=ax,
    )
    ax.axhline(0.5, color="#444444", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Five-fold macro F1")
    ax.set_title("Within-product Platform Classification")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(args.output_dir / "two_product_classifier_sensitivity.png", dpi=180)
    plt.close(fig)

    transfer["direction"] = transfer.apply(
        lambda row: f"{row['train_product'].replace('Baseus Bowie ', '')} to "
        f"{row['test_product'].replace('KOSPET TANK ', '')}",
        axis=1,
    )
    transfer.loc[
        transfer["train_product_id"].eq("kospet_tank_t3_ultra"), "direction"
    ] = "T3 Ultra to MA10"
    transfer.loc[
        transfer["train_product_id"].eq("baseus_bowie_ma10"), "direction"
    ] = "MA10 to T3 Ultra"
    transfer = transfer.sort_values("train_product_id").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(transfer["direction"], transfer["macro_f1"], color=ACCENT_COLORS[:2])
    ax.axhline(0.5, color="#444444", linewidth=1, linestyle="--", label="Chance benchmark")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in transfer["macro_f1"]], padding=4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Test-set macro F1")
    ax.set_title("Cross-product Transfer of Platform Language")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cross_product_transfer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(lexical["product"], lexical["jensen_shannon_distance"], color=ACCENT_COLORS[:2])
    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in lexical["jensen_shannon_distance"]],
        padding=4,
    )
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Jensen-Shannon distance")
    ax.set_title("Within-product Vocabulary Shift")
    fig.tight_layout()
    fig.savefig(args.output_dir / "two_product_vocabulary_shift.png", dpi=180)
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
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(
        data=aspect_plot,
        x="share",
        y="aspect",
        hue="platform",
        hue_order=platform_order,
        palette=PLATFORM_COLORS,
        ax=ax,
    )
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("Share of reviews mentioning aspect")
    ax.set_ylabel("")
    ax.set_title("KOSPET TANK T3 Ultra Aspect Mentions")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(args.output_dir / "kospet_aspect_mentions.png", dpi=180)
    plt.close(fig)

    print(f"Two-product figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
