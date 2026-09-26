from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "cross_platform" / "matched_product",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "matched_product_platform_comparison",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "results" / "matched_product_figures",
    )
    args = parser.parse_args()

    corpus_path = args.data_dir / "baseus_bowie_ma10_matched_reviews.csv"
    manifest_path = args.data_dir / "matched_product_manifest.json"
    summary_path = args.results_dir / "analysis_summary.json"
    metrics_path = args.results_dir / "source_classifier_metrics.csv"
    aspects_path = args.results_dir / "aspect_comparison.csv"
    required_figures = [
        "matched_product_review_length.png",
        "matched_product_aspect_mentions.png",
        "matched_product_source_confusion.png",
        "matched_product_distinctive_terms.png",
        "matched_product_model_score.png",
        "matched_product_classifier_sensitivity.png",
    ]

    required = [
        corpus_path,
        manifest_path,
        summary_path,
        metrics_path,
        aspects_path,
        *[args.figures_dir / name for name in required_figures],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required outputs:\n- " + "\n- ".join(missing))

    corpus = pd.read_csv(corpus_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path).iloc[0]
    aspects = pd.read_csv(aspects_path)

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(len(corpus) == 200, f"expected 200 corpus rows, found {len(corpus)}")
    check(
        corpus["platform"].value_counts().to_dict()
        == {"Amazon": 100, "AliExpress": 100},
        "expected 100 reviews from each platform",
    )
    check(corpus["matched_product_id"].nunique() == 1, "expected one matched product")
    check(corpus["rating"].eq(5).all(), "all reviews must be five-star")
    check(corpus["language"].str.lower().eq("en").all(), "all reviews must be English")
    check(
        manifest["matched_product"]["amazon"]["asin"] == "B0C89QQ5XK",
        "unexpected Amazon ASIN",
    )
    check(
        manifest["matched_product"]["aliexpress"]["item_id"]
        == "1005005742938199",
        "unexpected AliExpress item id",
    )
    check(
        manifest["corpus_sha256"] == sha256(corpus_path),
        "corpus SHA-256 does not match the manifest",
    )
    check(abs(float(metrics["macro_f1"]) - 0.9349853717) < 1e-8, "full-text macro F1 drift")
    check(
        abs(float(metrics["fifteen_word_macro_f1"]) - 0.8648344222) < 1e-8,
        "15-word macro F1 drift",
    )
    check(float(metrics["permutation_p_value"]) < 0.01, "full-text permutation test failed")
    check(
        float(metrics["fifteen_word_permutation_p_value"]) < 0.01,
        "15-word permutation test failed",
    )
    check(len(aspects) == 8, f"expected 8 aspect tests, found {len(aspects)}")
    check(aspects["fdr_adjusted_p"].lt(0.05).all(), "not all aspect tests survive FDR correction")
    check(
        summary["interpretation"].startswith("The analysis tests platform-market language shift"),
        "interpretation boundary is missing",
    )

    if failures:
        print("Matched-product validation: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Matched-product validation: PASS")
    print("- 200 reviews: 100 Amazon and 100 AliExpress")
    print("- exact product, five-star rating and English language checks passed")
    print(f"- full-text macro F1: {float(metrics['macro_f1']):.4f}")
    print(f"- first-15-word macro F1: {float(metrics['fifteen_word_macro_f1']):.4f}")
    print("- all eight aspect comparisons remain significant after FDR correction")


if __name__ == "__main__":
    main()
