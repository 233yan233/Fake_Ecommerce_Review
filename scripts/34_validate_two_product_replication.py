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
        default=ROOT / "data" / "cross_platform" / "matched_product_replication",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "two_matched_product_replication",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "results" / "two_matched_product_figures",
    )
    args = parser.parse_args()

    corpus_path = args.data_dir / "kospet_tank_t3_ultra_matched_reviews.csv"
    manifest_path = args.data_dir / "matched_product_replication_manifest.json"
    summary_path = args.results_dir / "analysis_summary.json"
    within_path = args.results_dir / "within_product_classifier_metrics.csv"
    transfer_path = args.results_dir / "cross_product_transfer_metrics.csv"
    length_path = args.results_dir / "two_product_length_tests.csv"
    aspect_path = args.results_dir / "kospet_aspect_comparison.csv"
    required_figures = [
        "two_product_review_length.png",
        "two_product_classifier_sensitivity.png",
        "cross_product_transfer.png",
        "two_product_vocabulary_shift.png",
        "kospet_aspect_mentions.png",
    ]
    required = [
        corpus_path,
        manifest_path,
        summary_path,
        within_path,
        transfer_path,
        length_path,
        aspect_path,
        *[args.figures_dir / name for name in required_figures],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required outputs:\n- " + "\n- ".join(missing))

    corpus = pd.read_csv(corpus_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    within = pd.read_csv(within_path).set_index("matched_product_id")
    transfer = pd.read_csv(transfer_path).set_index(
        ["train_product_id", "test_product_id"]
    )
    length = pd.read_csv(length_path).set_index("matched_product_id")
    aspects = pd.read_csv(aspect_path)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(len(corpus) == 70, f"expected 70 replication rows, found {len(corpus)}")
    check(
        corpus["platform"].value_counts().to_dict()
        == {"Amazon": 35, "AliExpress": 35},
        "expected 35 reviews from each platform",
    )
    check(
        corpus["matched_product_id"].eq("kospet_tank_t3_ultra").all(),
        "replication corpus contains an unexpected product",
    )
    check(corpus["rating"].eq(5).all(), "all replication reviews must be five-star")
    check(
        corpus["language"].str.lower().eq("en").all(),
        "all replication reviews must be English",
    )
    check(
        manifest["matched_product"]["amazon"]["asin"] == "B0CQXB54JC",
        "unexpected Amazon ASIN",
    )
    check(
        manifest["matched_product"]["aliexpress"]["item_id"]
        == "1005006851800028",
        "unexpected AliExpress item id",
    )
    check(
        manifest["corpus_sha256"] == sha256(corpus_path),
        "replication corpus SHA-256 does not match its manifest",
    )

    baseus = within.loc["baseus_bowie_ma10"]
    kospet = within.loc["kospet_tank_t3_ultra"]
    check(abs(float(baseus["macro_f1"]) - 0.9349853717) < 1e-8, "Baseus F1 drift")
    check(abs(float(kospet["macro_f1"]) - 0.7285160237) < 1e-8, "KOSPET F1 drift")
    check(
        abs(float(kospet["first15_macro_f1"]) - 0.6338146056) < 1e-8,
        "KOSPET first-15-word F1 drift",
    )
    check(float(kospet["permutation_p_value"]) < 0.01, "KOSPET full-text test failed")
    check(
        float(kospet["first15_permutation_p_value"]) < 0.05,
        "KOSPET first-15-word test failed",
    )

    baseus_to_kospet = transfer.loc[("baseus_bowie_ma10", "kospet_tank_t3_ultra")]
    kospet_to_baseus = transfer.loc[("kospet_tank_t3_ultra", "baseus_bowie_ma10")]
    check(
        abs(float(baseus_to_kospet["macro_f1"]) - 0.8273026316) < 1e-8,
        "Baseus-to-KOSPET transfer F1 drift",
    )
    check(
        abs(float(kospet_to_baseus["macro_f1"]) - 0.7943779934) < 1e-8,
        "KOSPET-to-Baseus transfer F1 drift",
    )
    check(
        transfer["permutation_p_value"].lt(0.01).all(),
        "one or more cross-product transfer tests failed",
    )
    check(
        length["fdr_adjusted_p"].lt(0.001).all(),
        "review-length differences did not survive FDR correction",
    )
    check(
        aspects.loc[aspects["fdr_adjusted_p"].lt(0.05), "aspect"].tolist()
        == ["display"],
        "unexpected KOSPET FDR-significant aspect set",
    )
    check(summary["total_rows"] == 270, "expected 270 rows in the combined analysis")
    check(
        summary["interpretation"].endswith(
            "The evidence does not isolate national culture or estimate fraud prevalence."
        ),
        "interpretation boundary is missing",
    )

    if failures:
        print("Two-product replication validation: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Two-product replication validation: PASS")
    print("- exact KOSPET TANK T3 Ultra match: 35 Amazon and 35 AliExpress reviews")
    print(f"- KOSPET within-product macro F1: {float(kospet['macro_f1']):.4f}")
    print(
        "- cross-product macro F1: "
        f"{float(baseus_to_kospet['macro_f1']):.4f} and "
        f"{float(kospet_to_baseus['macro_f1']):.4f}"
    )
    print("- corpus checksum, significance tests and five figures passed")


if __name__ == "__main__":
    main()
