from __future__ import annotations

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
    matched_dir = ROOT / "data" / "cross_platform" / "matched_product"
    external_dir = ROOT / "data" / "external" / "derev2018" / "processed"
    matched_results = ROOT / "results" / "matched_product_platform_comparison"
    external_results = ROOT / "results" / "external_validation_derev2018"
    matched_figures = ROOT / "results" / "matched_product_figures"
    external_figures = ROOT / "results" / "external_validation_figures"

    required = [
        matched_dir / "baseus_bowie_ma10_matched_reviews.csv",
        matched_dir / "matched_product_manifest.json",
        external_dir / "derev2018_published.csv",
        external_dir / "derev2018_crowdsourced.csv",
        matched_results / "source_classifier_metrics.csv",
        matched_results / "aspect_comparison.csv",
        external_results / "external_validation_metrics.csv",
        ROOT / "results" / "screen_share_baseline" / "best_baseline_model.joblib",
        ROOT / "results" / "screen_share_hybrid" / "best_hybrid_model.joblib",
    ]
    required.extend(
        matched_figures / name
        for name in [
            "matched_product_review_length.png",
            "matched_product_aspect_mentions.png",
            "matched_product_source_confusion.png",
            "matched_product_distinctive_terms.png",
            "matched_product_model_score.png",
            "matched_product_classifier_sensitivity.png",
        ]
    )
    required.extend(
        external_figures / name
        for name in [
            "external_validation_f1.png",
            "external_validation_zero_shot_confusion.png",
        ]
    )

    failures = [f"missing required file: {path.relative_to(ROOT)}" for path in required if not path.is_file()]

    matched = pd.read_csv(matched_dir / "baseus_bowie_ma10_matched_reviews.csv")
    manifest = json.loads((matched_dir / "matched_product_manifest.json").read_text(encoding="utf-8"))
    matched_metrics = pd.read_csv(matched_results / "source_classifier_metrics.csv").iloc[0]
    aspects = pd.read_csv(matched_results / "aspect_comparison.csv")
    published = pd.read_csv(external_dir / "derev2018_published.csv")
    crowdsourced = pd.read_csv(external_dir / "derev2018_crowdsourced.csv")
    external_metrics = pd.read_csv(external_results / "external_validation_metrics.csv")

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(len(matched) == 200, f"matched corpus has {len(matched)} rows instead of 200")
    check(
        matched["platform"].value_counts().to_dict() == {"Amazon": 100, "AliExpress": 100},
        "matched corpus is not balanced at 100 reviews per platform",
    )
    check(matched["matched_product_id"].nunique() == 1, "matched corpus contains more than one product")
    check(matched["rating"].eq(5).all(), "matched corpus contains ratings other than five stars")
    check(matched["language"].str.lower().eq("en").all(), "matched corpus contains non-English rows")
    check(
        manifest["corpus_sha256"] == sha256(matched_dir / "baseus_bowie_ma10_matched_reviews.csv"),
        "matched corpus checksum does not agree with its source manifest",
    )
    check(abs(float(matched_metrics["macro_f1"]) - 0.9349853717) < 1e-8, "full-text platform F1 drift")
    check(
        abs(float(matched_metrics["fifteen_word_macro_f1"]) - 0.8648344222) < 1e-8,
        "15-word platform F1 drift",
    )
    check(float(matched_metrics["permutation_p_value"]) < 0.01, "full-text permutation test failed")
    check(
        float(matched_metrics["fifteen_word_permutation_p_value"]) < 0.01,
        "15-word permutation test failed",
    )
    check(len(aspects) == 8 and aspects["fdr_adjusted_p"].lt(0.05).all(), "aspect-analysis FDR check failed")

    for name, frame in [("published", published), ("crowdsourced", crowdsourced)]:
        check(len(frame) == 1552, f"DeRev {name} subset has {len(frame)} rows instead of 1552")
        check(
            frame["is_fake"].value_counts().to_dict() == {0: 776, 1: 776},
            f"DeRev {name} subset is not label balanced",
        )

    indexed = external_metrics.set_index(["experiment", "dataset"])
    zero_shot = float(indexed.loc[("main_hybrid_model_zero_shot", "published"), "f1_fake"])
    in_domain = float(indexed.loc[("derev_text_model_in_domain_oof", "published"), "f1_fake"])
    check(abs(zero_shot - 0.0676328502) < 1e-8, "published DeRev zero-shot F1 drift")
    check(abs(in_domain - 0.9572649573) < 1e-8, "published DeRev in-domain F1 drift")

    if failures:
        print("Supervisor revision validation: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Supervisor revision validation: PASS")
    print("- matched product: 100 Amazon and 100 AliExpress English five-star reviews")
    print(f"- platform classifier macro F1: {float(matched_metrics['macro_f1']):.4f}")
    print(f"- first-15-word sensitivity macro F1: {float(matched_metrics['fifteen_word_macro_f1']):.4f}")
    print(f"- published DeRev zero-shot fake-class F1: {zero_shot:.4f}")
    print(f"- published DeRev in-domain fake-class F1: {in_domain:.4f}")


if __name__ == "__main__":
    main()
