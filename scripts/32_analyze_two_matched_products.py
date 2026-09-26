from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from review_detection.features import add_simple_sentiment_features, add_text_statistics
from scripts_compat import add_week2_features_compat


WORD_RE = re.compile(r"\b[a-z][a-z'-]*\b")
PRODUCT_LABELS = {
    "baseus_bowie_ma10": "Baseus Bowie MA10",
    "kospet_tank_t3_ultra": "KOSPET TANK T3 Ultra",
}
KOSPET_ASPECTS = {
    "display": {"amoled", "brightness", "bright", "display", "screen", "touch"},
    "battery": {"battery", "charge", "charging", "day", "days", "week"},
    "health_tracking": {"heart", "pulse", "sleep", "spo2", "step", "steps"},
    "gps_navigation": {"altitude", "compass", "gps", "location", "track"},
    "app_connectivity": {"app", "application", "bluetooth", "connect", "connection", "phone"},
    "calling_notifications": {"call", "calls", "message", "notification", "speaker"},
    "durability_water": {"durable", "metal", "rugged", "steel", "swim", "water", "waterproof"},
    "comfort_design": {"band", "bracelet", "comfortable", "design", "heavy", "strap", "wrist"},
    "delivery_packaging": {"arrived", "box", "delivery", "package", "shipping"},
}


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted.clip(0, 1).tolist()


def prediction_score(model, data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(data), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(data)[:, 1], dtype=float)
    return np.full(len(data), np.nan)


def text_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=10000,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def save_confusion(path: Path, actual, predicted) -> None:
    labels = ["AliExpress", "Amazon"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(path)


def lexical_shift(frame: pd.DataFrame) -> dict:
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
        min_df=2,
    )
    matrix = vectorizer.fit_transform(frame["review_text"])
    terms = vectorizer.get_feature_names_out()
    counts = {
        platform: np.asarray(matrix[frame["platform"].eq(platform).to_numpy()].sum(axis=0))
        .ravel()
        .astype(float)
        for platform in ["Amazon", "AliExpress"]
    }
    distributions = {
        platform: (values + 0.5) / (values.sum() + 0.5 * len(values))
        for platform, values in counts.items()
    }
    top_n = min(100, len(terms))
    top_sets = {
        platform: set(terms[np.argsort(values)[::-1][:top_n]])
        for platform, values in counts.items()
    }
    union = top_sets["Amazon"] | top_sets["AliExpress"]
    return {
        "jensen_shannon_distance": float(
            jensenshannon(distributions["Amazon"], distributions["AliExpress"], base=2)
        ),
        "top_term_jaccard": len(top_sets["Amazon"] & top_sets["AliExpress"]) / len(union),
        "top_n": top_n,
        "vocabulary_size": len(terms),
    }


def within_product_classification(frame: pd.DataFrame, results_dir: Path) -> dict:
    product_id = frame["matched_product_id"].iloc[0]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prediction = cross_val_predict(
        text_model(), frame["review_text"], frame["platform"], cv=splitter, n_jobs=-1
    )
    truncated = frame["review_text"].map(lambda value: " ".join(str(value).split()[:15]))
    truncated_prediction = cross_val_predict(
        text_model(), truncated, frame["platform"], cv=splitter, n_jobs=-1
    )
    observed, permutation_scores, p_value = permutation_test_score(
        text_model(),
        frame["review_text"],
        frame["platform"],
        scoring="f1_macro",
        cv=splitter,
        n_permutations=500,
        n_jobs=-1,
        random_state=42,
    )
    truncated_observed, truncated_scores, truncated_p = permutation_test_score(
        text_model(),
        truncated,
        frame["platform"],
        scoring="f1_macro",
        cv=splitter,
        n_permutations=500,
        n_jobs=-1,
        random_state=42,
    )
    save_confusion(
        results_dir / f"{product_id}_within_product_confusion_matrix.csv",
        frame["platform"],
        prediction,
    )
    save_confusion(
        results_dir / f"{product_id}_first15_confusion_matrix.csv",
        frame["platform"],
        truncated_prediction,
    )
    pd.DataFrame(
        {
            "review_id": frame["review_id"],
            "matched_product_id": product_id,
            "platform": frame["platform"],
            "predicted_platform": prediction,
            "first15_predicted_platform": truncated_prediction,
        }
    ).to_csv(results_dir / f"{product_id}_within_product_predictions.csv", index=False)
    return {
        "matched_product_id": product_id,
        "product": PRODUCT_LABELS[product_id],
        "rows": len(frame),
        "sample_per_platform": int(frame["platform"].value_counts().min()),
        "accuracy": accuracy_score(frame["platform"], prediction),
        "macro_f1": f1_score(frame["platform"], prediction, average="macro"),
        "permutation_observed_macro_f1": observed,
        "permutation_mean_macro_f1": float(np.mean(permutation_scores)),
        "permutation_p_value": p_value,
        "first15_accuracy": accuracy_score(frame["platform"], truncated_prediction),
        "first15_macro_f1": f1_score(
            frame["platform"], truncated_prediction, average="macro"
        ),
        "first15_permutation_observed_macro_f1": truncated_observed,
        "first15_permutation_mean_macro_f1": float(np.mean(truncated_scores)),
        "first15_permutation_p_value": truncated_p,
    }


def cross_product_transfer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    results_dir: Path,
    n_permutations: int = 500,
) -> dict:
    train_id = train["matched_product_id"].iloc[0]
    test_id = test["matched_product_id"].iloc[0]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=10000,
    )
    x_train = vectorizer.fit_transform(train["review_text"])
    x_test = vectorizer.transform(test["review_text"])
    y_train = train["platform"].to_numpy()
    y_test = test["platform"].to_numpy()
    classifier = LinearSVC(class_weight="balanced", random_state=42)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    observed = f1_score(y_test, prediction, average="macro")

    rng = np.random.default_rng(42)
    permutation_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_train)
        permuted_classifier = LinearSVC(class_weight="balanced", random_state=42)
        permuted_classifier.fit(x_train, shuffled)
        permutation_scores.append(
            f1_score(y_test, permuted_classifier.predict(x_test), average="macro")
        )
    permutation_scores = np.asarray(permutation_scores)
    p_value = (1 + int(np.sum(permutation_scores >= observed))) / (n_permutations + 1)

    stem = f"train_{train_id}_test_{test_id}"
    save_confusion(results_dir / f"{stem}_confusion_matrix.csv", y_test, prediction)
    pd.DataFrame(
        {
            "review_id": test["review_id"],
            "matched_product_id": test_id,
            "platform": y_test,
            "predicted_platform": prediction,
        }
    ).to_csv(results_dir / f"{stem}_predictions.csv", index=False)
    return {
        "train_product_id": train_id,
        "train_product": PRODUCT_LABELS[train_id],
        "test_product_id": test_id,
        "test_product": PRODUCT_LABELS[test_id],
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(y_test, prediction),
        "macro_f1": observed,
        "permutation_mean_macro_f1": float(np.mean(permutation_scores)),
        "permutation_p_value": p_value,
    }


def kospet_aspect_analysis(frame: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    token_sets = frame["review_text"].map(lambda text: set(WORD_RE.findall(text.lower())))
    rows = []
    for aspect, terms in KOSPET_ASPECTS.items():
        mentioned = token_sets.map(lambda tokens: bool(tokens & terms))
        amazon = frame["platform"].eq("Amazon")
        ali = frame["platform"].eq("AliExpress")
        amazon_yes = int((mentioned & amazon).sum())
        ali_yes = int((mentioned & ali).sum())
        odds_ratio, p_value = fisher_exact(
            [
                [amazon_yes, int(amazon.sum()) - amazon_yes],
                [ali_yes, int(ali.sum()) - ali_yes],
            ],
            alternative="two-sided",
        )
        rows.append(
            {
                "aspect": aspect,
                "amazon_mentions": amazon_yes,
                "amazon_share": amazon_yes / amazon.sum(),
                "aliexpress_mentions": ali_yes,
                "aliexpress_share": ali_yes / ali.sum(),
                "odds_ratio_amazon_vs_aliexpress": odds_ratio,
                "fisher_p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    output["fdr_adjusted_p"] = bh_adjust(output["fisher_p_value"].tolist())
    output.to_csv(results_dir / "kospet_aspect_comparison.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseus-input",
        type=Path,
        default=(
            ROOT
            / "data"
            / "cross_platform"
            / "matched_product"
            / "baseus_bowie_ma10_matched_reviews.csv"
        ),
    )
    parser.add_argument(
        "--kospet-input",
        type=Path,
        default=(
            ROOT
            / "data"
            / "cross_platform"
            / "matched_product_replication"
            / "kospet_tank_t3_ultra_matched_reviews.csv"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results" / "screen_share_hybrid" / "best_hybrid_model.joblib",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "two_matched_product_replication",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseus = pd.read_csv(args.baseus_input)
    kospet = pd.read_csv(args.kospet_input)
    expected = {
        "baseus_bowie_ma10": {"Amazon": 100, "AliExpress": 100},
        "kospet_tank_t3_ultra": {"Amazon": 35, "AliExpress": 35},
    }
    for frame in [baseus, kospet]:
        product_id = frame["matched_product_id"].iloc[0]
        if frame["platform"].value_counts().to_dict() != expected[product_id]:
            raise ValueError(f"Unexpected sample structure for {product_id}")
        if not frame["rating"].eq(5).all() or not frame["language"].eq("en").all():
            raise ValueError(f"Control failure for {product_id}")

    combined = pd.concat([baseus, kospet], ignore_index=True)
    combined = add_text_statistics(combined)
    combined = add_simple_sentiment_features(combined)
    combined["product"] = combined["matched_product_id"].map(PRODUCT_LABELS)
    hybrid = joblib.load(args.model)
    combined["category"] = "Electronics"
    model_input = add_week2_features_compat(combined)
    combined["main_model_prediction"] = hybrid.predict(model_input)
    combined["main_model_score"] = prediction_score(hybrid, model_input)
    combined.to_csv(args.results_dir / "two_product_predictions.csv", index=False)

    platform_summary = (
        combined.groupby(["matched_product_id", "product", "platform"])
        .agg(
            rows=("review_id", "size"),
            avg_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            avg_sentiment_score=("sentiment_score", "mean"),
            main_model_flagged_share=("main_model_prediction", "mean"),
            avg_main_model_score=("main_model_score", "mean"),
        )
        .reset_index()
    )
    platform_summary.to_csv(args.results_dir / "two_product_platform_summary.csv", index=False)

    length_tests = []
    lexical_rows = []
    within_rows = []
    for product_id, frame in combined.groupby("matched_product_id", sort=False):
        amazon_words = frame.loc[frame["platform"].eq("Amazon"), "word_count"]
        ali_words = frame.loc[frame["platform"].eq("AliExpress"), "word_count"]
        statistic, p_value = mannwhitneyu(amazon_words, ali_words, alternative="two-sided")
        length_tests.append(
            {
                "matched_product_id": product_id,
                "product": PRODUCT_LABELS[product_id],
                "amazon_median_words": amazon_words.median(),
                "aliexpress_median_words": ali_words.median(),
                "mann_whitney_u": statistic,
                "p_value": p_value,
                "rank_biserial_amazon_minus_aliexpress": (
                    2 * statistic / (len(amazon_words) * len(ali_words)) - 1
                ),
            }
        )
        lexical_rows.append(
            {
                "matched_product_id": product_id,
                "product": PRODUCT_LABELS[product_id],
                **lexical_shift(frame),
            }
        )
        within_rows.append(within_product_classification(frame, args.results_dir))
    length_frame = pd.DataFrame(length_tests)
    length_frame["fdr_adjusted_p"] = bh_adjust(length_frame["p_value"].tolist())
    length_frame.to_csv(args.results_dir / "two_product_length_tests.csv", index=False)
    pd.DataFrame(lexical_rows).to_csv(
        args.results_dir / "two_product_lexical_shift.csv", index=False
    )
    within_frame = pd.DataFrame(within_rows)
    within_frame.to_csv(args.results_dir / "within_product_classifier_metrics.csv", index=False)

    transfer_rows = [
        cross_product_transfer(baseus, kospet, args.results_dir),
        cross_product_transfer(kospet, baseus, args.results_dir),
    ]
    transfer_frame = pd.DataFrame(transfer_rows)
    transfer_frame.to_csv(args.results_dir / "cross_product_transfer_metrics.csv", index=False)
    aspects = kospet_aspect_analysis(kospet, args.results_dir)

    summary = {
        "design": (
            "Two exact-product Amazon-AliExpress comparisons with product, rating, language and "
            "within-product sample size controlled."
        ),
        "total_rows": len(combined),
        "within_product_classification": within_frame.to_dict(orient="records"),
        "cross_product_transfer": transfer_frame.to_dict(orient="records"),
        "kospet_fdr_significant_aspects": aspects.loc[
            aspects["fdr_adjusted_p"].lt(0.05), "aspect"
        ].tolist(),
        "interpretation": (
            "Platform-source language remains detectable for a second product and transfers across "
            "the two products. The evidence does not isolate national culture or estimate fraud prevalence."
        ),
    }
    (args.results_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
