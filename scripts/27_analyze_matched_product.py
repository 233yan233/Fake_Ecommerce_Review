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
ASPECTS = {
    "sound": {"audio", "bass", "music", "sound", "treble", "volume"},
    "noise_cancellation": {
        "anc",
        "cancel",
        "cancellation",
        "cancelling",
        "noise",
        "transparency",
    },
    "battery_charging": {"battery", "case", "charge", "charging", "hour", "power"},
    "comfort_fit": {"bulky", "comfort", "comfortable", "ear", "earbud", "fit", "heavy"},
    "connectivity": {"app", "bluetooth", "connect", "connection", "latency", "pair"},
    "value_price": {"cheap", "cost", "money", "price", "value", "worth"},
    "delivery_packaging": {"arrived", "box", "delivery", "package", "packaging", "shipping"},
    "quality_durability": {"broken", "build", "durable", "plastic", "quality", "work", "working"},
}


def prediction_score(model, data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(data), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(data)[:, 1], dtype=float)
    return np.full(len(data), np.nan)


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


def platform_feature_test(frame: pd.DataFrame, column: str) -> dict:
    amazon = frame.loc[frame["platform"].eq("Amazon"), column].dropna().astype(float)
    aliexpress = frame.loc[frame["platform"].eq("AliExpress"), column].dropna().astype(float)
    statistic, p_value = mannwhitneyu(amazon, aliexpress, alternative="two-sided")
    effect = 2 * statistic / (len(amazon) * len(aliexpress)) - 1
    return {
        "feature": column,
        "amazon_median": amazon.median(),
        "aliexpress_median": aliexpress.median(),
        "mann_whitney_u": statistic,
        "p_value": p_value,
        "rank_biserial_amazon_minus_aliexpress": effect,
    }


def lexical_analysis(frame: pd.DataFrame, results_dir: Path) -> dict:
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
        min_df=2,
        ngram_range=(1, 1),
    )
    matrix = vectorizer.fit_transform(frame["review_text"])
    terms = vectorizer.get_feature_names_out()
    platforms = ["Amazon", "AliExpress"]
    counts = {
        platform: np.asarray(matrix[frame["platform"].eq(platform).to_numpy()].sum(axis=0))
        .ravel()
        .astype(float)
        for platform in platforms
    }
    distributions = {
        platform: (values + 0.5) / (values.sum() + 0.5 * len(values))
        for platform, values in counts.items()
    }
    js_distance = float(
        jensenshannon(distributions["Amazon"], distributions["AliExpress"], base=2)
    )

    top_n = min(100, len(terms))
    top_sets = {
        platform: set(terms[np.argsort(values)[::-1][:top_n]])
        for platform, values in counts.items()
    }
    union = top_sets["Amazon"] | top_sets["AliExpress"]
    top_jaccard = len(top_sets["Amazon"] & top_sets["AliExpress"]) / len(union)

    total_amazon = counts["Amazon"].sum()
    total_aliexpress = counts["AliExpress"].sum()
    amazon_log_odds = np.log(
        (counts["Amazon"] + 0.5) / (total_amazon - counts["Amazon"] + 0.5)
    )
    aliexpress_log_odds = np.log(
        (counts["AliExpress"] + 0.5)
        / (total_aliexpress - counts["AliExpress"] + 0.5)
    )
    delta = amazon_log_odds - aliexpress_log_odds
    variance = 1 / (counts["Amazon"] + 0.5) + 1 / (counts["AliExpress"] + 0.5)
    z_score = delta / np.sqrt(variance)

    rows = []
    for platform, indices in [
        ("Amazon", np.argsort(z_score)[-25:][::-1]),
        ("AliExpress", np.argsort(z_score)[:25]),
    ]:
        for rank, index in enumerate(indices, start=1):
            rows.append(
                {
                    "platform": platform,
                    "rank": rank,
                    "term": terms[index],
                    "z_score_amazon_minus_aliexpress": z_score[index],
                    "amazon_count": int(counts["Amazon"][index]),
                    "aliexpress_count": int(counts["AliExpress"][index]),
                }
            )
    pd.DataFrame(rows).to_csv(results_dir / "distinctive_terms.csv", index=False)
    return {
        "jensen_shannon_distance": js_distance,
        "top_100_term_jaccard": top_jaccard,
        "vocabulary_size": len(terms),
    }


def aspect_analysis(frame: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    token_sets = frame["review_text"].map(lambda text: set(WORD_RE.findall(text.lower())))
    rows = []
    for aspect, terms in ASPECTS.items():
        mentioned = token_sets.map(lambda tokens: bool(tokens & terms))
        amazon_mask = frame["platform"].eq("Amazon")
        ali_mask = frame["platform"].eq("AliExpress")
        amazon_yes = int((mentioned & amazon_mask).sum())
        ali_yes = int((mentioned & ali_mask).sum())
        amazon_total = int(amazon_mask.sum())
        ali_total = int(ali_mask.sum())
        odds_ratio, p_value = fisher_exact(
            [
                [amazon_yes, amazon_total - amazon_yes],
                [ali_yes, ali_total - ali_yes],
            ],
            alternative="two-sided",
        )
        rows.append(
            {
                "aspect": aspect,
                "amazon_mentions": amazon_yes,
                "amazon_share": amazon_yes / amazon_total,
                "aliexpress_mentions": ali_yes,
                "aliexpress_share": ali_yes / ali_total,
                "odds_ratio_amazon_vs_aliexpress": odds_ratio,
                "fisher_p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    output["fdr_adjusted_p"] = bh_adjust(output["fisher_p_value"].tolist())
    output.to_csv(results_dir / "aspect_comparison.csv", index=False)
    return output


def source_classifier(frame: pd.DataFrame, results_dir: Path) -> dict:
    model = Pipeline(
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
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prediction = cross_val_predict(
        model,
        frame["review_text"],
        frame["platform"],
        cv=splitter,
        n_jobs=-1,
    )
    labels = ["AliExpress", "Amazon"]
    matrix = confusion_matrix(frame["platform"], prediction, labels=labels)
    pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(results_dir / "source_classifier_confusion_matrix.csv")
    pd.DataFrame(
        {
            "review_id": frame["review_id"],
            "platform": frame["platform"],
            "predicted_platform": prediction,
        }
    ).to_csv(results_dir / "source_classifier_predictions.csv", index=False)

    observed_score, permutation_scores, p_value = permutation_test_score(
        model,
        frame["review_text"],
        frame["platform"],
        scoring="f1_macro",
        cv=splitter,
        n_permutations=500,
        n_jobs=-1,
        random_state=42,
    )
    truncated_text = frame["review_text"].map(
        lambda value: " ".join(str(value).split()[:15])
    )
    truncated_prediction = cross_val_predict(
        model,
        truncated_text,
        frame["platform"],
        cv=splitter,
        n_jobs=-1,
    )
    truncated_matrix = confusion_matrix(
        frame["platform"], truncated_prediction, labels=labels
    )
    pd.DataFrame(
        truncated_matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(results_dir / "source_classifier_15_word_confusion_matrix.csv")
    pd.DataFrame(
        {
            "review_id": frame["review_id"],
            "platform": frame["platform"],
            "predicted_platform": truncated_prediction,
        }
    ).to_csv(results_dir / "source_classifier_15_word_predictions.csv", index=False)
    truncated_score, truncated_permutations, truncated_p_value = permutation_test_score(
        model,
        truncated_text,
        frame["platform"],
        scoring="f1_macro",
        cv=splitter,
        n_permutations=500,
        n_jobs=-1,
        random_state=42,
    )
    return {
        "rows": len(frame),
        "folds": 5,
        "accuracy": accuracy_score(frame["platform"], prediction),
        "macro_f1": f1_score(frame["platform"], prediction, average="macro"),
        "permutation_observed_macro_f1": observed_score,
        "permutation_mean_macro_f1": float(np.mean(permutation_scores)),
        "permutation_p_value": p_value,
        "fifteen_word_accuracy": accuracy_score(
            frame["platform"], truncated_prediction
        ),
        "fifteen_word_macro_f1": f1_score(
            frame["platform"], truncated_prediction, average="macro"
        ),
        "fifteen_word_permutation_observed_macro_f1": truncated_score,
        "fifteen_word_permutation_mean_macro_f1": float(
            np.mean(truncated_permutations)
        ),
        "fifteen_word_permutation_p_value": truncated_p_value,
        "features": "review text only",
        "sensitivity_test": "first 15 whitespace-delimited words per review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
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
        "--model",
        type=Path,
        default=ROOT / "results" / "screen_share_hybrid" / "best_hybrid_model.joblib",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "matched_product_platform_comparison",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    expected = {"Amazon": 100, "AliExpress": 100}
    counts = frame["platform"].value_counts().to_dict()
    if counts != expected or frame["matched_product_id"].nunique() != 1:
        raise ValueError(f"Unexpected matched corpus structure: {counts}")
    if not frame["rating"].eq(5).all() or not frame["language"].eq("en").all():
        raise ValueError("Matched comparison must contain English five-star reviews only.")

    frame = add_text_statistics(frame)
    frame = add_simple_sentiment_features(frame)
    hybrid = joblib.load(args.model)
    frame["category"] = "Electronics"
    model_input = add_week2_features_compat(frame)
    frame["suspicious_prediction"] = hybrid.predict(model_input)
    frame["prediction_score"] = prediction_score(hybrid, model_input)
    frame.to_csv(args.results_dir / "matched_product_predictions.csv", index=False)

    platform_summary = (
        frame.groupby("platform")
        .agg(
            rows=("review_id", "size"),
            avg_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            avg_char_length=("char_length", "mean"),
            exclamation_share=("exclamation_count", lambda values: values.gt(0).mean()),
            avg_sentiment_score=("sentiment_score", "mean"),
            suspicious_prediction_rate=("suspicious_prediction", "mean"),
            avg_prediction_score=("prediction_score", "mean"),
            reviewer_countries=("reviewer_country", "nunique"),
        )
        .reset_index()
        .sort_values("platform")
    )
    platform_summary.to_csv(args.results_dir / "platform_summary.csv", index=False)

    feature_tests = pd.DataFrame(
        [
            platform_feature_test(frame, "word_count"),
            platform_feature_test(frame, "sentiment_score"),
            platform_feature_test(frame, "prediction_score"),
        ]
    )
    feature_tests["fdr_adjusted_p"] = bh_adjust(feature_tests["p_value"].tolist())
    feature_tests.to_csv(args.results_dir / "feature_tests.csv", index=False)

    lexical = lexical_analysis(frame, args.results_dir)
    pd.DataFrame([lexical]).to_csv(args.results_dir / "lexical_shift.csv", index=False)
    aspects = aspect_analysis(frame, args.results_dir)
    classifier = source_classifier(frame, args.results_dir)
    pd.DataFrame([classifier]).to_csv(
        args.results_dir / "source_classifier_metrics.csv", index=False
    )

    summary = {
        "design": (
            "Exact same product, English language, five-star rating and 100 reviews per platform."
        ),
        "matched_product": "Baseus Bowie MA10",
        "platform_summary": platform_summary.to_dict(orient="records"),
        "lexical_shift": lexical,
        "source_classifier": classifier,
        "fdr_significant_aspects": aspects.loc[
            aspects["fdr_adjusted_p"].lt(0.05), "aspect"
        ].tolist(),
        "interpretation": (
            "The analysis tests platform-market language shift after controlling product, rating, "
            "language and sample size. It does not isolate national culture or estimate fraud prevalence."
        ),
    }
    (args.results_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
