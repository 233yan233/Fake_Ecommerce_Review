from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts_compat import add_week2_features_compat


def decision_scores(model: Pipeline, data) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(data), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(data)[:, 1], dtype=float)
    return np.full(len(data), np.nan)


def metric_row(experiment: str, dataset: str, y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "experiment": experiment,
        "dataset": dataset,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
    }


def text_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    max_features=30000,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def save_evaluation(
    results_dir: Path,
    file_stem: str,
    frame: pd.DataFrame,
    prediction: np.ndarray,
    score: np.ndarray,
) -> None:
    y_true = frame["is_fake"].astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_real", "actual_fake"],
        columns=["pred_real", "pred_fake"],
    ).to_csv(results_dir / f"{file_stem}_confusion_matrix.csv")
    (results_dir / f"{file_stem}_classification_report.txt").write_text(
        classification_report(
            y_true,
            prediction,
            labels=[0, 1],
            target_names=["real", "fake"],
            digits=4,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    output = frame[["review_id", "source_group", "source_fold", "is_fake"]].copy()
    output["prediction"] = prediction
    output["decision_score"] = score
    output.to_csv(results_dir / f"{file_stem}_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--published",
        type=Path,
        default=ROOT / "data" / "external" / "derev2018" / "processed" / "derev2018_published.csv",
    )
    parser.add_argument(
        "--crowdsourced",
        type=Path,
        default=ROOT / "data" / "external" / "derev2018" / "processed" / "derev2018_crowdsourced.csv",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT / "results" / "screen_share_baseline" / "best_baseline_model.joblib",
    )
    parser.add_argument(
        "--hybrid-model",
        type=Path,
        default=ROOT / "results" / "screen_share_hybrid" / "best_hybrid_model.joblib",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "external_validation_derev2018",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "published": add_week2_features_compat(pd.read_csv(args.published)),
        "crowdsourced": add_week2_features_compat(pd.read_csv(args.crowdsourced)),
    }
    baseline = joblib.load(args.baseline_model)
    hybrid = joblib.load(args.hybrid_model)
    metrics = []

    for dataset_name, frame in datasets.items():
        y_true = frame["is_fake"].astype(int)
        evaluations = {
            "main_text_model_zero_shot": (
                baseline.predict(frame["review_text"]),
                decision_scores(baseline, frame["review_text"]),
            ),
            "main_hybrid_model_zero_shot": (
                hybrid.predict(frame),
                decision_scores(hybrid, frame),
            ),
        }
        for experiment, (prediction, score) in evaluations.items():
            metrics.append(metric_row(experiment, dataset_name, y_true, prediction))
            save_evaluation(
                args.results_dir,
                f"{experiment}_{dataset_name}",
                frame,
                np.asarray(prediction),
                score,
            )

        split = PredefinedSplit(frame["source_fold"].astype(int).to_numpy())
        prediction = cross_val_predict(
            text_model(),
            frame["review_text"],
            y_true,
            cv=split,
            n_jobs=-1,
            method="predict",
        )
        metrics.append(metric_row("derev_text_model_in_domain_oof", dataset_name, y_true, prediction))
        save_evaluation(
            args.results_dir,
            f"derev_text_model_in_domain_oof_{dataset_name}",
            frame,
            prediction,
            np.full(len(frame), np.nan),
        )

    for train_name, test_name in [("published", "crowdsourced"), ("crowdsourced", "published")]:
        train = datasets[train_name]
        test = datasets[test_name]
        model = text_model()
        model.fit(train["review_text"], train["is_fake"].astype(int))
        prediction = model.predict(test["review_text"])
        score = decision_scores(model, test["review_text"])
        experiment = f"derev_text_model_{train_name}_to_{test_name}"
        metrics.append(metric_row(experiment, test_name, test["is_fake"].astype(int), prediction))
        save_evaluation(args.results_dir, experiment, test, prediction, score)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.results_dir / "external_validation_metrics.csv", index=False)
    summary = {
        "external_dataset": "DeRev 2018",
        "purpose": "External and source-transfer validation; not additional tuning of the main model.",
        "published_rows": len(datasets["published"]),
        "crowdsourced_rows": len(datasets["crowdsourced"]),
        "headline_metrics": metrics_frame.to_dict(orient="records"),
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
