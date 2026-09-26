from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_detection.features import add_simple_sentiment_features, add_text_statistics


def add_week2_features_compat(df):
    out = df.copy()
    required_text_stats = {
        "char_length",
        "word_count",
        "exclamation_count",
        "question_count",
        "digit_count",
        "uppercase_ratio",
    }
    if not required_text_stats <= set(out.columns):
        out = add_text_statistics(out)
    out = add_simple_sentiment_features(out)
    numeric_features = [
        "rating",
        "char_length",
        "word_count",
        "exclamation_count",
        "question_count",
        "digit_count",
        "uppercase_ratio",
        "positive_word_count",
        "negative_word_count",
        "sentiment_score",
        "sentiment_rating_mismatch",
        "extreme_rating",
    ]
    for col in numeric_features:
        out[col] = out[col].fillna(0)
    out["category"] = out["category"].fillna("unknown").astype(str)
    return out
