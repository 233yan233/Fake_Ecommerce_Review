from __future__ import annotations

import re

import pandas as pd


WORD_RE = re.compile(r"\b\w+\b")
POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "comfortable",
    "durable",
    "easy",
    "excellent",
    "fast",
    "good",
    "great",
    "happy",
    "helpful",
    "love",
    "nice",
    "perfect",
    "quality",
    "recommend",
    "satisfied",
    "useful",
    "wonderful",
}
NEGATIVE_WORDS = {
    "awful",
    "bad",
    "broken",
    "cheap",
    "damaged",
    "defective",
    "disappointed",
    "fail",
    "fake",
    "hate",
    "poor",
    "refund",
    "returned",
    "slow",
    "terrible",
    "useless",
    "waste",
    "weak",
    "worst",
    "wrong",
}


def add_text_statistics(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    out = df.copy()
    text = out[text_col].fillna("").astype(str)

    out["char_length"] = text.str.len()
    out["word_count"] = text.map(lambda value: len(WORD_RE.findall(value)))
    out["exclamation_count"] = text.str.count("!")
    out["question_count"] = text.str.count(r"\?")
    out["digit_count"] = text.str.count(r"\d")
    out["uppercase_ratio"] = text.map(_uppercase_ratio)
    return out


def add_simple_sentiment_features(
    df: pd.DataFrame,
    text_col: str = "review_text",
    rating_col: str = "rating",
) -> pd.DataFrame:
    out = df.copy()
    text = out[text_col].fillna("").astype(str)
    tokens = text.map(lambda value: [token.lower() for token in WORD_RE.findall(value)])

    out["positive_word_count"] = tokens.map(lambda words: sum(word in POSITIVE_WORDS for word in words))
    out["negative_word_count"] = tokens.map(lambda words: sum(word in NEGATIVE_WORDS for word in words))
    out["sentiment_score"] = (
        (out["positive_word_count"] - out["negative_word_count"])
        / out["word_count"].clip(lower=1)
    )

    if rating_col in out.columns:
        rating = pd.to_numeric(out[rating_col], errors="coerce")
        positive_text_low_rating = (out["sentiment_score"] > 0.03) & (rating <= 2)
        negative_text_high_rating = (out["sentiment_score"] < -0.03) & (rating >= 4)
        out["sentiment_rating_mismatch"] = (positive_text_low_rating | negative_text_high_rating).astype(int)
        out["extreme_rating"] = rating.isin([1, 5]).astype(int)
    else:
        out["sentiment_rating_mismatch"] = 0
        out["extreme_rating"] = 0

    return out


def _uppercase_ratio(value: str) -> float:
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(char.isupper() for char in letters)
    return uppercase / len(letters)
