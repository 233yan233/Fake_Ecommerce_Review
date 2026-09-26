from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ID = "kospet_tank_t3_ultra"
PRODUCT_NAME = "KOSPET TANK T3 Ultra"
AMAZON_ASIN = "B0CQXB54JC"
ALIEXPRESS_ITEM_ID = "1005006851800028"
AMAZON_PRODUCT_URL = f"https://www.amazon.com/dp/{AMAZON_ASIN}"
ALIEXPRESS_PRODUCT_URL = f"https://www.aliexpress.com/item/{ALIEXPRESS_ITEM_ID}.html"
AMAZON_REVIEW_ENDPOINT = f"https://www.woot.com/review/Reviews/{AMAZON_ASIN}"
ALIEXPRESS_REVIEW_SOURCE = (
    "https://raw.githubusercontent.com/george07-t/"
    "AliExpress-Product-Review-Reply-Management-System/main/"
    "1.%20Dataset%20Collection/Product%20Review%20Of%20AliExpress.csv"
)
DetectorFactory.seed = 42
SPANISH_MARKERS = {
    "bateria",
    "con",
    "el",
    "es",
    "esta",
    "la",
    "las",
    "los",
    "muy",
    "para",
    "pero",
    "por",
    "que",
    "reloj",
    "una",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean_text(value: object) -> str:
    text = html.unescape(html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def parse_amazon_date(origin: str) -> str:
    match = re.search(r"on ([A-Z][a-z]+ \d{1,2}, \d{4})", origin)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
    except ValueError:
        return ""


def parse_amazon_country(origin: str) -> str:
    match = re.search(r"Reviewed in (?:the )?(.+?) on ", origin)
    return match.group(1).strip() if match else "unknown"


def is_english(value: str) -> bool:
    words = set(re.findall(r"\b[a-záéíóúñü]+\b", value.lower()))
    if len(words & SPANISH_MARKERS) >= 2:
        return False
    try:
        return detect(value) == "en"
    except LangDetectException:
        return False


def review_key(review: dict) -> str:
    return sha256_text(
        "|".join(
            [
                clean_text(review.get("Text")).lower(),
                clean_text(review.get("Title")).lower(),
                clean_text(review.get("OriginDescription")),
            ]
        )
    )


def fetch_amazon_pool(sort_value: int, delay: float) -> list[dict]:
    reviews: list[dict] = []
    seen: set[str] = set()
    paging_next = ""
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://www.woot.com/review/{AMAZON_ASIN}",
    }

    while True:
        params = {"filter": "5", "isVerified": "false", "sort": str(sort_value)}
        if paging_next:
            params["pagingNext"] = paging_next
        else:
            params["page"] = "1"
        request = urllib.request.Request(
            AMAZON_REVIEW_ENDPOINT + "?" + urllib.parse.urlencode(params),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        batch = payload.get("Reviews", [])
        if not batch:
            break
        for review in batch:
            text = clean_text(review.get("Text"))
            key = review_key(review)
            if text and int(review.get("OverallRating", 0)) == 5 and key not in seen:
                seen.add(key)
                reviews.append(review)

        paging_next = payload.get("PagingNext", "")
        if not paging_next:
            break
        time.sleep(delay)
    return reviews


def select_amazon_reviews(sample_size: int, delay: float) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    pool_counts: dict[str, int] = {}
    rejected_non_english = 0
    for sort_value, label in [(0, "top"), (1, "recent")]:
        pool = fetch_amazon_pool(sort_value, delay)
        pool_counts[label] = len(pool)
        for review in pool:
            key = review_key(review)
            if key in seen:
                continue
            seen.add(key)
            if not is_english(clean_text(review.get("Text"))):
                rejected_non_english += 1
                continue
            selected.append(review)
            if len(selected) >= sample_size:
                return selected, {
                    "source_pool_counts": pool_counts,
                    "non_english_rejected": rejected_non_english,
                    "selection_order": "top-ranked pool followed by recent-review pool",
                }
    raise RuntimeError(
        f"Amazon pools returned {len(selected)} unique English reviews; {sample_size} required."
    )


def standardize_amazon(reviews: list[dict]) -> pd.DataFrame:
    rows = []
    for review in reviews:
        text = clean_text(review.get("Text"))
        origin = clean_text(review.get("OriginDescription"))
        review_date = parse_amazon_date(origin)
        rows.append(
            {
                "review_id": "amazon_" + sha256_text(text + review_date)[:16],
                "matched_product_id": PRODUCT_ID,
                "brand": "KOSPET",
                "model": "TANK T3 Ultra",
                "platform": "Amazon",
                "marketplace_context": "Amazon US",
                "source_item_id": AMAZON_ASIN,
                "product_url": AMAZON_PRODUCT_URL,
                "review_text": text,
                "review_title": clean_text(review.get("Title")),
                "rating": 5.0,
                "review_date": review_date,
                "reviewer_country": parse_amazon_country(origin),
                "verified_purchase": bool(review.get("IsVerifiedPurchase")),
                "helpful_votes": int(review.get("HelpfulVotes") or 0),
                "language": "en",
                "collection_method": "public_woot_review_endpoint",
                "source_dataset": AMAZON_REVIEW_ENDPOINT,
                "sample_frame": "English written five-star reviews from the top-ranked endpoint pool.",
                "notes": "Reviewer name and media URLs were not retained.",
            }
        )
    return pd.DataFrame(rows)


def standardize_aliexpress(
    source: str | Path, sample_size: int, random_state: int
) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(source)
    item_id = raw["productURL"].fillna("").astype(str).str.extract(r"/item/(\d+)\.html")[0]
    selected = raw[
        item_id.eq(ALIEXPRESS_ITEM_ID)
        & raw["language"].fillna("").astype(str).str.lower().eq("en")
        & raw["reviewContent"].fillna("").astype(str).str.strip().ne("")
        & pd.to_numeric(raw["userStar"], errors="coerce").ge(5.0)
    ].copy()
    selected["review_text_clean"] = selected["reviewContent"].map(clean_text)
    selected = selected.drop_duplicates(subset=["review_text_clean", "reviewTime"])
    eligible_rows = len(selected)
    if eligible_rows < sample_size:
        raise RuntimeError(
            f"AliExpress source contains {eligible_rows} usable reviews; {sample_size} required."
        )
    selected = selected.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    rows = []
    for review in selected.to_dict(orient="records"):
        text = review["review_text_clean"]
        parsed_date = pd.to_datetime(
            review.get("reviewTime"), format="%d %b %Y %H:%M", errors="coerce"
        )
        review_date = "" if pd.isna(parsed_date) else parsed_date.date().isoformat()
        rows.append(
            {
                "review_id": "aliexpress_" + sha256_text(text + review_date)[:16],
                "matched_product_id": PRODUCT_ID,
                "brand": "KOSPET",
                "model": "TANK T3 Ultra",
                "platform": "AliExpress",
                "marketplace_context": "AliExpress global English-language reviews",
                "source_item_id": ALIEXPRESS_ITEM_ID,
                "product_url": ALIEXPRESS_PRODUCT_URL,
                "review_text": text,
                "review_title": "",
                "rating": 5.0,
                "review_date": review_date,
                "reviewer_country": clean_text(review.get("userCountry")) or "unknown",
                "verified_purchase": "unknown",
                "helpful_votes": "",
                "language": "en",
                "collection_method": "published_web_scraped_corpus",
                "source_dataset": ALIEXPRESS_REVIEW_SOURCE,
                "sample_frame": (
                    f"Deterministic random sample of {sample_size} from {eligible_rows} "
                    "English five-star reviews for the exact item."
                ),
                "notes": "Reviewer name was not retained.",
            }
        )
    return pd.DataFrame(rows), eligible_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aliexpress-source", default=ALIEXPRESS_REVIEW_SOURCE)
    parser.add_argument("--sample-size", type=int, default=35)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "cross_platform" / "matched_product_replication",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    amazon_reviews, amazon_selection = select_amazon_reviews(
        args.sample_size, args.request_delay
    )
    amazon = standardize_amazon(amazon_reviews)
    aliexpress, aliexpress_eligible = standardize_aliexpress(
        args.aliexpress_source, args.sample_size, args.random_state
    )
    corpus = pd.concat([amazon, aliexpress], ignore_index=True)
    corpus = corpus.sort_values(["platform", "review_id"]).reset_index(drop=True)
    corpus_path = args.output_dir / "kospet_tank_t3_ultra_matched_reviews.csv"
    corpus.to_csv(corpus_path, index=False)

    date_ranges = {}
    for platform, group in corpus.groupby("platform"):
        dates = pd.to_datetime(group["review_date"], errors="coerce").dropna()
        date_ranges[platform] = {
            "earliest": dates.min().date().isoformat() if not dates.empty else None,
            "latest": dates.max().date().isoformat() if not dates.empty else None,
        }

    manifest = {
        "created_on": date.today().isoformat(),
        "matched_product": {
            "matched_product_id": PRODUCT_ID,
            "brand": "KOSPET",
            "model": "TANK T3 Ultra",
            "match_rule": "Exact brand and model; the Amazon comparison uses the black variant.",
            "shared_specifications": [
                "1.43-inch AMOLED display",
                "470 mAh battery",
                "built-in GPS, altitude and compass functions",
                "5 ATM water resistance",
                "stainless-steel body",
            ],
            "amazon": {
                "asin": AMAZON_ASIN,
                "product_url": AMAZON_PRODUCT_URL,
                "indexed_evidence_url": "https://device.report/kospet/t3-ultra-black",
                "observed_title": "KOSPET TANK T3 Ultra Smart Watch, black",
            },
            "aliexpress": {
                "item_id": ALIEXPRESS_ITEM_ID,
                "product_url": ALIEXPRESS_PRODUCT_URL,
                "indexed_evidence_url": (
                    "https://www.pricearchive.org/aliexpress.com/item/1005006851800028"
                ),
                "observed_title": "KOSPET TANK T3 Ultra GPS smartwatch",
            },
        },
        "sampling": {
            "design": "Same product, English language, five-star rating and equal sample size.",
            "sample_size_per_platform": args.sample_size,
            "reason_for_sample_size": (
                "Thirty-five is the largest round sample supported by the "
                "38 unique English five-star Amazon texts available at collection time."
            ),
            "amazon_selection": amazon_selection,
            "aliexpress_selection": (
                f"Random-state-{args.random_state} sample from {aliexpress_eligible} eligible records."
            ),
            "date_ranges": date_ranges,
        },
        "sources": {
            "amazon_reviews": AMAZON_REVIEW_ENDPOINT,
            "aliexpress_reviews": str(args.aliexpress_source),
            "amazon_product_evidence": "https://device.report/kospet/t3-ultra-black",
            "aliexpress_product_evidence": (
                "https://www.pricearchive.org/aliexpress.com/item/1005006851800028"
            ),
        },
        "privacy": (
            "Reviewer names, profile identifiers and media URLs are not stored. Country is retained "
            "only as a coarse marketplace-composition variable."
        ),
        "limitations": [
            "The replication sample is smaller than the Bowie MA10 sample.",
            "The platforms use different review retrieval and ranking mechanisms.",
            "The review date ranges are not identical.",
            "AliExpress English reviews may contain machine-translated text.",
            "The comparison is a platform-market case study, not a direct test of national culture.",
        ],
        "rows": len(corpus),
        "platform_counts": corpus["platform"].value_counts().sort_index().to_dict(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "matched_product_replication_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
