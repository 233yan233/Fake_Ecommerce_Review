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
PRODUCT_ID = "baseus_bowie_ma10"
PRODUCT_NAME = "Baseus Bowie MA10"
AMAZON_ASIN = "B0C89QQ5XK"
ALIEXPRESS_ITEM_ID = "1005005742938199"
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
    "dura",
    "el",
    "es",
    "esta",
    "este",
    "la",
    "las",
    "los",
    "mucho",
    "muy",
    "para",
    "pero",
    "porque",
    "por",
    "que",
    "tiempo",
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


def fetch_amazon_review_pool(sort_value: int, max_reviews: int, delay: float) -> list[dict]:
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

    while len(reviews) < max_reviews:
        params = {
            "filter": "5",
            "isVerified": "false",
            "sort": str(sort_value),
        }
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
            if not text or int(review.get("OverallRating", 0)) != 5:
                continue
            key = sha256_text(
                "|".join(
                    [
                        text.lower(),
                        clean_text(review.get("Title")).lower(),
                        clean_text(review.get("OriginDescription")),
                    ]
                )
            )
            if key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= max_reviews:
                break

        paging_next = payload.get("PagingNext", "")
        if not paging_next:
            break
        time.sleep(delay)

    if len(reviews) < max_reviews:
        raise RuntimeError(
            f"Amazon endpoint returned {len(reviews)} usable reviews; {max_reviews} required."
        )
    return reviews


def select_amazon_english_reviews(max_reviews: int, delay: float) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    pool_counts: dict[str, int] = {}
    rejected_non_english = 0
    for sort_value, label in [(0, "top"), (1, "recent")]:
        pool = fetch_amazon_review_pool(sort_value, 100, delay)
        pool_counts[label] = len(pool)
        for review in pool:
            text = clean_text(review.get("Text"))
            key = sha256_text(
                "|".join(
                    [
                        text.lower(),
                        clean_text(review.get("Title")).lower(),
                        clean_text(review.get("OriginDescription")),
                    ]
                )
            )
            if key in seen:
                continue
            seen.add(key)
            if not is_english(text):
                rejected_non_english += 1
                continue
            selected.append(review)
            if len(selected) >= max_reviews:
                return selected, {
                    "source_pool_counts": pool_counts,
                    "non_english_rejected": rejected_non_english,
                    "selection_order": "top-ranked pool followed by recent-review pool",
                }
    raise RuntimeError(
        f"Amazon pools returned {len(selected)} unique English reviews; {max_reviews} required."
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
                "brand": "Baseus",
                "model": "Bowie MA10",
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
                "sample_frame": (
                    "English written five-star reviews selected from top-ranked and recent "
                    "endpoint pools."
                ),
                "notes": "Reviewer name and media URLs were not retained.",
            }
        )
    return pd.DataFrame(rows)


def standardize_aliexpress(
    source: str | Path,
    max_reviews: int,
    random_state: int,
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
    if eligible_rows < max_reviews:
        raise RuntimeError(
            f"AliExpress source contains {eligible_rows} usable reviews; {max_reviews} required."
        )
    selected = selected.sample(n=max_reviews, random_state=random_state).reset_index(drop=True)

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
                "brand": "Baseus",
                "model": "Bowie MA10",
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
                    f"Deterministic random sample of {max_reviews} from {eligible_rows} "
                    "English five-star reviews for the exact item."
                ),
                "notes": "Reviewer name was not retained.",
            }
        )
    return pd.DataFrame(rows), eligible_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aliexpress-source",
        default=ALIEXPRESS_REVIEW_SOURCE,
        help="Published AliExpress review CSV URL or local path.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--request-delay", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "cross_platform" / "matched_product",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    amazon_reviews, amazon_selection = select_amazon_english_reviews(
        args.sample_size, args.request_delay
    )
    amazon = standardize_amazon(amazon_reviews)
    aliexpress, aliexpress_eligible = standardize_aliexpress(
        args.aliexpress_source,
        args.sample_size,
        args.random_state,
    )
    corpus = pd.concat([amazon, aliexpress], ignore_index=True)
    corpus = corpus.sort_values(["platform", "review_id"]).reset_index(drop=True)
    corpus_path = args.output_dir / "baseus_bowie_ma10_matched_reviews.csv"
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
            "brand": "Baseus",
            "model": "Bowie MA10",
            "match_rule": "Exact brand and model, supported by overlapping product specifications.",
            "shared_specifications": [
                "up to 140 hours total playtime",
                "up to -48 dB active noise cancellation",
                "Bluetooth 5.3",
                "IPX6 water resistance",
            ],
            "amazon": {
                "asin": AMAZON_ASIN,
                "product_url": AMAZON_PRODUCT_URL,
                "observed_title": (
                    "Baseus Active Noise Cancelling Wireless Earbuds ... Bowie MA10"
                ),
            },
            "aliexpress": {
                "item_id": ALIEXPRESS_ITEM_ID,
                "product_url": ALIEXPRESS_PRODUCT_URL,
                "indexed_evidence_url": (
                    "https://www.pelando.com.br/d/baseus-bowie-ma10-anc-fone-de-ouvido-"
                    "sem-fio-imperme-vel-sport-earbud-48db-cancelamento-de-ru-do-140h-"
                    "playtime-bluetooth-5-3-ipx6-f457"
                ),
                "observed_title": (
                    "Baseus Bowie MA10 ANC wireless earbuds, 48 dB, 140h, Bluetooth 5.3, IPX6"
                ),
            },
        },
        "sampling": {
            "design": "Same product, English language, five-star rating and equal sample size.",
            "sample_size_per_platform": args.sample_size,
            "amazon_selection": amazon_selection,
            "aliexpress_selection": (
                f"Random-state-{args.random_state} sample from {aliexpress_eligible} eligible records."
            ),
            "date_ranges": date_ranges,
        },
        "sources": {
            "amazon_reviews": AMAZON_REVIEW_ENDPOINT,
            "aliexpress_reviews": str(args.aliexpress_source),
            "amazon_collection_reference": (
                "https://github.com/mrlong0129/amazon-review-scraper (MIT; endpoint and "
                "pagination behaviour independently reimplemented here)"
            ),
            "aliexpress_collection_documentation": (
                "https://github.com/george07-t/"
                "AliExpress-Product-Review-Reply-Management-System"
            ),
        },
        "privacy": (
            "Reviewer names, profile identifiers and media URLs are not stored. Country is retained "
            "only as a coarse marketplace-composition variable."
        ),
        "limitations": [
            "The platforms use different review retrieval and ranking mechanisms.",
            "The review date ranges are not identical.",
            "AliExpress English reviews come from multiple countries and may include translations.",
            "The comparison is a platform-market case study, not a direct test of national culture.",
        ],
        "rows": len(corpus),
        "platform_counts": corpus["platform"].value_counts().sort_index().to_dict(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "matched_product_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
