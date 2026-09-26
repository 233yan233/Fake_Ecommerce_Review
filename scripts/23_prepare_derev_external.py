from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://fornaciari.netlify.app/media/DeRev2018.zip"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def numeric_rating(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_review(xml_bytes: bytes, source_name: str) -> dict | None:
    review = ET.fromstring(xml_bytes)
    if source_name == "published":
        raw_label = review.attrib.get("gold2016")
        if raw_label not in {"0", "1"}:
            return None
    elif source_name == "crowdsourced":
        raw_label = review.attrib.get("goldTurk")
        if review.attrib.get("author") != "TURKER" or raw_label not in {"0", "1"}:
            return None
    else:
        raise ValueError(f"Unsupported source_name: {source_name}")

    text = (review.findtext("body") or "").strip()
    if not text:
        return None

    return {
        "review_id": f"derev_{source_name}_{review.attrib['ID']}",
        "review_text": text,
        "rating": numeric_rating(review.attrib.get("stars", "")),
        "category": "Books_5",
        "platform": "Amazon",
        "source_group": source_name,
        "source_fold": int(review.attrib["fold2016"]),
        "label_raw": raw_label,
        "is_fake": 1 if raw_label == "0" else 0,
        "label_name": "fake" if raw_label == "0" else "real",
    }


def read_source(archive: Path, source_name: str) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(
            name
            for name in zipped.namelist()
            if name.startswith("derev2016/ID") and name.endswith(".xml")
        )
        for name in names:
            row = parse_review(zipped.read(name), source_name)
            if row is not None:
                rows.append(row)

    frame = pd.DataFrame(rows).sort_values("review_id").reset_index(drop=True)
    if len(frame) != 1552 or frame["is_fake"].value_counts().to_dict() != {0: 776, 1: 776}:
        raise ValueError(
            f"Unexpected {source_name} label distribution: "
            f"rows={len(frame)}, labels={frame['is_fake'].value_counts().to_dict()}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data" / "external" / "derev2018" / "DeRev2018.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "external" / "derev2018" / "processed",
    )
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.force_download or not args.archive.exists():
        print(f"Downloading DeRev 2018 from {args.url}")
        download_archive(args.url, args.archive)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    published = read_source(args.archive, "published")
    crowdsourced = read_source(args.archive, "crowdsourced")
    published.to_csv(args.output_dir / "derev2018_published.csv", index=False)
    crowdsourced.to_csv(args.output_dir / "derev2018_crowdsourced.csv", index=False)

    manifest = {
        "dataset": "DeRev 2018",
        "source_url": args.url,
        "access_date": date.today().isoformat(),
        "archive_sha256": file_sha256(args.archive),
        "published_rows": len(published),
        "crowdsourced_rows": len(crowdsourced),
        "published_label_counts": published["label_name"].value_counts().to_dict(),
        "crowdsourced_label_counts": crowdsourced["label_name"].value_counts().to_dict(),
        "label_mapping": {"0": "fake", "1": "real"},
        "privacy": "Reviewer names are not exported to the processed files.",
        "citation": (
            "Fornaciari, T. et al. (2020), Fake opinion detection: how similar "
            "are crowdsourced datasets to real data?, Language Resources and Evaluation."
        ),
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
