import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_jsonl(file_path: Path) -> Iterable[dict]:
    """Yield dicts from a JSONL file, skipping malformed lines gracefully."""
    with file_path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                print(
                    f"Warning: Skipping malformed JSON on line {line_num} in {file_path.name}: {err}"
                )


def load_metadata(metadata_jsonl: Path) -> Dict[str, dict]:
    """Load metadata keyed by parent_asin (or asin if parent is missing)."""
    product_id_to_meta: Dict[str, dict] = {}
    for meta in read_jsonl(metadata_jsonl):
        asin: Optional[str] = meta.get("parent_asin") or meta.get("asin")
        if not asin:
            continue
        product_id_to_meta[asin] = meta
    print(
        f"Loaded metadata for {len(product_id_to_meta)} products from {metadata_jsonl}"
    )
    return product_id_to_meta


def build_merged_record(review: dict, meta: dict) -> dict:
    """Project the merged review + metadata into the required schema."""
    description_field = meta.get("description", [])
    if isinstance(description_field, list):
        product_description = " ".join([str(x) for x in description_field])
    else:
        product_description = str(description_field or "")

    return {
        "product_id": review.get("asin") or meta.get("parent_asin") or meta.get("asin"),
        "product_name": meta.get("title", ""),
        "product_description": product_description,
        "user_id": review.get("user_id", ""),
        "text": review.get("text", ""),
        "title": review.get("title", ""),
        "rating": review.get("rating", 0),
        "avg_rating": meta.get("average_rating", 0),
        "rating_count": meta.get("rating_number", 0),
        # Extras
        "category": meta.get("main_category", ""),
        "store": meta.get("store", ""),
        "price": meta.get("price"),
        "verified_purchase": review.get("verified_purchase", False),
        "helpful_vote": review.get("helpful_vote", 0),
        "timestamp": review.get("timestamp", 0),
    }


from typing import Union, Optional, List
from pathlib import Path


def merge_reviews_with_metadata(
    reviews_jsonl: Union[Path, str],
    metadata_jsonl: Union[Path, str],
    output_json: Optional[Path],
    output_jsonl: Optional[Path],
    sample_limit: Optional[int] = None,
) -> List[dict]:
    print("Loading metadata...")
    meta_index = load_metadata(metadata_jsonl)
    print(f"Loaded metadata for {len(meta_index)} products")

    merged_records: List[dict] = []
    jsonl_fp = None
    processed_count = 0
    skipped_count = 0

    try:
        if output_jsonl:
            jsonl_fp = output_jsonl.open("w", encoding="utf-8")

        print("Processing reviews...")
        for i, review in enumerate(read_jsonl(reviews_jsonl)):
            if i % 10000 == 0 and i > 0:
                print(
                    f"Processed {i} reviews, merged {processed_count}, skipped {skipped_count}"
                )

            asin = review.get("asin")
            if not asin:
                skipped_count += 1
                continue
            meta = meta_index.get(asin)
            if not meta:
                skipped_count += 1
                continue

            record = build_merged_record(review, meta)

            # Stream to JSONL to avoid high memory usage
            if jsonl_fp:
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

            merged_records.append(record)
            processed_count += 1

            if sample_limit is not None and len(merged_records) >= sample_limit:
                print(f"Reached sample limit of {sample_limit}")
                break

    finally:
        if jsonl_fp:
            jsonl_fp.close()

    print(f"Final counts: processed {processed_count}, skipped {skipped_count}")

    # Write JSON array if requested
    if output_json:
        print(f"Writing JSON file to {output_json}")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(merged_records, f, indent=2, ensure_ascii=False)

    return merged_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Amazon metadata and reviews into a unified JSON/JSONL dataset."
    )
    parser.add_argument(
        "--reviews",
        default=str(Path("data") / "Electronics.jsonl"),
        help="Path to reviews JSONL file",
    )
    parser.add_argument(
        "--metadata",
        default=str(Path("data") / "meta_Electronics.jsonl"),
        help="Path to metadata JSONL file",
    )
    parser.add_argument(
        "--out_json",
        default=str(Path("data") / "merged_electronics_data.json"),
        help="Output JSON file (array of objects)",
    )
    parser.add_argument(
        "--out_jsonl",
        default=str(Path("data") / "merged_electronics_data.jsonl"),
        help="Output JSONL file (one object per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of merged records for sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reviews_path = Path(args.reviews)
    metadata_path = Path(args.metadata)
    out_json_path = Path(args.out_json) if args.out_json else None
    out_jsonl_path = Path(args.out_jsonl) if args.out_jsonl else None

    # Ensure output directory exists
    for out in [out_json_path, out_jsonl_path]:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_reviews_with_metadata(
        reviews_jsonl=reviews_path,
        metadata_jsonl=metadata_path,
        output_json=out_json_path,
        output_jsonl=out_jsonl_path,
        sample_limit=args.limit,
    )

    print(
        f"Merged {len(merged)} records. "
        f"JSON: {out_json_path if out_json_path else 'skipped'}, "
        f"JSONL: {out_jsonl_path if out_jsonl_path else 'skipped'}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# python data/data.py --reviews data/Electronics.jsonl --metadata data/meta_Electronics.jsonl --out_json data/merged_electronics_data.json --out_jsonl data/merged_electronics_data.jsonl
