"""
Data Ingestion Pipeline — Stream-read JSONL, deduplicate, chunk, embed, insert to Supabase.

Processes in incremental mega-batches (embed + insert) so progress is saved continuously.
Supports resuming — skips products already in Supabase.
"""
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Set

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm

from config.config_loader import load_config

load_dotenv(override=True)

# ── Tokenizer for chunk size enforcement ──
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_by_sentences(text: str, max_tokens: int, overlap: int = 2) -> List[str]:
    """Split long text into sentence-based chunks with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]

    chunks = []
    step = max(1, 20 - overlap)
    for i in range(0, len(sentences), step):
        chunk_sents = sentences[i : i + 20]
        chunk = " ".join(chunk_sents)
        while count_tokens(chunk) > max_tokens and len(chunk_sents) > 1:
            chunk_sents.pop()
            chunk = " ".join(chunk_sents)
        if count_tokens(chunk) >= 10:
            chunks.append(chunk)

    return chunks if chunks else [text]


class DataIngestion:
    def __init__(self):
        logging.info("Initializing DataIngestion pipeline")
        self.config = load_config()
        self.ingestion_cfg = self.config.get("ingestion", {})

        self.supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )

        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embed_model = self.config["embedding_model"]["model"]
        self.table = self.config["supabase"]["table_name"]

    def _resolve_path(self, configured_path: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        path = (
            configured_path
            if os.path.isabs(configured_path)
            else os.path.join(project_root, configured_path)
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    # ── Resume support ──

    def get_ingested_doc_ids(self) -> Set[str]:
        """Fetch distinct doc_ids already in Supabase to skip on resume."""
        logging.info("Checking Supabase for already-ingested products...")
        ingested: Set[str] = set()
        try:
            offset = 0
            page_size = 1000
            while True:
                resp = (
                    self.supabase.table(self.table)
                    .select("doc_id")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                rows = resp.data or []
                if not rows:
                    break
                for row in rows:
                    ingested.add(row["doc_id"])
                offset += page_size
                if len(rows) < page_size:
                    break
            logging.info(f"Found {len(ingested):,} already-ingested doc_ids")
        except Exception as exc:
            logging.warning(f"Could not check existing data (table may be empty): {exc}")
        return ingested

    # ── Phase 1: Stream-read and deduplicate ──

    def load_and_deduplicate(self, skip_doc_ids: Set[str] = None) -> List[dict]:
        """Stream-read JSONL, group by product_id, keep top N reviews per product."""
        jsonl_path = self._resolve_path(self.config["data"]["jsonl_path"])
        max_per_product = int(self.ingestion_cfg.get("max_reviews_per_product", 10))
        min_length = int(self.ingestion_cfg.get("min_review_length", 50))
        min_helpful = int(self.ingestion_cfg.get("min_helpful_votes", 0))
        skip_doc_ids = skip_doc_ids or set()

        logging.info(f"Reading {jsonl_path}")
        logging.info(
            f"Dedup config: max_per_product={max_per_product}, "
            f"min_length={min_length}, min_helpful_votes={min_helpful}"
        )
        if skip_doc_ids:
            logging.info(f"Resuming — skipping {len(skip_doc_ids):,} already-ingested products")

        product_reviews: Dict[str, List[dict]] = defaultdict(list)
        total_read = 0
        skipped_short = 0
        skipped_helpful = 0
        skipped_existing = 0

        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_read += 1
                if total_read % 500_000 == 0:
                    logging.info(f"  ...read {total_read:,} lines")

                product_id = row.get("product_id") or row.get("asin")
                if not product_id:
                    continue

                if product_id in skip_doc_ids:
                    skipped_existing += 1
                    continue

                review_text = (row.get("text") or "").strip()
                if len(review_text) < min_length:
                    skipped_short += 1
                    continue

                helpful = row.get("helpful_vote", 0) or 0
                if helpful < min_helpful:
                    skipped_helpful += 1
                    continue

                product_reviews[product_id].append(row)

        logging.info(
            f"Read {total_read:,} total lines. "
            f"Skipped: {skipped_short:,} short, {skipped_helpful:,} low-helpful, "
            f"{skipped_existing:,} already-ingested. "
            f"Unique new products: {len(product_reviews):,}"
        )

        deduplicated = []
        for product_id, reviews in product_reviews.items():
            reviews.sort(key=lambda r: r.get("helpful_vote", 0) or 0, reverse=True)
            deduplicated.extend(reviews[:max_per_product])

        logging.info(
            f"After dedup: {len(deduplicated):,} reviews "
            f"(from {len(product_reviews):,} products)"
        )
        return deduplicated

    # ── Phase 2: Transform reviews into chunks ──

    def transform(self, reviews: List[dict]) -> List[dict]:
        """Convert reviews into chunks ready for embedding."""
        max_tokens = int(self.ingestion_cfg.get("max_tokens", 1300))
        min_tokens = int(self.ingestion_cfg.get("min_tokens", 50))

        chunks = []
        for row in reviews:
            title = (row.get("title") or "").strip()
            review_text = (row.get("text") or "").strip()
            content = f"{title}\n\n{review_text}".strip() if title else review_text

            if not content:
                continue

            product_id = row.get("product_id") or row.get("asin") or ""
            metadata = {
                k: v
                for k, v in {
                    "product_name": row.get("product_name"),
                    "rating": row.get("rating"),
                    "avg_rating": row.get("avg_rating"),
                    "category": row.get("category"),
                    "store": row.get("store"),
                    "price": row.get("price"),
                    "verified_purchase": row.get("verified_purchase"),
                }.items()
                if v is not None
            }

            if count_tokens(content) > max_tokens:
                text_chunks = split_by_sentences(content, max_tokens)
            else:
                text_chunks = [content]

            for idx, chunk_text in enumerate(text_chunks):
                if count_tokens(chunk_text) < min_tokens:
                    continue
                chunks.append({
                    "doc_id": product_id,
                    "chunk_index": idx,
                    "content": chunk_text,
                    "metadata": metadata,
                })

        logging.info(f"Transform complete: {len(chunks):,} chunks from {len(reviews):,} reviews")
        return chunks

    # ── Helpers: embed and insert with retries ──

    def _embed_batch(self, texts: List[str]) -> List[list]:
        """Embed a list of texts with retry + rate limit handling."""
        attempt = 0
        while True:
            try:
                resp = self.openai.embeddings.create(model=self.embed_model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as exc:
                attempt += 1
                if attempt > 10:
                    raise
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = min(30 * attempt, 120)
                else:
                    wait = min(2**attempt, 30)
                logging.warning(f"Embed failed (attempt {attempt}): {exc}. Retrying in {wait}s")
                time.sleep(wait)

    def _insert_batch(self, rows: List[dict]):
        """Insert rows into Supabase with retry."""
        attempt = 0
        while True:
            try:
                self.supabase.table(self.table).insert(rows).execute()
                return
            except Exception as exc:
                attempt += 1
                if attempt > 5:
                    raise
                wait = min(2**attempt, 30)
                logging.warning(f"Insert failed (attempt {attempt}): {exc}. Retrying in {wait}s")
                time.sleep(wait)

    # ── Phase 3+4: Embed and insert in mega-batches ──

    def process_incremental(self, chunks: List[dict], mega_batch_size: int = 2000):
        """Embed + insert in mega-batches of 2000. Progress saved after each batch."""
        embed_batch_size = int(self.ingestion_cfg.get("batch_embed", 100))
        insert_batch_size = int(self.ingestion_cfg.get("batch_insert", 200))
        total = len(chunks)
        total_inserted = 0

        logging.info(
            f"Processing {total:,} chunks in mega-batches of {mega_batch_size}"
        )

        for mega_start in range(0, total, mega_batch_size):
            mega_end = min(mega_start + mega_batch_size, total)
            mega_batch = chunks[mega_start:mega_end]
            mega_num = (mega_start // mega_batch_size) + 1
            total_megas = (total + mega_batch_size - 1) // mega_batch_size

            logging.info(
                f"-- Mega-batch {mega_num}/{total_megas} "
                f"({len(mega_batch):,} chunks) --"
            )

            # Step A: Embed this mega-batch
            texts = [c["content"] for c in mega_batch]
            all_embeddings = []
            for i in tqdm(
                range(0, len(texts), embed_batch_size),
                desc=f"Embed [{mega_num}/{total_megas}]",
                leave=False,
            ):
                batch = texts[i : i + embed_batch_size]
                embeddings = self._embed_batch(batch)
                all_embeddings.extend(embeddings)
                time.sleep(1)

            for i, chunk in enumerate(mega_batch):
                chunk["embedding"] = all_embeddings[i]

            # Step B: Insert this mega-batch into Supabase
            rows = [
                {
                    "doc_id": c["doc_id"],
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "metadata": c["metadata"],
                    "embedding": c["embedding"],
                }
                for c in mega_batch
            ]
            for i in range(0, len(rows), insert_batch_size):
                self._insert_batch(rows[i : i + insert_batch_size])

            total_inserted += len(mega_batch)

            # Free memory
            for c in mega_batch:
                c.pop("embedding", None)

            logging.info(
                f"Mega-batch {mega_num}/{total_megas} saved — "
                f"{total_inserted:,}/{total:,} total ({100 * total_inserted / total:.1f}%)"
            )

        logging.info(f"All done: {total_inserted:,} chunks embedded and inserted")

    # ── Full pipeline ──

    def run(self):
        """Run: check existing → load → dedup → transform → embed+insert incrementally."""
        logging.info("=" * 60)
        logging.info("STARTING DATA INGESTION PIPELINE")
        logging.info("=" * 60)

        existing_ids = self.get_ingested_doc_ids()
        reviews = self.load_and_deduplicate(skip_doc_ids=existing_ids)

        if not reviews:
            logging.info("No new reviews to ingest. Pipeline complete.")
            return

        chunks = self.transform(reviews)
        self.process_incremental(chunks)

        logging.info("=" * 60)
        logging.info("PIPELINE COMPLETE")
        logging.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    pipeline = DataIngestion()
    pipeline.run()
