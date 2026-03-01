"""
Test harness — Verify retrieval quality against Supabase vector store.
Run: python test_embeddings.py
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

from config.config_loader import load_config

load_dotenv(override=True)

config = load_config()
EMBED_MODEL = config["embedding_model"]["model"]
TOP_K = config["retriever"].get("top_k", 5)
TABLE = config["supabase"]["table_name"]

# Test queries for Amazon electronics
QUERIES = [
    "Best budget laptops for students under $500",
    "Compare noise cancelling headphones",
    "Most reliable external hard drives",
    "Good cameras for beginners",
    "Top rated wireless earbuds",
    "Smart home devices with best reviews",
    "Best gaming monitors under $300",
    "Portable Bluetooth speakers with long battery life",
]


def main():
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    for query in QUERIES:
        print("=" * 90)
        print(f"QUERY: {query}")
        print()

        # Embed the query
        resp = openai.embeddings.create(model=EMBED_MODEL, input=query)
        query_embedding = resp.data[0].embedding

        # Search via Supabase RPC
        results = supabase.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_count": TOP_K},
        ).execute()

        if not results.data:
            print("  No results found.\n")
            continue

        for i, row in enumerate(results.data, start=1):
            meta = row.get("metadata", {})
            product = meta.get("product_name", "Unknown")
            rating = meta.get("rating", "N/A")
            category = meta.get("category", "")
            sim = row.get("similarity", 0)
            content = row.get("content", "")[:160]

            print(f"  [{i}] sim={sim:.3f} | {product} | rating={rating} | {category}")
            print(f"      {content}...")
            print()

    print("=" * 90)
    print("TEST COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
