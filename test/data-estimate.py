"""Dry run — shows chunk count, storage, and cost estimate without embedding."""
import importlib.util
import logging
import os

# Load directly from data-ingestion folder to avoid conflict with data_ingestion/
spec = importlib.util.spec_from_file_location(
    "data_ingestion_mod",
    os.path.join(os.path.dirname(__file__), "data_ingestion", "data_ingestion.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DataIngestion = mod.DataIngestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

p = DataIngestion()
reviews = p.load_and_deduplicate()
chunks = p.transform(reviews)

est_storage_mb = len(chunks) * 7 / 1024
est_cost = len(chunks) * 100 * 0.02 / 1_000_000

print(f"\n{'='*40}")
print(f"Chunks:        {len(chunks):,}")
print(f"Est. storage:  {est_storage_mb:.0f} MB")
print(f"Est. cost:     ${est_cost:.2f}")
print(f"{'='*40}")

if est_storage_mb > 500:
    print("\nWARNING: Exceeds Supabase free tier (500MB)")
    print("Reduce max_reviews_per_product or raise min_helpful_votes")
else:
    print("\nFits within Supabase free tier (500MB)")
