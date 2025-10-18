from typing import List
import logging
import os
import json
import time
from dotenv import load_dotenv
from config.config_loader import load_config
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from utils.model_loader import ModelLoader


class DataIngestion:
    def __init__(self):
        """Initialize the DataIngestion pipeline"""
        logging.info("Initializing DataIngestion pipeline")
        self._load_env_variables()
        self.config = load_config()
        self.model_loader = ModelLoader()
        self.json_path = self.get_json_path()
        self.product_data = self.load_jsonl()
        try:
            jsonl_path = self.get_jsonl_path()
        except Exception:
            jsonl_path = "<unknown jsonl path>"
        logging.info(
            f"Loaded JSONL data from {jsonl_path} with {len(self.product_data)} rows"
        )

    def _load_env_variables(self):
        """Load the environment variables"""
        load_dotenv()
        required_env_variables = [
            "ASTRADB_API_ENDPOINT",
            "ASTRADB_APPLICATION_TOKEN",
            "ASTRADB_KEYSPACE",
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]
        for variable in required_env_variables:
            if variable not in os.environ:
                raise ValueError(f"Environment variable {variable} is not set")
        self.astradb_api_endpoint = os.environ["ASTRADB_API_ENDPOINT"]
        self.astradb_application_token = os.environ["ASTRADB_APPLICATION_TOKEN"]
        self.astradb_keyspace = os.environ["ASTRADB_KEYSPACE"]
        self.google_api_key = os.environ["GOOGLE_API_KEY"]
        self.groq_api_key = os.environ["GROQ_API_KEY"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

    def get_json_path(self):
        """
        Get the path to the JSON file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        configured_path = self.config["data"]["json_path"]
        json_path = (
            configured_path
            if os.path.isabs(configured_path)
            else os.path.join(project_root, configured_path)
        )

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
        return json_path

    def get_jsonl_path(self):
        """
        Get the path to the JSONL file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        configured_path = self.config["data"]["jsonl_path"]
        jsonl_path = (
            configured_path
            if os.path.isabs(configured_path)
            else os.path.join(project_root, configured_path)
        )

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at {jsonl_path}")
        return jsonl_path

    def load_jsonl(self):
        """
        Load the JSONL file
        """
        with open(self.get_jsonl_path(), "r", encoding="utf-8", errors="replace") as f:
            return [json.loads(line) for line in f]

    def transform(self):
        """
        Transform the JSONL file into a list of documents
        """
        product_list = []
        limit = self.config.get("ingestion", {}).get("limit")
        shuffle = bool(self.config.get("ingestion", {}).get("shuffle", False))
        data = list(self.product_data)
        if shuffle:
            try:
                import random

                random.shuffle(data)
            except Exception as exc:
                logging.warning(f"Shuffle failed; proceeding without shuffle: {exc}")
        if isinstance(limit, int) and limit > 0:
            data = data[:limit]
        logging.info(
            f"Transform will process rows: {len(data)} (limit={limit}, shuffle={shuffle})"
        )

        empty_skipped = 0
        for idx, row in enumerate(data):
            title = (row.get("title") or "").strip()
            review_text = (row.get("text") or "").strip()
            page_content = f"{title}\n\n{review_text}".strip() if title else review_text

            # Skip entries without any content to embed
            if not page_content:
                empty_skipped += 1
                continue

            # Optional: write full description externally and generate summary
            description_full = row.get("product_description")

            product_entry = {
                "page_content": page_content,
                "product_name": row.get("product_name"),
                # store full description only if we keep it locally (ref stored separately)
                "product_description": description_full,
                "product_rating": row.get("rating"),
                "product_category": row.get("category"),
                "user_id": row.get("user_id"),
                "product_id": row.get("product_id"),
                "avg_rating": row.get("avg_rating"),
                "rating_count": row.get("rating_count"),
                "verified_purchase": row.get("verified_purchase"),
                "helpful_vote": row.get("helpful_vote"),
                "store": row.get("store"),
                "price": row.get("price"),
                "timestamp": row.get("timestamp"),
            }
            product_list.append(product_entry)

        documents = []

        for entry in product_list:
            metadata = {
                "product_id": entry.get("product_id"),
                "product_name": entry.get("product_name"),
                "product_rating": entry.get("product_rating"),
                "product_category": entry.get("product_category"),
                "user_id": entry.get("user_id"),
            }

            metadata = {k: v for k, v in metadata.items() if v is not None}
            doc = Document(page_content=entry["page_content"], metadata=metadata)
            documents.append(doc)

        logging.info(
            f"Transform complete. Built {len(product_list)} product entries; "
            f"skipped {empty_skipped} empty contents. Output docs: {len(documents)}"
        )
        return documents

    def ingest(self, documents: List[Document]):
        """
        Ingest the documents into the vector store Astradb
        """
        logging.info(
            "Initializing AstraDBVectorStore (collection=%s, namespace=%s)",
            self.config["astradb"]["collection_name"],
            self.astradb_keyspace,
        )
        vectorStore = AstraDBVectorStore(
            api_endpoint=self.astradb_api_endpoint,
            token=self.astradb_application_token,
            namespace=self.astradb_keyspace,
            embedding=self.model_loader.load_embeddings(),
            collection_name=self.config["astradb"]["collection_name"],
        )
        # Insert in batches with retries to mitigate transient provider errors
        batch_size = int(self.config.get("ingestion", {}).get("batch_size", 50))
        max_retries = int(self.config.get("ingestion", {}).get("max_retries", 5))
        backoff_initial_seconds = float(
            self.config.get("ingestion", {}).get("backoff_initial_seconds", 1.0)
        )

        inserted_ids: List[str] = []
        total = len(documents)
        logging.info(
            f"Starting ingestion: total_docs={total}, batch_size={batch_size}, max_retries={max_retries}"
        )
        for start_index in range(0, total, batch_size):
            batch = documents[start_index : start_index + batch_size]
            attempt = 0
            backoff = backoff_initial_seconds
            while True:
                try:
                    batch_ids = vectorStore.add_documents(batch)
                    inserted_ids.extend(batch_ids)
                    logging.info(
                        f"Inserted batch {start_index // batch_size + 1} "
                        f"({len(batch)} docs). Cumulative: {len(inserted_ids)}/{total}"
                    )
                    break
                except Exception as exc:  # retry on transient provider/transport errors
                    attempt += 1
                    if attempt > max_retries:
                        logging.exception(
                            "Failed to insert batch after retries. Aborting ingestion."
                        )
                        raise
                    logging.warning(
                        f"Batch insert failed (attempt {attempt}/{max_retries}): {exc}. "
                        f"Retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)

        logging.info(f"Data inserted successfully. Total ids: {len(inserted_ids)}")

        return vectorStore, inserted_ids

    def run(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        logging.info("Running transform phase...")
        try:
            documents = self.transform()
        except Exception as exc:
            logging.exception(f"Transform phase failed: {exc}")
            raise

        logging.info("Running ingest phase...")
        try:
            vectorStore, inserted_ids = self.ingest(documents)
        except Exception as exc:
            logging.exception(f"Ingest phase failed: {exc}")
            raise

        # TEST (best-effort; do not fail the ingestion on query issues)
        try:
            test_query = "Can you recommend me binoculars for hunting?"
            results = vectorStore.similarity_search(test_query)
            logging.info(f"Query: {test_query}")
            for result in results:
                logging.info(f"Page Content: {result.page_content}")
                logging.info(f"Metadata: {result.metadata}")
                logging.info("-" * 80)
        except Exception as exc:
            logging.warning(f"Similarity search test skipped due to error: {exc}")


if __name__ == "__main__":
    # Basic logging setup if not configured by the host process
    logging.basicConfig(
        level=logging.INFO,
        format="% (asctime)s | %(levelname)s | %(name)s | %(message)s".replace(" ", ""),
    )
    data_ingest = DataIngestion()
    data_ingest.run()
