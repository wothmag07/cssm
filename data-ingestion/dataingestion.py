from json import load
from typing import List, Dict, Any
import logging
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
from configloader import load_config
from langchain_astradb import AstradbVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
from utils.modelloader import ModelLoader


class DataIngestion:
    def __init__(self, data: pd.DataFrame):
        """Initialize the DataIngestion pipeline"""
        logging.info("Initializing DataIngestion pipeline")
        self._load_env_variables()
        self.config = load_config()
        self.model_loader = ModelLoader()
        self.json_path = self.get_json_path()
        self.product_data = self.load_jsonl()

    def _load_env_variables(self):
        """Load the environment variables"""
        load_dotenv()
        required_env_variables = [
            "ASTRADB_API_ENDPOINT",
            "ASTRADB_APPLICATION_TOKEN",
            "ASTRADB_KEYSPACE",
            "GOOGLE_API_KEY",
        ]
        for variable in required_env_variables:
            if variable not in os.environ:
                raise ValueError(f"Environment variable {variable} is not set")
        self.astradb_api_endpoint = os.environ["ASTRADB_API_ENDPOINT"]
        self.astradb_application_token = os.environ["ASTRADB_APPLICATION_TOKEN"]
        self.astradb_keyspace = os.environ["ASTRADB_KEYSPACE"]
        self.google_api_key = os.environ["GOOGLE_API_KEY"]

    def get_json_path(self):
        """
        Get the path to the JSON file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, self.config["data"]["json_path"])

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
        return json_path

    def get_jsonl_path(self):
        """
        Get the path to the JSONL file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_path = os.path.join(current_dir, self.config["data"]["jsonl_path"])

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at {jsonl_path}")
        return jsonl_path

    def load_jsonl(self):
        """
        Load the JSONL file
        """
        with open(self.get_jsonl_path(), "r") as f:
            return [json.loads(line) for line in f]

    def transform(self):
        """
        Transform the JSONL file into a list of documents
        """
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                "product_name": row["product_name"],
                "product_description": row["product_description"],
                "product_rating": row["rating"],
                "product_summary": row["review_summary"],
                "product_review": row["review_text"],
                "product_category": row["category"],
            }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                "product_name": entry["product_name"],
                "product_description": entry["product_description"],
                "product_rating": entry["product_rating"],
                "product_summary": entry["product_summary"],
            }
            doc = Document(page_content=entry["product_review"], metadata=metadata)
            documents.append(doc)

        logging.info(f"Transformed {len(documents)} documents.")
        return documents

    def ingest(self, documents: List[Document]):
        """
        Ingest the documents into the vector store Astradb
        """
        vectorStore = AstradbVectorStore(
            api_endpoint=self.astradb_api_endpoint,
            token=self.astradb_application_token,
            namespace=self.astradb_keyspace,
            embeddings=self.model_loader.load_embeddings(),
            collection_name=self.config["astradb"]["collection_name"],
        )
        inserted_ids = vectorStore.add_documents(documents)
        logging.info(f"Data inserted successfully with ids: {inserted_ids}")

        return vectorStore, inserted_ids

    def run(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform()
        vectorStore, inserted_ids = self.ingest(documents)

        # TEST
        test_query = "Can you recommend me binoculars for hunting?"
        results = vectorStore.similarity_search(test_query)
        logging.info(f"Query: {test_query}")

        for result in results:
            logging.info(f"Result: {result.page_content}")
            logging.info(f"Result: {result.metadata}")
            logging.info("-" * 80)


if __name__ == "__main__":
    data_ingest = DataIngestion()
    data_ingest.run()
