import os
import logging
from utils.model_loader import ModelLoader
from langchain_astradb import AstraDBVectorStore
from config.config_loader import load_config
from dotenv import load_dotenv


class Retriever:
    def __init__(self):
        self.config = load_config()
        self.model_loader = ModelLoader()
        self.vector_store = None
        self._load_env_variables()
        self.retriever = None

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
                raise ValueError(f"Environment variable {variable} is missing")
        self.astradb_api_endpoint = os.environ["ASTRADB_API_ENDPOINT"]
        self.astradb_application_token = os.environ["ASTRADB_APPLICATION_TOKEN"]
        self.astradb_keyspace = os.environ["ASTRADB_KEYSPACE"]
        self.google_api_key = os.environ["GOOGLE_API_KEY"]
        self.groq_api_key = os.environ["GROQ_API_KEY"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

    def load_retriever(self):
        """Load the retriever"""

        if not self.vector_store:
            collection_name = self.config["astradb"]["collection_name"]

            self.vector_store = AstraDBVectorStore(
                api_endpoint=self.astradb_api_endpoint,
                token=self.astradb_application_token,
                namespace=self.astradb_keyspace,
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
            )
            logging.info(f"Loaded vector store for collection {collection_name}")

        if not self.retriever:
            logging.info("Loading retriever")
            top_k = (
                self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            logging.info(f"Loaded retriever with top_k {top_k}")

        return self.retriever

    def retrieve(self, query: str):
        """Retrieve the documents"""
        logging.info(f"Retrieving documents for query: {query}")
        retriever = self.load_retriever()
        output = retriever.invoke(query)
        logging.info(f"Output: {output}")
        return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="% (asctime)s | %(levelname)s | %(name)s | %(message)s".replace(" ", ""),
    )
    retriever = Retriever()
    query = "Can you suggest good budget laptops?"
    results = retriever.retrieve(query)
    logging.info(f"Results: {results}")
    for idx, result in enumerate(results, start=1):
        logging.info(
            f"Result {idx}: {result.page_content} \n Metadata: {result.metadata}"
        )
        logging.info("-" * 80)
