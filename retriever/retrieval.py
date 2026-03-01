import os
import logging

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI
from supabase import create_client

from config.config_loader import load_config

load_dotenv(override=True)


class Retriever:
    def __init__(self):
        self.config = load_config()
        self._client = None
        self._openai = None
        self.embed_model = self.config["embedding_model"]["model"]
        self.table = self.config["supabase"]["table_name"]
        self.query_name = self.config["supabase"]["query_name"]
        self.top_k = self.config["retriever"].get("top_k", 8)

    @property
    def client(self):
        if self._client is None:
            self._client = create_client(
                os.environ["SUPABASE_URL"],
                os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            )
            logging.info("Connected to Supabase")
        return self._client

    @property
    def openai(self):
        if self._openai is None:
            self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._openai

    def _embed_query(self, query: str) -> list:
        """Embed a query using OpenAI."""
        resp = self.openai.embeddings.create(model=self.embed_model, input=query)
        return resp.data[0].embedding

    def retrieve(self, query: str):
        """Retrieve documents with similarity scores via Supabase RPC."""
        embedding = self._embed_query(query)

        # Call the match_documents RPC function directly
        result = self.client.rpc(
            self.query_name,
            {
                "query_embedding": embedding,
                "match_count": self.top_k,
                "filter": {},
            },
        ).execute()

        docs_with_scores = []
        for row in result.data or []:
            doc = Document(
                page_content=row["content"],
                metadata=row.get("metadata", {}),
            )
            score = row.get("similarity", 0.0)
            docs_with_scores.append((doc, score))

        logging.info(f"Retrieved {len(docs_with_scores)} docs for: {query[:80]}")
        return docs_with_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    retriever = Retriever()
    query = "Can you suggest good budget laptops?"
    results = retriever.retrieve(query)
    for idx, (doc, score) in enumerate(results, start=1):
        logging.info(f"[{idx}] sim={score:.3f} | {doc.page_content[:120]}")
        logging.info(f"     metadata={doc.metadata}")
        logging.info("-" * 80)
