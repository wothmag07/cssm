import os
from typing import List, Dict, Any
import logging
from utils.modelloader import ModelLoader
from langchain_astradb import AstradbVectorStore
from langchain_core.documents import Document
from configloader import load_config
from dotenv import load_dotenv


class Retrieval:
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
