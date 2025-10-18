from config.config_loader import load_config
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

import logging


class ModelLoader:
    """
    Utility class to load the LLM models and embeddings
    """

    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.validate_env()

    def validate_env(self):
        """
        Validate the environment variables
        """
        required_env_variables = [
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]
        self.groq_api_key = os.environ["GROQ_API_KEY"]
        self.google_api_key = os.environ["GOOGLE_API_KEY"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

        missing_env_variables = [
            var for var in required_env_variables if var not in os.environ
        ]
        if missing_env_variables:
            raise ValueError(f"Missing environment variables: {missing_env_variables}")

    def load_embeddings(self):
        """
        Load the embedding model
        """
        logging.info(
            f"Loading embedding model: {self.config['embedding_model']['model']}"
        )
        model_name = self.config["embedding_model"]["model"]
        if self.config["embedding_model"]["provider"] == "google":
            return GoogleGenerativeAIEmbeddings(
                api_key=self.google_api_key, model=model_name
            )
        elif self.config["embedding_model"]["provider"] == "openai":
            return OpenAIEmbeddings(api_key=self.openai_api_key, model=model_name)
        # elif self.config["embedding_model"]["provider"] == "ollama":
        #     return OllamaEmbeddings(api_key=self.ollama_api_key, model=model_name)
        else:
            raise ValueError(
                f"Invalid embedding provider: {self.config['embedding_model']['provider']}"
            )

    def load_llm(self):
        """
        Load the LLM model
        """
        logging.info(f"Loading LLM model: {self.config['llm_model']['model']}")
        model_name = self.config["llm_model"]["model"]
        if self.config["llm_model"]["provider"] == "google":
            return ChatGoogleGenerativeAI(api_key=self.google_api_key, model=model_name)
        elif self.config["llm_model"]["provider"] == "groq":
            return ChatGroq(api_key=self.groq_api_key, model=model_name)
        elif self.config["llm_model"]["provider"] == "openai":
            return ChatOpenAI(api_key=self.openai_api_key, model=model_name)
        elif self.config["llm_model"]["provider"] == "anthropic":
            return ChatAnthropic(api_key=self.anthropic_api_key, model=model_name)
        # elif self.config["llm_model"]["provider"] == "ollama":
        #     return Ollama(api_key=self.ollama_api_key, model=model_name)

        else:
            raise ValueError(
                f"Invalid LLM provider: {self.config['llm_model']['provider']}"
            )
