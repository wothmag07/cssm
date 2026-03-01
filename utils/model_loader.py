import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.config_loader import load_config

load_dotenv()


class ModelLoader:
    """Load LLM and embedding models based on config."""

    def __init__(self):
        self.config = load_config()
        self._validate_env()

    def _validate_env(self):
        provider = self.config["llm_model"]["provider"]
        key_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        required_key = key_map.get(provider)
        if required_key and required_key not in os.environ:
            raise ValueError(
                f"Missing {required_key} for provider '{provider}'. Set it in .env"
            )

        embed_provider = self.config["embedding_model"]["provider"]
        embed_key = key_map.get(embed_provider)
        if embed_key and embed_key not in os.environ:
            raise ValueError(
                f"Missing {embed_key} for embedding provider '{embed_provider}'. Set it in .env"
            )

    def load_embeddings(self):
        provider = self.config["embedding_model"]["provider"]
        model = self.config["embedding_model"]["model"]
        logging.info(f"Loading embeddings: {provider}/{model}")

        if provider == "openai":
            return OpenAIEmbeddings(model=model)

        # Optional providers — import only if needed
        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model)

        raise ValueError(f"Unsupported embedding provider: {provider}")

    def load_llm(self):
        provider = self.config["llm_model"]["provider"]
        model = self.config["llm_model"]["model"]
        temperature = self.config["llm_model"].get("temperature", 0.2)
        logging.info(f"Loading LLM: {provider}/{model} (temp={temperature})")

        if provider == "openai":
            return ChatOpenAI(model=model, temperature=temperature)

        # Optional providers — import only if needed
        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, temperature=temperature)
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)

        raise ValueError(f"Unsupported LLM provider: {provider}")
