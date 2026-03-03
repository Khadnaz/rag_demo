"""OpenAI embedding generation."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

_client = None


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set. Copy .env.example to .env and add your key.")
        _client = OpenAI(api_key=api_key)
    return _client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings using OpenAI.
    
    Args:
        texts: List of text strings to embed.
    
    Returns:
        List of embedding vectors (each a list of floats).
    """
    client = _get_client()
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]
