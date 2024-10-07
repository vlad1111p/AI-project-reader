from typing import Sequence

from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain.embeddings.base import Embeddings


class OllamaLangchainEmbeddings(Embeddings):
    """Wrapper to adapt OllamaEmbeddingFunction to LangChain's Embeddings interface."""

    def __init__(self, model_name: str, url: str):
        self.ollama_embedding = OllamaEmbeddingFunction(model_name=model_name, url=url)

    def embed_query(self, text: str) -> Sequence[float] | Sequence[int]:
        """Generate embedding for a single query."""
        return self.ollama_embedding([text])[0]

    def embed_documents(self, texts: list) -> list:
        """Generate embeddings for a list of documents."""
        return [self.ollama_embedding([doc])[0] for doc in texts]
