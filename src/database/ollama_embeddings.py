from langchain.embeddings.base import Embeddings


class OllamaEmbeddings(Embeddings):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def embed_documents(self, texts):
        """Generate embeddings for a list of texts."""
        return [self.embedding_function([text])[0] for text in texts]

    def embed_query(self, text):
        """Generate embeddings for a single query."""
        return self.embedding_function([text])[0]
