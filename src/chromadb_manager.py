import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with Ollama embedding functions"""
        self.client = chromadb.PersistentClient(path="ollama")

        self.embedding_function = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings",
        )

        try:
            self.collection = self.client.get_collection("documents")
        except chromadb.api.errors.CollectionNotFoundError:
            self.collection = self.client.create_collection("documents")

    def embed_text(self, text: str):
        """Generate embeddings for the given text using Ollama's embedding model"""
        try:
            to_embedding = self.embedding_function([text])
            return to_embedding[0] if to_embedding else None
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")

    def add_file_to_db(self, file_name: str, file_content: str):
        """Add a file to the ChromaDB collection using Ollama embeddings"""
        to_embedding = self.embed_text(file_content)  # Generate embeddings
        if to_embedding:
            self.collection.add(
                documents=[file_content], ids=[file_name], embeddings=[to_embedding]
            )
            print(f"File '{file_name}' added to ChromaDB with embedding.")
        else:
            print(f"Embedding for file '{file_name}' is empty, skipping.")

    def query_db(self, query_text: str):
        """Query the ChromaDB with text and retrieve similar documents"""
        query_embedding = self.embed_text(query_text)  # Generate query embedding
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=4
            )
            return results["documents"] if results else None
        else:
            print(f"Embedding for query '{query_text}' is empty, skipping query.")
            return None


if __name__ == "__main__":
    db_manager = ChromaDBManager()
    test_text = "Here is an article about llamas..."
    embedding = db_manager.embed_text(test_text)
    print(f"Embedding for test text: {embedding}")
