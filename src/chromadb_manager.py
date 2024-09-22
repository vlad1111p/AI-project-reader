import logging

import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with Ollama embedding functions"""
        self.client = chromadb.PersistentClient(path="ollama")

        self.embedding_function = OllamaEmbeddingFunction(
            model_name="mxbai-embed-large",
            url="http://localhost:11434/api/embeddings",
        )
        try:
            self.collection = self.client.get_collection("documents")
            logging.info("Collection 'documents' retrieved.")
        except InvalidCollectionException:
            self.collection = self.client.create_collection("documents")
            logging.info("Collection 'documents' created.")

    def embed_text(self, text: str):
        """Generate embeddings for the given text using Ollama's embedding model"""
        try:
            to_embedding = self.embedding_function([text])
            return to_embedding[0] if to_embedding else None
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")

    def add_file_to_db(self, file_path, file_content):
        embedding_id = file_path

        existing_docs = self.collection.get(ids=[embedding_id])

        if existing_docs["documents"]:
            logging.warning(f"Embedding ID '{embedding_id}' already exists.")
        else:
            to_embedding = self.embed_text(file_content)
            if to_embedding:
                self.collection.add(
                    documents=[file_content],
                    ids=[embedding_id],
                    embeddings=[to_embedding],
                )
                logging.info(f"File '{file_path}' added to ChromaDB with embedding.")
            else:
                logging.warning(f"Embedding for '{file_path}' is empty.")

    def query_db(self, query_text: str):
        """Query the ChromaDB with text and retrieve similar documents"""
        query_embedding = self.embed_text(query_text)  # Generate query embedding
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=4
            )
            return results["documents"] if results else None
        else:
            logging.info(
                f"Embedding for query '{query_text}' is empty, skipping query."
            )
            return None


if __name__ == "__main__":
    db_manager = ChromaDBManager()
    test_text = "Here is an article about llamas..."
    embedding = db_manager.embed_text(test_text)
    logging.info(f"Embedding for test text: {embedding}")
