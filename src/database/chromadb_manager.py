import logging

import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from src.util.file_reader import FileReader


class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with Ollama embedding functions"""
        self.client = chromadb.PersistentClient(path="../ollama")

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

    def add_files_from_project_to_db(self, project_path, language):
        """Add all files from the project path to the database."""
        reader = FileReader(project_path, language)
        files_contents = reader.read_all_files()

        for file_path, content in files_contents.items():
            self.add_file_to_db(project_path, file_path, content, language)

    def add_file_to_db(self, project_path, file_path, file_content, language):
        """Add a single file to the ChromaDB along with its metadata."""
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
                    metadatas=[{"project_path": project_path, "language": language}],
                )
                logging.info(f"File '{file_path}' added to ChromaDB with embedding.")
            else:
                logging.warning(f"Embedding for '{file_path}' is empty.")

    def query_db_by_project_path_and_language(
        self, query_text: str, project_path: str, language: str
    ):
        """Query the ChromaDB with text and retrieve similar documents filtered by project path and language."""
        query_embedding = self.embed_text(query_text)
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=4,
                where={
                    "$and": [
                        {"project_path": {"$eq": project_path}},
                        {"language": {"$eq": language}},
                    ]
                },
            )
            return results["documents"] if results else None
        else:
            logging.info(
                f"Embedding for query '{query_text}' is empty, skipping query."
            )
            return None

    def embed_text(self, text: str):
        """Generate embeddings for the given text using Ollama's embedding model."""
        try:
            logging.info(f"Embedding for test text: {str}")
            to_embedding = self.embedding_function([text])
            return to_embedding[0] if to_embedding else None
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")


if __name__ == "__main__":
    python_project_path = "/"
    db_manager = ChromaDBManager()
    test_text = "py file."
    db_manager.add_files_from_project_to_db(python_project_path, "python")
    response = db_manager.query_db_by_project_path_and_language(
        test_text, python_project_path, "python"
    )
    logging.info(f"Embedding for test text: {response}")
