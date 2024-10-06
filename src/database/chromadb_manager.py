import logging
from hashlib import md5

import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.database.ollama_embeddings import OllamaEmbeddings
from src.util.file_reader import FileReader


class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with Ollama embedding functions"""
        self.retriever = None
        self.client = chromadb.PersistentClient(path="../ollama")
        self.embedding_function = OllamaEmbeddingFunction(
            model_name="mxbai-embed-large",
            url="http://localhost:11434/api/embeddings",
        )
        self.embeddings = OllamaEmbeddings(self.embedding_function)

        try:
            self.collection = self.client.get_collection("documents")
            logging.info("Collection 'documents' retrieved.")
        except InvalidCollectionException:
            self.collection = self.client.create_collection("documents")
            logging.info("Collection 'documents' created.")

    def add_files_from_project_to_db(self, project_path, language):
        """Add or update files from the project path to the database and update the retriever."""
        reader = FileReader(project_path, language)
        files_contents = reader.read_all_files()

        for file_path, content in files_contents.items():
            embedding_id = file_path
            existing_docs = self.collection.get(ids=[embedding_id])
            if existing_docs["documents"]:
                stored_content = existing_docs["documents"][0]
                if md5(stored_content.encode()).hexdigest() == md5(content.encode()).hexdigest():
                    logging.info(f"File '{file_path}' is unchanged.")
                else:
                    logging.info(f"File '{file_path}' has been modified, updating...")
                    self.collection.delete(ids=[embedding_id])
                    self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)
            else:
                self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)

        self.update_retriever()

    def add_file_to_db_by_project_and_language(self, project_path, file_path, file_content, language):
        """Add or update a file and its embeddings in ChromaDB."""
        embedding_id = file_path
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
            logging.info(f"Embedding for '{file_path}' is empty.")

        self.update_retriever()

    def embed_text(self, text: str):
        """Generate embeddings for the given text."""
        try:
            logging.info(f"Embedding for text: {text}")
            to_embedding = self.embedding_function([text])
            return to_embedding[0] if to_embedding else None
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")

    def query_db_by_project_path_and_language(self, query_text: str, project_path: str, language: str):
        """Use the retriever to query the ChromaDB for relevant documents."""
        logging.info("Querying ChromaDB using retriever...")
        query_result = self.retriever.get_relevant_documents(query_text)
        return query_result

    def update_retriever(self):
        """Update the retriever after documents are added or updated."""
        documents = self.collection.get(ids=None)  # Retrieve all documents
        formatted_documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents["documents"], documents["metadatas"])
        ]
        vectorstore = Chroma.from_documents(formatted_documents, embedding=self.embeddings)
        self.retriever = vectorstore.as_retriever()
