import logging
from hashlib import md5

import chromadb
from langchain.vectorstores import Chroma

from src.ai.OllamaLangchainEmbeddings import OllamaLangchainEmbeddings
from src.util.file_reader import FileReader


class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with Ollama embedding functions through a LangChain wrapper."""

        self.persistent_client = chromadb.PersistentClient(path="../ollama")
        self.persistent_client.get_or_create_collection("documents")

        self.embedding_function = OllamaLangchainEmbeddings(
            model_name="mxbai-embed-large",
            url="http://localhost:11434/api/embeddings"
        )
        self.vectorstore = Chroma(client=self.persistent_client, collection_name="documents",
                                  embedding_function=self.embedding_function)

    def add_files_from_project_to_db(self,
                                     project_path: str, language: str):
        """Add or update files from the project path to the database using langchain-chroma."""
        reader = FileReader(project_path, language)
        files_contents = reader.read_all_files()

        for file_path, content in files_contents.items():
            existing_docs = self.vectorstore.similarity_search(content)

            if existing_docs:
                stored_content = existing_docs[0].page_content
                if md5(stored_content.encode()).hexdigest() != md5(content.encode()).hexdigest():
                    logging.info(f"File '{file_path}' has been modified, updating...")
                    existing_ids = [doc.metadata['id'] for doc in existing_docs]
                    if file_path in existing_ids:
                        self.vectorstore.delete(ids=[file_path])
                    self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)
            else:
                self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)

    def add_file_to_db_by_project_and_language(self, project_path: str, file_path: str, file_content: str,
                                               language: str):
        """Add a single file to the ChromaDB along with its metadata."""
        self.vectorstore.add_texts(
            texts=[file_content],
            metadatas=[{
                "project_path": project_path,
                "language": language,
                "id": file_path
            }]
        )
        logging.info(f"File '{file_path}' added to ChromaDB with embedding.")

    def query_db_by_project_path_and_language(self, query_text: str, project_path: str, language: str):
        """Query the ChromaDB with text and retrieve similar documents filtered by project path and language."""

        query_embedding = self.embedding_function.embed_query(query_text)

        if query_embedding:
            filter_conditions = {
                "$and": [
                    {"project_path": {"$eq": project_path}},
                    {"language": {"$eq": language}}
                ]
            }
            # noinspection PyTypeChecker
            results = self.vectorstore.similarity_search(query_text, k=10, filter=filter_conditions)
            unique_files = {result.metadata['id']: result for result in results}.values()

            return list(unique_files) if unique_files else None
        else:
            logging.info(f"Embedding for query '{query_text}' is empty, skipping query.")
            return None
