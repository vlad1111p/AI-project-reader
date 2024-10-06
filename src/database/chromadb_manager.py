import logging
from hashlib import md5

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = Chroma(client=self.persistent_client, collection_name="documents",
                                  embedding_function=self.embedding_function)

    def add_files_from_project_to_db(self, project_path, language):
        """Add or update files from the project path to the database using langchain-chroma."""
        reader = FileReader(project_path, language)
        files_contents = reader.read_all_files()

        for file_path, content in files_contents.items():
            existing_docs = self.vectorstore.similarity_search(content)

            if existing_docs:
                stored_content = existing_docs[0].page_content
                if md5(stored_content.encode()).hexdigest() == md5(content.encode()).hexdigest():
                    logging.info(f"File '{file_path}' is unchanged.")
                else:
                    logging.info(f"File '{file_path}' has been modified, updating...")
                    self.vectorstore.delete(
                        ids=[existing_docs[0].metadata['id']])
                    self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)
            else:
                self.add_file_to_db_by_project_and_language(project_path, file_path, content, language)

    def add_file_to_db_by_project_and_language(self, project_path, file_path, file_content, language):
        """Add a single file to the ChromaDB along with its metadata."""
        chunks = self.text_splitter.split_text(file_content)
        self.vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"project_path": project_path, "language": language, "id": file_path}] * len(chunks)
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
            results = self.vectorstore.similarity_search(query_text, k=4, filter=filter_conditions)
            return results if results else None
        else:
            logging.info(f"Embedding for query '{query_text}' is empty, skipping query.")
            return None
