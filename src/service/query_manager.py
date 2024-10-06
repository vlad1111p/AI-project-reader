import logging

from src.ai.llama_ai import OllamaAI
from src.database.chromadb_manager import ChromaDBManager


class QueryManager:
    def __init__(self, language: str):
        self.chroma_db_manager = ChromaDBManager()
        self.ollama_ai = OllamaAI()
        self.language = language

    def ingest_relevant_files_from_project(self, project_path: str):
        self.chroma_db_manager.add_files_from_project_to_db(project_path, self.language)

    def process_query(self, query: str, project_path: str):
        logging.info("Querying ChromaDB for relevant embeddings...")

        query_result = self.chroma_db_manager.query_db_by_project_path_and_language(
            query, project_path, self.language
        )

        prompt = (
            f"The following is the content of files related to your query: '{query}'. "
            f"Based on your query, please provide an answer or further explanation related to the content.\n\n"
            f"{query_result}\n\n"
            "Respond based on the context of the query."
        )

        response = self.ollama_ai.query_ollama(query, prompt, project_path)
        print("----------------------Response----------------")
        print(f"Response for Document : {response}")
