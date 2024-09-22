import logging

from src.database.chromadb_manager import ChromaDBManager
from src.database.sql_database_manager import DatabaseManager
from src.llama_ai import OllamaAI


class QueryManager:
    def __init__(self, language: str):
        self.db_manager = ChromaDBManager()
        self.ollama_ai = OllamaAI()
        self.sql_db_manager = DatabaseManager()
        self.language = language

    def ingest_relevant_files_from_project(self, project_path):
        self.db_manager.add_files_from_project_to_db(project_path, self.language)

    def process_query(self, query: str, project_path: str):
        logging.info("Querying ChromaDB for relevant embeddings...")

        query_result = self.db_manager.query_db_by_project_path_and_language(
            query, project_path, self.language
        )

        if query_result:
            logging.info("Processing query results with OllamaAI...")
            for i, document_content in enumerate(query_result):
                logging.info(f"Processing document {i + 1}...")

                prompt = (
                    f"The following is the content of a file related to your query: '{query}'. "
                    f"Based on your query, please provide an answer or further explanation related to the content.\n\n"
                    f"{document_content}\n\n"
                    "Respond based on the context of the query."
                )

                response = self.ollama_ai.query_ollama(prompt)
                print(f"Response for Document {i + 1}: {response}")
                self.sql_db_manager.connect_and_execute_query(query, response)
                print(f"SQLDatabase content {self.sql_db_manager.get_queries()}")

        else:
            logging.info("No relevant files found for the query.")
