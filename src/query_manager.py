import logging

from src.chromadb_manager import ChromaDBManager
from src.llama_ai import OllamaAI


class QueryManager:
    def __init__(self):
        self.db_manager = ChromaDBManager()
        self.ollama_ai = OllamaAI()

    def ingest_relevant_files_from_project(self, project_folder):
        self.db_manager.add_files_from_project_to_db(project_folder)

    def process_query(self, query):
        logging.info("Querying ChromaDB for relevant embeddings...")

        query_result = self.db_manager.query_db(query)

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
        else:
            logging.info("No relevant files found for the query.")
