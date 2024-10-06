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

        if query_result:
            logging.info("Processing query results with OllamaAI...")
            for i, document_content in enumerate(query_result):
                logging.info(f"Processing document {i + 1}...")

                prompt = (
                    f"The following are documents related to the project '{project_path}' in response to the query '{query}':\n\n"
                    f"Document {i + 1} content:\n"
                    f"{document_content}\n\n"
                    "Please provide an analysis or explanation based on the document and your knowledge.\n"
                    "Use the query context and recent chat history, if relevant, to provide a response.\n"
                    "Ensure that the response incorporates the relevant project context."
                )

                response = self.ollama_ai.query_ollama(prompt, query, project_path)
                print("----------------------Response----------------")
                print(f"Response for Document {i + 1}: {response}")

        else:
            logging.info("No relevant files found for the query.")
