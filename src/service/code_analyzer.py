import logging

from src.ai.ai_handler import AiHandler
from src.database.chromadb_manager import ChromaDBManager


class CodeAnalyzer:
    def __init__(self, language: str):
        self.chroma_db_manager = ChromaDBManager()
        self.ai_handler = AiHandler()
        self.language = language

    def analyze(self, query: str, project_path: str):
        logging.info("Querying ChromaDB for relevant embeddings...")
        self.chroma_db_manager.add_files_from_project_to_db(project_path, self.language)
        query_result = self.chroma_db_manager.query_db_by_project_path_and_language(
            query, project_path, self.language
        )

        response = self.ai_handler.analyze_code(query, query_result, project_path)
        print("----------------------Response----------------")
        print(f"Response for Document : {response}")
