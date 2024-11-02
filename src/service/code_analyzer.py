from src.ai.ai_handler import AiHandler
from src.database.chromadb_manager import ChromaDBManager


class CodeAnalyzer:
    def __init__(self, language: str):
        self.chroma_db_manager = ChromaDBManager()
        self.ai_handler = AiHandler()
        self.language = language

    def analyze(self, query: str, project_path: str):
        """Analyze the user query and retrieve relevant embeddings from ChromaDB."""
        self.chroma_db_manager.add_files_from_project_to_db(project_path, self.language)
        query_result = self.chroma_db_manager.query_db(
            query, project_path, self.language
        )
        response = self.ai_handler.analyze_code(query, query_result, project_path)
        print("----------------------Response----------------")
        print(f"Response for Document : {response}")
