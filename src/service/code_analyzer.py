from src.ai.ai_code_analyzer.ai_analyzer import AiProjectAnalyzer
from src.ai.ai_handler import AiHandler


def analyze(query: str, project_path: str, language: str):
    """Analyze the user query and retrieve relevant embeddings from ChromaDB."""

    llm = AiHandler().llm
    project_analyzer = AiProjectAnalyzer(llm, project_path)

    response = project_analyzer.query_model(query, project_path, language)
    print("----------------------Response----------------")
    print(f"Response for Document : {response}")
