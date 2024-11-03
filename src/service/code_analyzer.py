from src.ai.ai_code_analyzer.ai_analyzer import AiProjectAnalyzer
from src.ai.ai_handler import AiHandler


def analyze(query: str, project_path: str, language: str):
    """This function processes a given user query, using the project path and programming language
    as additional context. It leverages an AI model to analyze the query in the specified project scope
    and returns a structured response based on relevant documents and context."""

    llm = AiHandler().llm
    project_analyzer = AiProjectAnalyzer(llm, project_path)

    response = project_analyzer.query_model(query, project_path, language)
    print("----------------------Response----------------")
    print(f"Response for Document : {response}")
