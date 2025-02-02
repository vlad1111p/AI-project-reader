from src.ai.ai_code_analyzer.ai_analyzer import AiProjectAnalyzer


def analyze(query: str, project_path: str, language: str, model_type: str, model_name: str):
    """This function processes a given user query, using the project path and programming language
    as additional context. It leverages an AI model to analyze the query in the specified project scope
    and returns a structured response based on relevant documents and context."""

    project_analyzer = AiProjectAnalyzer(project_path, model_type=model_type, model_name=model_name)

    response = project_analyzer.query_model(query, project_path, language)
    print("----------------------Response----------------")
    print(f"Response for Document : {response}")
    return response
