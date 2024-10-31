from typing import List

from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.ai.ai_code_analyzer.ai_analyzer import AiAnalyze


def create_llm(model_name, model_type, temperature):
    if model_type == "llama":
        return ChatOllama(model=model_name, temperature=temperature)
    elif model_type == "chatgpt":
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class AiHandler:
    def __init__(self, model_name="llama3.1", model_type="llama", temperature=0.5):
        """Initialize the appropriate LLM for chat."""
        self.llm = create_llm(model_name, model_type, temperature)
        self.code_analyzer = AiAnalyze(self.llm)

    def analyze_code(self, query: str, retrieved_files: List[Document], project_path: str):
        return self.code_analyzer.query_model(query, retrieved_files, project_path)
