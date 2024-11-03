from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama


def create_llm(model_name, model_type, temperature):
    if model_type == "llama":
        return ChatOllama(model=model_name, temperature=temperature)
    elif model_type == "chatgpt":
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class AiHandler:
    def __init__(self, model_name="llama3.2", model_type="llama", temperature=0.2):
        """Initialize the appropriate LLM for chat."""
        self.llm = create_llm(model_name, model_type, temperature)
