from langchain_ollama import ChatOllama


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and bind tools"""
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def query_ollama(self, user_input: str) -> str | list[str | dict]:
        """Generate a chat response using ChatOllama"""
        messages = [
            (
                "system",
                "You are an assistant that provides help with Java Spring Boot applications.",
            ),
            ("human", user_input),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content
