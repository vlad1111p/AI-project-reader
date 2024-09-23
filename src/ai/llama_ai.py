from langchain.memory import ConversationSummaryBufferMemory
from langchain_ollama import ChatOllama
from oauthlib.uri_validate import query

from src.database.sql_database_manager import DatabaseManager


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5, memory_max_tokens=2000):
        """Initialize the Ollama model for chat and bind tools"""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm, max_token_limit=memory_max_tokens
        )
        self.sql_db_manager = DatabaseManager()

    def query_ollama(
            self, user_input: str, project_path: str
    ) -> str | list[str | dict]:
        """Generate a chat response using ChatOllama and save conversation to memory"""

        messages = [
            (
                "system",
                "You are an assistant that provides help with python or java applications.",
            ),
            ("human", user_input),
        ]
        ai_msg = self.llm.invoke(messages)
        self.memory.save_context({"input": user_input}, {"output": ai_msg.content})
        self.sql_db_manager.connect_and_execute_query(
            query, ai_msg.content, project_path
        )

        return ai_msg.content
