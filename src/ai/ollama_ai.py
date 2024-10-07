from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from src.ai.conversational_history import CustomConversationBufferMemory
from src.database.sql_database_manager import DatabaseManager


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.sql_db_manager = DatabaseManager()

        self.memory = CustomConversationBufferMemory(llm=self.llm)

        self.prompt_template = ChatPromptTemplate(
            [
                SystemMessage(
                    content="You are an assistant that provides help with Python or Java applications by analyzing "
                            "project files."),
                MessagesPlaceholder(variable_name="chat_history"),
                SystemMessage(content="Let me analyze the following file content:  {query_result}"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        self.conversation_chain = RunnableSequence(self.prompt_template | self.llm)

    def query_ollama(self, query: str, query_result: list, project_path: str) -> str:
        """Generate a chat response using ChatOllama and store conversation."""
        self.memory.set_db_manager_and_project(self.sql_db_manager, project_path)

        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        if not isinstance(chat_history, list):
            chat_history = []

        prompt_variables = {
            "query": query,
            "query_result": query_result,
            "chat_history": chat_history
        }
        response = self.conversation_chain.invoke(prompt_variables)

        if isinstance(response, AIMessage):
            response = response.content

        self.memory.save_context({"input": query}, {"output": response})

        return response
