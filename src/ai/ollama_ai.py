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
from src.util.tools import analyze_file, fetch_api_doc, get_git_history


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.sql_db_manager = DatabaseManager()

        self.memory = CustomConversationBufferMemory(llm=self.llm)

        self.llm.bind_tools([analyze_file, fetch_api_doc, get_git_history])
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

    def query_ollama(self, query: str, query_result: list, project_path: str, language: str) -> str:
        """Generate a chat response using ChatOllama and store conversation."""
        self.memory.set_db_manager_and_project(self.sql_db_manager, project_path)

        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        if not isinstance(chat_history, list):
            chat_history = []

        file_analysis_result = self.llm.tools["analyze_file"](file_path=project_path)
        api_doc_result = self.llm.tools["fetch_api_doc"](library="langchain", language=language)
        git_history_result = self.llm.tools["get_git_history"](file_path=project_path)

        tool_output_summary = (f"File Analysis: {file_analysis_result}\nAPI Documentation: {api_doc_result}\nGit "
                               f"History: {git_history_result}")

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
