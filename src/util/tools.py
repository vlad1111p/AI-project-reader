from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.database.sql_database_manager import DatabaseManager


class RetrieveChatHistoryArgs(BaseModel):
    project_path: str


class AnalyzeFilesArgs(BaseModel):
    retrieved_files: List[Document]


class ToolService:
    def __init__(self):
        self.sql_db_manager = DatabaseManager()

    def create_tools(self):
        """Create structured tools and return them."""
        retrieve_chat_history_tool = StructuredTool.from_function(
            args_schema=RetrieveChatHistoryArgs,
            name="retrieve_chat_history",
            description="Retrieve the chat history for a project.",
            func=self.retrieve_chat_history
        )

        analyze_files_tool = StructuredTool.from_function(
            args_schema=AnalyzeFilesArgs,
            name="analyze_files",
            description="Analyze the retrieved files.",
            func=self.analyze_files
        )

        return [retrieve_chat_history_tool, analyze_files_tool]

    def retrieve_chat_history(self, args) -> dict:
        """Retrieve chat history for the given project as a dictionary."""
        args_obj = RetrieveChatHistoryArgs(**args)
        project_path = args_obj.project_path
        chat_history = []

        if project_path:
            chat_contexts = self.sql_db_manager.get_project_chat_context(project_path)
            for context in chat_contexts:
                chat_history.append(HumanMessage(content=context.question))
                if context.response:
                    chat_history.append(AIMessage(content=context.response))

        return {"chat_history": chat_history}

    def analyze_files(self, args) -> dict:
        """Analyze retrieved files and return a dictionary."""
        args_model = AnalyzeFilesArgs(**args) if isinstance(args, dict) else args
        retrieved_files = args_model.retrieved_files

        analyzed_results = [f"Analyzed content of {doc.page_content}" for doc in retrieved_files]

        return {"analyzed_files": analyzed_results}
