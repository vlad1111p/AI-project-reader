from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class QueryState(TypedDict):
    query: str
    retrieved_files: List[Document]
    analyzed_files: List[str]
    project_path: str
    chat_history: List[BaseMessage]
    messages: Annotated[list, add_messages]
    response: str
