from typing import TypedDict, List

from langchain_core.documents import Document
from langgraph.graph import add_messages
from typing_extensions import Annotated


class QueryState(TypedDict):
    query: str
    retrieved_files: List[Document]
    project_path: str
    messages: Annotated[list, add_messages]
    response: str
