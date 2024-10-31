from typing import List

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class AnalyzeFilesArgs(BaseModel):
    retrieved_files: List[Document]


def analyze_files(args) -> dict:
    """Analyze retrieved files and return a dictionary."""
    args_model = AnalyzeFilesArgs(**args) if isinstance(args, dict) else args
    retrieved_files = args_model.retrieved_files

    analyzed_results = [f"Analyzed content of {doc.page_content}" for doc in retrieved_files]

    return {"analyzed_files": analyzed_results}


def create_tools():
    """Create structured tools and return them."""

    analyze_files_tool = StructuredTool.from_function(
        args_schema=AnalyzeFilesArgs,
        name="analyze_files",
        description="Analyze the retrieved files.",
        func=analyze_files
    )

    return [analyze_files_tool]
