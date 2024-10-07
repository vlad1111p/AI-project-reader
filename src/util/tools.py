import logging

import requests
from langchain_core.tools import tool


@tool("file_analysis")
def analyze_file(file_path: str) -> str:
    """Analyzes a file and returns information about its structure and complexity."""
    with open(file_path, 'r') as f:
        content = f.read()
        logging.info("analyzing file")
        lines_of_code = len(content.splitlines())
        return f"File: {file_path}, Lines of Code: {lines_of_code}"


# @tool("dependency_mapping")
# def dependency_mapping(file_path: str) -> str:
#     """Identifies dependencies between files in a project."""
#     dependencies = find_dependencies(file_path)
#     return f"Dependencies for {file_path}: {dependencies}"


@tool("git_history")
def get_git_history(file_path: str) -> str:
    """Fetches the Git history for a given file or project."""
    import subprocess
    logging.info("github history")
    result = subprocess.run(["git", "log", file_path], capture_output=True, text=True)
    return result.stdout


@tool("api_doc_lookup")
def fetch_api_doc(library: str, language: str) -> str:
    """Fetches API documentation for a given library or module based on the project language."""
    logging.info("fetch api doc")
    if language.lower() == 'python':
        return fetch_python_api_doc(library)
    elif language.lower() == 'java':
        return fetch_java_api_doc(library)
    else:
        return f"Language '{language}' is not supported. Please provide 'python' or 'java'."


def fetch_python_api_doc(library: str) -> str:
    url = f"https://pypi.org/project/{library}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Could not retrieve documentation for Python library: {library}"


def fetch_java_api_doc(library: str) -> str:
    url = f"https://javadoc.io/doc/{library}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Could not retrieve documentation for Java library: {library}"
