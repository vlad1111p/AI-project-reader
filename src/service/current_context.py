import hashlib
import os


def hash_project_path(project_path: str) -> str:
    """Hash the project path to create a unique, deterministic string using MD5."""
    return hashlib.md5(project_path.encode()).hexdigest()


def create_or_update_context_file(project_path: str, summary_text: str):
    """Ensure that a context file for the given project exists and write the summary to it."""
    file_name = f"current_context_{hash_project_path(project_path)}.txt"
    file_path = os.path.join("context", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        with open(file_path, "w") as file:
            file.write(summary_text)
        print(f"Created context file: {file_path} and wrote summary.")
    else:
        print(f"Context file {file_path} already exists.")
