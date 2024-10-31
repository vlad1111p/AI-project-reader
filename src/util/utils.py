import hashlib


def hash_project_path(project_path: str) -> str:
    """Hash the project path to create a unique, deterministic string using MD5."""
    return hashlib.md5(project_path.encode()).hexdigest()
