import logging
import os
from glob import glob


def _read_file(file_path):
    """Return content of a single file."""
    if os.path.getsize(file_path) == 0:
        logging.warning(f"Skipping empty file: {file_path}")
        return "empty file"
    with open(file_path, "r") as file:
        return file.read().strip()


class FileReader:
    def __init__(self, project_path, language):
        self.project_path = project_path
        self.language = language
        self.allowed_extensions = self.get_allowed_extensions()

    def get_allowed_extensions(self):
        """Return the allowed file extensions based on the selected language."""
        if self.language == "java":
            return [".java", ".xml", ".properties", ".yml", ".yaml", "pom.xml"]
        elif self.language == "python":
            return [".py", ".yaml", ".yml", "requirements.txt"]
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def read_files(self):
        """Read all files in the project directory."""
        return {file: _read_file(file) for file in self._get_files()}

    def _get_files(self):
        """Fetch all relevant files."""
        files = []
        for ext in self.allowed_extensions:
            files.extend(glob(os.path.join(self.project_path, "**", f"*{ext}"), recursive=True))
        return files
