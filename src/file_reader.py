import os
from glob import glob


def read_file(file_path_param):
    """Reads a single file's content."""
    with open(file_path_param, "r", encoding="utf-8") as file:
        return file.read()


class FileReader:
    def __init__(self, project_path, language="java"):
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

    def read_all_files(self):
        """Read all relevant files in the project directory."""
        files = self.get_files()
        return {file: read_file(file) for file in files}

    def get_files(self):
        """Get all relevant files in the project directory based on file extensions."""
        files = []
        for extension in self.allowed_extensions:
            files.extend(
                glob(
                    os.path.join(self.project_path, "**", f"*{extension}"),
                    recursive=True,
                )
            )
        return files


if __name__ == "__main__":
    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    file_reader = FileReader(project_folder)
    files_content = file_reader.read_all_files()
    for file_path, content in files_content.items():
        print(f"File: {file_path}\nContent: {content[:]}...\n")
