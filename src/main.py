from src.chromadb_manager import ChromaDBManager
from src.file_reader import FileReader

if __name__ == "__main__":
    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    file_reader = FileReader(project_folder)
    files_content = file_reader.read_all_files()

    db_manager = ChromaDBManager()

    for file_path, content in files_content.items():
        print(f"File: {file_path}\nContent: {content[:]}...\n")
        db_manager.add_file_to_db(file_path, content)

    # Try a more general query
    query_result = db_manager.query_db("What classes are in this project?")
    print(query_result)
