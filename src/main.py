from src.chromadb_manager import ChromaDBManager
from src.file_reader import FileReader
from src.llama_ai import OllamaAI  # Updated OllamaAI class

if __name__ == "__main__":

    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"

    file_reader = FileReader(project_folder)
    files_content = file_reader.read_all_files()

    db_manager = ChromaDBManager()
    for file_path, content in files_content.items():
        print(f"Ingesting file: {file_path}")
        db_manager.add_file_to_db(file_path, content)

    query_result = db_manager.query_db(
        "What is the purpose of DemoHazelcastApplication class?"
    )

    if query_result:
        document_content = query_result[0].page_content

        ollama_ai = OllamaAI()

        prompt = f"how can i further improve the class? {document_content}"

        print(prompt)

        response = ollama_ai.query_ollama(prompt)
        print(f"Ollama Response:\n{response}")
