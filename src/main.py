import logging

from chromadb_manager import ChromaDBManager
from file_reader import FileReader
from llama_ai import OllamaAI


def process_files_as_chunks(project_folder):
    db_manager = ChromaDBManager()
    ollama_ai = OllamaAI()

    reader = FileReader(project_folder)
    files_contents = reader.read_all_files()

    for file_path, content in files_contents.items():
        db_manager.add_file_to_db(file_path, content)

    query = "how can I further improve the class?"
    logging.info("Querying ChromaDB for relevant embeddings...")
    query_result = db_manager.query_db(query)
    if query_result:
        logging.info("Processing query results with OllamaAI...")
        for i, document_content in enumerate(query_result):
            logging.info(f"Processing document {i + 1}...")
            prompt = f"Here is a file content. How can I improve the class?\n\n{document_content}"
            response = ollama_ai.query_ollama(prompt)
            print(f"Response for Document {i + 1}: {response}")


if __name__ == "__main__":
    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    process_files_as_chunks(project_folder)
