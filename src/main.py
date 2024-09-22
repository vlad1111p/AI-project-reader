import logging

from chromadb_manager import ChromaDBManager
from llama_ai import OllamaAI


def process_files_as_chunks(llm):
    query = "how can I further improve the class?"
    logging.info("Querying ChromaDB for relevant embeddings...")
    query_result = db_manager.query_db(query)
    if query_result:
        logging.info("Processing query results with OllamaAI...")
        for i, document_content in enumerate(query_result):
            logging.info(f"Processing document {i + 1}...")
            prompt = f"Here is a file content. How can I improve the class?\n\n{document_content}"
            response = llm.query_ollama(prompt)
            print(f"Response for Document {i + 1}: {response}")


if __name__ == "__main__":
    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"

    ollama_ai = OllamaAI()

    db_manager = ChromaDBManager()

    db_manager.add_files_from_project(project_folder)

    process_files_as_chunks(ollama_ai)
