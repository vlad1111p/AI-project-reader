import logging

from src.query_manager import QueryManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    project_folder = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    user_query = "how can I further improve the class?"

    query_manager = QueryManager()
    query_manager.ingest_relevant_files_from_project(project_folder)
    query_manager.process_query(user_query)
