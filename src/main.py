import logging

from src.database.sql_database_manager import DatabaseManager
from src.service.query_manager import QueryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s",
)

if __name__ == "__main__":
    # java_project_path = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    # java_user_query = "how can I further improve the code?"
    # java_language = "java"
    # java_query_manager = QueryManager(java_language)
    # java_query_manager.process_query(java_user_query, java_project_path)
    python_project_path = "C:/Users/vlad/PycharmProjects/ai-project-reader"
    python_language = "python"
    python_user_query = "tell me how i should refactor chromadb_manager.py and write the refactored version directly "
    python_query_manager = QueryManager(python_language)
    python_query_manager.process_query(python_user_query, python_project_path)

    sql_manager = DatabaseManager()
