import logging

from src.service.code_analyzer import CodeAnalyzer

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
    # python_user_query = "what was my previous message."
    python_user_query = "tell me how to improve the code and also explain what you did "
    python_query_manager = CodeAnalyzer(python_language)
    python_query_manager.analyze(python_user_query, python_project_path)
