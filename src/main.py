import logging

from src.service.code_analyzer import CodeAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s",
)

if __name__ == "__main__":
    java_project_path = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    java_user_query = "what is my name"
    java_language = "java"
    java_query_manager = CodeAnalyzer(java_language)
    java_query_manager.analyze(java_user_query, java_project_path)
    # python_project_path = "C:/Users/vlad/PycharmProjects/ai-project-reader"
    # python_language = "python"
    # python_user_query = "what is my name"
    # # python_user_query = "Refactor me the ai_analyzer.py"
    # python_query_manager = CodeAnalyzer(python_language)
    # python_query_manager.analyze(python_user_query, python_project_path)
