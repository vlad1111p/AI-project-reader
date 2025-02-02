import logging

from src.service.code_analyzer import analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s",
)

if __name__ == "__main__":
    # java_project_path = "D:/work/spring projects/demoHazelcast/src/main/java/com/demoHazelcast/demoHazelcast"
    # java_user_query = "How should i improve PremierLeague class"
    # java_language = "java"
    #
    # analyze(java_user_query, java_project_path, java_language)
    python_project_path = "C:/Users/vlad/PycharmProjects/ai-project-reader"
    python_language = "python"
    python_user_query = "can you make me README for github of the whole project also mention in the usage that it can be run from main by specifying the language python or java model_type=chatgpt, model_name=gpt-4o or llama and project path "
    # python_user_query = "Refactor me the ai_analyzer.py"
    analyze(python_user_query, python_project_path, python_language, model_type="Ollama", model_name="llama3.2")
