from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.ai.ai_code_analyzer.prompts import summary_prompt


def update_summary(query: str, response: str):
    try:
        with open('summary.txt', 'r') as file:
            previous_summary = file.read()
    except FileNotFoundError:
        previous_summary = ""

    new_entry = f"\nUser Query: {query}\nAI Response: {response}\n"
    combined_summary = summarize_conversation(previous_summary, new_entry)
    with open('summary.txt', 'w') as file:
        file.write(combined_summary)
    return combined_summary


def summarize_conversation(previous_entry: str, new_entry: str) -> str:
    """Summarize the previous conversation and the new entry, combining them into a concise summary."""
    acceptable_size = 10000
    if len(previous_entry) + len(new_entry) > acceptable_size:
        messages = [HumanMessage(content=summary_prompt(previous_entry, new_entry, acceptable_size))]
        summary_response = ChatOllama(model="llama3.1").invoke(messages)
        return summary_response.content
    else:
        return previous_entry + new_entry
