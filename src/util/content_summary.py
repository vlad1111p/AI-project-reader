from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def update_summary(query: str, response: str):
    try:
        with open('summary.txt', 'r') as file:
            previous_summary = file.read()
    except FileNotFoundError:
        previous_summary = ""

    new_entry = f"\nUser Query: {query}\nAI Response: {response}\n"
    combined_summary = summarize_conversation(previous_summary + new_entry)
    with open('summary.txt', 'w') as file:
        file.write(combined_summary)
    return combined_summary


def summarize_conversation(content: str) -> str:
    """Summarize the conversation history to keep it concise."""
    acceptable_size = 10000
    if len(content) > acceptable_size:
        content = summarize_conversation(content)
    summary_prompt = (f"Here is the summary so far:\n "
                      f"{content} "
                      f"\n Please condense the summary to be more concise."
                      f"The content must have up to {acceptable_size / 2} letters"
                      f"Do not add the code to the summary")

    messages = [HumanMessage(content=summary_prompt)]
    summary_response = ChatOllama(model="llama3.1").invoke(messages)
    return summary_response.content
