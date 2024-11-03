from langchain_core.prompts import ChatPromptTemplate


def query_prompt(query: str) -> str:
    return (
        f"[NEW QUERY] User Query:"
        f"{query}"
        "Please prioritize the new query while using the context if relevant.")


def supporting_code_prompt(combined_files_content: str) -> str:
    return (
        f"Reference File (for context only):"
        f"{combined_files_content}"
        f"Please focus on the user query above. Use the code details only to clarify or support "
        f"the main response as needed."
    )


def summary_prompt(previous_entry: str, new_entry: str, acceptable_size: int):
    return (
        f"The following is a summary task. First, summarize the 'Previous Entry' section. "
        f"Then incorporate key points from the 'New Entry' to produce a combined, concise summary."
        f"Previous Entry:\n{previous_entry}\n\nNew Entry:\n{new_entry}"
        f"\n Please condense the summary to be more concise."
        f"The content must have up to {acceptable_size} letters"
        f"Do not add the code to the summary"
    )


def system_prompt() -> ChatPromptTemplate:
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    human_prompt = (
        "User Query:"
        "{query}"
        "Please prioritize the user query while using the context if relevant.")

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", human_prompt),
        ]
    )
