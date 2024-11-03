from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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
        "You are an assistant specialized in providing structured responses for question-answering tasks. "
        "If the provided context allows, start with a **summary section** around 300 words long, "
        "explaining the main points, background, or relevant context. "
        "After the summary, clearly separate it with a line, then write a **direct answer** to the user's query. "
        "Use the phrase '### Summary' to indicate the start of the summary, and '### Answer' to start the direct "
        "answer."
        "If the context is insufficient for a detailed summary, skip the summary and go directly to the answer."
        "\n\n"
        "{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "{input}"),
        ]
    )


def contextualize_q_prompt() -> ChatPromptTemplate:
    prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
