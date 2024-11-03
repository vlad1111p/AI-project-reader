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
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "{input}"),
        ]
    )


def contextualize_q_prompt():
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
