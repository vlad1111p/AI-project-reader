def history_prompt(summary_content: str) -> str:
    return (
        "This content serves as the historical context for the current conversation.\n\n"
        f"{summary_content}\n\n"
        "Please use this information as the conversation history and context for responding to the user query."
    )


def user_query_prompt(query: str) -> str:
    return (
        f"Primary User Query: {query}. "
        f"Please address this query directly and only refer to the provided code context if it "
        f"enhances or supports your response to the user’s question."
    )


def supporting_code_prompt(combined_files_content: str) -> str:
    return (
        f"Reference Files (for context only):\n{combined_files_content}\n"
        f"Please focus on the user query above. Use the code details only to clarify or support "
        f"the main response as needed."
    )
