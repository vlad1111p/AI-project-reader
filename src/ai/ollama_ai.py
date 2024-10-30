from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import StateGraph, START

from src.model.query_state import QueryState
from src.util.tools import ToolService, analyze_files


def update_summary(query: str, response: str):
    try:
        with open('summary.txt', 'r') as file:
            previous_summary = file.read()
    except FileNotFoundError:
        previous_summary = ""

    new_entry = f"\nUser Query: {query}\nAI Response: {response}\n"
    combined_summary = previous_summary + new_entry
    if len(combined_summary) > 2000:
        combined_summary = summarize_conversation(combined_summary)
    with open('summary.txt', 'w') as file:
        file.write(combined_summary)
    return combined_summary


def summarize_conversation(content: str) -> str:
    """Summarize the conversation history to keep it concise."""
    summary_prompt = "Here is the summary so far:\n" + content + "\nPlease condense the summary to be more concise."
    messages = [HumanMessage(content=summary_prompt)]
    summary_response = ChatOllama(model="llama3.1").invoke(messages)
    return summary_response.content


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up tools."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.tool_service = ToolService()
        self.state_graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(QueryState)
        
        workflow.add_node("analyze_files", analyze_files)
        workflow.add_node("process_query", self.process_query)

        workflow.add_edge(START, "analyze_files")
        workflow.add_edge("analyze_files", "process_query")
        workflow.add_edge("process_query", END)
        return workflow.compile()

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query using chat history from summary.txt and generate a response."""
        query = state.get("query", "")
        retrieved_files = state.get("retrieved_files", [])

        try:
            with open('summary.txt', 'r') as file:
                summary_content = file.read().strip()
        except FileNotFoundError:
            summary_content = "No previous conversation history available."

        messages = [
            HumanMessage(content="Use the following as conversation history/context:"),
            HumanMessage(content=summary_content),
            HumanMessage(content=f"User Query: {query}")
        ]

        file_contents = [doc.page_content for doc in retrieved_files]
        if file_contents:
            combined_files_content = "\n".join(file_contents)
            messages.append(HumanMessage(content=f"Retrieved Files:\n{combined_files_content}"))

        response_message = self.llm.invoke(messages)

        if not response_message.content.strip():
            response_message.content = "No valid response generated."

        state["response"] = response_message.content
        messages.append(AIMessage(content=response_message.content))
        state["messages"] = messages

        # Update the summary with the latest interaction
        update_summary(query, response_message.content)

        return state

    def query_ollama(self, query: str, retrieved_files: List[Document], project_path: str) -> str:
        """Generate a chat response using ChatOllama with summary.txt as context."""
        input_data = {
            "query": query,
            "project_path": project_path,
            "retrieved_files": retrieved_files,
            "chat_history": [],
            "response": ""
        }

        final_state = None
        for state in self.state_graph.stream(input_data):
            final_state = state

        response = final_state["process_query"]["response"]

        # Ensure the latest summary update after response generation
        update_summary(query, response)

        return response
