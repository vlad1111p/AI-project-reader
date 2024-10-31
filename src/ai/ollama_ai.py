from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import StateGraph, START

from src.model.query_state import QueryState
from src.util.content_summary import update_summary


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up tools."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.state_graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("process_query", self.process_query)

        workflow.add_edge(START, "process_query")
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
        update_summary(query, response)
        return response
