from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import StateGraph, START

from src.model.query_state import QueryState
from src.util.tools import ToolService


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up tools."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.tool_service = ToolService()
        self.state_graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("analyze_files", self.tool_service.analyze_files)
        workflow.add_node("retrieve_chat_history", self.tool_service.retrieve_chat_history)
        workflow.add_node("process_query", self.process_query)

        workflow.add_edge(START, "analyze_files")
        workflow.add_edge("analyze_files", "retrieve_chat_history")
        workflow.add_edge("retrieve_chat_history", "process_query")
        workflow.add_edge("process_query", END)

        return workflow.compile()

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query and update the response in the state."""
        chat_history = state.get("chat_history", [])
        retrieved_files = state.get("retrieved_files", [])
        query = state.get("query", "")

        messages = []
        for entry in chat_history:
            if isinstance(entry, str):
                messages.append(HumanMessage(content=entry))
            elif isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                messages.append(HumanMessage(content=entry['content']) if entry['role'] == 'human' else AIMessage(
                    content=entry['content']))

        file_contents = [doc.page_content if isinstance(doc, Document) else doc["page_content"] for doc in
                         retrieved_files]
        combined_input = f"Query: {query}\nRetrieved Files:\n" + "\n".join(file_contents)
        messages.append(HumanMessage(content=combined_input))

        response_message = self.llm.invoke(messages)

        state["response"] = response_message.content if response_message else "No valid response generated."
        messages.append(AIMessage(content=state["response"]))
        state["messages"] = messages

        return state

    def query_ollama(self, query: str, retrieved_files: List[Document], project_path: str) -> str:
        """Generate a chat response using ChatOllama and include tools in the process."""
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

        return final_state["process_query"]["response"]
