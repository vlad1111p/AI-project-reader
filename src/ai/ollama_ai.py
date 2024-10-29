from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import StateGraph, START

from src.database.sql_database_manager import DatabaseManager
from src.model.query_state import QueryState
from src.util.tools import ToolService


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up tools."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.tool_service = ToolService()
        self.state_graph = self.build_graph()
        self.sql_db_manager = DatabaseManager()

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("analyze_files", self.tool_service.analyze_files)
        workflow.add_node("retrieve_chat_history", self.tool_service.retrieve_chat_history)
        workflow.add_node("process_query", self.process_query)

        workflow.add_edge(START, "retrieve_chat_history")
        workflow.add_edge("retrieve_chat_history", "analyze_files")
        workflow.add_edge("analyze_files", "process_query")
        workflow.add_edge("process_query", END)

        return workflow.compile()

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query by including chat history and generating a response."""

        query = state.get("query", "")
        retrieved_files = state.get("retrieved_files", [])
        chat_history = state.get("chat_history", [])

        messages = []
        for entry in chat_history:
            if isinstance(entry, HumanMessage) or isinstance(entry, AIMessage):
                messages.append(entry)
            elif isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                if entry['role'] == 'human':
                    messages.append(HumanMessage(content=entry['content']))
                else:
                    messages.append(AIMessage(content=entry['content']))

        messages.append(HumanMessage(content=f"Query: {query}"))

        file_contents = [doc.page_content for doc in retrieved_files]
        if file_contents:
            combined_files_content = "\n".join(file_contents)
            messages.append(HumanMessage(content=f"Retrieved Files:\n{combined_files_content}"))

        print(messages)
        response_message = self.llm.invoke(messages)

        print(response_message)
        if not response_message.content.strip():
            response_message.content = "No valid response generated."

        state["response"] = response_message.content
        messages.append(AIMessage(content=response_message.content))
        state["messages"] = messages

        return state

    def query_ollama(self, query: str, retrieved_files: List[Document], project_path: str) -> str:
        """Generate a chat response using ChatOllama and include tools in the process."""
        chat_history = self.sql_db_manager.get_project_chat_context(project_path)
        messages = []
        print(chat_history)

        for entry in chat_history:
            messages.append(HumanMessage(content=entry.question))
            if entry.response:
                messages.append(AIMessage(content=entry.response))

        input_data = {
            "query": query,
            "project_path": project_path,
            "retrieved_files": retrieved_files,
            "chat_history": messages,
            "response": ""
        }

        final_state = None
        for state in self.state_graph.stream(input_data):
            final_state = state

        response = final_state["process_query"]["response"]
        self.sql_db_manager.connect_and_store_chat_context(query, response, project_path)
        return response
