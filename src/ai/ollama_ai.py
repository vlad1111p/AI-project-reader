import uuid
from typing import TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from src.database.sql_database_manager import DatabaseManager


class QueryState(TypedDict):
    query: str
    query_result: List[str]
    chat_history: List[str]


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory and graph."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.sql_db_manager = DatabaseManager()
        self.checkpointer = MemorySaver()
        self.memory_store = InMemoryStore()
        self.state_graph = self.build_graph()

    def build_graph(self):
        """Build a LangGraph workflow with state schema for query processing."""
        workflow = StateGraph(QueryState)
        workflow.add_node("process_query", self.process_query)
        workflow.add_edge(START, "process_query")
        workflow.add_edge("process_query", END)
        # TODO currently the memory is still not stored correctly, check if maybe the memory store needs to be updated before
        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)

    def process_query(self, state: QueryState) -> QueryState:
        """Node function to process the query."""
        messages = [
            SystemMessage(content="You are an AI assistant that helps with Python or Java code analysis."),
            HumanMessage(content=state["query"])
        ]

        for history_item in state["chat_history"]:
            if hasattr(history_item, 'question') and isinstance(history_item.question, str):
                messages.append(HumanMessage(content=history_item.question))

            if hasattr(history_item, 'response') and isinstance(history_item.response, str):
                messages.append(HumanMessage(content=history_item.response))

        # TODO check solution
        state["query_result"] = self.llm.invoke(messages)

        return state

    def query_ollama(self, query: str, query_result: list, project_path: str) -> str:
        """Generate a chat response using ChatOllama and store conversation in the graph."""
        thread_id = str(uuid.uuid4())

        chat_history = self.sql_db_manager.get_project_chat_context(project_path)

        serializable_chat_history = []
        for history_item in chat_history:
            if hasattr(history_item, 'question') and isinstance(history_item.question, str):
                serializable_chat_history.append(history_item.question)
            if hasattr(history_item, 'response') and isinstance(history_item.response, str):
                serializable_chat_history.append(history_item.response)

        input_data = {
            "query": query,
            "query_result": query_result,
            "chat_history": serializable_chat_history
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = self.state_graph.invoke(input_data, config=config)

        response = result.get("query_result", "No response")
        if isinstance(response, AIMessage):
            response = response.content

        self.sql_db_manager.connect_and_store_chat_context(query, response, project_path)

        return response

    def handle_query(self, user_input: str, project_path: str):
        """Main handler for user input and processing query via graph."""
        query_result = []
        return self.query_ollama(user_input, query_result, project_path)
