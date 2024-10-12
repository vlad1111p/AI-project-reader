import uuid
from typing import TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.store.memory import InMemoryStore
from typing_extensions import Annotated

from src.database.sql_database_manager import DatabaseManager


class QueryState(TypedDict):
    query: str
    retrieved_files: List[str]
    chat_history: List[str]
    messages: Annotated[list, add_messages]


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
        """Node function to process the query and analyze the query_result."""
        messages = [
            SystemMessage(content="You are an AI assistant that helps with Python or Java code analysis."),
            HumanMessage(content=state["query"])
        ]

        if state["retrieved_files"]:
            messages.append(SystemMessage(content="Here is the code/data that needs to be analyzed:"))
            for result in state["retrieved_files"]:
                if isinstance(result, str):
                    messages.append(HumanMessage(content=result))
                elif hasattr(result, 'page_content') and isinstance(result.page_content, str):
                    messages.append(HumanMessage(content=result.page_content))
                else:
                    messages.append(HumanMessage(content=str(result)))
        for history_item in state["chat_history"]:
            if hasattr(history_item, 'question') and isinstance(history_item.question, str):
                messages.append(HumanMessage(content=history_item.question))
            if hasattr(history_item, 'response') and isinstance(history_item.response, str):
                messages.append(HumanMessage(content=history_item.response))

        state["retrieved_files"] = self.llm.invoke(messages)

        return state

    def query_ollama(self, query: str, retrieved_files: list, project_path: str) -> str:
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
            "retrieved_files": retrieved_files,
            "chat_history": serializable_chat_history
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = self.state_graph.invoke(input_data, config=config)

        response = result.get("retrieved_files", "No response")
        if isinstance(response, AIMessage):
            response = response.content

        self.sql_db_manager.connect_and_store_chat_context(query, response, project_path)

        return response
