import sqlite3
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from src.ai.ai_code_analyzer.prompts import supporting_code_prompt, query_prompt
from src.domain.query_state import QueryState


class AiAnalyze:

    def __init__(self, llm):
        self.llm = llm
        self.conn_string = "checkpoints.sqlite"
        self.connection = sqlite3.connect(self.conn_string, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.state_graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("process_query", self.process_query)

        workflow.set_entry_point("process_query")
        workflow.set_finish_point("process_query")

        return workflow.compile(checkpointer=self.checkpointer)

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query using chat history from summary.txt and generate a response."""
        query = state.get("query", "")
        retrieved_files = state.get("retrieved_files", [])

        messages = state.get("messages", [])

        for doc in retrieved_files:
            retrieved_files_message = SystemMessage(content=supporting_code_prompt(doc.page_content))
            messages.append(retrieved_files_message)
            
        messages.append(HumanMessage(content=query_prompt(query)))

        response_message = self.llm.invoke(messages)

        if not response_message.content.strip():
            response_message.content = "No valid response generated."

        state["response"] = response_message.content
        messages.append(AIMessage(content=response_message.content))
        state["messages"] = messages

        return state

    def query_model(self, query: str, retrieved_files: List[Document], project_path: str) -> str:
        """Generate a chat response using ChatOllama with summary.txt as context."""
        input_data = {
            "query": query,
            "project_path": project_path,
            "retrieved_files": retrieved_files,
            "response": ""
        }

        final_state = None
        thread = {"configurable": {"thread_id": project_path}}

        for state in self.state_graph.stream(input_data, thread):
            final_state = state
        response = final_state["process_query"]["response"]

        return response
