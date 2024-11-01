from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.ai.ai_code_analyzer.prompts import supporting_code_prompt, user_query_prompt, history_prompt
from src.domain.query_state import QueryState
from src.util.content_summary import update_summary


class AiAnalyze:
    def __init__(self, llm):
        self.llm = llm
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

        history_message = HumanMessage(content=history_prompt(summary_content))
        summary_message = HumanMessage(content=summary_content)
        user_query_message = HumanMessage(content=user_query_prompt(query))

        messages = [history_message, summary_message, user_query_message]
        state["messages"].extend(messages)

        file_contents = [doc.page_content for doc in retrieved_files]
        retrieved_files_message = None
        if file_contents:
            combined_files_content = "\n".join(file_contents)
            retrieved_files_message = HumanMessage(content=supporting_code_prompt(combined_files_content))
            messages.append(retrieved_files_message)

        response_message = self.llm.invoke(messages)

        if not response_message.content.strip():
            response_message.content = "No valid response generated."

        state["response"] = response_message.content
        messages.append(AIMessage(content=response_message.content))
        state["messages"] = messages

        if retrieved_files_message in messages:
            messages.remove(retrieved_files_message)
        for msg in [history_message, summary_message, user_query_message]:
            if msg in state["messages"]:
                state["messages"].remove(msg)

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
        for state in self.state_graph.stream(input_data):
            final_state = state
        response = final_state["process_query"]["response"]
        update_summary(query, response)
        return response
