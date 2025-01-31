from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.ai.ai_handler import AiHandler
from src.domain.query_state import QueryState


class AiProjectAnalyzer:

    def __init__(self, project_path: str, model_name: str, model_type: str):
        self.ai_handler = AiHandler(project_path=project_path, model_name=model_name, model_type=model_type)
        self.state_graph = self.build_graph()

    def build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(QueryState)

        workflow.add_node("process_query", self.process_query)

        workflow.set_entry_point("process_query")
        workflow.set_finish_point("process_query")

        return workflow.compile(checkpointer=MemorySaver())

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query using chat history from summary.txt and generate a response."""
        query = state.get("query", "")
        project_path = state.get("project_path", "")

        llm_input = {
            "input": query,
            "configurable": {"thread_id": project_path}
        }

        response = self.ai_handler.rag_chain.invoke(llm_input)
        state["response"] = response["answer"]

        return state

    def query_model(self, query: str, project_path: str, language: str) -> str:
        """Generate a chat response using ChatOllama with summary.txt as context."""
        input_data = {
            "query": query,
            "project_path": project_path,
            "response": ""
        }

        self.ai_handler.chroma_db.add_files_from_project_to_db(project_path, language)

        final_state = None
        thread = {"configurable": {"thread_id": project_path}}

        for state in self.state_graph.stream(input_data, thread):
            final_state = state
        response = final_state["process_query"]["response"]

        return response
