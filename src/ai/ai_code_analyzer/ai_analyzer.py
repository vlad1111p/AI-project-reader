import sqlite3
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from src.ai.ai_code_analyzer.prompts import supporting_code_prompt
from src.domain.query_state import QueryState
from src.service.grade import Grade


class AiAnalyze:

    def __init__(self, llm):
        self.llm = llm
        self.conn_string = "checkpoints.sqlite"
        self.connection = sqlite3.connect(self.conn_string, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.state_graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("filter_documents", self.filter_documents)
        # workflow.add_node("rewrite", self.rewrite_question)
        workflow.add_node("process_query", self.process_query)

        workflow.set_entry_point("filter_documents")
        workflow.add_edge("filter_documents", "process_query")
        workflow.set_finish_point("process_query")

        return workflow.compile(checkpointer=self.checkpointer)

    # def rewrite_question(self, state: QueryState):
    #     """Transform the query to produce a better question."""
    #     question = state["query"]
    #
    #     msg = [HumanMessage(content=f"""
    #     Look at the input and try to reason about the underlying semantic intent / meaning.
    #     Here is the initial question:
    #     -------
    #     {question}
    #     -------
    #     Formulate an improved question: """)]
    #
    #     response_message = self.llm.invoke(msg)
    #     improved_question = response_message.content.strip()
    #
    #     state["query"] = improved_question
    #     return state

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query using chat history from summary.txt and generate a response."""
        retrieved_files = state.get("retrieved_files", [])

        query = state.get("query", "")
        messages = state.get("messages", [])
        project_path = state.get("project_path", "")

        for doc in retrieved_files:
            retrieved_files_message = SystemMessage(content=supporting_code_prompt(doc.page_content))
            messages.append(retrieved_files_message)

        messages.append(HumanMessage(content=query))

        thread = {"configurable": {"thread_id": project_path}}

        response_message = self.llm.invoke(messages, thread)

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

    def filter_documents(self, state: QueryState) -> QueryState:
        """ Filters documents in the QueryState to retain only those relevant to the user's query."""
        llm_with_tool = self.llm.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. 
            Here is the retrieved document: 
            -------
            {context} 
            -------
            Here is the user question: {question} 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
            Provide a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        retrieved_files = state.get("retrieved_files", [])
        question = state["query"]
        relevant_docs = []

        for doc in retrieved_files:
            scored_result = chain.invoke({"question": question, "context": doc.page_content})

            if scored_result.binary_score == "yes":
                relevant_docs.append(doc)

        state["retrieved_files"] = relevant_docs

        return state
