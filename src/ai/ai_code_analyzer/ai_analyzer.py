import sqlite3
from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from src.ai.ai_code_analyzer.prompts import supporting_code_prompt, system_prompt
from src.database.chromadb_manager import ChromaDBManager
from src.domain.query_state import QueryState
from src.service.grade import Grade


def setup_checkpointer():
    conn_string = "checkpoints.sqlite"
    connection = sqlite3.connect(conn_string, check_same_thread=False)
    return SqliteSaver(connection)


class AiAnalyze:

    def __init__(self, llm):
        self.llm = llm
        self.checkpointer = setup_checkpointer()
        self.state_graph = self.build_graph()
        self.chroma_db = ChromaDBManager()
        self.question_answer_chain = create_stuff_documents_chain(self.llm, system_prompt())
        self.retriever = self.chroma_db.vectorstore.as_retriever()
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        self.history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )

    def build_graph(self):
        workflow = StateGraph(QueryState)

        workflow.add_node("filter_documents", self.filter_documents)
        workflow.add_node("process_query", self.process_query)

        workflow.set_entry_point("filter_documents")
        workflow.add_edge("filter_documents", "process_query")
        workflow.set_finish_point("process_query")

        return workflow.compile(checkpointer=self.checkpointer)

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
