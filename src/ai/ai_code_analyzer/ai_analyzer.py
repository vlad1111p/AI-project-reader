import sqlite3
from typing import List, Literal

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

        workflow.add_node("rewrite", self.rewrite_question)
        workflow.add_node("process_query", self.process_query)

        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "process_query")
        workflow.set_finish_point("process_query")

        return workflow.compile(checkpointer=self.checkpointer)

    def rewrite_question(self, state: QueryState):
        """Transform the query to produce a better question."""
        question = state["query"]

        msg = [HumanMessage(content=f""" 
        Look at the input and try to reason about the underlying semantic intent / meaning. 
        Here is the initial question:
        ------- 
        {question} 
        ------- 
        Formulate an improved question: """)]

        response_message = self.llm.invoke(msg)
        improved_question = response_message.content.strip()

        state["query"] = improved_question
        return state

    def process_query(self, state: QueryState) -> QueryState:
        """Process the query using chat history from summary.txt and generate a response."""
        retrieved_files = state.get("retrieved_files", [])

        query = state.get("query", "")
        messages = state.get("messages", [])

        for doc in retrieved_files:
            retrieved_files_message = SystemMessage(content=supporting_code_prompt(doc.page_content))
            messages.append(retrieved_files_message)

        messages.append(HumanMessage(content=query))

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

    def grade_documents(self, state: QueryState) -> Literal["generate", "rewrite"]:
        llm_with_tool = self.llm.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {context} \n\n
                Here is the user question: {question} \n
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        retrieved_files = state.get("retrieved_files", [])

        question = state["query"]
        docs = "\n\n".join(
            doc.page_content for doc in retrieved_files) if retrieved_files else "No relevant documents found."

        scored_result = chain.invoke({"question": question, "context": docs})

        if scored_result.binary_score == "yes":
            return "generate"
        else:
            return "rewrite"

    # def agent(self, state: QueryState,retriever):
    #     """
    #     Invokes the agent model to generate a response based on the current state. Given
    #     the question, it will decide to retrieve using the retriever tool, or simply end.
    #     """
    #     print("---CALL AGENT---")
    #     messages = state["messages"]
    #     model = self.bind_tools(tools)
    #     response = model.invoke(messages)
    #     create_retriever_tool(retriever=retriever,)
    #     return {"messages": [response]}
