import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.ai.ai_code_analyzer.prompts import system_prompt, contextualize_q_prompt
from src.database.chromadb_manager import ChromaDBManager

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_llm(model_name, model_type, temperature):
    if model_type == "Ollama":
        return ChatOllama(model=model_name,
                          temperature=temperature)
    elif model_type == "Chatgpt":
        return ChatOpenAI(model=model_name,
                          temperature=temperature,
                          openai_api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class AiHandler:
    def __init__(self, model_name, model_type, temperature=0.2, project_path=""):
        """Initialize the appropriate LLM for chat."""
        self.llm = create_llm(model_name, model_type, temperature)
        self.chroma_db = ChromaDBManager()
        self.question_answer_chain = create_stuff_documents_chain(self.llm, system_prompt())
        self.retriever = self.chroma_db.vectorstore.as_retriever(
            search_kwargs={'filter': {'project_path': project_path}, 'k': 20})
        self.history_aware_retriever = create_history_aware_retriever(self.llm,
                                                                      self.retriever,
                                                                      contextualize_q_prompt())
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    # def retrieve_documents(self, query):
    #     """Retrieve documents and log queried files"""
    #     retrieved_docs = self.retriever.invoke(query)
    #
    #     if retrieved_docs:
    #         logging.info("Queried Files:")
    #         for doc in retrieved_docs:
    #             file_id = doc.metadata.get('id', 'Unknown File')
    #             logging.info(f"Retrieved: {file_id}")
    #     else:
    #         logging.info("No documents retrieved for query.")
    #
    #     return retrieved_docs
