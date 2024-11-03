from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

from src.ai.ai_code_analyzer.prompts import system_prompt, contextualize_q_prompt
from src.database.chromadb_manager import ChromaDBManager


def create_llm(model_name, model_type, temperature):
    if model_type == "llama":
        return ChatOllama(model=model_name, temperature=temperature)
    elif model_type == "chatgpt":
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class AiHandler:
    def __init__(self, model_name="llama3.2", model_type="llama", temperature=0.2, project_path=""):
        """Initialize the appropriate LLM for chat."""
        self.llm = create_llm(model_name, model_type, temperature)
        self.chroma_db = ChromaDBManager()
        self.question_answer_chain = create_stuff_documents_chain(self.llm, system_prompt())
        self.retriever = self.chroma_db.vectorstore.as_retriever(
            search_kwargs={'filter': {'project_path': project_path}})
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt()
        )
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
