from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from src.ai.conversational_history import CustomConversationBufferMemory
from src.database.sql_database_manager import DatabaseManager


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory"""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.sql_db_manager = DatabaseManager()

        self.memory = CustomConversationBufferMemory(llm=self.llm)
        self.prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessage(
                    content="You are an assistant that provides help with python or java applications."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        self.conversation_chain = RunnableSequence(self.prompt_template | self.llm)

    def query_ollama(self, query: str, query_result, project_path: str) -> str:
        """Generate a chat response using ChatOllama and store conversation."""
        self.memory.set_db_manager_and_project(self.sql_db_manager, project_path)

        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("history", [])

        if not isinstance(chat_history, list):
            chat_history = []

        prompt = (
            f"The following is the content of files related to your query: '{query}'. "
            f"Based on your query, please provide an answer or further explanation related to the content.\n\n"
            f"{query_result}\n\n"
            "Respond based on the context of the query."
        )

        # TODO change ChatPromptTemplate to include section of content and not add prompt as query
        # https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent/ use this to change project to use tools

        response = self.conversation_chain.invoke({
            "query": prompt,
            "chat_history": chat_history
        })

        if isinstance(response, AIMessage):
            response = response.content

        self.memory.save_context({"input": query}, {"output": response})

        return response
