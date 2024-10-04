import logging

from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from transformers import GPT2Tokenizer

from src.database.sql_database_manager import DatabaseManager


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory"""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.sql_db_manager = DatabaseManager()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessage(
                    content="You are an assistant that provides help with python or java applications."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        self.conversation_chain = RunnableSequence(
            self.prompt_template | self.llm
        )

    def query_ollama(self, user_input: str, project_path: str) -> str:
        """Generate a chat response using ChatOllama and store conversation"""

        truncated_input = self.truncate(user_input)

        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])

        response = self.conversation_chain.invoke({
            "query": truncated_input,
            "chat_history": chat_history
        })

        logging.info("-----------------------------------------------")
        logging.info(truncated_input)
        logging.info(response)

        if isinstance(response, AIMessage):
            response = response.content
            
        self.sql_db_manager.connect_and_execute_query(
            truncated_input, response, project_path
        )

        return response

    def truncate(self, text: str, max_tokens: int = 512) -> str:
        """Truncate the input text to fit within the model's token limit."""
        tokens = self.tokenizer.encode(text, return_tensors="pt").tolist()
        if len(tokens[0]) > max_tokens:
            truncated_tokens = tokens[0][:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text
        return text
