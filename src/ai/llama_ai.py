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

from src.ai.conversational_history import CustomConversationBufferMemory
from src.database.sql_database_manager import DatabaseManager


class OllamaAI:
    def __init__(self, model_name="llama3.1", temperature=0.5):
        """Initialize the Ollama model for chat and set up memory"""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.sql_db_manager = DatabaseManager()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

    def query_ollama(self, query: str, user_input: str, project_path: str) -> str:
        """Generate a chat response using ChatOllama and store conversation."""
        self.memory.set_db_manager_and_project(self.sql_db_manager, project_path)

        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("history", [])

        if not isinstance(chat_history, list):
            chat_history = []

        print("-------------------------chat history")
        print(memory_vars)

        response = self.conversation_chain.invoke({
            "query": user_input,
            "chat_history": chat_history
        })

        if isinstance(response, AIMessage):
            response = response.content

        self.memory.save_context({"input": query}, {"output": response})

        return response

    def truncate(self, text: str, max_tokens: int = 512) -> str:
        """Truncate the input text to fit within the model's token limit."""
        tokens = self.tokenizer.encode(text, return_tensors="pt").tolist()
        if len(tokens[0]) > max_tokens:
            truncated_tokens = tokens[0][:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text
        return text
