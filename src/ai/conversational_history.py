from typing import Any, Dict, List

from langchain.memory.chat_memory import InMemoryChatMessageHistory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import PrivateAttr


def format_message(message: BaseMessage) -> str:
    """Format a message for history."""
    if isinstance(message, SystemMessage):
        role = "System"
    elif isinstance(message, HumanMessage):
        role = "Human"
    elif isinstance(message, AIMessage):
        role = "AI"
    else:
        role = "Unknown"
    return f"{role}: {message.content}"


def format_db_history(chat_history):
    """Format chat history from the DB into the format required by memory."""
    formatted_history = []
    for row in chat_history:
        question = row[0]
        response = row[1]

        if isinstance(question, str) and question.strip():
            formatted_history.append(HumanMessage(content=question))
        if isinstance(response, str) and response.strip():
            formatted_history.append(AIMessage(content=response))

    return formatted_history


class CustomConversationBufferMemory(ConversationSummaryBufferMemory):
    """Custom buffer with database support for loading memory and saving summaries."""

    _db_manager: Any = PrivateAttr()
    _project_path: str = PrivateAttr()

    conversation_history: List[BaseMessage] = []

    def set_db_manager_and_project(self, db_manager, project_path: str):
        """Set the database manager and project path."""
        self._db_manager = db_manager
        self._project_path = project_path
        self.load_db_history()

    def load_db_history(self):
        """Load history from the database and convert it into memory."""
        chat_history = self._db_manager.get_project_chat_context(self._project_path)

        # print("------------chat_history------------")
        # print(chat_history)
        formatted_history = format_db_history(chat_history)
        self.set_conversation_history(formatted_history)

    def set_conversation_history(self, history: List[BaseMessage]) -> None:
        """Set the conversation history in memory."""
        self.conversation_history = history
        self.chat_memory = InMemoryChatMessageHistory(messages=self.conversation_history)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from the conversation to buffer and database."""
        new_input = inputs.get("input", "")
        new_output = outputs.get("output", "")
        self.conversation_history.append(HumanMessage(content=new_input))
        self.conversation_history.append(AIMessage(content=new_output))

        self.chat_memory = InMemoryChatMessageHistory(messages=self.conversation_history)
        super().save_context(inputs, outputs)
        self.conversation_history = self.chat_memory.messages
        self.prune()
        #
        # print("---------Buffer---------------")
        # print(self.moving_summary_buffer)
        self.save_to_db(new_input, new_output)

    def save_to_db(self, question: str, response: str):
        """Save the latest question and response to the database."""
        self._db_manager.connect_and_store_chat_context(question, response, self._project_path)

    def load_memory_variables(self, variables: Dict) -> Dict[str, Any]:
        """Load memory variables, including the conversation history."""
        history = "\n".join(
            [format_message(message) for message in self.conversation_history]
        )
        return {"history": history}
