from sqlalchemy import Column, String, Text, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.util.utils import hash_project_path

Base = declarative_base()


class ChatContext(Base):
    __tablename__ = 'chat_context'

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    project_path = Column(String, nullable=False)


class DatabaseManager:
    def __init__(self, db_url="sqlite:///queries.db"):
        """Initialize the database manager and create a session."""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def store_chat_context(self, question: str, response: str, project_path: str):
        """Store chat context in the chat_context table."""
        self.create_table()
        chat_context = ChatContext(
            question=question,
            response=response,
            project_path=project_path
        )
        self.session.add(chat_context)
        self.session.commit()

    def get_project_chat_context(self, project_path: str):
        """Retrieve chat context for a specific project path."""
        hashed_project_path = hash_project_path(project_path)
        return self.session.query(ChatContext).filter_by(project_path=hashed_project_path).all()

    def create_table(self):
        """Ensure the chat_context table exists."""
        Base.metadata.create_all(self.engine)

    def close(self):
        """Close the session."""
        self.session.close()


if __name__ == "__main__":
    database = DatabaseManager()
    context = database.get_project_chat_context(
        "C:/Users/vlad/PycharmProjects/ai-project-reader"
    )
    for row in context:
        print(f"Question: {row.question}, Response: {row.response}, Project Path: {row.project_path}")
    database.close()
