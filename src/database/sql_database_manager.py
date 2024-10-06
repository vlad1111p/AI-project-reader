import sqlite3

from src.service.utils import hash_project_path


def connect():
    """Establish a new database connection."""
    return sqlite3.connect("queries.db")


class DatabaseManager:
    def __init__(self):
        self.conn = connect()

    def connect_and_store_chat_context(self, question: str, response: str, project_path: str):
        self.create_table()
        self.store_chat_context(
            question, response, hash_project_path(project_path)
        )

    def store_chat_context(self, question: str, response: str, project_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chat_context (question, response, project_path) VALUES (?, ?, ?)",
            (question, response, project_path),
        )
        self.conn.commit()

    def get_project_chat_context(self, project_path):
        cursor = self.conn.cursor()
        self.create_table()
        cursor.execute(
            "SELECT * FROM chat_context WHERE project_path = ?",
            (hash_project_path(project_path),),
        )

        return cursor.fetchall()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_context (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                response TEXT,
                project_path TEXT
            )
        """
        )
        self.conn.commit()


if __name__ == "__main__":
    database = DatabaseManager()
    print(
        database.get_project_chat_context(
            "C:/Users/vlad/PycharmProjects/ai-project-reader"
        )
    )
