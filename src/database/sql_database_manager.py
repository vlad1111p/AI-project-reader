import sqlite3

from src.service.utils import hash_project_path


def store_chat_context(conn, question, response, project_path):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_context (question, response,project_path) VALUES (?, ?, ?)",
        (question, response, project_path),
    )
    conn.commit()


def create_table(conn):
    cursor = conn.cursor()
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
    conn.commit()


class DatabaseManager:
    def __init__(self):
        self.conn = None

    def connect_and_execute_query(self, question, response, project_path):
        with self.connect() as conn:
            create_table(conn)
            store_chat_context(
                conn, question, response, hash_project_path(project_path)
            )

    def connect(self):
        if not self.conn:
            return sqlite3.connect("queries.db")
        else:
            return self.conn

    def get_project_chat_context(self, project_path):
        with self.connect() as conn:
            cursor = conn.cursor()
            create_table(conn)
            cursor.execute(
                "SELECT * FROM chat_context WHERE project_path = ?",
                (hash_project_path(project_path),),
            )

            return cursor.fetchall()


if __name__ == "__main__":
    database = DatabaseManager()
    print(
        database.get_project_chat_context(
            "C:/Users/vlad/PycharmProjects/ai-project-reader"
        )
    )
