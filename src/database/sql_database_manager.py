import sqlite3


def add_query(conn, question, response):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (question, response) VALUES (?, ?)",
        (question, response),
    )
    conn.commit()


def create_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            response TEXT
        )
    """
    )
    conn.commit()


class DatabaseManager:
    def __init__(self):
        self.conn = None

    def connect_and_execute_query(self, question, response):
        with self.connect() as conn:
            create_table(conn)
            add_query(conn, question, response)

    def connect(self):
        if not self.conn:
            return sqlite3.connect("queries.db")
        else:
            return self.conn

    def get_queries(self):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queries")
            return cursor.fetchall()
