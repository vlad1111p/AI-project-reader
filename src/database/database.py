import sqlite3


class DatabaseManager:
    def __init__(self):
        self.conn = None

    def connect_and_execute_query(self, question, response):
        self.connect()
        self.create_table()
        self.add_query(question, response)

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect("queries.db")
        return self.conn

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                response TEXT
            )
        """
        )
        self.conn.commit()

    def add_query(self, question, response):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO queries (question, response) VALUES (?, ?)",
            (question, response),
        )
        self.conn.commit()

    def get_queries(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM queries")
        return cursor.fetchall()
