import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaDBManager:
    def __init__(self, collection_name="code_collection"):
        # Initialize Chroma vector store with Hugging Face Embeddings
        self.client = chromadb.Client()
        self.vector_store = Chroma(
            collection_name=collection_name,
            client=self.client,
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    def add_file_to_db(self, file_path, file_content):
        """Adds a file and its content to ChromaDB."""
        document = Document(page_content=file_content, metadata={"source": file_path})
        self.vector_store.add_documents([document])

    def query_db(self, query_text):
        """Queries the ChromaDB using similarity search."""
        return self.vector_store.similarity_search(query_text)


# Example usage
if __name__ == "__main__":
    db_manager = ChromaDBManager()
    db_manager.add_file_to_db("example.java", "public class Example {}")
    result = db_manager.query_db("What does Example class do?")
    print(result)
