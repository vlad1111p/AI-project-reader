# AI Project Reader

AI Project Reader is a tool designed to process and interact with code projects using natural language. The project
enables the ingestion of source code files into ChromaDB for querying, uses AI models to process natural language
queries related to the project files, and stores both queries and AI responses in a SQL database.

## Features

- **Ingest Source Code Files**: Adds `.java`, `.xml`, and other relevant files to ChromaDB for later querying.
- **Natural Language Query**: Allows users to query their project files using natural language prompts.
- **AI Response Generation**: Uses the OllamaAI language model to generate relevant responses based on project files.
- **Conversation Summary Memory**: Custom memory implementation that tracks and summarizes the conversation for improved
  AI responses.
- **SQL Database**: Stores conversations, queries, and responses for later retrieval and use.

## Technologies Used

- **Python**: Core programming language.
- **LangChain**: Framework for chaining together different LLM models and prompts.
- **OllamaAI**: AI model used for processing queries.
- **ChromaDB**: Vector store database for embedding and retrieving project files.
- **SQLite**: SQL database for storing queries and AI responses.

## Project Structure

```bash
├── src/
│   ├── ai/
│   │   ├── ai_handler.py         # Manages AI queries and responses using OllamaAI
│   ├── database/
│   │   ├── chromadb_manager.py # Handles ChromaDB for file ingestion and querying
│   │   ├── sql_database_manager.py # Manages SQLite database interactions
│   └── service/
│       └── code_analyzer.py     # Orchestrates query processing and interactions with AI and databases
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

