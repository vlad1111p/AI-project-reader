# AI Project Reader

AI Project Reader is a tool designed to analyze code projects using AI models. It supports both Java and Python projects and provides insights or improvements based on user queries. The tool leverages advanced AI models like ChatGPT and LLaMA to process and respond to queries in the context of the project's codebase.

## Features

- **Multi-language Support**: Works with Java and Python projects.
- **AI-Powered Analysis**: Utilizes AI models to provide code insights and improvements.
- **ChromaDB Integration**: Stores and retrieves document embeddings for efficient query processing.
- **Extensible Architecture**: Easily add support for new languages or AI models.

## Components

- **FileReader**: Reads and processes files from the project directory.
- **ChromaDBManager**: Manages document embeddings using ChromaDB.
- **AiProjectAnalyzer**: Processes user queries and generates responses using AI models.
- **Logging**: Provides detailed logs for monitoring and debugging.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-project-reader.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-project-reader
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To analyze a project, run the following command:

```bash
python src/service/code_analyzer.py
```

Modify the script to specify the project path, language, and query.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License.

---

Feel free to customize the README further based on your specific project details and requirements.
