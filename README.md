# AskMyDocs

A lightweight document Q&A application that lets you upload documents and ask questions about them in natural language.

## Overview

AskMyDocs uses vector embeddings and large language models to answer questions about your documents. The application processes uploaded documents into text chunks, creates vector embeddings for each chunk, and then uses these embeddings to find the most relevant context when answering questions.

## Architecture

AskMyDocs follows a modular architecture designed for flexibility and resource efficiency:

### API Design

The API is built around three core endpoints:

1. **Document Upload**: `POST /api/documents` - Upload and process documents
2. **Document Query**: `POST /api/query` - Ask questions about documents
3. **System Status**: `GET /api/health` - Check system status and configuration

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- sentence-transformers
- FAISS
- PyMuPDF (for PDF processing)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/askmydocs.git
   cd askmydocs
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Start the server:
   ```bash
   python app.py
   ```

### Docker Deployment

1. Build the Docker image:

   ```bash
   docker build -t askmydocs .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -e ASKMYDOCS_ENV=production askmydocs
   ```

## Hosting Options

AskMyDocs is designed to be deployable on free hosting platforms. Here are recommended options:

### Hugging Face Spaces

The most suitable option for hosting, as it provides up to 16GB RAM and is designed for ML applications:

1. Fork this repository to your GitHub account
2. Create a new Space on Hugging Face
3. Connect to your GitHub repository
4. Select "Docker" as the Space SDK
5. Configure environment variables in the Space settings

### LLM Integration

Free LLM APIs are accessed through OpenRouter:

```python
# Example of how OpenRouter is integrated
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "meta-llama/llama-4-scout:free", "messages": [...]}
)
```

## 📄 API Documentation

### Document Upload

```
POST /api/documents
```

Request:

- Form data with `file` and `session_id`

Response:

```json
{
  "id": "doc123",
  "filename": "document.pdf",
  "content_type": "application/pdf"
}
```

### Query Documents

```
POST /api/query
```

Request:

```json
{
  "session_id": "session123",
  "query": "What does the document say about X?",
  "max_results": 3
}
```

Response:

```json
{
  "answer": "According to the documents, X is...",
  "sources": ["document1.pdf", "document2.txt"],
  "context_used": true
}
```

### Error: "Model not found"

Make sure you have internet access when first loading the model, as it needs to download from Hugging Face.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
