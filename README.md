# AskMyDocs

A lightweight, production-ready document Q&A application that lets you upload documents and ask questions about them using natural language and free LLM models.

## =€ Features

- **Multi-format Support**: PDF, Markdown, and plain text documents
- **Free LLM Models**: Uses OpenRouter's free models (no API costs)
- **Vector Search**: FAISS-powered semantic search with HuggingFace embeddings
- **Production Ready**: Security features, file validation, and robust error handling
- **Developer Friendly**: Full type hints, linting, and pre-commit hooks
- **Thread Safe**: Proper concurrent request handling

## <× Architecture

AskMyDocs follows a modular, production-ready architecture:

### Core Components
- **FastAPI Backend**: RESTful API with comprehensive validation
- **Vector Store**: FAISS for efficient similarity search
- **LLM Integration**: OpenRouter adapter with automatic model selection
- **Document Processing**: Robust text extraction with custom error handling
- **React Frontend**: Modern UI built with Vite

### Security Features
- Environment-based CORS configuration
- File upload validation (size, type, collection limits)
- Proper secret management
- Input sanitization and validation

## =€ Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/yourusername/askmydocs.git
   cd askmydocs
   ```

2. **Backend setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

3. **Environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenRouter API key
   ```

4. **Frontend setup** (optional):
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Start the server**:
   ```bash
   python app.py
   ```

## =à Development

### Code Quality Tools
```bash
# Format code
black . && isort .

# Run linting
flake8 api/ core/

# Type checking
mypy api/ core/

# Set up pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Run with Development Tools
```bash
# Start with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Frontend development server
cd frontend && npm run dev
```

## =á API Endpoints

### Document Management
- `POST /upload` - Upload documents to a collection
- `GET /collections/{id}/documents` - List documents in collection
- `DELETE /collections/{id}/documents/{filename}` - Delete document

### Querying
- `POST /query` - Ask questions about documents
- `GET /health` - System health check

### LLM Management
- `GET /llm/info` - Get current model info
- `GET /llm/models` - List available free models
- `POST /llm/change-model` - Switch LLM model

## =3 Docker Deployment

### Development
```bash
docker build -t askmydocs .
docker run -p 8000:8000 askmydocs
```

### Production
```bash
docker run -p 8000:8000 -e ASKMYDOCS_ENV=production askmydocs
```

##  Hosting Options

### Hugging Face Spaces (Recommended)
Perfect for ML applications with up to 16GB RAM:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your GitHub repository
4. Select "Docker" as the Space SDK
5. Configure environment variables

### Other Platforms
- **Railway**: Good for quick deployments
- **Fly.io**: Global deployment with good free tier
- **DigitalOcean App Platform**: Managed container deployment

## =Ë Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional
APP_NAME=AskMyDocs
APP_URL=http://localhost:8000
HF_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
UPLOAD_DIR=./uploads
DB_DIR=./vectordb
ASKMYDOCS_ENV=development
```

## =' Configuration

### File Upload Limits
- Max file size: 50MB
- Max files per collection: 100
- Supported formats: PDF, Markdown (.md), Plain text (.txt)

### LLM Models
- Automatic selection of best available free model
- Fallback hierarchy: Llama 4 ’ Gemini ’ Mistral ’ DeepSeek
- Real-time model switching support

## = Troubleshooting

### Common Issues

**"Model not found" error**:
- Ensure internet connection for initial model download
- Check OpenRouter API key is valid

**File upload fails**:
- Verify file size is under 50MB
- Check file format is supported (.pdf, .md, .txt)

**Memory issues**:
- Reduce chunk size in document processing
- Use smaller embedding models for limited RAM

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Run quality checks: `black . && isort . && flake8 api/ core/`
5. Submit a pull request

## =Ü License

This project is licensed under the MIT License - see the LICENSE file for details.

## =O Acknowledgments

- OpenRouter for free LLM access
- HuggingFace for embedding models
- FAISS for efficient vector search