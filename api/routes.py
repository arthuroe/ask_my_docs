import logging
import shutil
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any
import threading
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Form,
    BackgroundTasks,
    Depends,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional

from core.llm.openrouter import OpenRouterFreeChain, OpenRouterFreeAdapter
from core.documents.extractor import extract_text_from_file

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
DB_DIR = os.getenv("DB_DIR", "./vectordb")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# File upload limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILES_PER_COLLECTION = 100
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# Thread-safe storage for vector databases
class VectorDBManager:
    def __init__(self):
        self._dbs = {}
        self._lock = threading.RLock()

    def get(self, collection_id: str):
        with self._lock:
            return self._dbs.get(collection_id)

    def set(self, collection_id: str, db):
        with self._lock:
            self._dbs[collection_id] = db

    def remove(self, collection_id: str):
        with self._lock:
            self._dbs.pop(collection_id, None)


vector_db_manager = VectorDBManager()


# Singleton for OpenRouter adapter
class OpenRouterManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.adapter = None
        return cls._instance

    def get_adapter(self):
        if self.adapter is None:
            with self._lock:
                if self.adapter is None:
                    self.adapter = OpenRouterFreeAdapter(api_key=OPENROUTER_API_KEY)
        return self.adapter


openrouter_manager = OpenRouterManager()


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    collection_id: str

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    collection_id: str


class Document(BaseModel):
    id: str
    filename: str
    content_type: str
    upload_date: datetime
    size_bytes: int


class DocumentList(BaseModel):
    documents: List[Document]
    total_count: int


class LLMInfo(BaseModel):
    model: str
    is_free: bool = True
    provider: str = "openrouter"


class LLMModelsList(BaseModel):
    current_model: str
    free_models: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    message: str


# Helper functions
def get_embeddings():
    """Get HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)


def validate_file_upload(file: UploadFile, collection_id: str) -> None:
    """Validate file upload constraints."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Check file size
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    # Check collection file limit
    collection_dir = os.path.join(UPLOAD_DIR, collection_id)
    if os.path.exists(collection_dir):
        file_count = len(
            [
                f
                for f in os.listdir(collection_dir)
                if os.path.isfile(os.path.join(collection_dir, f))
            ]
        )
        if file_count >= MAX_FILES_PER_COLLECTION:
            raise HTTPException(
                status_code=413,
                detail=f"Collection has too many files. Max: {MAX_FILES_PER_COLLECTION}",
            )


def determine_content_type(filename: str) -> str:
    """Determine content type from filename."""
    if filename.endswith(".pdf"):
        return "application/pdf"
    elif filename.endswith(".md"):
        return "text/markdown"
    elif filename.endswith(".txt"):
        return "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


def process_documents(collection_id: str, file_paths: List[tuple]):
    """Process documents and create vector store."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        all_docs = []
        for file_path, content_type, filename in file_paths:
            try:
                text_content = extract_text_from_file(file_path, content_type)
                if not text_content.strip():
                    logger.warning(f"No text content extracted from {filename}")
                    continue

                chunks = text_splitter.split_text(text_content)
                docs = [
                    LangchainDocument(
                        page_content=chunk, metadata={"source": filename, "chunk": i}
                    )
                    for i, chunk in enumerate(chunks)
                ]
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue

        if not all_docs:
            logger.warning(f"No documents processed for collection {collection_id}")
            return

        embeddings = get_embeddings()
        vector_db = FAISS.from_documents(all_docs, embeddings)

        collection_path = os.path.join(DB_DIR, collection_id)
        os.makedirs(collection_path, exist_ok=True)
        vector_db.save_local(collection_path)

        vector_db_manager.set(collection_id, vector_db)
        logger.info(
            f"Successfully processed {len(all_docs)} chunks from {len(file_paths)} documents"
        )

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise


# API Routes
@router.get("/")
async def index():
    return {"message": "Welcome to AskMyDocs API"}


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    collection_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a document and process it for querying."""
    try:
        validate_file_upload(file, collection_id)

        doc_id = str(uuid.uuid4())
        collection_dir = os.path.join(UPLOAD_DIR, collection_id)
        os.makedirs(collection_dir, exist_ok=True)

        # Sanitize filename
        safe_filename = Path(file.filename).name
        file_path = os.path.join(collection_dir, safe_filename)

        # Prevent overwrite
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name_parts = Path(original_path).stem, Path(original_path).suffix
            file_path = os.path.join(
                collection_dir, f"{name_parts[0]}_{counter}{name_parts[1]}"
            )
            counter += 1

        content_type = determine_content_type(safe_filename)

        # Save file
        file_size = 0
        with open(file_path, "wb") as f:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="File too large")
            file_size = len(content)
            f.write(content)

        # Process in background
        background_tasks.add_task(
            process_documents,
            collection_id,
            [(file_path, content_type, Path(file_path).name)],
        )

        return UploadResponse(
            id=doc_id,
            filename=Path(file_path).name,
            content_type=content_type,
            message="File uploaded and processing started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/collections/{collection_id}/documents", response_model=DocumentList)
async def list_documents(collection_id: str):
    """List all documents in a collection."""
    try:
        collection_dir = os.path.join(UPLOAD_DIR, collection_id)
        if not os.path.exists(collection_dir):
            return DocumentList(documents=[], total_count=0)

        documents = []
        for filename in os.listdir(collection_dir):
            file_path = os.path.join(collection_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                try:
                    content_type = determine_content_type(filename)
                    documents.append(
                        Document(
                            id=str(uuid.uuid4()),
                            filename=filename,
                            content_type=content_type,
                            upload_date=datetime.fromtimestamp(stat.st_mtime),
                            size_bytes=stat.st_size,
                        )
                    )
                except HTTPException:
                    continue  # Skip unsupported files

        return DocumentList(documents=documents, total_count=len(documents))

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using natural language."""
    try:
        collection_id = request.collection_id

        # Get vector DB
        vector_db = vector_db_manager.get(collection_id)
        if vector_db is None:
            collection_path = os.path.join(DB_DIR, collection_id)
            if not os.path.exists(collection_path):
                raise HTTPException(
                    status_code=404, detail=f"Collection '{collection_id}' not found"
                )

            embeddings = get_embeddings()
            vector_db = FAISS.load_local(
                collection_path, embeddings, allow_dangerous_deserialization=True
            )
            vector_db_manager.set(collection_id, vector_db)

        # Retrieve relevant documents
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(request.query)

        if not docs:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                collection_id=collection_id,
            )

        # Extract sources
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in docs))
        context = [doc.page_content for doc in docs]

        # Generate answer
        adapter = openrouter_manager.get_adapter()
        chain = OpenRouterFreeChain(adapter)
        answer = chain.run(request.query, context)

        return QueryResponse(
            answer=answer, sources=sources, collection_id=collection_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/collections/{collection_id}/documents/{filename}")
async def delete_document(collection_id: str, filename: str):
    """Delete a document from a collection."""
    try:
        file_path = os.path.join(UPLOAD_DIR, collection_id, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail=f"Document '{filename}' not found"
            )

        os.remove(file_path)

        # Clean up vector store
        collection_path = os.path.join(DB_DIR, collection_id)
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
            vector_db_manager.remove(collection_id)

            # Rebuild if other documents exist
            collection_dir = os.path.join(UPLOAD_DIR, collection_id)
            if os.path.exists(collection_dir) and os.listdir(collection_dir):
                file_paths = []
                for fname in os.listdir(collection_dir):
                    fpath = os.path.join(collection_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            content_type = determine_content_type(fname)
                            file_paths.append((fpath, content_type, fname))
                        except HTTPException:
                            continue

                if file_paths:
                    process_documents(collection_id, file_paths)

        return JSONResponse(
            content={"message": f"Document '{filename}' deleted successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/llm/info", response_model=LLMInfo)
async def get_llm_info():
    """Get the current LLM information."""
    try:
        adapter = openrouter_manager.get_adapter()
        return LLMInfo(model=adapter.model, is_free=True, provider="openrouter")
    except Exception as e:
        logger.error(f"Error getting LLM info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/llm/models", response_model=LLMModelsList)
async def list_free_models():
    """List all available free models."""
    try:
        adapter = openrouter_manager.get_adapter()
        free_models = adapter.list_free_models()

        model_list = []
        for model in free_models:
            model_info = {
                "id": model.get("id"),
                "name": model.get("name", model.get("id")),
                "context_length": model.get("context_length", 4096),
                "provider": (
                    model.get("id", "").split("/")[0]
                    if "/" in model.get("id", "")
                    else "unknown"
                ),
            }
            model_list.append(model_info)

        return LLMModelsList(current_model=adapter.model, free_models=model_list)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/llm/change-model")
async def change_model(model_info: LLMInfo):
    """Change the LLM model (only to another free model)."""
    try:
        adapter = openrouter_manager.get_adapter()

        model_id = model_info.model
        if not model_id.endswith(":free") and ":free" not in model_id:
            model_id = f"{model_id}:free"

        adapter.model = model_id
        return JSONResponse(content={"message": f"Model changed to {model_id}"})

    except Exception as e:
        logger.error(f"Error changing model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
