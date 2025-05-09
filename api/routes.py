import logging
import shutil
import uuid
import os

from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from core.llm.openrouter import OpenRouterFreeChain, OpenRouterFreeAdapter
from core.documents.extractor import extract_text_from_file

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
DB_DIR = os.getenv("DB_DIR", "./vectordb")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# Initialize OpenRouter adapter (singleton)
openrouter_adapter = None

# Global variable to store vector databases (in memory for simplicity)
# In production, you would use persistent storage
vector_dbs = {}

class QueryRequest(BaseModel):
    query: str
    collection_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class Document(BaseModel):
    id: str
    filename: str
    content_type: str


class DocumentList(BaseModel):
    documents: List[Document]


class LLMInfo(BaseModel):
    model: str
    is_free: bool = True
    provider: str = "openrouter"


class LLMModelsList(BaseModel):
    current_model: str
    free_models: List[Dict[str, Any]]


def get_embeddings():
    """Get HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)


def get_openrouter_adapter():
    """Get or initialize the OpenRouter adapter for free models."""
    global openrouter_adapter

    if openrouter_adapter is None:
        openrouter_adapter = OpenRouterFreeAdapter(api_key=OPENROUTER_API_KEY)

    return openrouter_adapter


def process_documents(collection_id: str, file_paths: List[tuple]):
    """Process documents and create vector store."""
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        all_docs = []
        for file_path, content_type, filename in file_paths:
            text_content = extract_text_from_file(file_path, content_type)
            chunks = text_splitter.split_text(text_content)

            # Create Document objects with metadata
            docs = [
                LangchainDocument(
                    page_content=chunk,
                    metadata={"source": filename, "chunk": i}
                )
                for i, chunk in enumerate(chunks)
            ]
            all_docs.extend(docs)

        # Create vector store
        embeddings = get_embeddings()
        vector_db = FAISS.from_documents(all_docs, embeddings)

        # Save vector store
        collection_path = os.path.join(DB_DIR, collection_id)
        os.makedirs(collection_path, exist_ok=True)
        vector_db.save_local(collection_path)

        # Store in memory (would be replaced by database lookup in production)
        vector_dbs[collection_id] = vector_db

        logger.info(
            f"Successfully processed {len(all_docs)} chunks from {len(file_paths)} documents")
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}")


@router.post("/upload", response_model=Document)
async def upload_file(
    background_tasks: BackgroundTasks,
    collection_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a document and process it for querying."""
    try:
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Create collection directory if it doesn't exist
        collection_dir = os.path.join(UPLOAD_DIR, collection_id)
        os.makedirs(collection_dir, exist_ok=True)

        # Define the file path
        file_path = os.path.join(collection_dir, file.filename)

        # Determine content type
        content_type = file.content_type
        if not content_type:
            if file.filename.endswith('.pdf'):
                content_type = "application/pdf"
            elif file.filename.endswith('.md'):
                content_type = "text/markdown"
            elif file.filename.endswith('.txt'):
                content_type = "text/plain"
            else:
                raise HTTPException(
                    status_code=400, detail="Unsupported file type")

        # Save the file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process the document in the background
        background_tasks.add_task(
            process_documents,
            collection_id,
            [(file_path, content_type, file.filename)]
        )

        return Document(
            id=doc_id,
            filename=file.filename,
            content_type=content_type
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/collections/{collection_id}/documents", response_model=DocumentList)
async def list_documents(collection_id: str):
    """List all documents in a collection."""
    try:
        collection_dir = os.path.join(UPLOAD_DIR, collection_id)
        if not os.path.exists(collection_dir):
            return DocumentList(documents=[])

        documents = []
        for filename in os.listdir(collection_dir):
            file_path = os.path.join(collection_dir, filename)
            if os.path.isfile(file_path):
                content_type = "application/octet-stream"
                if filename.endswith('.pdf'):
                    content_type = "application/pdf"
                elif filename.endswith('.md'):
                    content_type = "text/markdown"
                elif filename.endswith('.txt'):
                    content_type = "text/plain"

                documents.append(Document(
                    # In production, store and retrieve actual IDs
                    id=str(uuid.uuid4()),
                    filename=filename,
                    content_type=content_type
                ))

        return DocumentList(documents=documents)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using natural language."""
    try:
        collection_id = request.collection_id
        # Check if vector DB exists in memory
        if collection_id in vector_dbs:
            vector_db = vector_dbs[collection_id]
        else:
            # Load from disk
            collection_path = os.path.join(DB_DIR, collection_id)
            if not os.path.exists(collection_path):
                raise HTTPException(
                    status_code=404, detail=f"Collection {collection_id} not found")

            embeddings = get_embeddings()
            vector_db = FAISS.load_local(collection_path, embeddings)
            vector_dbs[collection_id] = vector_db

        # Get the retriever
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # Get relevant documents
        docs = retriever.get_relevant_documents(request.query)

        # Extract sources
        sources = []
        for doc in docs:
            if doc.metadata.get("source") not in sources:
                sources.append(doc.metadata.get("source"))

        # Get context from documents
        context = [doc.page_content for doc in docs]

        # Get OpenRouter adapter for free LLMs
        adapter = get_openrouter_adapter()
        chain = OpenRouterFreeChain(adapter)

        # Generate answer
        answer = chain.run(request.query, context)

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error querying documents: {str(e)}")


@router.delete("/collections/{collection_id}/documents/{filename}")
async def delete_document(collection_id: str, filename: str):
    """Delete a document from a collection."""
    try:
        file_path = os.path.join(UPLOAD_DIR, collection_id, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail=f"Document {filename} not found")

        os.remove(file_path)

        # Rebuild vector store if needed
        collection_path = os.path.join(DB_DIR, collection_id)
        if os.path.exists(collection_path):
            # In production, you would selectively remove documents rather than rebuilding
            shutil.rmtree(collection_path)

            # If there are still documents, rebuild the vector store
            collection_dir = os.path.join(UPLOAD_DIR, collection_id)
            if os.path.exists(collection_dir) and os.listdir(collection_dir):
                file_paths = []
                for fname in os.listdir(collection_dir):
                    fpath = os.path.join(collection_dir, fname)
                    if os.path.isfile(fpath):
                        content_type = "application/octet-stream"
                        if fname.endswith('.pdf'):
                            content_type = "application/pdf"
                        elif fname.endswith('.md'):
                            content_type = "text/markdown"
                        elif fname.endswith('.txt'):
                            content_type = "text/plain"
                        file_paths.append((fpath, content_type, fname))

                if file_paths:
                    process_documents(collection_id, file_paths)

            # Remove from in-memory cache
            if collection_id in vector_dbs:
                del vector_dbs[collection_id]

        return JSONResponse(content={"message": f"Document {filename} deleted"})
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}")


@router.get("/llm/info", response_model=LLMInfo)
async def get_llm_info():
    """Get the current LLM information."""
    adapter = get_openrouter_adapter()

    return LLMInfo(
        model=adapter.model,
        is_free=True,
        provider="openrouter"
    )


@router.get("/llm/models", response_model=LLMModelsList)
async def list_free_models():
    """List all available free models."""
    adapter = get_openrouter_adapter()
    free_models = adapter.list_free_models()

    # Create a simplified list for the frontend
    model_list = []
    for model in free_models:
        model_info = {
            "id": model.get("id"),
            "name": model.get("name", model.get("id")),
            "context_length": model.get("context_length", 4096),
            "provider": model.get("id").split("/")[0] if "/" in model.get("id") else "unknown"
        }
        model_list.append(model_info)

    return LLMModelsList(
        current_model=adapter.model,
        free_models=model_list
    )


@router.post("/llm/change-model")
async def change_model(model_info: LLMInfo):
    """Change the LLM model (only to another free model)."""
    adapter = get_openrouter_adapter()

    # Make sure the model has the :free suffix if it doesn't already
    model_id = model_info.model
    if not model_id.endswith(":free") and ":free" not in model_id:
        model_id = f"{model_id}:free"

    # Set the new model
    adapter.model = model_id

    return JSONResponse(content={"message": f"Model changed to {model_id}"})
