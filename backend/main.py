"""FastAPI application for RAG-powered knowledge assistant."""

import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from document_processor import extract_text, chunk_text
from embeddings import get_embeddings
from vector_store import add_document, search, list_documents, delete_document

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ──────────────────────────────────────────────
# Upload endpoint
# ──────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT document, chunk it, embed it, and store in ChromaDB."""
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    extension = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if extension not in ("pdf", "txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Read and process
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_text(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text content could be extracted from the file.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text chunking produced no chunks.")

    # Generate embeddings and store
    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    doc_id = str(uuid.uuid4())
    add_document(doc_id, chunks, embeddings, file.filename)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "message": "Document uploaded and indexed successfully.",
    }


# ──────────────────────────────────────────────
# Chat endpoint
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful knowledge assistant. Answer the user's question based ONLY on the provided context. 
If the context doesn't contain enough information to answer, say so honestly.
Be concise and accurate."""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using RAG: retrieve relevant chunks, then generate a response."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Embed the question
    try:
        query_embedding = get_embeddings([request.question])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")

    # Retrieve relevant chunks
    results = search(query_embedding, top_k=5)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return ChatResponse(
            answer="No documents have been uploaded yet. Please upload some documents first.",
            sources=[],
        )

    # Build context
    context = "\n\n---\n\n".join(documents)
    source_files = list(set(m.get("filename", "unknown") for m in metadatas))

    # Generate answer with OpenAI
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

    return ChatResponse(answer=answer, sources=source_files)


# ──────────────────────────────────────────────
# Document management endpoints
# ──────────────────────────────────────────────

@app.get("/documents")
async def get_documents():
    """List all uploaded documents."""
    return {"documents": list_documents()}


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a document and all its chunks from the vector store."""
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
