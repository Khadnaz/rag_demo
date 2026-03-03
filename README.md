# RAG-Powered Personal Knowledge Assistant

A full-stack application that lets users upload documents (PDF/TXT), ask questions, and receive contextual answers powered by OpenAI and ChromaDB.

## Architecture

```
┌─────────────┐     HTTP     ┌──────────────┐     ┌──────────┐
│  React App  │ ◄──────────► │   FastAPI     │ ──► │ ChromaDB │
│  (Port 3000)│              │  (Port 8000)  │     │ (local)  │
└─────────────┘              └──────┬───────┘     └──────────┘
                                    │
                              ┌─────▼──────┐
                              │  OpenAI API │
                              │ Embeddings  │
                              │ + Chat GPT  │
                              └────────────┘
```

**Flow:** Upload → Extract text → Chunk → Embed (OpenAI) → Store (ChromaDB) → Query → Retrieve similar chunks → Generate answer (GPT-3.5-turbo)

## Tech Stack

| Layer     | Technology |
|-----------|-----------|
| Backend   | FastAPI, Python 3.9+ |
| Frontend  | React (CRA) |
| Vector DB | ChromaDB (persistent, local) |
| AI        | OpenAI `text-embedding-3-small` + `gpt-3.5-turbo` |
| File processing | PyPDF2 |
| API client | Axios |

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 16+
- OpenAI API key

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Run:**
```bash
uvicorn main:app --reload
```
Backend available at http://localhost:8000 (docs at http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
npm install
npm start
```
Frontend available at http://localhost:3000

### 3. Run Tests

```bash
cd backend
python -m pytest tests/ -v
```

## Architecture Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector DB | ChromaDB | Built-in persistence, pip-installable, no external services needed |
| Embedding model | `text-embedding-3-small` | Cost-effective, recommended in assignment |
| Chat model | `gpt-3.5-turbo` | Fast, affordable, sufficient for demo |
| Chunking strategy | Fixed-size (500 chars) with 50-char overlap | Simple and effective; overlap prevents context loss at boundaries |
| State management | React `useState` | No global state needed for this scope |
| No auth | — | Explicitly excluded by assignment requirements |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload PDF/TXT, auto-index into vector store |
| `POST` | `/chat` | Ask a question, get RAG-powered answer |
| `GET` | `/documents` | List all uploaded documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document and its vectors |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key | for local usage rename .env.example to .env and add your OPENAI_API_KEY

