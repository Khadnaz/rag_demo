"""ChromaDB vector store wrapper."""

import chromadb
from chromadb.config import Settings

# Persistent local storage
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def add_document(doc_id: str, chunks: list[str], embeddings: list[list[float]], filename: str) -> None:
    """Store document chunks and their embeddings in ChromaDB.
    
    Args:
        doc_id: Unique document identifier.
        chunks: List of text chunks.
        embeddings: Corresponding embedding vectors.
        filename: Original filename for metadata.
    """
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search(query_embedding: list[float], top_k: int = 5) -> dict:
    """Search for the most similar chunks to the query embedding.
    
    Args:
        query_embedding: The embedding vector of the user's query.
        top_k: Number of results to return.
    
    Returns:
        ChromaDB query results dict with documents, metadatas, distances.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


def list_documents() -> list[dict]:
    """List all unique documents stored in the collection.
    
    Returns:
        List of dicts with doc_id and filename.
    """
    all_metadata = collection.get()["metadatas"]
    if not all_metadata:
        return []

    seen = set()
    documents = []
    for meta in all_metadata:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            documents.append({
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
            })
    return documents


def delete_document(doc_id: str) -> bool:
    """Delete all chunks belonging to a document.
    
    Args:
        doc_id: The document ID to delete.
    
    Returns:
        True if chunks were found and deleted, False otherwise.
    """
    # Find all chunk IDs for this document
    results = collection.get(where={"doc_id": doc_id})
    if not results["ids"]:
        return False

    collection.delete(ids=results["ids"])
    return True
