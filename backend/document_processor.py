"""Document processing utilities: text extraction and chunking."""

from PyPDF2 import PdfReader
import io


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text content from a PDF or TXT file.
    
    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to determine file type).
    
    Returns:
        Extracted text as a string.
    
    Raises:
        ValueError: If the file type is not supported.
    """
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if extension == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    elif extension == "txt":
        return file_bytes.decode("utf-8").strip()
    else:
        raise ValueError(f"Unsupported file type: .{extension}. Only PDF and TXT are supported.")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: The full text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
    
    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
