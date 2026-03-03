"""Unit tests for document_processor module."""

import pytest
from document_processor import extract_text, chunk_text


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_basic_chunking(self):
        """Test that text is split into correct number of chunks with overlap."""
        text = "A" * 1000  # 1000 characters
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # With chunk_size=500 and overlap=50, step = 450
        # Chunks start at: 0, 450, 900 → 3 chunks
        assert len(chunks) == 3
        assert all(len(c) <= 500 for c in chunks)

    def test_empty_input(self):
        """Test that empty or whitespace-only input returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_small_text(self):
        """Test that text smaller than chunk_size produces a single chunk."""
        text = "Hello, this is a short document."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestExtractText:
    """Tests for the extract_text function."""

    def test_txt_extraction(self):
        """Test text extraction from a TXT file."""
        content = "Hello, this is a test document."
        file_bytes = content.encode("utf-8")
        result = extract_text(file_bytes, "test.txt")
        assert result == content

    def test_unsupported_format(self):
        """Test that unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(b"some data", "file.docx")

    def test_txt_with_whitespace(self):
        """Test that extracted text is stripped of leading/trailing whitespace."""
        content = "  \n  Hello world  \n  "
        file_bytes = content.encode("utf-8")
        result = extract_text(file_bytes, "notes.txt")
        assert result == content.strip()
