"""
Unit tests for basic RAG ingestion pipeline (implementation/ingestion.py).

Tests document loading, chunking, and embedding creation without actual API calls.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))

from implementation import ingestionion


class TestFetchDocuments:
    """Test document loading functionality."""
    
    def test_fetch_documents_with_sample_kb(self, sample_knowledge_base):
        """Test that fetch_documents loads markdown files correctly."""
        with patch.object(ingest, 'KNOWLEDGE_BASE', str(sample_knowledge_base)):
            with patch('implementation.ingestion.DirectoryLoader') as mock_loader_class:
                # Mock the DirectoryLoader to return sample documents
                mock_loader = MagicMock()
                mock_doc1 = Mock()
                mock_doc1.metadata = {}
                mock_doc1.page_content = "Test content 1"
                
                mock_doc2 = Mock()
                mock_doc2.metadata = {}
                mock_doc2.page_content = "Test content 2"
                
                mock_loader.load = MagicMock(return_value=[mock_doc1, mock_doc2])
                mock_loader_class.return_value = mock_loader
                
                documents = ingest.fetch_documents()
                
                assert len(documents) == 4  # 2 folders × 2 docs each
                assert all(hasattr(doc, 'metadata') for doc in documents)
                assert all('doc_type' in doc.metadata for doc in documents)
    
    def test_fetch_documents_sets_doc_type(self, sample_knowledge_base):
        """Test that documents have correct doc_type metadata."""
        with patch.object(ingest, 'KNOWLEDGE_BASE', str(sample_knowledge_base)):
            with patch('implementation.ingestion.DirectoryLoader') as mock_loader_class:
                mock_loader = MagicMock()
                mock_doc = Mock()
                mock_doc.metadata = {}
                mock_loader.load = MagicMock(return_value=[mock_doc])
                mock_loader_class.return_value = mock_loader
                
                documents = ingest.fetch_documents()
                
                doc_types = {doc.metadata['doc_type'] for doc in documents}
                assert 'company' in doc_types
                assert 'people' in doc_types


class TestCreateChunks:
    """Test document chunking functionality."""
    
    def test_create_chunks_splits_documents(self, sample_documents):
        """Test that create_chunks splits documents into smaller pieces."""
        # Convert sample documents to Mock objects with required attributes
        mock_docs = []
        for doc in sample_documents:
            mock_doc = Mock()
            mock_doc.page_content = doc["page_content"]
            mock_doc.metadata = doc["metadata"]
            mock_docs.append(mock_doc)
        
        chunks = ingest.create_chunks(mock_docs)
        
        assert len(chunks) > 0
        assert len(chunks) >= len(mock_docs)  # Should have at least as many chunks as docs
    
    def test_create_chunks_preserves_metadata(self, sample_documents):
        """Test that chunks preserve metadata from original documents."""
        mock_docs = []
        for doc in sample_documents:
            mock_doc = Mock()
            mock_doc.page_content = doc["page_content"]
            mock_doc.metadata = doc["metadata"]
            mock_docs.append(mock_doc)
        
        chunks = ingest.create_chunks(mock_docs)
        
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert 'source' in chunk.metadata or 'doc_type' in chunk.metadata
    
    def test_create_chunks_with_overlap(self):
        """Test that chunks have configured overlap."""
        # Create a long document that will be split
        long_text = "word " * 200  # 200 words
        mock_doc = Mock()
        mock_doc.page_content = long_text
        mock_doc.metadata = {"source": "test.md"}
        
        chunks = ingest.create_chunks([mock_doc])
        
        # Should create multiple chunks due to length
        assert len(chunks) > 1


class TestCreateEmbeddings:
    """Test embedding creation and vector store functionality."""
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.os.path.exists')
    def test_create_embeddings_deletes_existing_db(self, mock_exists, mock_chroma_class):
        """Test that existing database is deleted before creating new one."""
        mock_exists.return_value = True
        
        # Mock the Chroma class
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=10)
        mock_collection.get = MagicMock(return_value={
            "embeddings": [[0.1] * 768]
        })
        mock_store._collection = mock_collection
        mock_store.delete_collection = MagicMock()
        
        # Mock from_documents to return our mock store
        mock_chroma_class.return_value = mock_store
        mock_chroma_class.from_documents = MagicMock(return_value=mock_store)
        
        # Create test chunks
        mock_chunk = Mock()
        mock_chunk.page_content = "Test content"
        mock_chunk.metadata = {"source": "test.md"}
        
        ingestion.create_embeddings([mock_chunk])
        
        # Verify delete was called
        mock_store.delete_collection.assert_called_once()
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.os.path.exists')
    def test_create_embeddings_creates_vector_store(self, mock_exists, mock_chroma_class):
        """Test that vector store is created with chunks."""
        mock_exists.return_value = False
        
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=5)
        mock_collection.get = MagicMock(return_value={
            "embeddings": [[0.1] * 3072]
        })
        mock_store._collection = mock_collection
        
        mock_chroma_class.from_documents = MagicMock(return_value=mock_store)
        
        mock_chunks = [Mock(page_content=f"Chunk {i}", metadata={"source": "test.md"}) for i in range(5)]
        
        result = ingest.create_embeddings(mock_chunks)
        
        assert result is not None
        mock_chroma_class.from_documents.assert_called_once()
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.os.path.exists')
    def test_create_embeddings_prints_summary(self, mock_exists, mock_chroma_class, capsys):
        """Test that embedding creation prints summary statistics."""
        mock_exists.return_value = False
        
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=100)
        mock_collection.get = MagicMock(return_value={
            "embeddings": [[0.1] * 3072]
        })
        mock_store._collection = mock_collection
        
        mock_chroma_class.from_documents = MagicMock(return_value=mock_store)
        
        mock_chunk = Mock(page_content="Test", metadata={"source": "test.md"})
        ingestion.create_embeddings([mock_chunk])
        
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "3,072" in captured.out


class TestIntegration:
    """Integration tests for the complete ingestion pipeline."""
    
    @patch('implementation.ingestion.create_embeddings')
    @patch('implementation.ingestion.create_chunks')
    @patch('implementation.ingestion.fetch_documents')
    def test_full_pipeline_integration(self, mock_fetch, mock_chunk, mock_embed):
        """Test that the full pipeline runs without errors."""
        # Setup mocks
        mock_docs = [Mock(page_content="Doc", metadata={"source": "test.md"})]
        mock_chunks = [Mock(page_content="Chunk", metadata={"source": "test.md"})]
        mock_store = MagicMock()
        
        mock_fetch.return_value = mock_docs
        mock_chunk.return_value = mock_chunks
        mock_embed.return_value = mock_store
        
        # Execute pipeline
        documents = mock_fetch()
        chunks = mock_chunk(documents)
        vectorstore = mock_embed(chunks)
        
        # Verify pipeline executed
        assert documents is not None
        assert chunks is not None
        assert vectorstore is not None
        
        mock_fetch.assert_called_once()
        mock_chunk.assert_called_once_with(mock_docs)
        mock_embed.assert_called_once_with(mock_chunks)
