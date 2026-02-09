"""
Unit tests for advanced RAG ingestion pipeline (pro_implementation/ingest.py).

Tests LLM-powered chunking, parallel processing, and enhanced embedding creation.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))

from pro_implementation import ingest
from pro_implementation.ingest import Chunk, Chunks, Result


class TestPydanticModels:
    """Test Pydantic model definitions."""
    
    def test_chunk_model_validation(self):
        """Test that Chunk model validates correctly."""
        chunk = Chunk(
            headline="Test Headline",
            summary="This is a test summary.",
            original_text="Original text content here."
        )
        
        assert chunk.headline == "Test Headline"
        assert chunk.summary == "This is a test summary."
        assert chunk.original_text == "Original text content here."
    
    def test_chunk_as_result(self):
        """Test that Chunk.as_result() creates Result correctly."""
        chunk = Chunk(
            headline="Headline",
            summary="Summary",
            original_text="Original"
        )
        document = {"source": "test.md", "type": "company"}
        
        result = chunk.as_result(document)
        
        assert isinstance(result, Result)
        assert "Headline" in result.page_content
        assert "Summary" in result.page_content
        assert "Original" in result.page_content
        assert result.metadata["source"] == "test.md"
        assert result.metadata["type"] == "company"
    
    def test_chunks_model_with_list(self):
        """Test that Chunks model handles list of chunks."""
        chunks = Chunks(chunks=[
            Chunk(headline="H1", summary="S1", original_text="T1"),
            Chunk(headline="H2", summary="S2", original_text="T2")
        ])
        
        assert len(chunks.chunks) == 2
        assert chunks.chunks[0].headline == "H1"


class TestFetchDocuments:
    """Test document loading functionality."""
    
    def test_fetch_documents_loads_markdown(self, sample_knowledge_base):
        """Test that fetch_documents loads markdown files."""
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', sample_knowledge_base):
            documents = ingest.fetch_documents()
            
            assert len(documents) == 2  # 2 markdown files in sample KB
            assert all('type' in doc for doc in documents)
            assert all('source' in doc for doc in documents)
            assert all('text' in doc for doc in documents)
    
    def test_fetch_documents_preserves_metadata(self, sample_knowledge_base):
        """Test that document metadata is preserved."""
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', sample_knowledge_base):
            documents = ingest.fetch_documents()
            
            for doc in documents:
                assert doc['type'] in ['company', 'people']
                assert doc['source'].endswith('.md')
                assert len(doc['text']) > 0


class TestMakePrompt:
    """Test LLM prompt generation."""
    
    def test_make_prompt_includes_document_info(self):
        """Test that prompt includes document metadata."""
        document = {
            "type": "company",
            "source": "test.md",
            "text": "This is a test document with some content."
        }
        
        prompt = ingest.make_prompt(document)
        
        assert "company" in prompt
        assert "test.md" in prompt
        assert "This is a test document" in prompt
    
    def test_make_prompt_suggests_chunk_count(self):
        """Test that prompt suggests appropriate number of chunks."""
        short_doc = {"type": "test", "source": "short.md", "text": "Short text"}
        long_doc = {"type": "test", "source": "long.md", "text": "word " * 500}
        
        short_prompt = ingest.make_prompt(short_doc)
        long_prompt = ingest.make_prompt(long_doc)
        
        # Long document should suggest more chunks
        assert "chunks" in short_prompt
        assert "chunks" in long_prompt
    
    def test_make_prompt_mentions_overlap(self):
        """Test that prompt instructs LLM to create overlap."""
        document = {"type": "test", "source": "test.md", "text": "Test content"}
        
        prompt = ingest.make_prompt(document)
        
        assert "overlap" in prompt.lower()


class TestProcessDocument:
    """Test LLM-powered document processing."""
    
    @patch('pro_implementation.ingest.completion')
    def test_process_document_returns_results(self, mock_completion):
        """Test that process_document returns list of Results."""
        # Mock LLM response
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = '''
        {
            "chunks": [
                {
                    "headline": "Test Headline",
                    "summary": "Test summary",
                    "original_text": "Original text"
                }
            ]
        }
        '''
        mock_response.choices = [Mock(message=mock_message)]
        mock_completion.return_value = mock_response
        
        document = {
            "type": "company",
            "source": "test.md",
            "text": "Test document content"
        }
        
        results = ingest.process_document(document)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], Result)
    
    @patch('pro_implementation.ingest.completion')
    def test_process_document_preserves_metadata(self, mock_completion):
        """Test that processed chunks preserve document metadata."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = '''
        {
            "chunks": [
                {
                    "headline": "H1",
                    "summary": "S1",
                    "original_text": "T1"
                }
            ]
        }
        '''
        mock_response.choices = [Mock(message=mock_message)]
        mock_completion.return_value = mock_response
        
        document = {
            "type": "people",
            "source": "leadership.md",
            "text": "Content"
        }
        
        results = ingest.process_document(document)
        
        assert results[0].metadata["type"] == "people"
        assert results[0].metadata["source"] == "leadership.md"


class TestCreateChunks:
    """Test parallel document processing."""
    
    @patch('pro_implementation.ingest.Pool')
    @patch('pro_implementation.ingest.process_document')
    def test_create_chunks_uses_multiprocessing(self, mock_process, mock_pool_class):
        """Test that create_chunks uses multiprocessing pool."""
        # Mock process_document to return sample results
        mock_result = [Result(page_content="Test", metadata={"source": "test.md"})]
        mock_process.return_value = mock_result
        
        # Mock Pool
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap_unordered = Mock(return_value=[mock_result, mock_result])
        mock_pool_class.return_value = mock_pool
        
        documents = [
            {"type": "test", "source": "doc1.md", "text": "Content 1"},
            {"type": "test", "source": "doc2.md", "text": "Content 2"}
        ]
        
        chunks = ingest.create_chunks(documents)
        
        assert len(chunks) >= 2
        mock_pool_class.assert_called_once()
    
    @patch('pro_implementation.ingest.Pool')
    def test_create_chunks_flattens_results(self, mock_pool_class):
        """Test that create_chunks flattens nested results."""
        # Mock pool to return nested results
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        
        result1 = [Result(page_content="C1", metadata={}), Result(page_content="C2", metadata={})]
        result2 = [Result(page_content="C3", metadata={})]
        mock_pool.imap_unordered = Mock(return_value=[result1, result2])
        mock_pool_class.return_value = mock_pool
        
        documents = [{"type": "test", "source": "test.md", "text": "Content"}]
        
        chunks = ingest.create_chunks(documents)
        
        # Should flatten to 3 total chunks
        assert len(chunks) == 3


class TestCreateEmbeddings:
    """Test embedding creation with ChromaDB."""
    
    @patch('pro_implementation.ingest.PersistentClient')
    @patch('pro_implementation.ingest.openai')
    def test_create_embeddings_deletes_existing_collection(self, mock_openai, mock_client_class):
        """Test that existing collection is deleted."""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = Mock()
        mock_collection.name = "docs"
        mock_client.list_collections = Mock(return_value=[mock_collection])
        mock_client.delete_collection = Mock()
        mock_client.get_or_create_collection = Mock(return_value=Mock(count=Mock(return_value=5)))
        mock_client_class.return_value = mock_client
        
        # Mock OpenAI embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        chunks = [Result(page_content="Test", metadata={"source": "test.md"})]
        
        ingest.create_embeddings(chunks)
        
        mock_client.delete_collection.assert_called_once_with("docs")
    
    @patch('pro_implementation.ingest.PersistentClient')
    @patch('pro_implementation.ingest.openai')
    def test_create_embeddings_calls_openai(self, mock_openai, mock_client_class):
        """Test that OpenAI embeddings API is called."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_client.list_collections = Mock(return_value=[])
        mock_collection = Mock()
        mock_collection.count = Mock(return_value=3)
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)
        mock_client_class.return_value = mock_client
        
        # Mock OpenAI
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1] * 3072
        mock_embeddings_response = Mock()
        mock_embeddings_response.data = [mock_embedding_data] * 3
        mock_openai.embeddings.create = Mock(return_value=mock_embeddings_response)
        
        chunks = [
            Result(page_content="Chunk 1", metadata={"source": "test1.md"}),
            Result(page_content="Chunk 2", metadata={"source": "test2.md"}),
            Result(page_content="Chunk 3", metadata={"source": "test3.md"})
        ]
        
        ingest.create_embeddings(chunks)
        
        # Verify OpenAI was called with correct model and texts
        mock_openai.embeddings.create.assert_called_once()
        call_kwargs = mock_openai.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-large"
        assert len(call_kwargs["input"]) == 3
    
    @patch('pro_implementation.ingest.PersistentClient')
    @patch('pro_implementation.ingest.openai')
    def test_create_embeddings_stores_in_chromadb(self, mock_openai, mock_client_class):
        """Test that embeddings are stored in ChromaDB collection."""
        # Mock setup
        mock_client = MagicMock()
        mock_client.list_collections = Mock(return_value=[])
        mock_collection = Mock()
        mock_collection.count = Mock(return_value=2)
        mock_collection.add = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)
        mock_client_class.return_value = mock_client
        
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding, mock_embedding]))
        
        chunks = [
            Result(page_content="C1", metadata={"source": "s1.md"}),
            Result(page_content="C2", metadata={"source": "s2.md"})
        ]
        
        ingest.create_embeddings(chunks)
        
        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args.kwargs
        assert len(call_kwargs["ids"]) == 2
        assert len(call_kwargs["embeddings"]) == 2
        assert len(call_kwargs["documents"]) == 2
        assert len(call_kwargs["metadatas"]) == 2


class TestConfiguration:
    """Test configuration constants."""
    
    def test_model_is_configured(self):
        """Test that MODEL is set."""
        assert ingest.MODEL is not None
        assert "gpt" in ingest.MODEL.lower()
    
    def test_workers_is_positive(self):
        """Test that WORKERS is a positive integer."""
        assert ingest.WORKERS > 0
        assert isinstance(ingest.WORKERS, int)
    
    def test_average_chunk_size_is_reasonable(self):
        """Test that AVERAGE_CHUNK_SIZE is configured."""
        assert ingest.AVERAGE_CHUNK_SIZE > 0
        assert ingest.AVERAGE_CHUNK_SIZE < 1000  # Reasonable upper bound
