"""
Integration tests for the complete basic RAG pipeline.

Tests end-to-end workflow: ingestion → vector storage → retrieval → answer generation.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))


class TestBasicRAGIngestionPipeline:
    """Test complete ingestion workflow."""
    
    @pytest.fixture
    def temp_kb_and_db(self):
        """Create temporary knowledge base and database directories."""
        temp_dir = Path(tempfile.mkdtemp())
        kb_dir = temp_dir / "knowledge-base"
        kb_dir.mkdir()
        db_dir = temp_dir / "vector_db"
        
        # Create sample documents
        (kb_dir / "company").mkdir()
        (kb_dir / "company" / "overview.md").write_text(
            "# Insurellm\n\nInsurellm is an insurance technology company that provides AI-powered solutions."
        )
        
        yield kb_dir, db_dir
        
        shutil.rmtree(temp_dir)
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.OpenAIEmbeddings')
    def test_full_ingestion_pipeline(self, mock_embeddings_class, mock_chroma_class, temp_kb_and_db):
        """Test complete ingestion: load → chunk → embed → store."""
        from implementation import ingestionion
        
        kb_dir, db_dir = temp_kb_and_db
        
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock vector store
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=5)
        mock_collection.get = MagicMock(return_value={"embeddings": [[0.1] * 3072]})
        mock_store._collection = mock_collection
        mock_store.delete_collection = MagicMock()
        mock_chroma_class.from_documents = MagicMock(return_value=mock_store)
        mock_chroma_class.return_value = mock_store
        
        # Override paths
        with patch.object(ingestion, 'KNOWLEDGE_BASE', str(kb_dir)):
            with patch.object(ingestion, 'DB_NAME', str(db_dir)):
                with patch.object(ingestion, 'embeddings', mock_embeddings):
                    # Run pipeline
                    documents = ingestion.fetch_documents()
                    assert len(documents) > 0
                    
                    chunks = ingestion.create_chunks(documents)
                    assert len(chunks) > 0
                    assert len(chunks) >= len(documents)
                    
                    vectorstore = ingestion.create_embeddings(chunks)
                    assert vectorstore is not None
                    mock_chroma_class.from_documents.assert_called_once()
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.OpenAIEmbeddings')
    def test_ingestion_preserves_metadata_through_pipeline(self, mock_embeddings_class, mock_chroma_class, temp_kb_and_db):
        """Test that metadata is preserved from documents → chunks → vector store."""
        from implementation import ingestion
        
        kb_dir, db_dir = temp_kb_and_db
        
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=3)
        mock_collection.get = MagicMock(return_value={"embeddings": [[0.1] * 3072]})
        mock_store._collection = mock_collection
        mock_chroma_class.from_documents = MagicMock(return_value=mock_store)
        mock_chroma_class.return_value = mock_store
        
        with patch.object(ingestion, 'KNOWLEDGE_BASE', str(kb_dir)):
            with patch.object(ingestion, 'DB_NAME', str(db_dir)):
                with patch.object(ingestion, 'embeddings', mock_embeddings):
                    documents = ingestion.fetch_documents()
                    
                    # Verify documents have metadata
                    assert all('doc_type' in doc.metadata for doc in documents)
                    
                    chunks = ingest.create_chunks(documents)
                    
                    # Verify chunks preserve metadata
                    assert all(hasattr(chunk, 'metadata') for chunk in chunks)
                    
                    # When creating embeddings, metadata should be passed
                    ingestion.create_embeddings(chunks)
                    call_args = mock_chroma_class.from_documents.call_args
                    stored_chunks = call_args.kwargs['documents']
                    assert all(hasattr(c, 'metadata') for c in stored_chunks)


class TestBasicRAGQueryPipeline:
    """Test complete query workflow."""
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.retriever')
    def test_full_query_pipeline(self, mock_retriever, mock_llm):
        """Test complete query: question → retrieve → generate answer."""
        from implementation import inference
        
        # Mock retrieval
        mock_docs = [
            Mock(page_content="Insurellm is an insurance tech company.", metadata={"source": "overview.md"}),
            Mock(page_content="The CEO is Avery Lancaster.", metadata={"source": "leadership.md"})
        ]
        mock_retriever.invoke = MagicMock(return_value=mock_docs)
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Insurellm is an insurance technology company focused on AI solutions."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        # Execute query
        question = "What is Insurellm?"
        answer_text, retrieved_docs = inference.answer_question(question)
        
        # Verify pipeline
        assert answer_text == mock_response.content
        assert len(retrieved_docs) == 2
        assert retrieved_docs == mock_docs
        
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once()
        
        # Verify LLM was called with context
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg_content = str(call_args[0].content)
        assert "Insurellm is an insurance tech company" in system_msg_content
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.retriever')
    def test_query_pipeline_with_conversation_history(self, mock_retriever, mock_llm):
        """Test query pipeline with multi-turn conversation."""
        from implementation import inference
        
        mock_retriever.invoke = MagicMock(return_value=[
            Mock(page_content="Avery Lancaster is the CEO.", metadata={"source": "leadership.md"})
        ])
        
        mock_response = Mock()
        mock_response.content = "Avery Lancaster is the CEO of Insurellm."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        history = [
            {"role": "user", "content": "Tell me about Insurellm"},
            {"role": "assistant", "content": "It's an insurance tech company."}
        ]
        
        answer_text, docs = answer.answer_question("Who is the CEO?", history)
        
        # Verify history was used in retrieval (combined question)
        retriever_call_arg = mock_retriever.invoke.call_args[0][0]
        assert "Tell me about Insurellm" in retriever_call_arg
        assert "Who is the CEO?" in retriever_call_arg
        
        # Verify history was passed to LLM
        llm_call_messages = mock_llm.invoke.call_args[0][0]
        assert len(llm_call_messages) >= 4  # system + history + current


class TestBasicRAGEndToEnd:
    """End-to-end tests simulating real usage."""
    
    @patch('implementation.ingestion.Chroma')
    @patch('implementation.ingestion.OpenAIEmbeddings')
    @patch('implementation.inference.llm')
    @patch('implementation.inference.retriever')
    def test_ingest_then_query(self, mock_retriever, mock_answer_llm, mock_ingest_embeddings, mock_ingest_chroma):
        """Test: ingest documents → query system → get answer."""
        from implementation import ingestion, answer
        
        # Setup ingestion mocks
        temp_dir = Path(tempfile.mkdtemp())
        kb_dir = temp_dir / "knowledge-base"
        kb_dir.mkdir()
        (kb_dir / "docs").mkdir()
        (kb_dir / "docs" / "about.md").write_text("Insurellm provides AI insurance solutions.")
        
        mock_embeddings_inst = MagicMock()
        mock_ingest_embeddings.return_value = mock_embeddings_inst
        
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=1)
        mock_collection.get = MagicMock(return_value={"embeddings": [[0.1] * 3072]})
        mock_store._collection = mock_collection
        mock_ingest_chroma.from_documents = MagicMock(return_value=mock_store)
        mock_ingest_chroma.return_value = mock_store
        
        # Run ingestion
        with patch.object(ingest, 'KNOWLEDGE_BASE', str(kb_dir)):
            with patch.object(ingestion, 'embeddings', mock_embeddings_inst):
                docs = ingestion.fetch_documents()
                chunks = ingestion.create_chunks(docs)
                ingestion.create_embeddings(chunks)
        
        # Setup query mocks
        mock_retriever.invoke = MagicMock(return_value=[
            Mock(page_content="Insurellm provides AI insurance solutions.", metadata={"source": "about.md"})
        ])
        
        mock_response = Mock()
        mock_response.content = "Insurellm is a company that provides AI-powered insurance solutions."
        mock_answer_llm.invoke = MagicMock(return_value=mock_response)
        
        # Run query
        answer_text, retrieved_docs = answer.answer_question("What does Insurellm do?")
        
        # Verify end-to-end flow
        assert "AI" in answer_text or "insurance" in answer_text
        assert len(retrieved_docs) > 0
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.retriever')
    def test_multiple_queries_maintain_context(self, mock_retriever, mock_llm):
        """Test multiple queries building conversation context."""
        from implementation import inference
        
        # Query 1
        mock_retriever.invoke = MagicMock(return_value=[
            Mock(page_content="Insurellm is an insurance tech company.", metadata={"source": "s.md"})
        ])
        mock_llm.invoke = MagicMock(return_value=Mock(content="It's an insurance tech company."))
        
        answer1, _ = inference.answer_question("What is Insurellm?")
        history = [
            {"role": "user", "content": "What is Insurellm?"},
            {"role": "assistant", "content": answer1}
        ]
        
        # Query 2 with history
        mock_retriever.invoke = MagicMock(return_value=[
            Mock(page_content="CEO: Avery Lancaster", metadata={"source": "s.md"})
        ])
        mock_llm.invoke = MagicMock(return_value=Mock(content="Avery Lancaster is the CEO."))
        
        answer2, _ = inference.answer_question("Who is the CEO?", history)
        
        # Verify history was used
        retriever_arg = mock_retriever.invoke.call_args[0][0]
        assert "What is Insurellm?" in retriever_arg
        
        llm_messages = mock_llm.invoke.call_args[0][0]
        assert len(llm_messages) >= 4


class TestBasicRAGErrorHandling:
    """Test error handling in the pipeline."""
    
    @patch('implementation.ingestion.DirectoryLoader')
    def test_ingestion_handles_empty_knowledge_base(self, mock_loader_class):
        """Test graceful handling of empty knowledge base."""
        from implementation import ingestion
        
        # Mock empty loader
        mock_loader = MagicMock()
        mock_loader.load = MagicMock(return_value=[])
        mock_loader_class.return_value = mock_loader
        
        temp_dir = Path(tempfile.mkdtemp())
        kb_dir = temp_dir / "knowledge-base"
        kb_dir.mkdir()
        (kb_dir / "empty").mkdir()
        
            with patch.object(ingestion, 'KNOWLEDGE_BASE', str(kb_dir)):
                documents = ingestion.fetch_documents()
            
            # Should return empty list, not crash
            assert isinstance(documents, list)
            assert len(documents) == 0
        
        shutil.rmtree(temp_dir)
    
    @patch('implementation.inference.retriever')
    @patch('implementation.inference.llm')
    def test_query_handles_no_relevant_context(self, mock_llm, mock_retriever):
        """Test handling when no relevant documents are found."""
        from implementation import inference
        
        # Mock retriever returning empty results
        mock_retriever.invoke = MagicMock(return_value=[])
        
        mock_response = Mock()
        mock_response.content = "I don't have information about that."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        answer_text, docs = answer.answer_question("Completely unrelated question")
        
        # Should still generate an answer (even if it says "I don't know")
        assert isinstance(answer_text, str)
        assert len(answer_text) > 0
        assert len(docs) == 0
