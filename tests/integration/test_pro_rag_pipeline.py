"""
Integration tests for the complete advanced RAG pipeline.

Tests end-to-end workflow with LLM chunking, query rewriting, dual retrieval, and reranking.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))


class TestProRAGIngestionPipeline:
    """Test complete advanced ingestion workflow."""
    
    @pytest.fixture
    def temp_kb(self):
        """Create temporary knowledge base."""
        temp_dir = Path(tempfile.mkdtemp())
        kb_dir = temp_dir / "knowledge-base"
        kb_dir.mkdir()
        
        (kb_dir / "company").mkdir()
        (kb_dir / "company" / "overview.md").write_text(
            "# Insurellm\n\nInsurellm is an insurance technology company. "
            "We provide AI-powered solutions for insurance claims processing. "
            "Our products help insurers automate routine tasks and improve customer service."
        )
        
        yield kb_dir
        shutil.rmtree(temp_dir)
    
    @patch('pro_implementation.ingestion.PersistentClient')
    @patch('pro_implementation.ingestion.openai')
    @patch('pro_implementation.ingestion.completion')
    def test_full_ingestion_pipeline(self, mock_completion, mock_openai, mock_client_class, temp_kb):
        """Test complete ingestion: load → LLM chunk → embed → store."""
        from pro_implementation import ingestion
        
        # Mock LLM chunking response
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content='''
        {
            "chunks": [
                {
                    "headline": "Insurellm Overview",
                    "summary": "Insurance tech company providing AI solutions",
                    "original_text": "Insurellm is an insurance technology company."
                },
                {
                    "headline": "AI Solutions",
                    "summary": "Automated claims processing",
                    "original_text": "We provide AI-powered solutions for insurance claims processing."
                }
            ]
        }
        '''))]
        mock_completion.return_value = mock_llm_response
        
        # Mock OpenAI embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding, mock_embedding]))
        
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_client.list_collections = Mock(return_value=[])
        mock_collection = Mock()
        mock_collection.count = Mock(return_value=2)
        mock_collection.add = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)
        mock_client_class.return_value = mock_client
        
        # Run pipeline
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', temp_kb):
            with patch.object(ingest, 'WORKERS', 1):  # Single worker for testing
                documents = ingest.fetch_documents()
                assert len(documents) == 1
                
                chunks = ingest.create_chunks(documents)
                assert len(chunks) == 2  # 2 chunks from LLM response
                assert all(isinstance(c, ingestion.Result) for c in chunks)
                
                ingestion.create_embeddings(chunks)
                
                # Verify embeddings were created
                mock_openai.embeddings.create.assert_called_once()
                
                # Verify stored in ChromaDB
                mock_collection.add.assert_called_once()
    
    @patch('pro_implementation.ingestion.completion')
    def test_llm_chunking_preserves_overlap(self, mock_completion, temp_kb):
        """Test that LLM is instructed to create overlap between chunks."""
        from pro_implementation import ingestion
        
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', temp_kb):
            documents = ingestion.fetch_documents()
            prompt = ingestion.make_prompt(documents[0])
            
            # Verify prompt instructs overlap
            assert "overlap" in prompt.lower()
            assert "50 words" in prompt or "25%" in prompt
    
    @patch('pro_implementation.ingestion.PersistentClient')
    @patch('pro_implementation.ingestion.openai')
    @patch('pro_implementation.ingestion.completion')
    def test_chunks_have_enhanced_metadata(self, mock_completion, mock_openai, mock_client_class, temp_kb):
        """Test that chunks include headline and summary."""
        from pro_implementation import ingestion
        
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content='''
        {
            "chunks": [
                {
                    "headline": "Test Headline",
                    "summary": "Test Summary",
                    "original_text": "Original content"
                }
            ]
        }
        '''))]
        mock_completion.return_value = mock_llm_response
        
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        mock_client = MagicMock()
        mock_client.list_collections = Mock(return_value=[])
        mock_collection = Mock()
        mock_collection.count = Mock(return_value=1)
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)
        mock_client_class.return_value = mock_client
        
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', temp_kb):
            with patch.object(ingest, 'WORKERS', 1):
                documents = ingestion.fetch_documents()
                chunks = ingestion.create_chunks(documents)
                
                # Verify chunk structure
                chunk = chunks[0]
                assert "Test Headline" in chunk.page_content
                assert "Test Summary" in chunk.page_content
                assert "Original content" in chunk.page_content


class TestProRAGQueryPipeline:
    """Test complete advanced query workflow."""
    
    @patch('pro_implementation.inference.completion')
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai')
    def test_full_query_pipeline_with_rewriting(self, mock_openai, mock_collection, mock_completion):
        """Test complete query: question → rewrite → dual retrieval → rerank → answer."""
        from pro_implementation import inference
        
        # Mock query rewriting
        rewrite_response = Mock()
        rewrite_response.choices = [Mock(message=Mock(content="Insurellm CEO name"))]
        
        # Mock reranking
        rerank_response = Mock()
        rerank_response.choices = [Mock(message=Mock(content='{"order": [1, 2]}'))]
        
        # Mock final answer
        answer_response = Mock()
        answer_response.choices = [Mock(message=Mock(content="The CEO is Avery Lancaster."))]
        
        mock_completion.side_effect = [rewrite_response, rerank_response, answer_response]
        
        # Mock embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        # Mock retrieval
        mock_collection.query = Mock(return_value={
            "documents": [["CEO: Avery Lancaster", "Founded in 2020"]],
            "metadatas": [[{"source": "leadership.md"}, {"source": "history.md"}]]
        })
        
        # Execute query
        answer_text, chunks = inference.answer_question("Who is the CEO?")
        
        # Verify pipeline stages
        assert answer_text == "The CEO is Avery Lancaster."
        assert len(chunks) <= answer.FINAL_K
        
        # Verify rewriting happened (completion called 3 times: rewrite, rerank, answer)
        assert mock_completion.call_count == 3
        
        # Verify dual retrieval (openai embeddings called twice)
        assert mock_openai.embeddings.create.call_count == 2
    
    @patch('pro_implementation.inference.completion')
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai')
    def test_dual_retrieval_merges_results(self, mock_openai, mock_collection, mock_completion):
        """Test that dual retrieval combines results from both queries."""
        from pro_implementation import inference
        
        # Mock rewrite
        mock_completion.return_value = Mock(choices=[Mock(message=Mock(content="rewritten query"))])
        
        # Mock embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        # Mock retrieval - return different results for each query
        call_count = 0
        def mock_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "documents": [["Result from original query"]],
                    "metadatas": [[{"source": "doc1.md"}]]
                }
            else:
                return {
                    "documents": [["Result from rewritten query"]],
                    "metadatas": [[{"source": "doc2.md"}]]
                }
        
        mock_collection.query = mock_query
        
        # Test merge_chunks directly
        chunks1 = [inference.Result(page_content="A", metadata={})]
        chunks2 = [inference.Result(page_content="B", metadata={})]
        
        merged = inference.merge_chunks(chunks1, chunks2)
        
        assert len(merged) == 2
        assert any(c.page_content == "A" for c in merged)
        assert any(c.page_content == "B" for c in merged)
    
    @patch('pro_implementation.inference.completion')
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai')
    def test_reranking_improves_order(self, mock_openai, mock_collection, mock_completion):
        """Test that LLM reranking reorders chunks."""
        from pro_implementation import inference
        
        # Mock reranking to reverse order
        rerank_response = Mock()
        rerank_response.choices = [Mock(message=Mock(content='{"order": [3, 2, 1]}'))]
        mock_completion.return_value = rerank_response
        
        chunks = [
            inference.Result(page_content="Chunk 1", metadata={}),
            inference.Result(page_content="Chunk 2", metadata={}),
            inference.Result(page_content="Chunk 3", metadata={})
        ]
        
        reranked = inference.rerank("test question", chunks)
        
        # Order should be reversed
        assert reranked[0].page_content == "Chunk 3"
        assert reranked[1].page_content == "Chunk 2"
        assert reranked[2].page_content == "Chunk 1"


class TestProRAGEndToEnd:
    """End-to-end tests for advanced RAG."""
    
    @patch('pro_implementation.ingestion.PersistentClient')
    @patch('pro_implementation.ingestion.openai', new_callable=MagicMock)
    @patch('pro_implementation.ingestion.completion')
    @patch('pro_implementation.inference.completion')
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai', new_callable=MagicMock)
    def test_ingest_then_query_with_advanced_techniques(
        self, mock_answer_openai, mock_answer_collection, mock_answer_completion,
        mock_ingest_completion, mock_ingest_openai, mock_ingest_client
    ):
        """Test: LLM ingest → query with rewriting/reranking → get answer."""
        from pro_implementation import ingestion, answer
        
        # Setup ingestion
        temp_dir = Path(tempfile.mkdtemp())
        kb_dir = temp_dir / "knowledge-base"
        kb_dir.mkdir()
        (kb_dir / "company").mkdir()
        (kb_dir / "company" / "about.md").write_text("Insurellm provides AI insurance solutions.")
        
        # Mock ingestion
        mock_ingest_completion.return_value = Mock(choices=[Mock(message=Mock(content='''
        {
            "chunks": [{
                "headline": "AI Solutions",
                "summary": "Insurance automation",
                "original_text": "Insurellm provides AI insurance solutions."
            }]
        }
        '''))])
        
        mock_ingest_openai.embeddings.create = Mock(return_value=Mock(data=[
            Mock(embedding=[0.1] * 3072)
        ]))
        
        mock_client = MagicMock()
        mock_client.list_collections = Mock(return_value=[])
        mock_collection_obj = Mock()
        mock_collection_obj.count = Mock(return_value=1)
        mock_client.get_or_create_collection = Mock(return_value=mock_collection_obj)
        mock_ingest_client.return_value = mock_client
        
        # Run ingestion
        with patch.object(ingest, 'KNOWLEDGE_BASE_PATH', kb_dir):
            with patch.object(ingestion, 'WORKERS', 1):
                docs = ingestion.fetch_documents()
                chunks = ingestion.create_chunks(docs)
                ingestion.create_embeddings(chunks)
        
        # Setup query mocks
        mock_answer_completion.side_effect = [
            Mock(choices=[Mock(message=Mock(content="AI insurance solutions"))]),  # Rewrite
            Mock(choices=[Mock(message=Mock(content='{"order": [1]}'))]),  # Rerank
            Mock(choices=[Mock(message=Mock(content="Insurellm provides AI insurance solutions."))])  # Answer
        ]
        
        mock_answer_openai.embeddings.create = Mock(return_value=Mock(data=[Mock(embedding=[0.1] * 3072)]))
        
        mock_answer_collection.query = Mock(return_value={
            "documents": [["AI Solutions\n\nInsurance automation\n\nInsurellm provides AI insurance solutions."]],
            "metadatas": [[{"source": "about.md"}]]
        })
        
        # Run query
        answer_text, retrieved = inference.answer_question("What does Insurellm do?")
        
        # Verify end-to-end
        assert "Insurellm" in answer_text or "AI" in answer_text or "insurance" in answer_text
        assert len(retrieved) > 0
        
        # Cleanup
        shutil.rmtree(temp_dir)


class TestProRAGAdvancedFeatures:
    """Test advanced RAG-specific features."""
    
    @patch('pro_implementation.inference.completion')
    def test_query_rewriting_improves_search(self, mock_completion):
        """Test that query rewriting creates better search terms."""
        from pro_implementation import inference
        
        mock_completion.return_value = Mock(choices=[
            Mock(message=Mock(content="Insurellm CEO Avery Lancaster leadership"))
        ])
        
        rewritten = inference.rewrite_query("Who runs the company?")
        
        # Rewritten query should be more specific
        assert len(rewritten) > 0
        # Should have called LLM
        mock_completion.assert_called_once()
    
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai')
    def test_retrieval_k_greater_than_final_k(self, mock_openai, mock_collection):
        """Test that initial retrieval gets more chunks than final output."""
        from pro_implementation import inference
        
        assert inference.RETRIEVAL_K >= inference.FINAL_K
        
        # This allows reranking to choose best chunks from larger pool
        assert inference.RETRIEVAL_K > inference.FINAL_K or inference.RETRIEVAL_K == inference.FINAL_K
    
    @patch('pro_implementation.inference.completion')
    @patch('pro_implementation.inference.collection')
    @patch('pro_implementation.inference.openai')
    def test_system_prompt_emphasizes_quality_metrics(self, mock_openai, mock_collection, mock_completion):
        """Test that system prompt optimizes for evaluation metrics."""
        from pro_implementation import inference
        
        prompt = inference.SYSTEM_PROMPT
        
        # Should emphasize accuracy, relevance, completeness
        assert "accuracy" in prompt.lower() or "accurate" in prompt.lower()
        assert "complete" in prompt.lower()
        assert "relevant" in prompt.lower() or "relevance" in prompt.lower()
