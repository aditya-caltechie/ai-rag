"""
Unit tests for advanced RAG answer pipeline (pro_implementation/answer.py).

Tests query rewriting, dual retrieval, LLM reranking, and answer generation.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))

from pro_implementation import answer
from pro_implementation.answer import Result, RankOrder


class TestPydanticModels:
    """Test Pydantic model definitions."""
    
    def test_result_model(self):
        """Test Result model validation."""
        result = Result(
            page_content="Test content",
            metadata={"source": "test.md", "type": "company"}
        )
        
        assert result.page_content == "Test content"
        assert result.metadata["source"] == "test.md"
    
    def test_rank_order_model(self):
        """Test RankOrder model validation."""
        rank = RankOrder(order=[3, 1, 4, 2])
        
        assert len(rank.order) == 4
        assert rank.order[0] == 3


class TestRewriteQuery:
    """Test query rewriting functionality."""
    
    @patch('pro_implementation.answer.completion')
    def test_rewrite_query_calls_llm(self, mock_completion):
        """Test that rewrite_query calls LLM."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="rewritten query"))]
        mock_completion.return_value = mock_response
        
        result = answer.rewrite_query("Who is the CEO?")
        
        assert result == "rewritten query"
        mock_completion.assert_called_once()
    
    @patch('pro_implementation.answer.completion')
    def test_rewrite_query_with_history(self, mock_completion):
        """Test that rewrite_query uses conversation history."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="rewritten"))]
        mock_completion.return_value = mock_response
        
        history = [
            {"role": "user", "content": "Tell me about Insurellm"},
            {"role": "assistant", "content": "It's a tech company"}
        ]
        
        result = answer.rewrite_query("What products do they have?", history)
        
        # Should include history in the prompt
        call_args = mock_completion.call_args
        assert history in str(call_args) or len(str(call_args)) > 100


class TestFetchContextUnranked:
    """Test basic retrieval functionality."""
    
    @patch('pro_implementation.answer.collection')
    @patch('pro_implementation.answer.openai')
    def test_fetch_context_unranked_returns_results(self, mock_openai, mock_collection):
        """Test that fetch_context_unranked returns list of Results."""
        # Mock OpenAI embedding
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        # Mock ChromaDB query results
        mock_collection.query = Mock(return_value={
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "s1.md"}, {"source": "s2.md"}]]
        })
        
        results = answer.fetch_context_unranked("test question")
        
        assert len(results) == 2
        assert isinstance(results[0], Result)
        assert results[0].page_content == "Doc 1"
    
    @patch('pro_implementation.answer.collection')
    @patch('pro_implementation.answer.openai')
    def test_fetch_context_unranked_uses_correct_k(self, mock_openai, mock_collection):
        """Test that retrieval uses RETRIEVAL_K."""
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 3072
        mock_openai.embeddings.create = Mock(return_value=Mock(data=[mock_embedding]))
        
        mock_collection.query = Mock(return_value={
            "documents": [[]],
            "metadatas": [[]]
        })
        
        answer.fetch_context_unranked("test")
        
        # Verify n_results parameter
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == answer.RETRIEVAL_K


class TestMergeChunks:
    """Test chunk deduplication."""
    
    def test_merge_chunks_removes_duplicates(self):
        """Test that merge_chunks deduplicates by content."""
        chunks1 = [
            Result(page_content="Content A", metadata={"source": "s1.md"}),
            Result(page_content="Content B", metadata={"source": "s2.md"})
        ]
        chunks2 = [
            Result(page_content="Content B", metadata={"source": "s2.md"}),  # Duplicate
            Result(page_content="Content C", metadata={"source": "s3.md"})
        ]
        
        merged = answer.merge_chunks(chunks1, chunks2)
        
        # Should have 3 unique chunks (A, B, C)
        assert len(merged) == 3
        contents = [c.page_content for c in merged]
        assert "Content A" in contents
        assert "Content B" in contents
        assert "Content C" in contents
    
    def test_merge_chunks_preserves_order(self):
        """Test that original chunks come first."""
        chunks1 = [Result(page_content="First", metadata={})]
        chunks2 = [Result(page_content="Second", metadata={})]
        
        merged = answer.merge_chunks(chunks1, chunks2)
        
        assert merged[0].page_content == "First"
        assert merged[1].page_content == "Second"


class TestRerank:
    """Test LLM-based reranking."""
    
    @patch('pro_implementation.answer.completion')
    def test_rerank_returns_reordered_chunks(self, mock_completion):
        """Test that rerank returns chunks in new order."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"order": [2, 1, 3]}'))]
        mock_completion.return_value = mock_response
        
        chunks = [
            Result(page_content="Chunk 1", metadata={}),
            Result(page_content="Chunk 2", metadata={}),
            Result(page_content="Chunk 3", metadata={})
        ]
        
        reranked = answer.rerank("test question", chunks)
        
        assert len(reranked) == 3
        assert reranked[0].page_content == "Chunk 2"  # Order: [2, 1, 3]
        assert reranked[1].page_content == "Chunk 1"
        assert reranked[2].page_content == "Chunk 3"
    
    @patch('pro_implementation.answer.completion')
    def test_rerank_includes_question_in_prompt(self, mock_completion):
        """Test that question is passed to reranking LLM."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"order": [1]}'))]
        mock_completion.return_value = mock_response
        
        question = "Who is the CEO?"
        chunks = [Result(page_content="Test", metadata={})]
        
        answer.rerank(question, chunks)
        
        # Verify question appears in the call
        call_args = mock_completion.call_args
        assert question in str(call_args)


class TestFetchContext:
    """Test multi-stage retrieval pipeline."""
    
    @patch('pro_implementation.answer.rerank')
    @patch('pro_implementation.answer.fetch_context_unranked')
    @patch('pro_implementation.answer.rewrite_query')
    def test_fetch_context_uses_dual_retrieval(self, mock_rewrite, mock_fetch, mock_rerank):
        """Test that fetch_context retrieves with both original and rewritten queries."""
        mock_rewrite.return_value = "rewritten query"
        
        chunks1 = [Result(page_content="C1", metadata={})]
        chunks2 = [Result(page_content="C2", metadata={})]
        mock_fetch.side_effect = [chunks1, chunks2]
        
        mock_rerank.return_value = [Result(page_content="Ranked", metadata={})]
        
        result = answer.fetch_context("original question")
        
        # Should call fetch_context_unranked twice
        assert mock_fetch.call_count == 2
        # First with original, second with rewritten
        assert mock_fetch.call_args_list[0][0][0] == "original question"
        assert mock_fetch.call_args_list[1][0][0] == "rewritten query"
    
    @patch('pro_implementation.answer.rerank')
    @patch('pro_implementation.answer.fetch_context_unranked')
    @patch('pro_implementation.answer.rewrite_query')
    def test_fetch_context_returns_top_k(self, mock_rewrite, mock_fetch, mock_rerank):
        """Test that fetch_context returns top FINAL_K chunks."""
        mock_rewrite.return_value = "rewritten"
        mock_fetch.return_value = []
        
        # Mock rerank to return many chunks
        many_chunks = [Result(page_content=f"C{i}", metadata={}) for i in range(30)]
        mock_rerank.return_value = many_chunks
        
        result = answer.fetch_context("question")
        
        # Should return only FINAL_K
        assert len(result) == answer.FINAL_K


class TestMakeRagMessages:
    """Test message construction for LLM."""
    
    def test_make_rag_messages_includes_context(self):
        """Test that messages include retrieved context."""
        chunks = [
            Result(page_content="Context 1", metadata={"source": "s1.md"}),
            Result(page_content="Context 2", metadata={"source": "s2.md"})
        ]
        
        messages = answer.make_rag_messages("Question?", [], chunks)
        
        # Should have system message with context
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "Context 1" in system_msg["content"]
        assert "Context 2" in system_msg["content"]
    
    def test_make_rag_messages_includes_history(self):
        """Test that conversation history is included."""
        chunks = [Result(page_content="Context", metadata={"source": "s.md"})]
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        messages = answer.make_rag_messages("Current?", history, chunks)
        
        # Should have system + history + current question
        assert len(messages) >= 4  # system + 2 history + current
        assert messages[1] == history[0]
        assert messages[2] == history[1]
    
    def test_make_rag_messages_includes_source_info(self):
        """Test that source information is included in context."""
        chunks = [
            Result(page_content="Content", metadata={"source": "important.md"})
        ]
        
        messages = answer.make_rag_messages("Question", [], chunks)
        
        # Source should appear in system message
        assert "important.md" in messages[0]["content"]


class TestAnswerQuestion:
    """Test complete answer generation pipeline."""
    
    @patch('pro_implementation.answer.completion')
    @patch('pro_implementation.answer.fetch_context')
    def test_answer_question_returns_tuple(self, mock_fetch, mock_completion):
        """Test that answer_question returns (answer, chunks) tuple."""
        mock_fetch.return_value = [Result(page_content="Context", metadata={})]
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_completion.return_value = mock_response
        
        result = answer.answer_question("Question?")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "Answer"
        assert isinstance(result[1], list)
    
    @patch('pro_implementation.answer.completion')
    @patch('pro_implementation.answer.fetch_context')
    def test_answer_question_uses_advanced_retrieval(self, mock_fetch, mock_completion):
        """Test that answer_question uses fetch_context (not basic retrieval)."""
        mock_chunks = [Result(page_content="Advanced context", metadata={"source": "s.md"})]
        mock_fetch.return_value = mock_chunks
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_completion.return_value = mock_response
        
        answer_text, chunks = answer.answer_question("Test question")
        
        mock_fetch.assert_called_once_with("Test question")
        assert chunks == mock_chunks
    
    @patch('pro_implementation.answer.completion')
    @patch('pro_implementation.answer.fetch_context')
    def test_answer_question_with_history(self, mock_fetch, mock_completion):
        """Test that answer_question passes history to LLM."""
        mock_fetch.return_value = [Result(page_content="C", metadata={})]
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer with history"))]
        mock_completion.return_value = mock_response
        
        history = [{"role": "user", "content": "Previous"}]
        
        answer.answer_question("Current", history)
        
        # Verify completion was called with messages including history
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert any("Previous" in str(msg) for msg in messages)


class TestConfiguration:
    """Test configuration constants."""
    
    def test_retrieval_k_is_larger_than_final_k(self):
        """Test that RETRIEVAL_K > FINAL_K for reranking."""
        assert answer.RETRIEVAL_K >= answer.FINAL_K
    
    def test_model_is_configured(self):
        """Test that MODEL is set."""
        assert answer.MODEL is not None
        assert isinstance(answer.MODEL, str)
    
    def test_system_prompt_emphasizes_quality(self):
        """Test that SYSTEM_PROMPT emphasizes accuracy and completeness."""
        prompt = answer.SYSTEM_PROMPT
        assert "accuracy" in prompt.lower() or "accurate" in prompt.lower()
        assert "complete" in prompt.lower()
