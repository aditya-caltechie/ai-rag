"""
Unit tests for basic RAG answer pipeline (implementation/inference.py).

Tests retrieval, context combination, and answer generation without actual API calls.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))

from implementation import inference


class TestFetchContext:
    """Test context retrieval functionality."""
    
    @patch('implementation.inference.retriever')
    def test_fetch_context_returns_documents(self, mock_retriever):
        """Test that fetch_context returns list of documents."""
        mock_docs = [
            Mock(page_content="Doc 1", metadata={"source": "test1.md"}),
            Mock(page_content="Doc 2", metadata={"source": "test2.md"})
        ]
        mock_retriever.invoke = MagicMock(return_value=mock_docs)
        
        result = answer.fetch_context("What is Insurellm?")
        
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"
        mock_retriever.invoke.assert_called_once()
    
    @patch('implementation.inference.retriever')
    def test_fetch_context_with_different_k(self, mock_retriever):
        """Test that fetch_context respects RETRIEVAL_K parameter."""
        mock_retriever.invoke = MagicMock(return_value=[Mock() for _ in range(10)])
        
        result = answer.fetch_context("Test question")
        
        # Should request K documents (even if fewer returned)
        mock_retriever.invoke.assert_called_once()
        assert "k" in mock_retriever.invoke.call_args.kwargs or len(result) <= answer.RETRIEVAL_K


class TestCombinedQuestion:
    """Test conversation history handling."""
    
    def test_combined_question_with_empty_history(self):
        """Test combining question with empty history."""
        question = "What is Insurellm?"
        history = []
        
        result = answer.combined_question(question, history)
        
        assert question in result
        assert result.strip().endswith(question)
    
    def test_combined_question_with_history(self):
        """Test combining question with conversation history."""
        question = "Who is the CEO?"
        history = [
            {"role": "user", "content": "What is Insurellm?"},
            {"role": "assistant", "content": "It's a tech company."},
            {"role": "user", "content": "What products do they have?"}
        ]
        
        result = answer.combined_question(question, history)
        
        # Should include previous user questions
        assert "What is Insurellm?" in result
        assert "What products do they have?" in result
        assert question in result
        # Should NOT include assistant responses
        assert "It's a tech company." not in result
    
    def test_combined_question_filters_assistant_messages(self):
        """Test that only user messages are included in combined question."""
        question = "New question"
        history = [
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "User message 2"}
        ]
        
        result = answer.combined_question(question, history)
        
        assert "User message 1" in result
        assert "User message 2" in result
        assert "Assistant response" not in result


class TestAnswerQuestion:
    """Test answer generation functionality."""
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.fetch_context')
    def test_answer_question_returns_tuple(self, mock_fetch, mock_llm):
        """Test that answer_question returns (answer, docs) tuple."""
        # Setup mocks
        mock_docs = [Mock(page_content="Context", metadata={"source": "test.md"})]
        mock_fetch.return_value = mock_docs
        
        mock_response = Mock()
        mock_response.content = "This is the answer."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        result = answer.answer_question("Test question")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)  # Answer
        assert isinstance(result[1], list)  # Documents
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.fetch_context')
    def test_answer_question_uses_context(self, mock_fetch, mock_llm):
        """Test that retrieved context is used in LLM prompt."""
        mock_docs = [
            Mock(page_content="Insurellm is a tech company.", metadata={"source": "test.md"})
        ]
        mock_fetch.return_value = mock_docs
        
        mock_response = Mock()
        mock_response.content = "Answer based on context."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        answer_text, docs = answer.answer_question("What is Insurellm?")
        
        # Verify LLM was called with messages containing context
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Check that context appears in system message
        system_message_content = str(call_args[0].content)
        assert "Insurellm is a tech company." in system_message_content
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.fetch_context')
    def test_answer_question_with_history(self, mock_fetch, mock_llm):
        """Test that conversation history is included in LLM messages."""
        mock_fetch.return_value = [Mock(page_content="Context", metadata={"source": "test.md"})]
        
        mock_response = Mock()
        mock_response.content = "Answer"
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        inference.answer_question("Current question", history)
        
        # Verify LLM was called
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Should have system message + history + current question
        assert len(call_args) >= 3
    
    @patch('implementation.inference.llm')
    @patch('implementation.inference.fetch_context')
    def test_answer_question_returns_source_docs(self, mock_fetch, mock_llm):
        """Test that source documents are returned for citation."""
        mock_docs = [
            Mock(page_content="Doc 1", metadata={"source": "source1.md"}),
            Mock(page_content="Doc 2", metadata={"source": "source2.md"})
        ]
        mock_fetch.return_value = mock_docs
        
        mock_response = Mock()
        mock_response.content = "Answer"
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        answer_text, returned_docs = answer.answer_question("Question")
        
        assert len(returned_docs) == 2
        assert returned_docs[0].page_content == "Doc 1"
        assert returned_docs[1].metadata["source"] == "source2.md"


class TestSystemPrompt:
    """Test system prompt configuration."""
    
    def test_system_prompt_has_context_placeholder(self):
        """Test that SYSTEM_PROMPT has context placeholder."""
        assert "{context}" in answer.SYSTEM_PROMPT
    
    def test_system_prompt_mentions_insurellm(self):
        """Test that system prompt mentions the company."""
        assert "Insurellm" in answer.SYSTEM_PROMPT


class TestConfiguration:
    """Test configuration constants."""
    
    def test_model_is_configured(self):
        """Test that MODEL is set."""
        assert answer.MODEL is not None
        assert isinstance(answer.MODEL, str)
    
    def test_retrieval_k_is_positive(self):
        """Test that RETRIEVAL_K is a positive integer."""
        assert answer.RETRIEVAL_K > 0
        assert isinstance(answer.RETRIEVAL_K, int)
    
    def test_db_name_is_path(self):
        """Test that DB_NAME is a valid path string."""
        assert answer.DB_NAME is not None
        assert isinstance(answer.DB_NAME, str)
        assert "vector_db" in answer.DB_NAME
