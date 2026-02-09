"""
Integration tests for the evaluation framework.

Tests retrieval and answer quality evaluation metrics.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))


class TestRetrievalEvaluation:
    """Test retrieval metrics calculation."""
    
    @patch('evaluation.eval.fetch_context')
    def test_mrr_calculation_perfect_score(self, mock_fetch):
        """Test MRR calculation when relevant doc is first."""
        from evaluation.eval import calculate_mrr
        from evaluation.test import TestQuestion
        
        # Mock retrieval: relevant doc is first
        mock_fetch.return_value = [
            Mock(page_content="Insurellm is an insurance company with AI technology."),
            Mock(page_content="Other content")
        ]
        
        test = TestQuestion(
            question="What is Insurellm?",
            keywords=["insurellm", "insurance", "company"],
            reference_answer="Insurellm is an insurance company.",
            category="direct_fact"
        )
        
        mrr = calculate_mrr([test])
        
        # Should be 1.0 (perfect - found at position 1)
        assert mrr == 1.0
    
    @patch('evaluation.eval.fetch_context')
    def test_mrr_calculation_second_position(self, mock_fetch):
        """Test MRR when relevant doc is second."""
        from evaluation.eval import calculate_mrr
        from evaluation.test import TestQuestion
        
        # Relevant keywords in second doc
        mock_fetch.return_value = [
            Mock(page_content="Unrelated content about something else."),
            Mock(page_content="Insurellm insurance company technology.")
        ]
        
        test = TestQuestion(
            question="What is Insurellm?",
            keywords=["insurellm", "insurance"],
            reference_answer="Answer",
            category="direct_fact"
        )
        
        mrr = calculate_mrr([test])
        
        # Should be 0.5 (1/2)
        assert mrr == 0.5
    
    @patch('evaluation.eval.fetch_context')
    def test_keyword_coverage_calculation(self, mock_fetch):
        """Test keyword coverage percentage."""
        from evaluation.eval import calculate_keyword_coverage
        from evaluation.test import TestQuestion
        
        # Mock retrieval with some keywords
        mock_fetch.return_value = [
            Mock(page_content="Insurellm is a company."),  # Has "insurellm" and "company"
            Mock(page_content="Technology solutions.")  # Has "technology"
        ]
        
        test = TestQuestion(
            question="What is Insurellm?",
            keywords=["insurellm", "company", "technology", "insurance"],  # 4 keywords, 3 found
            reference_answer="Answer",
            category="direct_fact"
        )
        
        coverage = calculate_keyword_coverage([test])
        
        # Should be 0.75 (3 out of 4 keywords found)
        assert coverage == 0.75


class TestAnswerEvaluation:
    """Test answer quality evaluation."""
    
    @patch('evaluation.eval.llm')
    @patch('evaluation.eval.answer_question')
    def test_answer_quality_scoring(self, mock_answer, mock_llm):
        """Test that LLM-as-judge scores answer quality."""
        from evaluation.eval import evaluate_answer_quality
        from evaluation.test import TestQuestion
        
        # Mock RAG answer
        mock_answer.return_value = ("Insurellm is an insurance tech company.", [])
        
        # Mock LLM judge response
        mock_judge_response = Mock()
        mock_judge_response.content = '''
        {
            "accuracy": 5,
            "completeness": 4,
            "relevance": 5
        }
        '''
        mock_llm.invoke = Mock(return_value=mock_judge_response)
        
        test = TestQuestion(
            question="What is Insurellm?",
            keywords=["insurellm"],
            reference_answer="Insurellm is an insurance technology company.",
            category="direct_fact"
        )
        
        scores = evaluate_answer_quality([test])
        
        assert scores["accuracy"] == 5.0
        assert scores["completeness"] == 4.0
        assert scores["relevance"] == 5.0
    
    @patch('evaluation.eval.llm')
    @patch('evaluation.eval.answer_question')
    def test_answer_evaluation_uses_reference_answer(self, mock_answer, mock_llm):
        """Test that evaluation compares against reference answer."""
        from evaluation.eval import evaluate_answer_quality
        from evaluation.test import TestQuestion
        
        mock_answer.return_value = ("Generated answer", [])
        
        mock_judge_response = Mock()
        mock_judge_response.content = '{"accuracy": 4, "completeness": 4, "relevance": 4}'
        mock_llm.invoke = Mock(return_value=mock_judge_response)
        
        test = TestQuestion(
            question="Test question",
            keywords=[],
            reference_answer="This is the reference answer that should appear in the prompt.",
            category="direct_fact"
        )
        
        evaluate_answer_quality([test])
        
        # Verify reference answer was passed to LLM judge
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        prompt_text = str(messages)
        
        assert "reference answer" in prompt_text.lower() or test.reference_answer in prompt_text


class TestEvaluationByCategory:
    """Test category-specific evaluation."""
    
    @patch('evaluation.eval.fetch_context')
    def test_evaluation_by_category(self, mock_fetch):
        """Test that evaluations can be broken down by category."""
        from evaluation.eval import evaluate_by_category
        from evaluation.test import TestQuestion
        
        mock_fetch.return_value = [
            Mock(page_content="Relevant content with keywords.")
        ]
        
        tests = [
            TestQuestion(
                question="Q1", keywords=["keyword"], reference_answer="A1", category="direct_fact"
            ),
            TestQuestion(
                question="Q2", keywords=["keyword"], reference_answer="A2", category="spanning"
            ),
            TestQuestion(
                question="Q3", keywords=["keyword"], reference_answer="A3", category="direct_fact"
            )
        ]
        
        results = evaluate_by_category(tests)
        
        # Should have results for each category
        assert "direct_fact" in results
        assert "spanning" in results
        
        # direct_fact should have 2 tests
        assert results["direct_fact"]["count"] == 2
        assert results["spanning"]["count"] == 1


class TestEvaluationPipeline:
    """Test complete evaluation pipeline."""
    
    @patch('evaluation.eval.llm')
    @patch('evaluation.eval.answer_question')
    @patch('evaluation.eval.fetch_context')
    @patch('evaluation.eval.load_tests')
    def test_full_evaluation_pipeline(self, mock_load_tests, mock_fetch, mock_answer, mock_llm):
        """Test running complete evaluation suite."""
        from evaluation.eval import run_full_evaluation
        from evaluation.test import TestQuestion
        
        # Mock test data
        mock_load_tests.return_value = [
            TestQuestion(
                question="What is Insurellm?",
                keywords=["insurellm", "company"],
                reference_answer="Insurellm is a company.",
                category="direct_fact"
            )
        ]
        
        # Mock retrieval
        mock_fetch.return_value = [
            Mock(page_content="Insurellm is an insurance technology company.")
        ]
        
        # Mock answer generation
        mock_answer.return_value = ("Insurellm is a company.", [])
        
        # Mock LLM judge
        mock_judge = Mock()
        mock_judge.content = '{"accuracy": 5, "completeness": 5, "relevance": 5}'
        mock_llm.invoke = Mock(return_value=mock_judge)
        
        results = run_full_evaluation()
        
        # Should have both retrieval and answer metrics
        assert "retrieval" in results
        assert "answer" in results
        
        assert "mrr" in results["retrieval"]
        assert "ndcg" in results["retrieval"]
        assert "coverage" in results["retrieval"]
        
        assert "accuracy" in results["answer"]
        assert "completeness" in results["answer"]
        assert "relevance" in results["answer"]
    
    @patch('evaluation.eval.fetch_context')
    @patch('evaluation.eval.load_tests')
    def test_evaluation_handles_multiple_test_cases(self, mock_load_tests, mock_fetch):
        """Test evaluation with multiple test cases."""
        from evaluation.eval import calculate_mrr
        from evaluation.test import TestQuestion
        
        # Multiple test cases
        tests = [
            TestQuestion(q="Q1", keywords=["kw1"], reference_answer="A1", category="c1"),
            TestQuestion(q="Q2", keywords=["kw2"], reference_answer="A2", category="c2"),
            TestQuestion(q="Q3", keywords=["kw3"], reference_answer="A3", category="c3")
        ]
        mock_load_tests.return_value = tests
        
        # Mock different retrieval qualities
        call_count = 0
        def mock_retrieval(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [Mock(page_content="kw1 content")]  # Position 1
            elif call_count == 2:
                return [
                    Mock(page_content="irrelevant"),
                    Mock(page_content="kw2 content")  # Position 2
                ]
            else:
                return [
                    Mock(page_content="irrelevant"),
                    Mock(page_content="irrelevant"),
                    Mock(page_content="kw3 content")  # Position 3
                ]
        
        mock_fetch.side_effect = mock_retrieval
        
        mrr = calculate_mrr(tests)
        
        # MRR = average of (1/1 + 1/2 + 1/3) / 3 = (1 + 0.5 + 0.333) / 3 ≈ 0.611
        assert 0.6 <= mrr <= 0.65


class TestEvaluationMetrics:
    """Test specific metric calculations."""
    
    def test_mrr_formula(self):
        """Test MRR calculation formula."""
        from evaluation.eval import calculate_mrr_for_positions
        
        positions = [1, 2, 3, 1, 1]  # Positions where relevant docs were found
        
        # MRR = (1/1 + 1/2 + 1/3 + 1/1 + 1/1) / 5 = (1 + 0.5 + 0.333 + 1 + 1) / 5 = 0.767
        mrr = calculate_mrr_for_positions(positions)
        
        assert 0.76 <= mrr <= 0.77
    
    def test_ndcg_calculation(self):
        """Test nDCG metric calculation."""
        from evaluation.eval import calculate_ndcg
        
        # Ideal ranking: [3, 2, 1, 0]
        # Actual ranking: [2, 3, 0, 1]
        ideal_relevance = [3, 2, 1, 0]
        actual_relevance = [2, 3, 0, 1]
        
        ndcg = calculate_ndcg(actual_relevance, ideal_relevance)
        
        # nDCG should be between 0 and 1
        assert 0.0 <= ndcg <= 1.0
        
        # nDCG should be less than 1 since actual != ideal
        assert ndcg < 1.0
