"""
Pytest configuration and shared fixtures for RAG pipeline tests.

Provides common test fixtures, mock data, and test utilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "page_content": "Insurellm is an insurance technology company that provides AI-powered solutions.",
            "metadata": {"source": "company_overview.md", "doc_type": "company"}
        },
        {
            "page_content": "The CEO of Insurellm is Avery Lancaster. She founded the company in 2020.",
            "metadata": {"source": "leadership.md", "doc_type": "people"}
        },
        {
            "page_content": "Insurellm offers three main products: ClaimAssist, PolicyAI, and RiskAnalyzer.",
            "metadata": {"source": "products.md", "doc_type": "products"}
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        "Insurellm is an insurance technology company.",
        "The CEO is Avery Lancaster.",
        "ClaimAssist helps process insurance claims faster.",
        "PolicyAI provides personalized policy recommendations.",
        "RiskAnalyzer uses machine learning for risk assessment."
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embedding function that returns fake vectors."""
    def embed_fn(texts):
        # Return fake embeddings (768-dimensional vectors)
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1] * 768 for _ in texts]
    return embed_fn


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock = MagicMock()
    mock.content = "This is a test response from the LLM."
    return mock


@pytest.fixture
def sample_test_questions():
    """Sample test questions for evaluation testing."""
    return [
        {
            "question": "What is Insurellm?",
            "keywords": ["insurance", "technology", "company"],
            "reference_answer": "Insurellm is an insurance technology company.",
            "category": "direct_fact"
        },
        {
            "question": "Who is the CEO?",
            "keywords": ["CEO", "Avery", "Lancaster"],
            "reference_answer": "The CEO of Insurellm is Avery Lancaster.",
            "category": "direct_fact"
        }
    ]


@pytest.fixture
def mock_vector_store(temp_dir):
    """Mock vector store for testing retrieval."""
    mock_store = MagicMock()
    mock_store.persist_directory = str(temp_dir / "vector_db")
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke = MagicMock(return_value=[
        Mock(page_content="Test content 1", metadata={"source": "test1.md"}),
        Mock(page_content="Test content 2", metadata={"source": "test2.md"})
    ])
    mock_store.as_retriever = MagicMock(return_value=mock_retriever)
    
    return mock_store


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Mock embeddings.create()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 3072  # text-embedding-3-large dimensions
    mock_embeddings_response = MagicMock()
    mock_embeddings_response.data = [mock_embedding_data]
    mock_client.embeddings.create = MagicMock(return_value=mock_embeddings_response)
    
    return mock_client


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection for testing."""
    mock_collection = MagicMock()
    mock_collection.count = MagicMock(return_value=100)
    mock_collection.query = MagicMock(return_value={
        "documents": [["Test document 1", "Test document 2"]],
        "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
        "distances": [[0.1, 0.2]]
    })
    return mock_collection


@pytest.fixture
def sample_knowledge_base(temp_dir):
    """Create a sample knowledge base directory structure."""
    kb_path = temp_dir / "knowledge-base"
    kb_path.mkdir()
    
    # Create company folder
    company_folder = kb_path / "company"
    company_folder.mkdir()
    (company_folder / "overview.md").write_text(
        "# Insurellm\n\nInsurellm is an insurance technology company."
    )
    
    # Create people folder
    people_folder = kb_path / "people"
    people_folder.mkdir()
    (people_folder / "leadership.md").write_text(
        "# Leadership\n\nCEO: Avery Lancaster\nCTO: Sam Chen"
    )
    
    return kb_path


@pytest.fixture
def env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-key-12345"
    os.environ["RAG_MODE"] = "basic"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "What is Insurellm?"},
        {"role": "assistant", "content": "Insurellm is an insurance technology company."},
        {"role": "user", "content": "Who founded it?"}
    ]
