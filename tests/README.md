# RAG Pipeline Tests

Comprehensive test suite for both basic and advanced RAG implementations.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests for individual components
│   ├── test_basic_ingest.py      # Basic ingestion tests
│   ├── test_basic_answer.py      # Basic answer pipeline tests
│   ├── test_pro_ingest.py        # Advanced ingestion tests
│   └── test_pro_answer.py        # Advanced answer pipeline tests
└── integration/             # Integration tests for complete workflows
    ├── test_basic_rag_pipeline.py   # End-to-end basic RAG tests
    ├── test_pro_rag_pipeline.py     # End-to-end advanced RAG tests
    └── test_evaluation.py           # Evaluation framework tests
```

## Running Tests

### Prerequisites

```bash
# Install dependencies including pytest
uv sync
```

### Run All Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src/rag-pipeline --cov-report=html
```

### Run Specific Test Suites

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Specific test file
uv run pytest tests/unit/test_basic_ingest.py

# Specific test class
uv run pytest tests/unit/test_basic_ingest.py::TestFetchDocuments

# Specific test function
uv run pytest tests/unit/test_basic_ingest.py::TestFetchDocuments::test_fetch_documents_with_sample_kb
```

### Run with Markers

```bash
# Run fast tests only (unit tests)
uv run pytest tests/ -m "not slow"

# Run integration tests only
uv run pytest tests/integration/

# Run tests matching a keyword
uv run pytest tests/ -k "ingest"
```

## Test Coverage

### Unit Tests

Unit tests focus on individual functions and components:

- **Ingestion Tests**: Document loading, chunking, embedding creation
- **Answer Tests**: Context retrieval, question processing, LLM interaction
- **Pydantic Models**: Data validation and serialization
- **Configuration**: Environment variables and constants

Key features:
- ✅ Isolated components with mocked dependencies
- ✅ Fast execution (no API calls)
- ✅ Deterministic results
- ✅ Test edge cases and error handling

### Integration Tests

Integration tests verify complete workflows:

- **Full Pipeline Tests**: Ingestion → Storage → Retrieval → Answer
- **Multi-turn Conversations**: Context preservation across queries
- **Advanced Features**: Query rewriting, dual retrieval, reranking
- **Error Handling**: Graceful degradation with missing data

Key features:
- ✅ Components working together
- ✅ Real data flow (with mocked APIs)
- ✅ End-to-end user scenarios
- ✅ Performance characteristics

## Test Fixtures

Shared fixtures in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_documents`: Mock document data
- `sample_chunks`: Pre-chunked text samples
- `mock_embeddings`: Fake embedding function
- `mock_llm_response`: Mocked LLM output
- `sample_test_questions`: Evaluation test data
- `mock_vector_store`: Mocked ChromaDB store
- `mock_openai_client`: Mocked OpenAI API client
- `sample_knowledge_base`: Temporary KB with sample files
- `conversation_history`: Multi-turn conversation data

## Writing New Tests

### Unit Test Example

```python
import pytest
from unittest.mock import patch, MagicMock

def test_my_function():
    """Test that my_function works correctly."""
    # Arrange: Setup test data
    test_input = "test"
    
    # Act: Execute the function
    result = my_function(test_input)
    
    # Assert: Verify the result
    assert result == expected_output
```

### Integration Test Example

```python
@patch('module.external_api')
def test_full_pipeline(mock_api):
    """Test complete workflow from input to output."""
    # Setup mocks
    mock_api.call.return_value = {"data": "mocked"}
    
    # Run pipeline
    documents = load_documents()
    chunks = process(documents)
    result = generate_output(chunks)
    
    # Verify end-to-end flow
    assert result is not None
    mock_api.call.assert_called_once()
```

## Common Test Patterns

### Mocking LLM Calls

```python
@patch('implementation.answer.llm')
def test_with_mocked_llm(mock_llm):
    mock_response = Mock()
    mock_response.content = "Mocked answer"
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    result = answer_question("Question")
    
    assert result[0] == "Mocked answer"
```

### Mocking Vector Store

```python
@patch('implementation.answer.retriever')
def test_with_mocked_retriever(mock_retriever):
    mock_docs = [Mock(page_content="Doc", metadata={})]
    mock_retriever.invoke = MagicMock(return_value=mock_docs)
    
    docs = fetch_context("Query")
    
    assert len(docs) == 1
```

### Using Temporary Files

```python
def test_with_temp_files(temp_dir):
    # temp_dir is a fixture that's automatically cleaned up
    test_file = temp_dir / "test.md"
    test_file.write_text("Test content")
    
    result = process_file(test_file)
    
    assert result is not None
```

## Continuous Integration

Tests are designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    uv sync
    uv run pytest tests/ --cov=src --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Debugging Tests

### Run with Print Statements

```bash
# See print output
uv run pytest tests/ -s

# See detailed output
uv run pytest tests/ -vv
```

### Run with Debugger

```bash
# Drop into debugger on failure
uv run pytest tests/ --pdb

# Drop into debugger at start of test
uv run pytest tests/ --trace
```

### Show Warnings

```bash
# Show all warnings
uv run pytest tests/ -W all
```

## Best Practices

✅ **Do:**
- Write descriptive test names explaining what is tested
- Use fixtures for common setup
- Mock external dependencies (APIs, databases)
- Test both success and failure cases
- Keep tests independent (no shared state)
- Use assertions with clear error messages

❌ **Don't:**
- Make real API calls in tests
- Rely on external files or services
- Share state between tests
- Write tests that depend on execution order
- Skip cleanup of temporary resources

## Troubleshooting

### Import Errors

If you get import errors, ensure the path is added:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "rag-pipeline"))
```

### Fixture Not Found

Make sure `conftest.py` is in the correct location (tests root).

### Mock Not Working

Verify the patch path matches the actual import location:

```python
# If code does: from implementation.answer import llm
# Then patch:   @patch('implementation.answer.llm')
```

## Coverage Goals

Target coverage levels:

- **Unit Tests**: 80%+ coverage of core functions
- **Integration Tests**: 70%+ coverage of pipelines
- **Critical Paths**: 100% coverage (ingestion, retrieval, answer)

## Contributing

When adding new features:

1. Write unit tests for new functions
2. Write integration tests for new workflows
3. Update this README if adding new test categories
4. Ensure all tests pass before committing

---

**Test Philosophy**: Tests should be fast, reliable, and provide clear feedback when things break.

*Last updated: February 2026*
