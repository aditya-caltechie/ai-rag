## AGENTS — Project overview (ai-advance-rag)

This document is a **contributor-oriented map** of this repo: where things live, what each module does, and the common dev/run commands.

---

### Repository layout

```
ai-advance-rag/
├── .github/workflows/           # CI/CD (tests, lint, docs)
│   ├── test.yml                # Main test pipeline
│   ├── lint.yml                # Code quality checks
│   ├── docs.yml                # Documentation validation
│   └── ci.yml                  # Complete CI/CD pipeline
├── README.md                   # Project overview + quick start
├── .gitignore
├── .env.example                # Template for environment variables
├── pyproject.toml              # Dependencies (managed by uv)
├── uv.lock                     # Lockfile
├── docs/                       # Comprehensive documentation
│   ├── README.md              # Documentation index
│   ├── workflow_guide.md      # Step-by-step runbook
│   ├── architecture.md        # System architecture
│   ├── EVALS.md              # Evaluation methodology
│   ├── TESTS.md              # Test suite overview
│   └── ...                    # More guides
├── tests/                      # Unit + integration tests
│   ├── conftest.py            # Shared pytest fixtures
│   ├── unit/                  # Fast, isolated tests
│   └── integration/           # End-to-end tests
└── src/rag-pipeline/
    ├── config.py                    # RAG_MODE configuration
    ├── app.py                       # Gradio chatbot UI
    ├── evaluator.py                 # Gradio evaluation dashboard
    │
    ├── implementation/              # Basic RAG
    │   ├── ingest.py               # Rule-based chunking
    │   └── answer.py               # Single-query retrieval
    │
    ├── pro_implementation/          # Advanced RAG
    │   ├── ingest.py               # LLM-powered chunking
    │   └── answer.py               # Multi-stage retrieval
    │
    ├── evaluation/                  # Evaluation framework
    │   ├── eval.py                 # Metrics calculation (MRR, nDCG)
    │   ├── test.py                 # Test question loader
    │   └── tests.jsonl             # 150 test questions
    │
    └── knowledge-base/              # Source documents
        ├── company/
        ├── people/
        └── ...
```

Notes:

- This repo runs as a **regular Python app** (scripts under `src/rag-pipeline/`), not as a published package.
- Prefer running scripts via `uv run ...` to use the correct environment.
- The project demonstrates **Basic → Advanced RAG evolution** with comprehensive evaluation.

---

### Key components (where to start reading)

#### Configuration & Mode Switching

- **Configuration**: `src/rag-pipeline/config.py`
  - Loads `RAG_MODE` from `.env` (values: `"basic"` or `"pro"`)
  - Controls which implementation is used across all apps
  - No manual import changes needed

#### Basic RAG Implementation

- **Ingestion**: `src/rag-pipeline/implementation/ingest.py`
  - Loads markdown documents from `knowledge-base/`
  - Splits using `RecursiveCharacterTextSplitter` (500 chars, 200 overlap)
  - Creates embeddings with OpenAI `text-embedding-3-large`
  - Stores in ChromaDB (`vector_db/`)

- **Query pipeline**: `src/rag-pipeline/implementation/answer.py`
  - Retrieves top K=10 chunks via cosine similarity
  - Combines with conversation history
  - Generates answer using `ChatOpenAI(model="gpt-4.1-nano")`

#### Advanced RAG Implementation

- **Ingestion**: `src/rag-pipeline/pro_implementation/ingest.py`
  - LLM intelligently chunks documents with semantic understanding
  - Each chunk gets: `headline` + `summary` + `original_text`
  - Parallel processing with multiprocessing.Pool
  - Stores enhanced chunks in ChromaDB (`preprocessed_db/`)

- **Query pipeline**: `src/rag-pipeline/pro_implementation/answer.py`
  - **Step 1**: LLM rewrites query for better retrieval
  - **Step 2**: Dual retrieval (original + rewritten query, K=20 each)
  - **Step 3**: Merge and deduplicate chunks
  - **Step 4**: LLM reranks by semantic relevance
  - **Step 5**: Use top K=10 for answer generation

#### User Interfaces

- **Chatbot**: `src/rag-pipeline/app.py`
  - Gradio interface for interactive Q&A
  - Dynamically imports based on `RAG_MODE`
  - Shows retrieved context sources
  - Maintains conversation history

- **Evaluator**: `src/rag-pipeline/evaluator.py`
  - Gradio dashboard for metrics visualization
  - **Retrieval metrics**: MRR, nDCG, keyword coverage
  - **Answer quality**: Accuracy, completeness, relevance (LLM-as-judge)
  - Category-level analysis with charts

#### Evaluation Framework

- **Test data**: `src/rag-pipeline/evaluation/tests.jsonl`
  - 150 test questions across 7 categories
  - Each with: question, keywords, reference answer, category

- **Evaluation logic**: `src/rag-pipeline/evaluation/eval.py`
  - Calculates retrieval metrics (MRR, nDCG)
  - Uses LLM-as-judge for answer quality
  - Generates comprehensive reports

---

### Running locally

#### 1. Install dependencies

```bash
uv sync
```

#### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
# Set RAG_MODE=basic or RAG_MODE=pro
```

#### 3. Run Basic RAG

```bash
cd src/rag-pipeline

# Set RAG_MODE=basic in .env

# Ingest documents
uv run implementation/ingest.py

# Launch chatbot
uv run app.py

# Run evaluation
uv run evaluator.py
```

#### 4. Run Advanced RAG

```bash
cd src/rag-pipeline

# Set RAG_MODE=pro in .env

# Ingest with LLM chunking (slower, ~3-5 min for 76 docs)
uv run pro_implementation/ingest.py

# Launch chatbot (automatically uses pro mode)
uv run app.py

# Run evaluation (automatically uses pro mode)
uv run evaluator.py
```

#### 5. Run tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only (fast, ~5-10 seconds)
uv run pytest tests/unit/ -v

# Integration tests only (~20-30 seconds)
uv run pytest tests/integration/ -v

# With coverage
uv run pytest tests/ --cov=src/rag-pipeline --cov-report=html
```

---

*Last updated: February 8, 2026*
