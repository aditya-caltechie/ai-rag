

## RAG = Retrieval-Augmented Generation

It's an architecture that improves answers from LLMs by doing this in two main steps at query time:

- Retrieval — search a knowledge base (your documents, PDFs, database records, wiki pages, etc.) → find the most relevant chunks of text

- Augmented Generation — put those relevant chunks into the LLM's context window → let the LLM generate a much more accurate, up-to-date, and grounded answer

Without RAG → LLM only knows what it was trained on (cutoff date + general knowledge)
With RAG → LLM gets fresh/private/company-specific context on every question

### So what does "RAG deployment" specifically refer to?

It refers to all the engineering work needed to run a RAG system reliably in production, not just in a Jupyter notebook or local demo. This usually includes most or all of these aspects:

| Area | What "deployment" typically involves | Why it's hard / important |
|------|--------------------------------------|---------------------------|
| Ingestion pipeline | PDF → text → chunking → embedding creation → vector DB upsert (ongoing / incremental updates) | Data freshness, deduplication, handling updates/deletes |
| Vector database | Choose & operate Pinecone, Weaviate, Qdrant, pgvector, Elasticsearch, Redis, Astra DB, etc. | Latency, scale, hybrid search, metadata filtering |
| Retrieval logic | Hybrid search (dense + sparse), reranking (cross-encoder / LLM reranker), query rewriting/expansion | Recall & precision — bad retrieval = bad answers |
| Prompt engineering | Good system prompt + context formatting + citation instructions | Reduces hallucinations, improves answer quality |
| LLM inference | Hosting OpenAI, Anthropic, Grok, Llama-3.1/DeepSeek, Mistral, etc. (self-hosted or API) | Cost, speed, rate limits, fallback models |
| Serving layer | FastAPI / LangServe / Haystack / LlamaIndex Serve / BentoML / vLLM + streaming | Low latency, concurrency, rate limiting |
| Monitoring & eval | Answer quality (RAGAS, DeepEval), retrieval metrics (nDCG, recall@k), hallucinations, latency | You cannot improve what you don't measure |
| Security | AuthN/AuthZ, data isolation, PII redaction, prompt injection defense | Enterprise must-have |
| Freshness pipeline | Real-time / near real-time updates (webhooks, CDC, scheduled jobs) | Stale data = outdated answers |
| Cost & scaling | Auto-scaling, caching (semantic cache), query classification (some questions don't need RAG) | Production bills can explode quickly |

## Quick summary table of stages people actually use

Stage 0 — Notebook hell (LangChain / LlamaIndex notebook)
Stage 1 — Local demo (exactly what you described: Gradio / Streamlit + Ollama / local vLLM + Chroma / FAISS)
Stage 2 — "Internal tool" deployment (Docker Compose + vLLM + persistent vector DB + simple auth)
Stage 3 — Production / "real" RAG deployment (cloud-native, scalable, observable, monitored, secure, continuously updated data)

When engineers / companies say "we deployed RAG" or "RAG deployment is hard", they almost always mean stage 3, not stage 1.
Gradio + Ollama is fantastic for getting started fast and validating the idea (many production systems began exactly like that!), but it's rarely kept as-is when real people start depending on it.


#### Here is a realistic, prioritized roadmap / checklist that most teams follow when going from prototype → production-grade RAG.

### 1. Strong Retrieval Foundation (the #1 make-or-break factor)
Bad retrieval → bad everything else. Fix this first.

- Chunking strategy — move beyond fixed-size 512 tokens
Recursive character + sentence-aware + metadata-preserving (LlamaIndex / LangChain default recursive + headers)
Semantic / proposition-based chunking or small-to-big retrieval (very popular in 2025–2026)
Overlap 15–30%, experiment with 300–800 token chunks

- Hybrid retrieval (dense + sparse) as default
Dense: bge-large-en-v1.5 / snowflake-arctic-embed-m / voyage-3-large / mixedbread-ai
Sparse: BM25 or SPLADE
Weighted fusion or reciprocal rank fusion

Re-ranking — almost always worth it
Cohere Rerank3 / bge-reranker-v2 / flashrank / jina-reranker-v2
Rerank top-20–50 → final top-5–8

Metadata filtering + query routing early
Pre-filter by date / department / doc type before vector search
Classify query → simple → no RAG / complex → full RAG / needs-tool → agent


### 2. Data Pipeline & Freshness (the silent killer in production)
Static indexes die fast.

Ingestion as ETL / streaming pipeline
Unstructured.io / LlamaParse / Docling for PDF / tables / images
Airbyte / Kafka / webhook / CDC for live sources
Incremental upsert + delete-by-metadata + deduplication

Refresh cadence — decide per use-case
Real-time (<5 min): news/support tickets → use change streams
Daily/weekly: policies/handbooks

Version documents + embeddings (avoid silent drift)

### 3. Generation & Prompt Layer

Strict system prompt templates (few-shot examples + citation rules)
Chain-of-verification / self-reflection / corrective RAG patterns if hallucination still high
Context compression / long-context re-ranking if context > 32k tokens
Fallbacks: no relevant docs → honest "I don't know" + web fallback if allowed

### 4. Serving & Infrastructure

API layer — FastAPI / LitServe / LangServe / BentoML
LLM backend — vLLM / TGI / SGLang / LMDeploy (for self-hosted) + autoscaling
Or managed: OpenAI / Anthropic / Grok / Fireworks / Together

Vector DB choice (pick one and commit)

| Scenario | Recommendation (2026) | Why |
|----------|------------------------|-----|
| < 1M vectors, simple | pgvector / Chroma persistent | Cheap, easy |
| 1M–50M, high QPS | Qdrant / Weaviate / Milvus | Good hybrid + filtering + speed |
| Serverless / very large | Pinecone serverless / Zilliz | Zero ops, auto-scale |
| Already on cloud DB | MongoDB Atlas / Redis / Elastic | Leverage existing infra |

Caching — exact semantic cache (Redis / GPTCache) + exact-match cache
Rate limiting + query classification (avoid RAG on chit-chat)

### 5. Evaluation & Monitoring (without this you fly blind)
This is where 70–80% of teams fail long-term.

- Offline eval dataset — 100–500 golden question-answer pairs (build from logs)
Metrics: context relevance, faithfulness, answer correctness (RAGAS / DeepEval / G-Eval / LLM-as-judge)
Retrieval-specific: nDCG@10, recall@K, MRR

- Online monitoring
Langfuse / Phoenix / Helicone / PromptLayer / TruLens
Log every query + retrieved chunks + final answer + latency + cost
Auto-eval on production logs + user thumbs-up/down feedback loop
Alerts on ↑ hallucination rate / ↑ latency / ↓ retrieval quality

CI/CD for RAG — prompt versioned, eval runs on PRs, regression tests

### 6. Security, Compliance & Cost Control

- AuthN/Z (JWT / OAuth)
- PII redaction / guardrails (Nemo Guardrails / Lakera / Patronus)
- Data residency / encryption at rest
- Cost — semantic cache hit rate > 40%, route easy questions away from RAG+LLM
- Auto-scaling groups + spot instances where possible

### Typical Progression Timeline (real teams in 2025–2026)

MonthFocusTypical Stack Example0–1Get good offline metrics on ~200 q-aLlamaIndex / LangChain + Qdrant local + vLLM local1–3Hybrid + rerank + eval dataset + monitoringAdd Cohere rerank + Langfuse + pgvector3–6Ingestion pipeline + freshness + APIKafka / webhook → Airflow + FastAPI + autoscaling6+Advanced (GraphRAG / multi-modal / agents) + cost optimizationIf needed: Neo4j + multi-modal embeddings

### Quick "Production Readiness" Scorecard
Score yourself 1–10 on each (honestly):

Retrieval quality (recall + precision) ≥ 85–90% on golden set?
End-to-end faithfulness ≥ 90% (no made-up facts)?
P50 latency < 2.5 s, P99 < 6 s?
Data refreshes automatically without downtime?
Full observability (can debug why a bad answer happened)?
Cost per 1,000 queries < target (e.g. <$0.50–$2)?
Security & access controls in place?
