"""
Advanced RAG Answer Pipeline

Multi-stage retrieval with LLM optimization:
1. Query rewriting: LLM refines user question for better retrieval
2. Dual retrieval: Search with both original and rewritten queries
3. LLM reranking: Reorder chunks by semantic relevance
4. Answer generation: Use top-ranked context for accurate responses
"""

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# Configuration
# MODEL = "openai/gpt-4.1-nano"  # Alternative: OpenAI model
MODEL = "groq/openai/gpt-oss-120b"  # Fast open-source model via Groq
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

collection_name = "docs"
embedding_model = "text-embedding-3-large"
wait = wait_exponential(multiplier=1, min=10, max=240)  # Retry backoff

# Initialize clients
openai = OpenAI()
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

# Retrieval configuration
RETRIEVAL_K = 20  # Initial broad retrieval (per query)
FINAL_K = 10      # Final top chunks after reranking

# System prompt emphasizes accuracy and completeness
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


# Pydantic models for type-safe structured responses

class Result(BaseModel):
    """Represents a retrieved chunk with content and metadata"""
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    """LLM-generated ranking order for chunks"""
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait)
def rerank(question, chunks):
    """
    Use LLM to rerank chunks by semantic relevance to the question.
    
    Better than cosine similarity because LLM understands:
    - Semantic meaning and context
    - Query intent
    - Subtle relevance signals
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks):
    """Build message chain with context, history, and current question"""
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question, history=[]):
    """
    LLM rewrites user's question for better retrieval.
    
    Improves search by:
    - Extracting key terms
    - Removing conversational language
    - Focusing on specific details
    """
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    """Combine chunks from two queries, deduplicating by content"""
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    """Retrieve top K chunks via vector similarity search"""
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question):
    """
    Advanced multi-stage retrieval pipeline:
    
    1. LLM rewrites query for better retrieval
    2. Retrieve with original question (K=20)
    3. Retrieve with rewritten question (K=20)
    4. Merge and deduplicate chunks
    5. LLM reranks by relevance
    6. Return top K=10 chunks
    
    This catches more relevant docs than single-query retrieval.
    """
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Generate answer using advanced RAG pipeline.
    
    Returns:
        tuple: (answer_text, list_of_context_chunks)
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

