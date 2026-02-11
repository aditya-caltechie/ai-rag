"""
Advanced RAG Ingestion Pipeline

Uses LLM-powered intelligent chunking with parallel processing:
1. Load documents from knowledge base
2. LLM splits each document with semantic understanding
3. Each chunk enhanced with headline + summary + original text
4. Generate embeddings and store in vector database
"""

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# Configuration
MODEL = "openai/gpt-4.1-nano"  # LLM for intelligent chunking
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100  # Words per chunk (guideline for LLM)
wait = wait_exponential(multiplier=1, min=10, max=240)  # Retry backoff strategy

WORKERS = 3  # Parallel workers (reduce to 1 if rate limited)

openai = OpenAI()


# Pydantic models for type-safe structured data

class Result(BaseModel):
    """Represents a processed chunk with content and metadata"""
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """LLM-generated chunk with enhanced retrieval metadata"""
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        """Combine headline, summary, and original text into final chunk format"""
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    """Container for multiple chunks returned by LLM"""
    chunks: list[Chunk]


def fetch_documents():
    """
    Step 1: Load all markdown documents from knowledge base.
    
    Scans folders and reads markdown files, preserving document type and source metadata.
    Custom implementation (no LangChain dependency).
    """
    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    """Create prompt instructing LLM to intelligently chunk the document"""
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    """Format document into LLM message format"""
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=wait)
def process_document(document):
    """
    Step 2: Use LLM to intelligently split document into enhanced chunks.
    
    LLM generates headline + summary + original text for each chunk.
    Retry logic handles rate limits and transient errors.
    """
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Process all documents in parallel with multiple workers.
    
    Uses multiprocessing to speed up LLM chunking.
    Progress bar tracks completion.
    Note: Reduce WORKERS to 1 if hitting rate limits.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    """
    Step 3: Generate embeddings and store in ChromaDB.
    
    Creates embeddings for all enhanced chunks and persists to vector database.
    Direct ChromaDB API usage (no LangChain wrapper).
    """
    # Initialize ChromaDB and clear existing collection
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Generate embeddings for all chunks
    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    # Store in ChromaDB
    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


# Main execution pipeline
if __name__ == "__main__":
    """
    Execute advanced RAG ingestion pipeline:
    
    1. Load documents from knowledge base
    2. LLM intelligently chunks each document (parallel processing)
    3. Generate embeddings and store enhanced chunks in ChromaDB
    """
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")

