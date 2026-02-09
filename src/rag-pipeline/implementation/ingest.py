"""
RAG Pipeline Ingestion Script

This module handles the ingestion pipeline for a Retrieval-Augmented Generation (RAG) system:
1. Load documents from the knowledge base directory
2. Split documents into smaller chunks for better retrieval
3. Generate embeddings and store them in a vector database

The pipeline processes markdown files and creates a searchable vector store using ChromaDB.
"""

import os
import glob
from pathlib import Path

# LangChain imports for document loading and processing
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Embedding model options
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Configuration constants
MODEL = "gpt-4.1-nano"  # LLM model for downstream RAG tasks

# Directory paths for vector database and source documents
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Alternative: Use local HuggingFace embeddings (no API key required)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv(override=True)

# Initialize OpenAI embeddings model (3072 dimensions)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Step 1: Load all documents from the knowledge base.
    
    Recursively scans subdirectories in the knowledge base folder, loads all markdown files,
    and adds document type metadata based on the folder name.
    
    Returns:
        list: List of LangChain Document objects with metadata
    """
    # Get all subdirectories in the knowledge base
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    
    # Process each folder (e.g., "research", "docs", etc.)
    for folder in folders:
        # Extract folder name to use as document type
        doc_type = os.path.basename(folder)
        
        # Load all markdown files from the folder and its subdirectories
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        
        # Add document type to metadata for filtering during retrieval
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    
    return documents


def create_chunks(documents):
    """
    Step 2: Split documents into smaller chunks for optimal retrieval.
    
    Uses RecursiveCharacterTextSplitter to break down large documents while:
    - Maintaining semantic coherence within chunks
    - Creating overlap between chunks to preserve context at boundaries
    - Keeping chunks within a size suitable for embedding models
    """
    # Configure text splitter with chunk size and overlap
    # chunk_size=500: Maximum characters per chunk (balance between context and specificity)
    # chunk_overlap=200: Characters shared between adjacent chunks (preserves context at boundaries)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def create_embeddings(chunks):
    """
    Step 3: Generate embeddings and store in vector database.
    
    Converts text chunks into vector embeddings using OpenAI's embedding model
    and stores them in ChromaDB for efficient similarity search during retrieval.
    """
    # Delete existing vector database if it exists to ensure a fresh start
    # This prevents duplicate or stale embeddings from previous runs
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Create vector store by generating embeddings for all document chunks
    # and persisting them to disk for future retrieval
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    # Access the underlying collection to retrieve metadata about the vector store
    collection = vectorstore._collection
    count = collection.count()

    # Get a sample embedding to determine the dimensionality of the vectors
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    
    # Display summary statistics about the created vector store
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    
    return vectorstore


# Main execution pipeline
if __name__ == "__main__":
    """
    Execute the complete RAG ingestion pipeline:
    
    1. fetch_documents(): Load all markdown files from knowledge base
    2. create_chunks(): Split documents into smaller, overlapping chunks
    3. create_embeddings(): Generate vector embeddings and store in ChromaDB
    
    This creates a searchable vector database that can be used for
    semantic search and retrieval in the RAG application.
    """
    print("Starting RAG ingestion pipeline...")
    
    # Step 1: Load documents from knowledge base
    print("\n[Step 1/3] Loading documents from knowledge base...")
    documents = fetch_documents()
    print(f"✓ Loaded {len(documents)} documents")
    
    # Step 2: Split documents into chunks
    print("\n[Step 2/3] Splitting documents into chunks...")
    chunks = create_chunks(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings and store in vector database
    print("\n[Step 3/3] Generating embeddings and creating vector store...")
    create_embeddings(chunks)
    
    print("\n✓ Ingestion complete! Vector database is ready for queries.")

