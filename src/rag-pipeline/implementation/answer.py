"""
RAG Answer Pipeline

Handles question-answering using Retrieval-Augmented Generation:
1. Retrieve relevant context from vector database
2. Combine context with conversation history
3. Generate answer using LLM with retrieved context
"""

from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv(override=True)

# Configuration
MODEL = "gpt-4.1-nano"  # LLM model for generating answers
DB_NAME = str(Path(__file__).parent.parent / "vector_db")  # Path to vector database

# Initialize embeddings (must match the model used during ingestion)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Alternative: local embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Number of documents to retrieve from vector store
RETRIEVAL_K = 10

# System prompt template with placeholder for retrieved context
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Initialize RAG components
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)  # Load persisted vector DB
retriever = vectorstore.as_retriever()  # Create retriever interface for similarity search
llm = ChatOpenAI(temperature=0, model_name=MODEL)  # Initialize LLM with deterministic output


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    Uses semantic similarity search to find the most relevant chunks
    from the vector database.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine conversation history with current question.
    Concatenates all previous user questions to provide better context
    for retrieval, improving multi-turn conversation relevance.
    """
    # Extract only user messages from history
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    Pipeline:
    1. Combine current question with conversation history
    2. Retrieve relevant context documents from vector store
    3. Format system prompt with retrieved context
    4. Build message history for LLM
    5. Generate answer using LLM with context
    
    Args:
        question: Current user question
        history: Conversation history as list of message dicts
        
    Returns:
        Tuple of (answer_text, context_documents)
    """
    # Step 1: Combine question with history for better context retrieval
    combined = combined_question(question, history)
    
    # Step 2: Retrieve relevant documents using semantic search
    docs = fetch_context(combined)
    
    # Step 3: Format context from retrieved documents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Step 4: Build system prompt with context
    system_prompt = SYSTEM_PROMPT.format(context=context)
    
    # Step 5: Construct message chain with system prompt, history, and current question
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))  # Add conversation history
    messages.append(HumanMessage(content=question))  # Add current question
    
    # Step 6: Generate answer from LLM
    response = llm.invoke(messages)
    
    # Return both answer and source documents (for citation/verification)
    return response.content, docs
