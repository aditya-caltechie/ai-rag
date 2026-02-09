import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================
# CHANGE THIS TO SWITCH IMPLEMENTATIONS
# ============================================
RAG_MODE = os.getenv("RAG_MODE", "basic")  # Options: "basic" or "pro"
# ============================================

# Validate configuration
if RAG_MODE not in ["basic", "pro"]:
    raise ValueError(f"RAG_MODE must be 'basic' or 'pro', got: {RAG_MODE}")

# Display current mode
print(f"🔧 RAG Mode: {RAG_MODE.upper()}")
if RAG_MODE == "basic":
    print("   Using: implementation/ (Simple, Fast, Lower Cost)")
elif RAG_MODE == "pro":
    print("   Using: pro_implementation/ (Advanced, Slower, Higher Quality)")
