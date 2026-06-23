"""Central configuration — all values loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
INDEX_NAME        = os.getenv("PINECONE_INDEX", "academic-advisor-v2")
EMBED_DIMENSION   = 1536
CLAUDE_MODEL      = "claude-3-5-sonnet-20241022"
MAX_TOKENS        = 2048
TEMPERATURE       = 0.4
MEMORY_WINDOW     = 8
RAG_TOP_K         = 5
CHUNK_SIZE        = 900
CHUNK_OVERLAP     = 150
