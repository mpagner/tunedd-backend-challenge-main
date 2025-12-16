from ..ingest.pdf_ingest import get_embedding
import os
import logging

from google.adk.tools import FunctionTool
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)

def search_documents(query: str) -> str:
    """Searches the internal knowledge base for relevant snippets."""
    logger.info(f"Tool executing: Search for '{query}'")
    embedding = get_embedding(query)
    
    hits = qdrant_client.search(
        collection_name="ai_docs",
        query_vector=embedding,
        limit=3
    )
    
    if not hits:
        return "No relevant information found in the knowledge base."
    
    results = "\n\n".join([f"Source ({h.payload['source']}): {h.payload['text']}" for h in hits])
    return results

# Register tool with ADK
rag_tool = FunctionTool(func=search_documents)