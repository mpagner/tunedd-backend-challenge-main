import requests
from tunedd_api.ingest.pdf_ingest import get_embedding
import os
import logging

from litellm import completion
from google.adk.tools import FunctionTool
from qdrant_client import QdrantClient, models

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL")
RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)

def generate_response(prompt: str):
    response = completion(
        model=f"ollama/{RAG_CHAT_MODEL}",
        messages=[{"role": "assistant", "content": prompt}],
    )
    return response.json()

def qdrant_search_tool(input_text: str) -> dict:
    """Searches a local knowledge base for information pertaining to the input_text.
    
    Use this search tool ONLY when the user asks questions whose terms are
    related to Artificial Intelligence, Agents or LLM.
    Do not use this for general inquiries.

    Args:
        input_text: The user's text input. 
    
    Returns:
        A dictionary with a status and a digest text.
    """
    logger.info(f"Tool executing: Search for '{input_text}'")

    adjusted_prompt = f"Represent this sentence for searching relevant passages: {input_text}"

    vector_data = get_embedding(adjusted_prompt)

    results = qdrant_client.query_points(
        collection_name="ai_docs",
        query=vector_data,
        limit=3
    )

    passages = "\n".join([f"- {point.payload['text']}" for point in results.points])

    augmented_prompt = f"""
      The following are relevant passages: 
      <retrieved-data>
      {passages}
      </retrieved-data>
      
      Here's the original user prompt, summarize the passages in order to answer the prompt:
      <user-prompt>
      {input_text}
      </user-prompt>
    """

    response = generate_response(augmented_prompt)
    
    return {
        "status": "success",
        "message": response,
    }
