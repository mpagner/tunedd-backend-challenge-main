import requests
from tunedd_api.ingest.pdf_ingest import get_embedding
import os
import logging

from tunedd_api.settings import settings

from litellm import completion
from google.adk.tools import FunctionTool
from qdrant_client import QdrantClient, models

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=settings.QDRANT_URL)

def generate_response(prompt: str):
    response = completion(
        model=f"ollama/{settings.RAG_CHAT_MODEL}",
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
        collection_name=settings.QDRANT_COLLECTION,
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
