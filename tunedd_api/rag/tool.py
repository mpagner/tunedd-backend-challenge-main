from tunedd_api.ingest.pdf_ingest import get_embedding
import logging

from tunedd_api.settings import settings

from litellm import completion
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=settings.QDRANT_URL)

def generate_response(prompt: str):
    response = completion(
        model=f"ollama/{settings.RAG_CHAT_MODEL}",
        messages=[{"role": "assistant", "content": prompt}],
        api_base=settings.OLLAMA_BASE_URL
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
        A dictionary with a status and the retrieved passages as a digest.
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
    
    return {
        "status": "success",
        "message": passages,
    }
