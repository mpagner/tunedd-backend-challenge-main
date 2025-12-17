
import os
from pathlib import Path
import glob
import logging
from dotenv import load_dotenv
from pypdf import PdfReader

from qdrant_client import QdrantClient, models
import litellm

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#litellm._turn_on_debug()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL, timeout=30.0)

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL")
COLLECTION_NAME = "ai_docs"

PDF_DIR = Path("data/ai-agents-arxiv-papers")

def get_embedding(text: str):
    """Generates embeddings using LiteLLM/Ollama."""
    response = litellm.embedding(
        model=f"ollama/{EMBEDDING_MODEL}",
        input=text,
    )
    return response['data'][0]['embedding']

def ingest_data():
    """Reads files and indexes them into Qdrant."""
    logger.info("Checking Qdrant collection...")
    
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION_NAME}'.")

        files = glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
        if not files:
            logger.warning("No files found in ./data to ingest.")
            return


        logger.info("Adding...")

        points = []
        idx = 0
        for file_path in files:
            logger.info(f"Processing {file_path}...")
            try:
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n\n"

                # Simple chunking by paragraphs
                chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                
                for chunk in chunks:
                    vector = get_embedding(chunk)
                    points.append(models.PointStruct(
                        id=idx,
                        vector=vector,
                        payload={"source": file_path, "text": chunk}
                    ))
                    idx += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Ingested {len(points)} chunks into Qdrant.")
    else:
        logger.info(f"Collection already exists.")