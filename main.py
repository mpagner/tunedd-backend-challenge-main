import logging
from contextlib import asynccontextmanager
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from tunedd_api.ingest.pdf_ingest import ingest_data
from tunedd_api.agent import root_agent


logger = logging.getLogger(__name__)

#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
#)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def background_ingest():
        try:
            await asyncio.to_thread(ingest_data)
        except Exception:
            logger.exception("Ingestion failed")

    asyncio.create_task(background_ingest())
    yield

app = FastAPI(lifespan=lifespan, title="ADK RAG API")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)