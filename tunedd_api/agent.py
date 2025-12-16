import os
from dotenv import load_dotenv

import litellm
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.models.registry import LLMRegistry

from .rag.tool import rag_tool

load_dotenv()

RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
# Setup LiteLLM to point to Ollama
litellm.api_base = OLLAMA_BASE_URL
litellm.api_key = "fake-key"  # Required by LiteLLM but not used by Ollama
litellm.drop_params = True  # Optional: prevents LiteLLM from sending unsupported params
LITELLM_MODEL_STRING = f"ollama/{RAG_CHAT_MODEL}" 

session_service = InMemorySessionService()


rag_agent = LlmAgent(
    name="rag_assistant",
    model=LITELLM_MODEL_STRING,
    tools=[rag_tool],
    instruction=(
        "You are a RAG assistant. "
        "Always call rag_search before answering user questions."
    ),
)

root_agent = rag_agent
root_session_service = session_service

def get_runner(session_id: str) -> Runner:
    """Creates a Runner instance for a specific session."""
    return Runner(
        agent=root_agent,
        session_service=root_session_service,
        session_id=session_id
    )