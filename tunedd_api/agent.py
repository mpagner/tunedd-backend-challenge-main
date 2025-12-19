import os

from google.adk.models.lite_llm import LiteLlm 
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from .rag.tool import qdrant_search_tool
from tunedd_api.settings import settings


os.environ["LITELLM_API_BASE"] = settings.OLLAMA_BASE_URL 

session_service = InMemorySessionService()

llm_model = LiteLlm(
    model=f"ollama/{settings.RAG_CHAT_MODEL}",
    api_base=settings.OLLAMA_BASE_URL 
)

rag_agent = LlmAgent(
    name="rag_assistant",
    model=llm_model,
    tools=[qdrant_search_tool],
    instruction=("""
        You are an AI assistant with access to documents about artificial intelligence.
        Your role is to provide accurate and concise answers to questions and provide 
        additional context based on data from documents that are retrievable using
        qdrant_search_tool. 
                 
        ## CRITICAL RULES:
        - Provide your final answer as PLAIN TEXT only.
        - NEVER output JSON.
        - NEVER call nonexistent functions like 'output()', 'assistant()', or 'answer()'.
        - If you have an answer, just type it out naturally.
        - Do not reveal your internal process or that you used a tool.
                 
        ## TOOL USAGE:
        Only call the qdrant_search_tool when the user input contains any question about, or
        petition to explain, terms or topics related to AI, Artificial Intelligence, Agents
        or LLMs. In that case:
        1 - Use qdrant_search_tool once with the user input as a parameter.
        2 - Synthesize an answer based on the retrieved text and output it as plain text.
                 
        If you believe the user is just chatting, don't use the tool and answer casually.
        If you are not certain, ask a clarifying question.
        """
    ),
)

root_agent = rag_agent
root_session_service = session_service

def get_runner(session_id: str):
    return Runner(
        agent=root_agent,
        session_service=root_session_service,
        session_id=session_id
    )