import os
from dotenv import load_dotenv

from google.adk.models.lite_llm import LiteLlm 
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from .rag.tool import qdrant_search_tool

load_dotenv()

RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

os.environ["LITELLM_API_BASE"] = OLLAMA_BASE_URL 

session_service = InMemorySessionService()

llm_model = LiteLlm(
    model=f"ollama/{RAG_CHAT_MODEL}",
    api_base=OLLAMA_BASE_URL 
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
                 
        Do not reveal your internal chain-of-thought or how you used the chunks.
        Simply provide concise and factual answers. If you are not certain or the
        information is not available, clearly state that you do not have
        enough information.
                 
        Only call the qdrant_search_tool when the user input contains any question about, or
        petition to explain, terms or topics related to AI, Artificial Intelligence, Agents
        or LLMs. In that case:
        1 - Use qdrant_search_tool once with the user input as a parameter
        2 - Output the digest from qdrant_search_tool as plain text.
                 
        If you believe the user is just chatting and having casual conversation,
        don't use qdrant_search_tool and answer casually in plain text.
                 
        Provide your final answer as a DIRECT TEXT RESPONSE to the user. Do not call
        any tools like 'output', 'assistant' or 'answer' to provide your final response.
                 
        Do NOT answer in JSON format.
                 
        If you are not certain about the user intent, make sure to ask clarifying questions
        before answering. Once you have the information you need, you can use the retrieval tool
        If you cannot provide an answer, clearly explain why.
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