from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    RAG_CHAT_MODEL: str
    RAG_EMBEDDING_MODEL: str

    OLLAMA_BASE_URL: str

    QDRANT_URL: str
    QDRANT_COLLECTION: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()
