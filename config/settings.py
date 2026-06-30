from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # === LLM ===
    openrouter_api_key:  str
    llm_model:           str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # === Qdrant ===
    qdrant_url:      str
    qdrant_api_key:  str = ""
    collection_name: str = "role_based_rag"

    # === Postgres ===
    database_url: str

    # === Redis ===
    redis_url: str = "redis://localhost:6379/0"

    # === JWT ===
    secret_key:                  str
    algorithm:                   str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # === Embedding ===
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim:   int = 768

    # === App ===
    environment: str = "development"
    log_level:   str = "info"

    # ===========================
    # Monitoring & Observability
    # ===========================
    langsmith_api_key:     str  = ""
    langsmith_project:     str  = "role-based-rag"
    langsmith_enabled:     bool = False
    prometheus_port:       int  = 9090
    ragas_enabled:         bool = False
    pii_guardrail_enabled: bool = True
    model_config = SettingsConfigDict(
        env_file       = ".env",
        case_sensitive = False,
        extra          = "ignore",
    )

    def model_post_init(self, __context) -> None:
        self.qdrant_url         = self.qdrant_url.strip()
        self.openrouter_api_key = self.openrouter_api_key.strip()

@lru_cache()
def get_settings() -> Settings:
    return Settings()