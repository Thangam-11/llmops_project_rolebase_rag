from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # secrets api keys
    qdrant_api: str
    cluster_endpoint: str
    openrouter: str


 # ── App config ──

    cluster_name: str = "my_cluster"
    collection_name: str = "my_collection"
    embedding_model: str = "baai/bge-base-en-v1.5"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    retrieval_k: int = 5
    redis_url: str = "redis://localhost:6379/0"
    api_key: str = "changeme-in-production"
    environment: str = "development"
    log_level: str = "info"



    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",)
    
    def model_post_init(self,__context):

        # Additional validation or processing can be done here
        self.qdrant_api = self.qdrant_api.strip()
        self.cluster_endpoint = self.cluster_endpoint.strip()   
        self.openrouter = self.openrouter.strip()

@lru_cache()
def get_settings():
    return Settings()
        