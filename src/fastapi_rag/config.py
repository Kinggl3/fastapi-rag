from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = ""
    github_token: str = ""
    stackexchange_key: str = ""
    https_proxy: str = ""

    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gemini-2.5-flash"

    chunk_max_tokens: int = 512
    retrieval_top_k: int = 50
    reranker_top_k: int = 10


settings = Settings()
