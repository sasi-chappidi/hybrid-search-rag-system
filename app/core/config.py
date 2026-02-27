from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_DIR: str = "data/index"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
