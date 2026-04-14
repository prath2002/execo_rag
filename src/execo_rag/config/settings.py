"""Application settings loaded from environment variables and .env files."""

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    DEFAULT_APP_HOST,
    DEFAULT_APP_PORT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOG_FORMAT,
    DEFAULT_OPENROUTER_CHAT_MODEL,
    DEFAULT_PDF_EXTRACTOR,
    DEFAULT_PINECONE_NAMESPACE,
)


class AppSettings(BaseModel):
    """Application runtime settings."""

    name: str
    env: str
    host: str
    port: int


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str
    json_logs: bool
    format: str


class OpenRouterSettings(BaseModel):
    """Optional OpenRouter settings for LLM fallbacks and remote embeddings."""

    api_key: str
    chat_model: str


class EmbeddingSettings(BaseModel):
    """Embedding provider configuration."""

    provider: str
    model: str
    dimension: int
    batch_size: int


class PineconeSettings(BaseModel):
    """Pinecone connection settings."""

    api_key: str
    index_name: str
    namespace: str


class PdfExtractionSettings(BaseModel):
    """PDF extraction and chunking settings."""

    extractor: str
    max_chunk_tokens: int
    chunk_overlap_tokens: int


class RuntimeFlags(BaseModel):
    """Runtime feature flags and diagnostics options."""

    debug: bool
    reload: bool
    enable_llm_fallback: bool
    enable_metrics: bool


class Settings(BaseSettings):
    """Central application settings object."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="execo-rag", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default=DEFAULT_APP_HOST, alias="APP_HOST")
    app_port: int = Field(default=DEFAULT_APP_PORT, alias="APP_PORT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")
    log_format: str = Field(default=DEFAULT_LOG_FORMAT, alias="LOG_FORMAT")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_chat_model: str = Field(
        default=DEFAULT_OPENROUTER_CHAT_MODEL,
        alias="OPENROUTER_CHAT_MODEL",
    )

    embedding_provider: str = Field(
        default=DEFAULT_EMBEDDING_PROVIDER,
        alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION,
        alias="EMBEDDING_DIMENSION",
    )
    embedding_batch_size: int = Field(
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        alias="EMBEDDING_BATCH_SIZE",
    )

    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="", alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(
        default=DEFAULT_PINECONE_NAMESPACE,
        alias="PINECONE_NAMESPACE",
    )

    pdf_extractor: str = Field(default=DEFAULT_PDF_EXTRACTOR, alias="PDF_EXTRACTOR")
    max_chunk_tokens: int = Field(default=DEFAULT_CHUNK_SIZE, alias="MAX_CHUNK_TOKENS")
    chunk_overlap_tokens: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        alias="CHUNK_OVERLAP_TOKENS",
    )

    runtime_debug: bool = Field(default=False, alias="RUNTIME_DEBUG")
    runtime_reload: bool = Field(default=False, alias="RUNTIME_RELOAD")
    enable_llm_fallback: bool = Field(default=True, alias="ENABLE_LLM_FALLBACK")
    enable_metrics: bool = Field(default=False, alias="ENABLE_METRICS")

    @property
    def app(self) -> AppSettings:
        """Grouped application settings."""

        return AppSettings(
            name=self.app_name,
            env=self.app_env,
            host=self.app_host,
            port=self.app_port,
        )

    @property
    def logging(self) -> LoggingSettings:
        """Grouped logging settings."""

        return LoggingSettings(
            level=self.log_level,
            json_logs=self.log_json,
            format=self.log_format,
        )

    @property
    def openrouter(self) -> OpenRouterSettings:
        """Grouped OpenRouter settings."""

        return OpenRouterSettings(
            api_key=self.openrouter_api_key,
            chat_model=self.openrouter_chat_model,
        )

    @property
    def embeddings(self) -> EmbeddingSettings:
        """Grouped embedding settings."""

        return EmbeddingSettings(
            provider=self.embedding_provider,
            model=self.embedding_model,
            dimension=self.embedding_dimension,
            batch_size=self.embedding_batch_size,
        )

    @property
    def pinecone(self) -> PineconeSettings:
        """Grouped Pinecone settings."""

        return PineconeSettings(
            api_key=self.pinecone_api_key,
            index_name=self.pinecone_index_name,
            namespace=self.pinecone_namespace,
        )

    @property
    def pdf(self) -> PdfExtractionSettings:
        """Grouped PDF extraction settings."""

        return PdfExtractionSettings(
            extractor=self.pdf_extractor,
            max_chunk_tokens=self.max_chunk_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
        )

    @property
    def runtime(self) -> RuntimeFlags:
        """Grouped runtime flag settings."""

        return RuntimeFlags(
            debug=self.runtime_debug,
            reload=self.runtime_reload,
            enable_llm_fallback=self.enable_llm_fallback,
            enable_metrics=self.enable_metrics,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance."""

    return Settings()
