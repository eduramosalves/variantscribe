"""Central configuration, loaded from environment / .env (prefix VARIANTSCRIBE_)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VARIANTSCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NCBI E-utilities. Email is required by NCBI policy; api_key lifts rate limits.
    ncbi_email: str = "anonymous@example.com"
    ncbi_api_key: str | None = None

    # LLM agent.
    anthropic_api_key: str | None = None
    agent_model: str = "claude-sonnet-4-6"

    # Langfuse tracing (optional — tracing is a no-op unless these are set).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Retrieval models (HF hub ids). MedCPT uses asymmetric article/query encoders.
    embedding_model: str = "ncbi/MedCPT-Article-Encoder"
    query_embedding_model: str = "ncbi/MedCPT-Query-Encoder"
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"

    # ColPali visual document retrieval (guideline PDFs).
    colpali_model: str = "vidore/colpali-v1.3"

    # Filesystem.
    data_dir: Path = Field(default=Path("./data"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.index_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
