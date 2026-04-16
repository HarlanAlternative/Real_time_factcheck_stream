from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_claims_topic: str = "claims"
    kafka_results_topic: str = "results"
    kafka_consumer_group: str = "factcheck-workers"

    postgres_url: str = "postgresql+asyncpg://factcheck:factcheck@localhost:5432/factcheck"
    database_echo: bool = False

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "mistral-liar-lora"
    vllm_max_tokens: int = 200
    vllm_temperature: float = 0.1
    vllm_request_timeout_seconds: float = 60.0
    vllm_max_retries: int = 3
    vllm_retry_backoff_seconds: float = 0.5

    metrics_host: str = "0.0.0.0"
    metrics_port: int = 8001
    consumer_poll_timeout_ms: int = 1000

    generator_rate_per_sec: float = 1.0
    liar_dataset_name: str = "ucsbnlp/liar"

    model_path: str = "./fine_tuned/mistral-liar-lora"
    wandb_api_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

