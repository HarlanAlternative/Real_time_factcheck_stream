from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from common.config import Settings, get_settings
from common.db import create_engine, create_session_factory, init_db, persist_result
from common.parsing import parse_model_output
from common.prompts import build_inference_prompt
from common.schemas import ClaimMessage, FactCheckResult
from consumer.metrics import FactCheckMetrics

LOGGER = logging.getLogger("consumer.worker")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


class VLLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.vllm_request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def infer(self, claim_text: str) -> str:
        endpoint = f"{self._settings.vllm_base_url.rstrip('/')}/completions"
        payload = {
            "model": self._settings.vllm_model,
            "prompt": build_inference_prompt(claim_text),
            "max_tokens": self._settings.vllm_max_tokens,
            "temperature": self._settings.vllm_temperature,
        }

        for attempt in range(self._settings.vllm_max_retries):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["text"])
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                if attempt == self._settings.vllm_max_retries - 1:
                    raise RuntimeError("vLLM inference failed after all retries.") from exc
                sleep_seconds = self._settings.vllm_retry_backoff_seconds * (2**attempt)
                LOGGER.warning("vLLM call failed (attempt %s): %s", attempt + 1, exc)
                await asyncio.sleep(sleep_seconds)


class FactCheckWorker:
    def __init__(
        self,
        settings: Settings,
        metrics: FactCheckMetrics,
        engine: AsyncEngine,
        session_factory: async_sessionmaker,
    ) -> None:
        self._settings = settings
        self._metrics = metrics
        self._engine = engine
        self._session_factory = session_factory
        self._vllm_client = VLLMClient(settings)
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        await init_db(self._engine)
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
            value_serializer=lambda payload: payload.model_dump_json().encode("utf-8"),
        )
        self._consumer = AIOKafkaConsumer(
            self._settings.kafka_claims_topic,
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
            group_id=self._settings.kafka_consumer_group,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._producer.start()
        await self._consumer.start()
        self._task = asyncio.create_task(self._consume_loop(), name="factcheck-consumer")
        LOGGER.info("Fact-check worker started.")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        if self._consumer is not None:
            await self._consumer.stop()
        if self._producer is not None:
            await self._producer.stop()
        await self._vllm_client.close()
        await self._engine.dispose()
        LOGGER.info("Fact-check worker stopped.")

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        while not self._stop_event.is_set():
            batch = await self._consumer.getmany(
                timeout_ms=self._settings.consumer_poll_timeout_ms,
                max_records=32,
            )
            for topic_partition, messages in batch.items():
                for message in messages:
                    try:
                        await self._process_message(message.value)
                        await self._consumer.commit()
                    except Exception as exc:  # pragma: no cover
                        LOGGER.exception("Failed to process claim message: %s", exc)

                lag = self._consumer.highwater(topic_partition) - await self._consumer.position(topic_partition)
                self._metrics.set_consumer_lag(lag)

    async def _process_message(self, raw_value: bytes) -> None:
        claim = ClaimMessage.model_validate_json(raw_value)
        started_at = time.perf_counter()
        raw_response = await self._vllm_client.infer(claim.text)
        prediction = parse_model_output(raw_response)

        result = FactCheckResult(
            id=claim.id,
            text=claim.text,
            timestamp=claim.timestamp,
            label=prediction.label,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning,
            raw_response=raw_response,
            model_name=self._settings.vllm_model,
        )

        await persist_result(self._session_factory, result)
        assert self._producer is not None
        await self._producer.send_and_wait(self._settings.kafka_results_topic, result)

        latency_seconds = time.perf_counter() - started_at
        self._metrics.record_prediction(
            label=result.label,
            confidence=result.confidence,
            latency_seconds=latency_seconds,
        )
        LOGGER.info("Processed claim %s as %s", result.id, result.label)

settings = get_settings()
metrics = FactCheckMetrics()
engine = create_engine(settings.postgres_url, echo=settings.database_echo)
session_factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)
worker = FactCheckWorker(settings, metrics, engine, session_factory)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


configure_logging()
app = FastAPI(title="real-time-factcheck-stream", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/healthz")
async def healthcheck() -> dict[str, Any]:
    return {"status": "ok", "model": settings.vllm_model}
