# Architecture

## End-to-end flow

```text
LIAR test split
   |
   v
scripts/generate_claims.py
   |
   v
Kafka topic: claims
   |
   v
FastAPI worker (consumer/worker.py)
   |-- calls vLLM OpenAI-compatible completions API
   |-- parses structured JSON
   |-- writes Prometheus metrics
   |-- persists results
   v
Kafka topic: results
   |
   +--> PostgreSQL factcheck_results
   +--> Prometheus scrape target
            |
            v
         Grafana dashboard + alerting
```

## Components

- `fine_tuning/train.py`
  Fine-tunes `mistralai/Mistral-7B-v0.1` on LIAR using QLoRA, 4-bit NF4 quantization, PEFT LoRA adapters on `q_proj` and `v_proj`, and TRL `SFTTrainer`.
- `fine_tuning/evaluate.py`
  Loads the base model plus saved adapter, generates structured JSON predictions, computes accuracy and macro F1, and writes `reports/benchmark.md`.
- `scripts/generate_claims.py`
  Replays LIAR test examples into Kafka at a configurable rate.
- `scripts/setup_kafka_topics.py`
  Creates the `claims` and `results` topics before workers start.
- `consumer/worker.py`
  Runs as a FastAPI application with a background Kafka loop and `/metrics` exposure for Prometheus.
- `consumer/metrics.py`
  Encapsulates counters, histogram, and gauges for request volume, latency, labels, confidence, and lag.
- `infra/docker-compose.yml`
  Bootstraps Kafka, PostgreSQL, vLLM, Prometheus, Grafana, and the consumer worker.

## Data contracts

- `ClaimMessage`
  `{"id": "uuid", "text": "claim text", "timestamp": "ISO8601"}`
- `FactCheckResult`
  `{"id", "text", "timestamp", "label", "confidence", "reasoning", "processed_at", "model_name", "raw_response"}`

All message schemas are defined as Pydantic models in `common/schemas.py`.

## Model serving path

- vLLM serves an OpenAI-compatible `/v1/completions` endpoint.
- The worker sends the exact prompt template used during supervised fine-tuning.
- JSON parsing is strict-first, regex-fallback second, which keeps the stream resilient when the model returns code fences or minor formatting drift.

## Reliability notes

- vLLM requests retry up to 3 times with exponential backoff.
- FastAPI lifespan teardown stops the Kafka consumer and producer cleanly on shutdown.
- PostgreSQL schema creation is automatic on worker boot.
- Grafana alerting is provisioned to trigger when `false` plus `pants-fire` predictions exceed 60% of the rolling 5-minute window.

