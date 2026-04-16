# real-time-factcheck-stream

Production-style portfolio project that demonstrates LLM fine-tuning, streaming inference, and MLOps observability with Kafka, PostgreSQL, Prometheus, and Grafana.

## Architecture

```text
LIAR dataset (test split)
        |
        v
scripts/generate_claims.py
        |
        v
Kafka topic: claims
        |
        v
consumer/worker.py (FastAPI + background Kafka consumer)
        |
        +--> vLLM OpenAI-compatible API (Mistral 7B + LoRA adapter)
        |
        +--> PostgreSQL persistence
        |
        +--> Kafka topic: results
        |
        +--> Prometheus metrics (:8001/metrics)
                  |
                  v
             Grafana dashboard + drift alert
```

## Tech stack

| Layer | Choice |
| --- | --- |
| Base model | `mistralai/Mistral-7B-v0.1` |
| Fine-tuning | QLoRA, PEFT, TRL `SFTTrainer`, bitsandbytes NF4 |
| Dataset | Hugging Face LIAR (`ucsbnlp/liar`) |
| Stream transport | Kafka + Zookeeper |
| Inference serving | vLLM OpenAI-compatible `/v1/completions` |
| Stream worker | FastAPI + `aiokafka` + `httpx` |
| Persistence | PostgreSQL + SQLAlchemy |
| Metrics | Prometheus |
| Dashboards / alerts | Grafana |
| Tests | `pytest` |

## Quickstart

1. Create a Python 3.11 environment and install dependencies.

   ```bash
   pip install -r fine_tuning/requirements.txt -r consumer/requirements.txt -r scripts/requirements.txt
   ```

2. Copy `.env.example` to `.env` and adjust paths if needed.

3. Fine-tune the adapter.

   ```bash
   make train
   ```

4. Start infrastructure, serving, worker, and observability.

   ```bash
   make infra-up
   ```

5. Start the mock streaming load.

   ```bash
   make stream
   ```

6. Run offline evaluation and write `reports/benchmark.md`.

   ```bash
   make evaluate
   ```

### WSL2 / GPU notes

- On Windows, run the Docker and GPU workflow from WSL2 with NVIDIA Container Toolkit enabled.
- `MODEL_PATH` should point to the LoRA adapter directory created by `make train`.
- The compose stack serves the base model if the adapter directory does not exist yet.

## Make targets

- `make train` runs QLoRA fine-tuning.
- `make evaluate` generates benchmark reports under `reports/`.
- `make serve` starts only the vLLM service.
- `make stream` starts the mock claim generator and consumer path.
- `make infra-up` starts the full Docker Compose stack.
- `make all` runs train, infra boot, stream, and evaluation.

## Benchmark results

Evaluated on the LIAR test split (1,283 samples, 6-class classification).

| Metric | Value |
| --- | --- |
| Accuracy | **32.2%** |
| Macro F1 | **0.316** |
| Random baseline | ~16.7% |
| Invalid parses | 0 (100% structured output compliance) |

The fine-tuned model is ~1.9× the random baseline on a hard 6-class task.

### Per-class results

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| false | 0.368 | 0.444 | 0.402 | 250 |
| pants-fire | 0.472 | 0.272 | 0.345 | 92 |
| true | 0.315 | 0.374 | 0.342 | 211 |
| mostly-true | 0.296 | 0.325 | 0.310 | 249 |
| half-true | 0.281 | 0.330 | 0.303 | 267 |
| barely-true | 0.322 | 0.136 | 0.191 | 214 |

> `pants-fire` and `false` classes show highest precision, consistent with extreme labels being easier to distinguish. `barely-true` vs `half-true` confusion is a known challenge in this dataset.

Full per-class breakdown: [`docs/benchmark.md`](docs/benchmark.md) · [`docs/classification_report.json`](docs/classification_report.json)

## Performance

Stability test: 100 consecutive claims from the LIAR test split streamed through the full pipeline.

| Metric | Value |
| --- | --- |
| JSON parse success rate | 100% |
| Average latency | 1.31s / claim |
| P95 latency | 1.49s / claim |
| Regex fallback parses | 0 |
| Failed parses | 0 |
| Hardware | RTX 5080, 4-bit NF4 quantization, batch size 1 |

Label distribution across 100 streamed claims (consistent with LIAR dataset priors):

| Label | Count |
| --- | --- |
| half-true | 28 |
| mostly-true | 22 |
| false | 21 |
| true | 18 |
| barely-true | 7 |
| pants-fire | 4 |

## Grafana dashboard

![Grafana dashboard](docs/assets/grafana-dashboard.png)

## Environment variables

| Variable | Purpose | Example |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address for local scripts | `localhost:9092` |
| `POSTGRES_URL` | Async SQLAlchemy connection string | `postgresql+asyncpg://factcheck:factcheck@localhost:5432/factcheck` |
| `VLLM_BASE_URL` | Base URL for vLLM OpenAI-compatible API | `http://localhost:8000/v1` |
| `WANDB_API_KEY` | Optional Weights & Biases API key for training logs | `<secret>` |
| `MODEL_PATH` | Host path to the LoRA adapter directory | `./fine_tuned/mistral-liar-lora` |

## Repository layout

```text
real-time-factcheck-stream/
|- common/
|- fine_tuning/
|- consumer/
|- scripts/
|- infra/
|- docs/
|- tests/
|- reports/
|- .env.example
|- Makefile
`- README.md
```

## Testing

```bash
make test
```

## Notes

- All streaming message contracts use Pydantic models from `common/schemas.py`.
- The worker exposes Prometheus metrics at `http://localhost:8001/metrics`.
- Grafana ships with a provisioned dashboard and a drift alert rule.
- Additional design notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
