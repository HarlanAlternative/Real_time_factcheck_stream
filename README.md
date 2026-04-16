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
| Dataset | Hugging Face LIAR (`ucsbnlp/liar`), 3-class collapsed |
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

Evaluated on the LIAR test split (1,283 samples, 3-class classification).

| Metric | Value |
| --- | --- |
| Accuracy | **54.3%** |
| Macro F1 | **0.535** |
| Random baseline | ~33.3% |
| Invalid parses | 0 (100% structured output compliance) |

The fine-tuned model is ~1.6× the random baseline. Labels are collapsed from 6 to 3 classes: `true` (true + mostly-true), `mixed` (half-true + barely-true), `false` (false + pants-fire). Prompts include speaker name, title, party affiliation, and context from the LIAR dataset.

### Per-class results

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| true | 0.567 | 0.704 | **0.629** | 460 |
| mixed | 0.508 | 0.449 | **0.477** | 481 |
| false | 0.547 | 0.459 | **0.499** | 342 |

> `true` achieves the highest F1, benefiting most from speaker metadata context. `mixed` is hardest to classify, consistent with the inherent ambiguity of borderline truthfulness.

### vs. previous 6-class baseline

| Metric | 6-class (v1) | 3-class + speaker (v2) | Delta |
| --- | --- | --- | --- |
| Accuracy | 32.2% | **54.3%** | +22.1 pp |
| Macro F1 | 0.316 | **0.535** | +0.219 |
| Invalid parses | some | **0** | eliminated |

Full per-class breakdown: [`reports/benchmark.md`](reports/benchmark.md) · [`reports/classification_report.json`](reports/classification_report.json)

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

Label distribution across 100 streamed claims (consistent with LIAR dataset priors, collapsed to 3 classes):

| Label | Count |
| --- | --- |
| mixed (half-true + barely-true) | 35 |
| true (true + mostly-true) | 40 |
| false (false + pants-fire) | 25 |

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
