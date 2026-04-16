PYTHON ?= python
ENV_FILE ?= .env.example
COMPOSE ?= docker compose --env-file $(ENV_FILE) -f infra/docker-compose.yml

.PHONY: train evaluate serve stream infra-up test all

train:
	$(PYTHON) -m fine_tuning.train

evaluate:
	$(PYTHON) -m fine_tuning.evaluate

serve:
	$(COMPOSE) up -d vllm-server

stream:
	$(COMPOSE) --profile demo up -d claim-generator consumer-worker

infra-up:
	$(COMPOSE) up -d zookeeper kafka postgres kafka-init vllm-server consumer-worker prometheus grafana

test:
	$(PYTHON) -m pytest tests

all: train infra-up stream evaluate

