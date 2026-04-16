FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY consumer/requirements.txt consumer/requirements.txt
COPY scripts/requirements.txt scripts/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r consumer/requirements.txt -r scripts/requirements.txt

COPY common common
COPY consumer consumer
COPY scripts scripts

CMD ["uvicorn", "consumer.worker:app", "--host", "0.0.0.0", "--port", "8001"]

