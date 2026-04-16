from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY


class FactCheckMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.requests_total = Counter(
            "factcheck_requests",
            "Total number of processed fact-check requests.",
            registry=registry,
        )
        self.latency_seconds = Histogram(
            "factcheck_latency_seconds",
            "End-to-end latency for each processed claim.",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
            registry=registry,
        )
        self.label_distribution = Counter(
            "factcheck_label_distribution",
            "Distribution of predicted fact-check labels.",
            labelnames=("label",),
            registry=registry,
        )
        self.confidence_avg = Gauge(
            "factcheck_confidence_avg",
            "Running average confidence score.",
            registry=registry,
        )
        self.consumer_lag = Gauge(
            "kafka_consumer_lag",
            "Observed Kafka consumer lag.",
            registry=registry,
        )
        self._confidence_sum = 0.0
        self._confidence_count = 0

    def record_prediction(self, label: str, confidence: float, latency_seconds: float) -> None:
        self.requests_total.inc()
        self.latency_seconds.observe(latency_seconds)
        self.label_distribution.labels(label=label).inc()
        self._confidence_sum += confidence
        self._confidence_count += 1
        self.confidence_avg.set(self._confidence_sum / self._confidence_count)

    def set_consumer_lag(self, lag: int) -> None:
        self.consumer_lag.set(max(lag, 0))

