from __future__ import annotations

from prometheus_client import CollectorRegistry

from consumer.metrics import FactCheckMetrics


def test_metrics_emit_expected_values() -> None:
    registry = CollectorRegistry()
    metrics = FactCheckMetrics(registry=registry)

    metrics.record_prediction(label="false", confidence=0.8, latency_seconds=0.75)
    metrics.set_consumer_lag(4)

    assert registry.get_sample_value("factcheck_requests_total") == 1.0
    assert registry.get_sample_value(
        "factcheck_label_distribution_total",
        {"label": "false"},
    ) == 1.0
    assert registry.get_sample_value("factcheck_confidence_avg") == 0.8
    assert registry.get_sample_value("kafka_consumer_lag") == 4.0

