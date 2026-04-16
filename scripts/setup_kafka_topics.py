from __future__ import annotations

import argparse
import logging
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from common.config import get_settings

LOGGER = logging.getLogger("scripts.setup_kafka_topics")


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create Kafka topics for the fact-check stream.")
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    topics = [
        NewTopic(name=settings.kafka_claims_topic, num_partitions=3, replication_factor=1),
        NewTopic(name=settings.kafka_results_topic, num_partitions=3, replication_factor=1),
    ]

    for attempt in range(args.retries):
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=args.bootstrap_servers)
            for topic in topics:
                try:
                    admin_client.create_topics([topic], validate_only=False)
                    LOGGER.info("Created topic %s", topic.name)
                except TopicAlreadyExistsError:
                    LOGGER.info("Topic %s already exists", topic.name)
            admin_client.close()
            return
        except Exception as exc:
            LOGGER.warning("Kafka not ready yet (attempt %s/%s): %s", attempt + 1, args.retries, exc)
            time.sleep(args.retry_delay)

    raise RuntimeError("Failed to initialize Kafka topics after all retries.")


if __name__ == "__main__":
    main()

