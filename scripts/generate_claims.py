from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress
from datetime import datetime, timezone

from aiokafka import AIOKafkaProducer

from common.config import get_settings
from common.liar import get_split, load_liar_dataset, to_liar_example
from common.schemas import ClaimMessage

LOGGER = logging.getLogger("scripts.generate_claims")


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Publish LIAR claims into Kafka.")
    parser.add_argument("--dataset-name", default=settings.liar_dataset_name)
    parser.add_argument("--rate", type=float, default=settings.generator_rate_per_sec)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topic", default=settings.kafka_claims_topic)
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)


async def main() -> None:
    args = parse_args()
    configure_logging()

    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    dataset = load_liar_dataset(args.dataset_name)
    test_split = get_split(dataset, "test", fallback="validation").shuffle(seed=args.seed)
    if args.limit is not None:
        test_split = test_split.select(range(min(args.limit, len(test_split))))

    producer = AIOKafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda payload: payload.model_dump_json().encode("utf-8"),
    )
    await producer.start()

    try:
        for row in test_split:
            if stop_event.is_set():
                break
            example = to_liar_example(row)
            message = ClaimMessage(
                text=example.claim,
                timestamp=datetime.now(tz=timezone.utc),
            )
            await producer.send_and_wait(args.topic, message)
            LOGGER.info("Published claim %s", message.id)
            await asyncio.sleep(1.0 / max(args.rate, 0.1))
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())

