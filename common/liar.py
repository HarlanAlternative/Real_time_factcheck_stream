from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from common.parsing import normalize_label

DATASET_CANDIDATES: tuple[str, ...] = ("ucsbnlp/liar", "liar")
INTEGER_LABEL_MAP: tuple[str, ...] = (
    "false",
    "half-true",
    "mostly-true",
    "true",
    "barely-true",
    "pants-fire",
)
LIAR_ARCHIVE_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
LIAR_CACHE_DIR = Path.home() / ".cache" / "real-time-factcheck-stream" / "liar"
LIAR_SPLIT_FILES: dict[str, str] = {
    "train": "train.tsv",
    "validation": "valid.tsv",
    "test": "test.tsv",
}
LIAR_COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
)


@dataclass(frozen=True)
class LiarExample:
    claim: str
    label: str
    reasoning: str


def _cache_archive() -> Path:
    LIAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = LIAR_CACHE_DIR / "liar_dataset.zip"
    if archive_path.exists():
        return archive_path

    urllib.request.urlretrieve(LIAR_ARCHIVE_URL, archive_path)
    return archive_path


def _read_split_from_archive(archive_path: Path, split_name: str) -> Dataset:
    tsv_name = LIAR_SPLIT_FILES[split_name]
    records: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(tsv_name) as split_file:
            text_stream = io.TextIOWrapper(split_file, encoding="utf-8")
            reader = csv.reader(text_stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                records.append(dict(zip(LIAR_COLUMNS, row, strict=False)))

    return Dataset.from_list(records)


def _load_liar_from_archive() -> DatasetDict:
    archive_path = _cache_archive()
    return DatasetDict(
        {
            split_name: _read_split_from_archive(archive_path, split_name)
            for split_name in LIAR_SPLIT_FILES
        }
    )


def load_liar_dataset(dataset_name: str | None = None) -> DatasetDict:
    candidates = (dataset_name,) if dataset_name else DATASET_CANDIDATES
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return load_dataset(candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if any(candidate in DATASET_CANDIDATES for candidate in candidates):
        return _load_liar_from_archive()

    raise RuntimeError(f"Unable to load LIAR dataset from {candidates}.") from last_error


def get_split(dataset: DatasetDict, split_name: str, fallback: str | None = None) -> Dataset:
    if split_name in dataset:
        return dataset[split_name]
    if fallback and fallback in dataset:
        return dataset[fallback]
    available = ", ".join(dataset.keys())
    raise KeyError(f"Split '{split_name}' not present. Available splits: {available}")


def extract_claim(example: dict[str, Any]) -> str:
    value = example.get("statement") or example.get("claim") or example.get("text")
    if not value:
        raise ValueError(f"Claim text missing from example keys: {sorted(example.keys())}")
    return str(value).strip()


def extract_label(example: dict[str, Any]) -> str:
    if "label_text" in example and example["label_text"] is not None:
        raw_value = example["label_text"]
    elif "label" in example and example["label"] is not None:
        raw_value = example["label"]
    elif "truthfulness" in example and example["truthfulness"] is not None:
        raw_value = example["truthfulness"]
    else:
        raw_value = example.get("verdict")
    if raw_value is None:
        raise ValueError(f"Label missing from example keys: {sorted(example.keys())}")

    if isinstance(raw_value, int):
        return INTEGER_LABEL_MAP[raw_value]
    return normalize_label(str(raw_value))


def extract_reasoning(example: dict[str, Any]) -> str:
    value = example.get("justification") or example.get("explanation") or example.get("context")
    if value:
        return str(value).strip()
    return "Reasoning unavailable in source dataset; assign the closest truthfulness label."


def to_liar_example(example: dict[str, Any]) -> LiarExample:
    return LiarExample(
        claim=extract_claim(example),
        label=extract_label(example),
        reasoning=extract_reasoning(example),
    )
