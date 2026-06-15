"""
split_registry.py
-----------------
Central split manifest and leakage checker for RAG-Shield datasets.

This utility is intentionally lightweight: it does not run models and it does
not rewrite source datasets. It builds normalized sample manifests with stable
SHA-256 IDs so training, threshold tuning, and evaluation can be audited for
overlap before reporting results.

Usage:
    python split_registry.py build
    python split_registry.py build --include-raw-large
    python split_registry.py check
    python split_registry.py summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DATA_DIR = Path("data")
SPLIT_DIR = DATA_DIR / "splits"
MANIFEST_PATH = SPLIT_DIR / "manifest.jsonl"
SUMMARY_PATH = SPLIT_DIR / "summary.json"
LEAKAGE_REPORT_PATH = SPLIT_DIR / "leakage_report.json"

TRAIN_CUTOFF = 80
DEV_CUTOFF = 90

TEXT_FIELDS = ("text", "query", "prompt", "question", "attack", "user_input")
FINAL_HOLDOUT_DIRS = (DATA_DIR / "final_holdout",)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text_hash: str
    split: str
    label: int
    source: str
    sample_type: str
    text: str
    metadata: dict

    def to_json(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "text_hash": self.text_hash,
            "split": self.split,
            "label": self.label,
            "source": self.source,
            "sample_type": self.sample_type,
            "text": self.text,
            "metadata": self.metadata,
        }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def sample_id(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\0{normalize_text(text)}".encode("utf-8")).hexdigest()[:16]


def split_for_text(text: str) -> str:
    digest = text_hash(text)
    bucket = int(digest[:8], 16) % 100
    if bucket < TRAIN_CUTOFF:
        return "train"
    if bucket < DEV_CUTOFF:
        return "dev"
    return "test"


def _first_text(row: dict, fields: Iterable[str] = TEXT_FIELDS) -> str:
    for field in fields:
        value = row.get(field)
        if value:
            return str(value).strip()
    return ""


def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read_csv(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _make_sample(
    *,
    source: str,
    text: str,
    label: int,
    sample_type: str,
    metadata: dict | None = None,
    split: str | None = None,
) -> Sample | None:
    text = str(text or "").strip()
    if not text:
        return None
    th = text_hash(text)
    return Sample(
        sample_id=sample_id(source, text),
        text_hash=th,
        split=split or split_for_text(text),
        label=int(label),
        source=source,
        sample_type=sample_type,
        text=text,
        metadata=metadata or {},
    )


def iter_source_samples(include_raw_large: bool = False) -> Iterator[Sample]:
    paths = {
        "hackaprompt_holdout_seed42": DATA_DIR / "hackaprompt_holdout_seed42.csv",
        "injecagent": DATA_DIR / "injecagent.jsonl",
        "tensortrust": DATA_DIR / "tensortrust.jsonl",
        "bipia": DATA_DIR / "bipia.jsonl",
        "extended_benign": DATA_DIR / "extended_benign.csv",
        "validation_benign_expanded": DATA_DIR / "validation_benign_expanded.jsonl",
        "benign_queries": DATA_DIR / "benign_queries.jsonl",
        "wildchat_benign": DATA_DIR / "wildchat_benign.jsonl",
        "xstest": DATA_DIR / "xstest.jsonl",
        "adversarial_validation_curated": DATA_DIR / "adversarial_validation_curated.csv",
        "evasion_validation_curated": DATA_DIR / "evasion_validation_curated.csv",
        "evasion_benchmark_n100": DATA_DIR / "evasion_benchmark_n100.csv",
    }

    if include_raw_large:
        paths = {"hackaprompt": DATA_DIR / "hackaprompt.jsonl", **paths}

    for source, path in paths.items():
        if not path.exists():
            continue
        rows = _read_csv(path) if path.suffix.lower() == ".csv" else _read_jsonl(path)
        for row in rows:
            text = _first_text(row)
            if not text:
                continue
            if source in {"extended_benign", "validation_benign_expanded", "benign_queries", "wildchat_benign", "xstest"}:
                label = 0
                sample_type = "benign"
            elif source == "adversarial_validation_curated":
                label = int(row.get("label", 1) or 1)
                sample_type = "adversarial_attack"
            elif source in {"evasion_validation_curated", "evasion_benchmark_n100"}:
                label = int(row.get("label", 1) or 1)
                sample_type = "evasion"
            else:
                label = int(row.get("label", 1) or 1)
                sample_type = "attack"
            sample = _make_sample(
                source=source,
                text=text,
                label=label,
                sample_type=sample_type,
                metadata={k: v for k, v in row.items() if k not in TEXT_FIELDS},
            )
            if sample is not None:
                yield sample


def iter_final_holdout_samples() -> Iterator[Sample]:
    for root in FINAL_HOLDOUT_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".jsonl", ".csv"}:
                continue
            rows = _read_csv(path) if path.suffix.lower() == ".csv" else _read_jsonl(path)
            for row in rows:
                text = _first_text(row)
                if not text:
                    continue
                label = int(row.get("label", 1) or 1)
                sample_type = row.get("sample_type") or ("benign" if label == 0 else "attack")
                sample = _make_sample(
                    source=f"final_holdout/{path.name}",
                    text=text,
                    label=label,
                    sample_type=str(sample_type),
                    metadata={k: v for k, v in row.items() if k not in TEXT_FIELDS},
                    split="final_holdout",
                )
                if sample is not None:
                    yield sample


def build_manifest(include_raw_large: bool = False) -> list[Sample]:
    seen_ids: set[str] = set()
    samples: list[Sample] = []
    for sample in list(iter_source_samples(include_raw_large=include_raw_large)) + list(iter_final_holdout_samples()):
        if sample.sample_id in seen_ids:
            continue
        seen_ids.add(sample.sample_id)
        samples.append(sample)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")

    summary = summarize(samples)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return samples


def load_manifest(include_raw_large: bool = False) -> list[Sample]:
    if not MANIFEST_PATH.exists():
        return build_manifest(include_raw_large=include_raw_large)
    samples = []
    for row in _read_jsonl(MANIFEST_PATH):
        samples.append(Sample(
            sample_id=row["sample_id"],
            text_hash=row["text_hash"],
            split=row["split"],
            label=int(row["label"]),
            source=row["source"],
            sample_type=row["sample_type"],
            text=row["text"],
            metadata=row.get("metadata") or {},
        ))
    return samples


def summarize(samples: list[Sample]) -> dict:
    out: dict[str, dict] = {}
    for sample in samples:
        split_bucket = out.setdefault(sample.split, {"total": 0, "by_type": {}, "by_source": {}})
        split_bucket["total"] += 1
        split_bucket["by_type"][sample.sample_type] = split_bucket["by_type"].get(sample.sample_type, 0) + 1
        split_bucket["by_source"][sample.source] = split_bucket["by_source"].get(sample.source, 0) + 1
    return out


def check_leakage(samples: list[Sample]) -> dict:
    by_hash: dict[str, list[Sample]] = {}
    for sample in samples:
        by_hash.setdefault(sample.text_hash, []).append(sample)

    leaks = []
    for th, group in by_hash.items():
        splits = sorted({s.split for s in group})
        if len(splits) <= 1:
            continue
        leaks.append({
            "text_hash": th,
            "splits": splits,
            "sources": sorted({s.source for s in group}),
            "sample_ids": [s.sample_id for s in group[:20]],
            "text_preview": group[0].text[:200],
        })

    report = {
        "status": "failed" if leaks else "passed",
        "leak_count": len(leaks),
        "leaks": leaks[:100],
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and check RAG-Shield split manifests.")
    parser.add_argument("command", choices=["build", "check", "summary"])
    parser.add_argument(
        "--include-raw-large",
        action="store_true",
        help="Include very large raw source files such as data/hackaprompt.jsonl.",
    )
    args = parser.parse_args()

    if args.command == "build":
        samples = build_manifest(include_raw_large=args.include_raw_large)
        print(f"[splits] Wrote {len(samples)} samples -> {MANIFEST_PATH}")
        print(f"[splits] Summary -> {SUMMARY_PATH}")
        return 0

    samples = load_manifest()
    if args.command == "summary":
        print(json.dumps(summarize(samples), indent=2))
        return 0

    report = check_leakage(samples)
    print(f"[splits] Leakage check {report['status']}: {report['leak_count']} cross-split duplicate hashes")
    print(f"[splits] Report -> {LEAKAGE_REPORT_PATH}")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
