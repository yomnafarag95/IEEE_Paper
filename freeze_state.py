"""
freeze_state.py
---------------
Freeze and verify the RAG-Shield evaluation state.

The freeze file records thresholds, selected config flags, git commit, and
SHA-256 hashes for active code/model artifacts. Use it before final evaluation
so future metric reports can prove exactly which state produced them.

Usage:
    python freeze_state.py freeze
    python freeze_state.py check
    python freeze_state.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


FREEZE_PATH = Path("logs/frozen_thresholds.json")

THRESHOLD_NAMES = [
    "L1_BLOCK_THRESHOLD",
    "L2_STAGE1_THRESHOLD",
    "L2_DOC_PATTERN_THRESHOLD",
    "L3_CONSISTENCY_THRESHOLD",
    "META_HARD_BLOCK_SINGLE",
    "META_HARD_BLOCK_VIOLS",
    "META_BLOCK_THRESHOLD",
    "META_MONITOR_THRESHOLD",
    "META_BLOCK_CONSENSUS_THRESHOLD",
    "STATEFUL_DRIFT_THRESHOLD",
]

CONFIG_FLAG_NAMES = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "L2_DOC_SCAN_CHUNKS",
    "L2_USE_FINETUNED",
    "ENABLE_L1_EARLY_EXIT",
    "ENABLE_L2_EARLY_EXIT",
    "SEMANTIC_CACHE_THRESHOLD",
    "CANARY_DETECTION_ENABLED",
    "CANARY_INJECT_COUNT",
    "STATEFUL_HISTORY_LIMIT",
]

CODE_ARTIFACTS = [
    Path("config.py"),
    Path("orchestrator.py"),
    Path("layer1_anomaly.py"),
    Path("layer2_classifier.py"),
    Path("layer3_enhanced.py"),
    Path("keyword_detector.py"),
    Path("obfuscation_decoder.py"),
    Path("eval_suite.py"),
    Path("split_registry.py"),
    Path("build_validation_sets.py"),
]

MODEL_ARTIFACTS = [
    Path("models/layer1_models.pkl.bounds.pkl"),
    Path("models/layer1_models.pkl.ecod.pkl"),
    Path("models/layer1_models.pkl.iforest.pkl"),
    Path("models/layer1_models.pkl.svm.pkl"),
    Path("models/layer1_models.pkl.threshold.pkl"),
    Path("models/deberta_onnx/model.onnx"),
    Path("models/deberta_onnx/tokenizer/added_tokens.json"),
    Path("models/deberta_onnx/tokenizer/special_tokens_map.json"),
    Path("models/deberta_onnx/tokenizer/spm.model"),
    Path("models/deberta_onnx/tokenizer/tokenizer.json"),
    Path("models/deberta_onnx/tokenizer/tokenizer_config.json"),
    Path("models/layer3_consistency/config.json"),
    Path("models/layer3_consistency/model.safetensors"),
    Path("models/layer3_consistency/special_tokens_map.json"),
    Path("models/layer3_consistency/tokenizer.json"),
    Path("models/layer3_consistency/tokenizer_config.json"),
    Path("models/layer3_consistency/vocab.txt"),
    Path("models/meta_aggregator.pkl"),
    Path("models/meta_scaler.pkl"),
]

DATA_ARTIFACTS = [
    Path("data/splits/summary.json"),
    Path("data/splits/leakage_report.json"),
    Path("data/validation_benign_expanded.jsonl"),
    Path("data/evasion_validation_curated.csv"),
    Path("data/adversarial_validation_curated.csv"),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path).replace("\\", "/"),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    return {
        "path": str(path).replace("\\", "/"),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _collect_state() -> dict[str, Any]:
    thresholds = {name: getattr(config, name) for name in THRESHOLD_NAMES}
    config_flags = {name: getattr(config, name) for name in CONFIG_FLAG_NAMES}

    # Do not record secret values. A hash is useful for reproducibility without
    # exposing the token in logs or papers.
    canary_token = getattr(config, "CANARY_TOKEN", "") or ""
    canary = {
        "token_set": bool(canary_token),
        "token_sha256": hashlib.sha256(canary_token.encode("utf-8")).hexdigest() if canary_token else None,
    }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "dirty": bool(_git_value(["status", "--porcelain"])),
        },
        "thresholds": thresholds,
        "config_flags": config_flags,
        "canary": canary,
        "artifacts": {
            "code": [_file_record(path) for path in CODE_ARTIFACTS],
            "models": [_file_record(path) for path in MODEL_ARTIFACTS],
            "data": [_file_record(path) for path in DATA_ARTIFACTS],
        },
    }


def freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    state = _collect_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def _flatten_artifacts(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for group in state.get("artifacts", {}).values():
        for record in group:
            records[record["path"]] = record
    return records


def compare_to_frozen(path: Path = FREEZE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "message": f"Freeze file not found: {path}",
            "differences": [],
        }

    with open(path, encoding="utf-8") as f:
        frozen = json.load(f)
    current = _collect_state()

    differences = []
    for name, frozen_value in frozen.get("thresholds", {}).items():
        current_value = current.get("thresholds", {}).get(name)
        if frozen_value != current_value:
            differences.append({
                "type": "threshold",
                "name": name,
                "frozen": frozen_value,
                "current": current_value,
            })

    for name, frozen_value in frozen.get("config_flags", {}).items():
        current_value = current.get("config_flags", {}).get(name)
        if frozen_value != current_value:
            differences.append({
                "type": "config_flag",
                "name": name,
                "frozen": frozen_value,
                "current": current_value,
            })

    frozen_artifacts = _flatten_artifacts(frozen)
    current_artifacts = _flatten_artifacts(current)
    for artifact_path, frozen_record in frozen_artifacts.items():
        current_record = current_artifacts.get(artifact_path)
        if current_record != frozen_record:
            differences.append({
                "type": "artifact",
                "path": artifact_path,
                "frozen": frozen_record,
                "current": current_record,
            })

    return {
        "status": "failed" if differences else "passed",
        "freeze_file": str(path).replace("\\", "/"),
        "frozen_created_at_utc": frozen.get("created_at_utc"),
        "differences": differences,
    }


def _print_summary(state: dict[str, Any]) -> None:
    print(json.dumps({
        "created_at_utc": state.get("created_at_utc"),
        "git": state.get("git"),
        "thresholds": state.get("thresholds"),
        "config_flags": state.get("config_flags"),
        "canary": state.get("canary"),
        "artifact_counts": {
            group: len(records)
            for group, records in state.get("artifacts", {}).items()
        },
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify RAG-Shield evaluation state.")
    parser.add_argument("command", choices=["freeze", "check", "summary"])
    parser.add_argument("--path", default=str(FREEZE_PATH), help="Freeze JSON path.")
    args = parser.parse_args()

    path = Path(args.path)

    if args.command == "freeze":
        state = freeze(path)
        print(f"[freeze] Wrote state snapshot -> {path}")
        print(f"[freeze] Hashed {sum(len(v) for v in state['artifacts'].values())} artifacts.")
        return 0

    if args.command == "summary":
        if not path.exists():
            raise SystemExit(f"[freeze] Missing freeze file: {path}")
        with open(path, encoding="utf-8") as f:
            _print_summary(json.load(f))
        return 0

    report = compare_to_frozen(path)
    if report["status"] == "missing":
        print(f"[freeze] {report['message']}")
        return 1
    if report["status"] == "passed":
        print(f"[freeze] State check passed against {path}")
        return 0
    print(f"[freeze] State check failed against {path}: {len(report['differences'])} differences")
    for diff in report["differences"][:10]:
        print(json.dumps(diff, indent=2))
    if len(report["differences"]) > 10:
        print(f"[freeze] ... {len(report['differences']) - 10} more differences omitted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
