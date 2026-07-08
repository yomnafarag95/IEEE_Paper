"""
run_holdout_ablation_fast.py

Final-holdout ablation runner aligned with run_final_holdout_and_report.py.

Important: the "Full (Pipeline)" row calls orchestrator.run_pipeline() with the
same model objects and sample construction as the headline holdout evaluation.
This makes that row directly comparable to Table II / the final report.
"""

from __future__ import annotations

# Import onnxruntime first to avoid OpenMP DLL conflicts on Windows.
try:
    import onnxruntime as _ort  # noqa: F401
except ImportError:
    pass

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(__file__))

from eval_suite import _is_detected
from layer1_anomaly import load_detector, split_chunks
from layer2_classifier import load_classifier
from layer3_enhanced import load_monitor
from orchestrator import MetaAggregator, run_pipeline
from run_final_holdout_and_report import (
    ATTACKS_FILE,
    BENIGN_FILE,
    EVASION_FILE,
    load_jsonl,
    process_sample,
)


SYSTEM_PROMPT = "Answer using only the provided knowledge base."
RESULTS_PATH = Path("logs/ablation_results_final_holdout.json")
SAMPLE_LOG_PATH = Path("logs/ablation_results_final_holdout_samples.jsonl")


def _load_samples(path: str, label: int, source: str) -> list[dict]:
    samples = []
    rows = load_jsonl(path)
    for idx, row in enumerate(rows):
        doc, query, atype = process_sample(row, label)
        samples.append(
            {
                "sample_id": f"{source}:{idx}",
                "source": source,
                "source_index": idx,
                "source_file": path,
                "document": doc,
                "query": query,
                "label": label,
                "attack_type": atype,
            }
        )
    print(f"  {Path(path).name:<14} : {len(samples)} samples")
    return samples


def _neutral_l1() -> dict:
    return {
        "blocked": False,
        "max_score": 0.0,
        "window_scores": [],
        "full_score": 0.0,
        "flagged_chunks": [],
        "ev": [("Skipped", "Not part of this ablation configuration")],
    }


def _neutral_l2() -> dict:
    return {
        "stage1_prob": 0.0,
        "stage2_label": None,
        "stage2_conf": 0.0,
        "consistency_score": 0.0,
        "doc_score": 0.0,
        "doc_source": None,
        "doc_cache_hit": False,
        "blocked": False,
        "ev": [("Skipped", "Not part of this ablation configuration")],
    }


def _empty_l3() -> dict:
    return {
        "schema_valid": True,
        "schema_issues": [],
        "boundary_violations": [],
        "consistency_score": 0.0,
        "blocked": False,
        "confidence": 0.0,
        "ev": [("Skipped", "Not part of this ablation configuration")],
    }


def _run_component_ablation(
    document: str,
    query: str,
    system_prompt: str,
    l1_detector,
    l2_classifier,
    l3_monitor,
    meta_aggregator,
    mode: str,
) -> dict:
    chunks = split_chunks(document)
    l1 = _neutral_l1()
    l2 = _neutral_l2()
    l3 = _empty_l3()

    if mode in {"l1_only", "l1_l2_or", "l1_l3_or"}:
        l1 = l1_detector.scan(chunks)

    if mode == "l1_only":
        blocked = bool(l1["blocked"])
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": "Layer 1 - Anomaly Detection" if blocked else None,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(l1["max_score"])},
            "early_exit": False,
        }

    if mode in {"l2_only", "l1_l2_or", "l2_l3_or"}:
        l2 = l2_classifier.classify(query, chunks)

    if mode == "l2_only":
        blocked = bool(l2["blocked"])
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": "Layer 2 - Intent Classifier" if blocked else None,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(l2["stage1_prob"])},
            "early_exit": False,
        }

    if mode == "l1_l2_or":
        blocked = bool(l1["blocked"] or l2["blocked"])
        if l1["blocked"]:
            blocking_layer = "Layer 1 - Anomaly Detection"
        elif l2["blocked"]:
            blocking_layer = "Layer 2 - Intent Classifier"
        else:
            blocking_layer = None
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": blocking_layer,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(max(l1["max_score"], l2["stage1_prob"]))},
            "early_exit": False,
        }

    if mode in {"l3_only", "l1_l3_or", "l2_l3_or"}:
        l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2)

    if mode == "l3_only":
        blocked = bool(l3["blocked"])
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": "Layer 3 - Behavioral Monitor" if blocked else None,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(l3["consistency_score"])},
            "early_exit": False,
        }

    if mode == "l1_l3_or":
        blocked = bool(l1["blocked"] or l3["blocked"])
        if l1["blocked"]:
            blocking_layer = "Layer 1 - Anomaly Detection"
        elif l3["blocked"]:
            blocking_layer = "Layer 3 - Behavioral Monitor"
        else:
            blocking_layer = None
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": blocking_layer,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(max(l1["max_score"], l3["consistency_score"]))},
            "early_exit": False,
        }

    if mode == "l2_l3_or":
        blocked = bool(l2["blocked"] or l3["blocked"])
        if l2["blocked"]:
            blocking_layer = "Layer 2 - Intent Classifier"
        elif l3["blocked"]:
            blocking_layer = "Layer 3 - Behavioral Monitor"
        else:
            blocking_layer = None
        return {
            "blocked": blocked,
            "action": "blocked" if blocked else "allow",
            "blocking_layer": blocking_layer,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": {"risk_score": float(max(l2["stage1_prob"], l3["consistency_score"]))},
            "early_exit": False,
        }

    raise ValueError(f"Unknown ablation mode: {mode}")


def _run_config(samples: list[dict], l1, l2, l3, agg, mode: str, desc: str):
    y_true, y_pred, latencies, records = [], [], [], []
    t_start = time.time()
    n = len(samples)

    import unicodedata
    def sanitize_text(text: str) -> str:
        if not text:
            return ""
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t')
        return unicodedata.normalize('NFKC', text)

    for i, sample in enumerate(samples, 1):
        t0 = time.perf_counter()
        doc = sanitize_text(sample["document"])
        qry = sanitize_text(sample["query"])
        if mode == "full_pipeline":
            result = run_pipeline(
                document=doc,
                query=qry,
                system_prompt=SYSTEM_PROMPT,
                l1_detector=l1,
                l2_classifier=l2,
                l3_monitor=l3,
                meta_aggregator=agg,
            )
        else:
            result = _run_component_ablation(
                document=doc,
                query=qry,
                system_prompt=SYSTEM_PROMPT,
                l1_detector=l1,
                l2_classifier=l2,
                l3_monitor=l3,
                meta_aggregator=agg,
                mode=mode,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        predicted = 1 if _is_detected(result.get("action", "allow")) else 0

        y_true.append(sample["label"])
        y_pred.append(predicted)
        latencies.append(latency_ms)
        records.append(
            {
                "config": desc,
                "sample_id": sample["sample_id"],
                "source": sample["source"],
                "source_index": sample["source_index"],
                "source_file": sample["source_file"],
                "true_label": sample["label"],
                "pred_detected": predicted,
                "action": result.get("action", "allow"),
                "risk_score": float(result.get("meta", {}).get("risk_score", 0.0)),
                "l1_score": float(result.get("l1", {}).get("max_score", 0.0)),
                "l2_score": float(result.get("l2", {}).get("stage1_prob", 0.0)),
                "l3_score": float(result.get("l3", {}).get("consistency_score", 0.0)),
                "blocking_layer": result.get("blocking_layer"),
                "early_exit": bool(result.get("early_exit", False)),
                "latency_ms": round(latency_ms, 2),
                "timings": result.get("timings", {}),
                "attack_type": sample.get("attack_type"),
            }
        )

        if i % 50 == 0 or i == n:
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 0.01)
            eta = (n - i) / max(rate, 0.01)
            print(f"    [{desc}] {i}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return y_true, y_pred, latencies, records


def _compute_row(y_true: list[int], y_pred: list[int], latencies: list[float]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    adr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = (2 * adr * prec) / max(adr + prec, 1e-8)
    arr = np.asarray(latencies, dtype=float)

    return {
        "n": len(y_true),
        "ADR": round(adr, 4),
        "FPR": round(fpr, 4),
        "Prec": round(prec, 4),
        "F1": round(f1, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "mean_latency_ms": round(float(np.mean(arr)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
    }


def main() -> None:
    print("=" * 60)
    print("  HOLDOUT ABLATION RUNNER (Pipeline-aligned)")
    print("=" * 60)

    print("Loading pipeline components ...")
    l1 = load_detector()
    l2 = load_classifier()
    l3 = load_monitor()
    agg = MetaAggregator.load()

    print("\nLoading evaluation data ...")
    samples = (
        _load_samples(ATTACKS_FILE, 1, "attacks")
        + _load_samples(EVASION_FILE, 1, "evasions")
        + _load_samples(BENIGN_FILE, 0, "benign")
    )
    print(f"  Total samples : {len(samples)}")

    configs = [
        ("l1_only", "L1 only"),
        ("l2_only", "L2 only"),
        ("l3_only", "L3 only"),
        ("l1_l2_or", "L1+L2 Union"),
        ("l1_l3_or", "L1+L3 Union"),
        ("l2_l3_or", "L2+L3 Union"),
        ("full_pipeline", "Full (Pipeline)"),
    ]

    config_tp_sets = {}
    results = {}
    sample_records = []
    total_start = time.time()

    for mode, desc in configs:
        print(f"\nConfig : {desc}")
        yt, yp, latencies, records = _run_config(samples, l1, l2, l3, agg, mode, desc)
        tp_ids = {
            sample["sample_id"]
            for sample, true_label, pred in zip(samples, yt, yp)
            if true_label == 1 and pred == 1
        }
        config_tp_sets[desc] = tp_ids

        row = _compute_row(yt, yp, latencies)
        results[desc] = {
            "metrics": row,
            "methodology": (
                "Full (Pipeline) uses orchestrator.run_pipeline; component rows use "
                "direct layer calls on the same processed document/query pairs."
            ),
        }
        sample_records.extend(records)

        print(
            "  Result : "
            f"ADR={row['ADR']:.4f}  FPR={row['FPR']:.4f}  "
            f"F1={row['F1']:.4f}  TP={row['TP']}  FP={row['FP']}  FN={row['FN']}"
        )

    for desc, tp_set in config_tp_sets.items():
        other_union = set()
        for other_desc, other_set in config_tp_sets.items():
            if other_desc != desc:
                other_union |= other_set
        unique_ids = sorted(tp_set - other_union)
        results[desc]["unique_tp_count"] = len(unique_ids)
        results[desc]["unique_tp_sample_ids"] = unique_ids
        results[desc]["caught_attack_count"] = len(tp_set)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with SAMPLE_LOG_PATH.open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    total_elapsed = time.time() - total_start
    print(f"\nAblation study completed in {total_elapsed / 60:.2f} minutes.")
    print(f"Saved -> {RESULTS_PATH}")
    print(f"Saved per-sample records -> {SAMPLE_LOG_PATH}")


if __name__ == "__main__":
    main()
