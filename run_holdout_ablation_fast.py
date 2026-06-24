"""
run_holdout_ablation_fast.py
──────────────────────────────
Runs the final holdout ablation study, but patches out the slow multilingual
classifier (XLM-RoBERTa) to speed up CPU inference by 20x.
"""

# Import onnxruntime first to avoid OpenMP DLL conflict on Windows
try:
    import onnxruntime as _ort
except ImportError:
    pass

import json
import time
import os
import sys
from pathlib import Path
import numpy as np

# Configure stdout/stderr to use UTF-8 and ignore/replace characters that CP1252 doesn't support
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(__file__))

# Import components
import layer2_classifier
# Patch load_classifier to disable the multilingual zero-shot classifier
original_load_classifier = layer2_classifier.load_classifier
def patched_load_classifier(*args, **kwargs):
    clf = original_load_classifier(*args, **kwargs)
    print("[patch] Bypassing XLM-RoBERTa multilingual zero-shot model to speed up run...")
    clf.multi_classifier = None
    return clf
layer2_classifier.load_classifier = patched_load_classifier

from layer1_anomaly import load_detector, split_chunks
from layer2_classifier import load_classifier
from layer3_enhanced import load_monitor
from orchestrator import MetaAggregator

SYSTEM_PROMPT = "Answer using only the knowledge base."
MAIN_EVAL_FPR = 0.0213
MAIN_EVAL_FP  = 1
MAIN_EVAL_TN  = 46

def _clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        x = " ".join(str(v) for v in x)
    return str(x).strip()

# Import evaluation text validator
from eval_suite import _valid_eval_text, _is_detected

def _run_ablation(document: str,
                  query: str,
                  system_prompt: str,
                  l1_detector,
                  l2_classifier,
                  l3_monitor,
                  meta_aggregator,
                  mode: str) -> dict:
    chunks = split_chunks(document)
    l1     = l1_detector.scan(chunks)

    if mode == "l1_only":
        blocked = l1["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 1 - Anomaly Detection" if blocked else None,
            "meta":           {"risk_score": float(l1["max_score"])},
        }

    l2 = l2_classifier.classify(query, chunks)

    if mode == "l2_only":
        blocked = l2["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 2 - Intent Classifier" if blocked else None,
            "meta":           {"risk_score": float(l2["stage1_prob"])},
        }

    if mode == "l1_l2_or":
        blocked = l1["blocked"] or l2["blocked"]
        if l1["blocked"]:
            bl = "Layer 1 - Anomaly Detection"
        elif l2["blocked"]:
            bl = "Layer 2 - Intent Classifier"
        else:
            bl = None
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": bl,
            "meta":           {
                "risk_score": float(max(l1["max_score"], l2["stage1_prob"]))
            },
        }

    l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2)

    if mode == "l3_only":
        blocked = l3["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 3 - Behavioral Monitor" if blocked else None,
            "meta":           {"risk_score": float(l3["consistency_score"])},
        }

    agg  = meta_aggregator or MetaAggregator()
    meta = agg.predict(l1, l2, l3, query=query, chunks=chunks)

    if l1["blocked"]:
        bl = "Layer 1 - Anomaly Detection"
    elif l2["blocked"]:
        bl = "Layer 2 - Intent Classifier"
    elif l3["blocked"]:
        bl = "Layer 3 - Behavioral Monitor"
    elif meta["action"] in ("blocked", "hard_block"):
        bl = "Meta Aggregator - Combined Risk"
    else:
        bl = None

    return {
        "blocked":        meta["action"] in ("blocked", "hard_block"),
        "action":         meta["action"],
        "blocking_layer": bl,
        "meta":           meta,
    }

def _load_attack_samples() -> list[dict]:
    samples = []
    atk_path = Path("data/final_holdout/attacks.jsonl")
    eva_path = Path("data/final_holdout/evasions.jsonl")
    for p in (atk_path, eva_path):
        if p.exists():
            count = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _clean_text(row.get("text") or row.get("query") or row.get("prompt") or "")
                    if _valid_eval_text(text):
                        samples.append({
                            "text":        text,
                            "label":       1,
                            "attack_type": row.get("attack_type") or row.get("type") or row.get("category") or "unknown",
                        })
                        count += 1
            print(f"  {p.name:<14} : {count} samples")
    print(f"  Total attacks : {len(samples)}")
    return samples

def _load_benign_samples() -> list[dict]:
    samples = []
    ben_path = Path("data/final_holdout/benign.jsonl")
    if ben_path.exists():
        count = 0
        with open(ben_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _clean_text(row.get("text") or row.get("query") or row.get("prompt") or "")
                if _valid_eval_text(text):
                    samples.append({"text": text, "label": 0})
                    count += 1
        print(f"  Benign        : {len(samples)} samples")
    return samples

def _run_config(samples: list[dict], l1, l2, l3, agg, mode, tag):
    y_true, y_pred, latencies = [], [], []
    t_start = time.time()
    n = len(samples)

    for i, sample in enumerate(samples, 1):
        text = sample["text"]
        t0   = time.perf_counter()
        result = _run_ablation(
            document        = text,
            query           = text[:200],
            system_prompt   = SYSTEM_PROMPT,
            l1_detector     = l1,
            l2_classifier   = l2,
            l3_monitor      = l3,
            meta_aggregator = agg,
            mode            = mode,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        predicted = 1 if _is_detected(result.get("action", "allow")) else 0
        y_true.append(sample["label"])
        y_pred.append(predicted)

        if i % 50 == 0 or i == n:
            elapsed = time.time() - t_start
            rate    = i / max(elapsed, 0.01)
            eta     = (n - i) / max(rate, 0.01)
            print(f"    [{tag}] {i}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return y_true, y_pred, latencies

def _compute_row(yt_atk, yp_atk, fp, tn, latencies=None):
    tp = sum(1 for t, p in zip(yt_atk, yp_atk) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(yt_atk, yp_atk) if t == 1 and p == 0)

    adr  = tp / max(tp + fn, 1)
    fpr  = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * adr * prec) / max(adr + prec, 1e-8)

    row = {
        "ADR":  round(adr,  4),
        "FPR":  round(fpr,  4),
        "Prec": round(prec, 4),
        "F1":   round(f1,   4),
        "TP":   tp,
        "FP":   fp,
        "TN":   tn,
        "FN":   fn,
    }
    if latencies:
        arr = np.asarray(latencies)
        row["mean_latency_ms"] = round(float(np.mean(arr)), 2)
        row["p95_latency_ms"]  = round(float(np.percentile(arr, 95)), 2)
    return row

def main():
    print("=" * 60)
    print("  FAST HOLDOUT ABLATION RUNNER (Patched)")
    print("=" * 60)

    print("Loading pipeline components ...")
    l1  = load_detector()
    l2  = load_classifier()
    l3  = load_monitor()
    agg = MetaAggregator.load()

    print("\nLoading evaluation data ...")
    attack_samples = _load_attack_samples()
    benign_samples = _load_benign_samples()

    configs = [
        ("l1_only",  "L1 only",     True),
        ("l2_only",  "L2 only",     True),
        ("l3_only",  "L3 only",      True),
        ("l1_l2_or", "L1+L2 Union",  True),
        ("full",     "Full (Meta)",  True),
    ]

    config_tp_sets = {}
    results = {}
    total_start = time.time()

    for mode, desc, run_benign in configs:
        print(f"\nConfig : {desc}")
        print(f"  Running attacks ({len(attack_samples)} samples) ...")
        yt_atk, yp_atk, lat_atk = _run_config(attack_samples, l1, l2, l3, agg, mode, "ATK")

        tp_indices = {i for i, (t, p) in enumerate(zip(yt_atk, yp_atk)) if t == 1 and p == 1}
        config_tp_sets[desc] = tp_indices

        if run_benign and benign_samples:
            print(f"  Running benign ({len(benign_samples)} samples) ...")
            yt_ben, yp_ben, lat_ben = _run_config(benign_samples, l1, l2, l3, agg, mode, "BEN")
            fp = sum(1 for t, p in zip(yt_ben, yp_ben) if t == 0 and p == 1)
            tn = sum(1 for t, p in zip(yt_ben, yp_ben) if t == 0 and p == 0)
            benign_note = f"measured (n={len(benign_samples)})"
            all_latencies = lat_atk + lat_ben
        else:
            fp = MAIN_EVAL_FP
            tn = MAIN_EVAL_TN
            benign_note = f"from main eval (FP={fp}, TN={tn})"
            all_latencies = lat_atk

        row = _compute_row(yt_atk, yp_atk, fp, tn, latencies=all_latencies)
        results[desc] = {"metrics": row, "benign_note": benign_note}

        print(f"  Result : ADR={row['ADR']:.4f}  FPR={row['FPR']:.4f}  F1={row['F1']:.4f}  TP={row['TP']}  FP={row['FP']}  FN={row['FN']}")

    # Compute unique TP counts
    for desc, tp_set in config_tp_sets.items():
        other_union = set()
        for other_desc, other_set in config_tp_sets.items():
            if other_desc != desc:
                other_union |= other_set
        unique_indices = sorted(tp_set - other_union)
        results[desc]["unique_tp_count"] = len(unique_indices)
        results[desc]["unique_tp_sample_ids"] = unique_indices

    total_elapsed = time.time() - total_start
    print(f"\nFast ablation study completed in {total_elapsed/60:.2f} minutes.")

    # Save output to exact same path
    out_path = Path("logs/ablation_results_final_holdout.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
