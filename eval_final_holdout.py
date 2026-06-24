"""
eval_final_holdout.py
─────────────────────
Evaluates RAG-Shield against the sealed final_holdout dataset.
This data was NEVER used in:
  - training the meta-aggregator
  - tuning regex patterns
  - selecting thresholds

Provides a true, unbiased measure of generalization.
"""

# Import onnxruntime first to avoid OpenMP DLL conflict on Windows
try:
    import onnxruntime as _ort
except ImportError:
    pass

import json, time, sys, os
import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_auc_score

# Configure stdout/stderr to use UTF-8 and ignore/replace characters that CP1252 doesn't support
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import run_pipeline, MetaAggregator
from layer1_anomaly import load_detector
from layer2_classifier import load_classifier
from layer3_enhanced import load_monitor

HOLDOUT_DIR   = "data/final_holdout"
ATTACKS_FILE  = os.path.join(HOLDOUT_DIR, "attacks.jsonl")
BENIGN_FILE   = os.path.join(HOLDOUT_DIR, "benign.jsonl")
EVASION_FILE  = os.path.join(HOLDOUT_DIR, "evasions.jsonl")

SYS_PROMPT = "Answer using only the provided knowledge base."


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval(l1, l2, l3, meta, samples, label_value, split_name):
    """Run pipeline on all samples, return metrics dict."""
    y_true, y_pred, y_score = [], [], []
    latencies = []
    fp_examples, fn_examples = [], []
    type_results = {}

    print(f"\n[holdout] Evaluating {split_name} ({len(samples)} samples)...")

    for i, row in enumerate(samples):
        # Determine query and document
        text      = row.get("text", "")
        query     = row.get("query") or text
        doc       = row.get("document") or row.get("context", "")
        placement = row.get("placement", "user_query")

        # Indirect injection: text IS the malicious document
        if placement == "retrieved_document" and not doc:
            doc   = text
            query = "Please summarize the retrieved document."

        # Ensure both are non-empty
        if not doc or not doc.strip():
            doc = "This is a standard policy document with no special content."
        if not query or not query.strip():
            query = "What is the summary of this document?"

        t0 = time.perf_counter()
        try:
            result = run_pipeline(
                document=doc,
                query=query,
                system_prompt=SYS_PROMPT,
                l1_detector=l1,
                l2_classifier=l2,
                l3_monitor=l3,
                meta_aggregator=meta,
            )
        except Exception as e:
            print(f"  [ERROR] sample {i}: {e}")
            result = {"action": "allow", "meta": {"risk_score": 0.0}}
        latency_ms = (time.perf_counter() - t0) * 1000

        blocked = result.get("action") in ("blocked", "hard_block")
        risk    = result.get("meta", {}).get("risk_score", 0.0)

        y_true.append(label_value)
        y_pred.append(1 if blocked else 0)
        y_score.append(float(risk))
        latencies.append(latency_ms)

        # Per-type tracking
        atype = row.get("attack_type") or row.get("sample_type") or row.get("type", "unknown")
        if atype not in type_results:
            type_results[atype] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        tr = type_results[atype]

        if label_value == 1:
            if blocked: tr["tp"] += 1
            else:
                tr["fn"] += 1
                fn_examples.append({"text": query[:120], "doc_snippet": doc[:80], "type": atype, "risk": round(risk, 3)})
        else:
            if blocked:
                tr["fp"] += 1
                fp_examples.append({"text": query[:120], "type": atype, "risk": round(risk, 3)})
            else:
                tr["tn"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [progress] {i+1}/{len(samples)}")

    # Aggregate
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    n_pos = tp + fn
    n_neg = fp + tn

    adr  = tp / n_pos if n_pos > 0 else float("nan")
    fpr  = fp / n_neg if n_neg > 0 else float("nan")
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    mean_lat = float(np.mean(latencies))
    p95_lat  = float(np.percentile(latencies, 95))

    return {
        "split":       split_name,
        "n":           len(samples),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "ADR":         round(adr,  4),
        "FPR":         round(fpr,  4),
        "F1":          round(f1,   4),
        "Precision":   round(prec, 4),
        "AUC_ROC":     round(auc,  4) if not np.isnan(auc) else "N/A",
        "mean_lat_ms": round(mean_lat, 1),
        "p95_lat_ms":  round(p95_lat,  1),
        "fp_examples": fp_examples[:5],
        "fn_examples": fn_examples[:5],
        "per_type": {
            t: {
                "ADR": round(v["tp"] / max(v["tp"] + v["fn"], 1), 4),
                "n":   v["tp"] + v["fn"],
                "TP":  v["tp"], "FN": v["fn"],
            }
            for t, v in type_results.items()
            if v["tp"] + v["fn"] > 0
        },
    }


def print_result(r):
    print(f"\n{'='*62}")
    print(f"  {r['split'].upper()}")
    print(f"{'='*62}")
    print(f"  n                  : {r['n']}")
    print(f"  TP={r['TP']}  FP={r['FP']}  TN={r['TN']}  FN={r['FN']}")
    if not np.isnan(r["ADR"]):
        print(f"  ADR  (Recall)      : {r['ADR']:.4f}  ({r['ADR']*100:.2f}%)")
    if not np.isnan(r["FPR"]):
        print(f"  FPR                : {r['FPR']:.4f}  ({r['FPR']*100:.2f}%)")
    print(f"  F1                 : {r['F1']:.4f}")
    print(f"  Precision          : {r['Precision']:.4f}")
    print(f"  AUC-ROC            : {r['AUC_ROC']}")
    print(f"  Mean latency (ms)  : {r['mean_lat_ms']}")
    print(f"  P95  latency (ms)  : {r['p95_lat_ms']}")
    if r.get("per_type"):
        print(f"  Per-type ADR:")
        for t, v in sorted(r["per_type"].items(), key=lambda x: -x[1]["ADR"]):
            print(f"    {t:<38} {v['ADR']*100:5.1f}%  (n={v['n']}, TP={v['TP']}, FN={v['FN']})")
    if r.get("fn_examples"):
        print(f"\n  -- Missed attacks (FN) --")
        for ex in r["fn_examples"]:
            print(f"    risk={ex['risk']:.3f}  type={ex['type']}")
            print(f"      Q: {ex['text']!r}")
            if ex.get("doc_snippet"):
                print(f"      D: {ex['doc_snippet']!r}")
    if r.get("fp_examples"):
        print(f"\n  -- False positives (FP) --")
        for ex in r["fp_examples"]:
            print(f"    risk={ex['risk']:.3f}  type={ex['type']}")
            print(f"      Q: {ex['text']!r}")


if __name__ == "__main__":
    print("=" * 62)
    print("  RAG-Shield  -  FINAL HOLDOUT EVALUATION")
    print("  Sealed data: data/final_holdout/")
    print("  provenance : blind_llm_generated_post_freeze")
    print("=" * 62)

    print("\n[holdout] Loading models (once)...")
    l1   = load_detector()
    l2   = load_classifier()
    l3   = load_monitor()
    meta = MetaAggregator.load()
    print("[holdout] All models loaded.\n")

    attacks  = load_jsonl(ATTACKS_FILE)
    benign   = load_jsonl(BENIGN_FILE)
    evasions = load_jsonl(EVASION_FILE)

    print(f"[holdout] Dataset sizes:")
    print(f"  attacks  : {len(attacks)}")
    print(f"  benign   : {len(benign)}")
    print(f"  evasions : {len(evasions)}")

    r_att = run_eval(l1, l2, l3, meta, attacks,  1, "Attacks (standard)")
    r_ben = run_eval(l1, l2, l3, meta, benign,   0, "Benign (FPR test)")
    r_ev  = run_eval(l1, l2, l3, meta, evasions, 1, "Evasions (obfuscated/indirect)")

    print_result(r_att)
    print_result(r_ben)
    print_result(r_ev)

    print(f"\n{'='*62}")
    print("  PAPER-READY SUMMARY")
    print(f"{'='*62}")
    print(f"  ADR  attacks  (n={len(attacks):>3}) : {r_att['ADR']*100:.2f}%  (F1={r_att['F1']*100:.2f}%)")
    print(f"  ADR  evasions (n={len(evasions):>3}) : {r_ev['ADR']*100:.2f}%")
    print(f"  FPR  benign   (n={len(benign):>3}) : {r_ben['FPR']*100:.2f}%")
    print(f"  Mean latency        : {r_att['mean_lat_ms']} ms")
    print()

    # Save JSON
    out = {"attacks": r_att, "benign": r_ben, "evasions": r_ev}
    for v in out.values():
        v.pop("fp_examples", None)
        v.pop("fn_examples", None)
    os.makedirs("logs", exist_ok=True)
    with open("logs/final_holdout_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("[holdout] Saved -> logs/final_holdout_results.json")
