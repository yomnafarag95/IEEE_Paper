"""
run_final_holdout_and_report.py
─────────────────────────────────
Runs the final holdout evaluation using BenchmarkRunner from eval_suite.py
to generate both the ROC/PR curves and the detailed JSON report expected
by generate_paper_figures.py.
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
from eval_suite import BenchmarkRunner, _clean_text

HOLDOUT_DIR   = "data/final_holdout"
ATTACKS_FILE  = os.path.join(HOLDOUT_DIR, "attacks.jsonl")
BENIGN_FILE   = os.path.join(HOLDOUT_DIR, "benign.jsonl")
EVASION_FILE  = os.path.join(HOLDOUT_DIR, "evasions.jsonl")
SYS_PROMPT    = "Answer using only the provided knowledge base."
REPORT_PATH   = "logs/eval_report_final_holdout_2026-06-16.json"

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def process_sample(row, label_value):
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
        
    atype = row.get("attack_type") or row.get("sample_type") or row.get("type", "unknown")
    return doc, query, atype

def main():
    print("=" * 62)
    print("  RAG-Shield - FINAL HOLDOUT DETAILED REPORT RUNNER")
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

    print(f"[holdout] Loaded {len(attacks)} attacks, {len(benign)} benign, {len(evasions)} evasions.")

    def pipeline_fn(document, query, system_prompt):
        return run_pipeline(
            document=document,
            query=query,
            system_prompt=system_prompt,
            l1_detector=l1,
            l2_classifier=l2,
            l3_monitor=l3,
            meta_aggregator=meta,
        )

    runner = BenchmarkRunner(pipeline_fn, plots_dir="logs/curves")
    
    # We will log everything in runner.results_log using _run_one
    print("\n[holdout] Running attacks...")
    for idx, row in enumerate(attacks):
        doc, query, atype = process_sample(row, 1)
        runner._run_one(doc=doc, query=query, label=1, attack_type=atype, source="attacks")
        if (idx + 1) % 50 == 0:
            print(f"  [progress] {idx+1}/{len(attacks)}")

    print("\n[holdout] Running benign...")
    for idx, row in enumerate(benign):
        doc, query, atype = process_sample(row, 0)
        runner._run_one(doc=doc, query=query, label=0, attack_type=atype, source="benign")
        if (idx + 1) % 50 == 0:
            print(f"  [progress] {idx+1}/{len(benign)}")

    print("\n[holdout] Running evasions...")
    for idx, row in enumerate(evasions):
        doc, query, atype = process_sample(row, 1)
        runner._run_one(doc=doc, query=query, label=1, attack_type=atype, source="evasions")
        if (idx + 1) % 50 == 0:
            print(f"  [progress] {idx+1}/{len(evasions)}")

    print("\n[holdout] Computing combined metrics & generating curves...")
    metrics = runner._metrics(runner.results_log, name="final_holdout")

    # Add source counts and file info similar to original report
    metrics["source_counts"] = {
        "attacks": len(attacks),
        "benign": len(benign),
        "evasions": len(evasions)
    }

    # Add evaluation state
    try:
        from freeze_state import FREEZE_PATH, compare_to_frozen
        freeze_report = compare_to_frozen()
        freeze_meta = {
            "freeze_file": str(FREEZE_PATH).replace("\\", "/"),
            "freeze_status": freeze_report.get("status"),
            "frozen_created_at_utc": freeze_report.get("frozen_created_at_utc"),
            "difference_count": len(freeze_report.get("differences", [])),
        }
        if FREEZE_PATH.exists():
            with open(FREEZE_PATH, encoding="utf-8") as f:
                frozen = json.load(f)
            freeze_meta["thresholds"] = frozen.get("thresholds", {})
            freeze_meta["config_flags"] = frozen.get("config_flags", {})
            freeze_meta["git"] = frozen.get("git", {})
        metrics["_evaluation_state"] = freeze_meta
    except Exception as exc:
        metrics["_evaluation_state"] = {
            "freeze_status": "unavailable",
            "error": str(exc),
        }

    # Save to the specific REPORT_PATH
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n[holdout] Completed! Saved report to {REPORT_PATH}")
    print(f"[holdout] Curves generated at logs/curves/roc_final_holdout.png and pr_final_holdout.png")

if __name__ == "__main__":
    main()
