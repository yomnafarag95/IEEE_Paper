"""
Update paper_draft.md with correct numbers from eval_report_final_holdout_2026-06-16.json
and ablation_results_final_holdout.json, then build the PDF.
"""
from __future__ import annotations
import re
from pathlib import Path

DRAFT = Path("paper_draft.md")
content = DRAFT.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENTS — ordered from most specific to least specific
# ─────────────────────────────────────────────────────────────────────────────

replacements = [
    # Abstract — full sentence swap
    (
        r"Evaluated on a sealed, hash-verified final holdout benchmark \(\$n = 867\$, comprising 249 attacks, 543 benign, and 75 adversarial evasions after deduplication\), RAG-Shield achieves a Prevention Attack Detection Rate \(ADR\) of \*\*87\.04%\*\* and a Detection ADR of \*\*87\.35%\*\* at a False Positive Rate of only \*\*0\.37%\*\* \(2/543 benign queries\)\. The system achieves F1 scores of \*\*92\.76%\*\* under the prevention policy and \*\*92\.94%\*\* under the detection policy, with ROC-AUC of \*\*0\.9643\*\* and AUC-PR of \*\*0\.9701\*\*\. In head-to-head comparisons against commercial alternatives on a separate shared subset, RAG-Shield \(ADR = 93\.54%, F1 = 0\.945\) outperforms Llama Prompt Guard 2 \(ADR = 79\.49%, F1 = 0\.883, \$p < 10\^\{-4\}\$\), Llama-3\.1-8B Guardrails \(ADR = 29\.78%\), and NVIDIA NeMo Injection Rails \(ADR = 7\.58%\)\. Final-holdout mean local pipeline latency is \*\*1,214 ms\*\*; live commercial-comparison latency for RAG-Shield is \*\*2,187 ms\*\* including the comparison protocol overhead\.",
        "Evaluated on a sealed, hash-verified final holdout benchmark ($n = 868$, comprising 250 attacks, 543 benign, and 75 adversarial evasions), RAG-Shield achieves an Attack Detection Rate (ADR) of **91.38%** at a False Positive Rate of only **0.55%** (3/543 benign queries). The system achieves an F1 score of **95.04%**, with ROC-AUC of **0.9576** and AUC-PR of **0.9723**. Mean local pipeline latency is **600.3 ms**. In head-to-head comparisons against commercial alternatives on a separate shared subset, RAG-Shield (ADR = 93.54%, F1 = 0.945) outperforms Llama Prompt Guard 2 (ADR = 79.49%, F1 = 0.883, $p < 10^{-4}$), Llama-3.1-8B Guardrails (ADR = 29.78%), and NVIDIA NeMo Injection Rails (ADR = 7.58%). Live commercial-comparison latency for RAG-Shield is **2,187 ms** including the comparison protocol overhead.",
    ),
    # Early exit count
    (
        r"\*\*284 total samples\*\* triggered the early-exit path; \*\*282\*\* of these were true-positive prevention decisions and \*\*2\*\* were benign false positives\.",
        "**287 total samples** triggered the early-exit path; **284** of these were true-positive prevention decisions and **3** were benign false positives.",
    ),
    # Layer attribution (early exit section)
    (
        r"Layer 2 provided \*\*260 of the 282 true-positive prevention attributions\*\* in the final holdout, confirming it as the pipeline's primary detection engine\.",
        "Layer 2 provided **272 of the 297 true-positive attributions** in the final holdout, confirming it as the pipeline's primary detection engine. Layer 1 contributed **13 attributions** and the Meta-Aggregator contributed **12 additional detections**.",
    ),
    # Benchmark n=867 → 868 in dataset description
    (
        r"The raw files contain 868 rows; the evaluator removes one duplicate attack row, yielding 867 evaluated samples\. The benchmark is split across three files:",
        "The benchmark comprises 868 evaluated samples split across three files:",
    ),
    # attacks.jsonl count
    (
        r"\| `attacks\.jsonl` \| 249 \|",
        "| `attacks.jsonl` | 250 |",
    ),
    # Dataset total row
    (
        r"\| \*\*Total\*\* \| \*\*867\*\* \|",
        "| **Total** | **868** |",
    ),
    # 324/867 fraction
    (
        r"324/867 = 37\.4",
        "325/868 = 37.4",
    ),
    (
        r"543/867 = 62\.6",
        "543/868 = 62.6",
    ),
    # Table caption
    (
        r"RAG-Shield Final Holdout Benchmark \(\$n = 867\$",
        "RAG-Shield Final Holdout Benchmark ($n = 868$",
    ),
    # Table I row: "Table I summarizes..."
    (
        r"Table I summarizes RAG-Shield's performance on the sealed final holdout benchmark \(\$n = 867\$\)\.",
        "Table I summarizes RAG-Shield's performance on the sealed final holdout benchmark ($n = 868$).",
    ),
    # Table I header and rows
    (
        r"\| Metric \| Prevention Policy \| Detection Policy \| 95% CI \|[\s\S]*?\| Early Exit Count \| 284 total \| 284 total \| - \|",
        "| Metric | Value | 95% CI |\n|--------|-------|--------|\n| Total Samples ($n$) | 868 | -- |\n| Attacks + Evasions | 325 | -- |\n| Benign | 543 | -- |\n| True Positives (TP) | 297 | -- |\n| False Positives (FP) | 3 | -- |\n| False Negatives (FN) | 28 | -- |\n| True Negatives (TN) | 540 | -- |\n| **ADR (Attack Detection Rate)** | **91.38%** | [87.83%, 93.97%] |\n| **FPR (False Positive Rate)** | **0.55%** | [0.19%, 1.61%] |\n| **Precision** | **99.00%** | [97.10%, 99.66%] |\n| **F1 Score** | **95.04%** | [93.16%, 96.68%] |\n| **ROC-AUC** | **0.9576** | [94.18%, 97.20%] |\n| **AUC-PR** | **0.9723** | -- |\n| Mean Latency (ms) | 600.3 | -- |\n| P95 Latency (ms) | 2,663.8 | -- |\n| Early Exit Count | 287 | -- |",
    ),
    # Specificity sentence after Table I
    (
        r"The system correctly allows 541 of 543 benign queries \(99\.63% specificity\) and blocks 282-283 of 324 attack\+evasion samples\.",
        "The system correctly allows 540 of 543 benign queries (99.45% specificity) and blocks 297 of 325 attack+evasion samples.",
    ),
    # Verbatim confusion matrix
    (
        r"Benign  \(N = 543\)\s+TN = 541\s+FP = 2[\s\S]*?Attacks \(N = 324\)\s+FN = 41\s+TP = 283[\s\S]*?\(87\.35%\)",
        "Benign  (N = 543)      TN = 540              FP = 3\n                       (99.45%)             (0.55%)\nAttacks (N = 325)      FN = 28               TP = 297\n                       (8.62%)              (91.38%)",
    ),
    # Latency table rows
    (
        r"\| Layer 1 \(Anomaly: IForest \+ ECOD \+ OCSVM\) \| 695\.75 \| 4,165\.41 \|",
        "| Layer 1 (Anomaly: IForest + ECOD + OCSVM) | 434.11 | 2,351.42 |",
    ),
    (
        r"\| Layer 2 \(DeBERTa ONNX, query \+ docs\) \| 330\.22 \| 1,115\.33 \|",
        "| Layer 2 (DeBERTa ONNX, query + docs) | 164.85 | 629.74 |",
    ),
    (
        r"\| L1 \+ L2 Wall Clock \(parallel\) \| 879\.45 \| 4,166\.47 \|",
        "| L1 + L2 Wall Clock (parallel) | 558.31 | 2,631.45 |",
    ),
    (
        r"\| Layer 3 \(Cross-Encoder \+ canary \+ regex\) \| 0\.25 \| 0\.46 \|",
        "| Layer 3 (Cross-Encoder + canary + regex) | 0.18 | 0.40 |",
    ),
    (
        r"\| Meta-Aggregator \(Logistic Regression\) \| 0\.64 \| 1\.10 \|",
        "| Meta-Aggregator (Logistic Regression) | 1.00 | 2.25 |",
    ),
    (
        r"\| \*\*Total Pipeline\*\* \| \*\*1,211\.93\*\* \| \*\*4,633\.85\*\* \|",
        "| **Total Pipeline** | **600.3** | **2,663.8** |",
    ),
    # Latency prose
    (
        r"The top-level evaluator reports a mean latency of 1,214\.0 ms and P95 latency of 4,635\.1 ms; Table II reports the internal per-component timing totals from the same final-holdout JSON artifact\.",
        "The evaluator reports a mean latency of 600.3 ms and P95 latency of 2,663.8 ms.",
    ),
    (
        r"Parallel execution of L1 and L2 reduces the combined wall clock to 879 ms despite L1's 695 ms mean latency\. Layer 3 and the meta-aggregator together contribute less than 1 ms of overhead\.",
        "Parallel execution of L1 and L2 reduces the combined wall clock to 558 ms. Layer 3 and the meta-aggregator together contribute less than 1.2 ms of overhead.",
    ),
    (
        r"L1 registered \*\*692 cache hits\*\* and L2 document scanning registered \*\*692 cache hits\*\* across the 867 evaluation runs, corresponding to a cache hit rate of approximately 79\.8%\.",
        "L1 registered **757 cache hits** and L2 document scanning registered **757 cache hits** across the 868 evaluation runs, corresponding to a cache hit rate of approximately 87.2%.",
    ),
    # Layer attribution subsection
    (
        r"Layer 2 \(Intent Classifier\) was the dominant blocking layer, receiving \*\*260 of 282 true-positive prevention attributions\*\* \(92\.2%\) in the final holdout\. Layer 1 \(Anomaly Detector\) received \*\*22 true-positive attributions\*\* \(7\.8%\)\. The ablation study below shows that the marginal prevention gain over Layer 2 alone is smaller: L1\+L2 catches one additional attack relative to L2-only under the frozen policy\.",
        "Layer 2 (Intent Classifier) was the dominant blocking layer, receiving **272 of 297 true-positive attributions** (91.6%) in the final holdout. Layer 1 (Anomaly Detector) received **13 unique attributions** (4.4%), and the Meta-Aggregator contributed **12 additional detections** (4.0%) via its probability-threshold border decisions.",
    ),
    # Conclusion
    (
        r"RAG-Shield achieves a comprehensive attack detection rate of \*\*87\.35%\*\* at an enterprise-grade False Positive Rate of \*\*0\.37%\*\*\.",
        "RAG-Shield achieves an Attack Detection Rate of **91.38%** at an enterprise-grade False Positive Rate of **0.55%** on the sealed final holdout ($n = 868$).",
    ),
    (
        r"Our ablation study shows that the full prevention policy improves by \*\*0\.31 pp\*\* over Layer 2 alone \(87\.04% vs\. 86\.73%\), while the full detection policy adds one monitored true positive \(87\.35%\)\. This means the current final-holdout result is driven primarily by document-level DeBERTa scanning, with Layer 1 adding a small number of structurally anomalous detections and Layer 3 requiring separate canary/output and multi-turn benchmarks for full evaluation\. Head-to-head comparisons against commercial guardrails show a \*\*\+14\.05 pp ADR advantage over Llama Prompt Guard 2\*\* and a \*\*\+63\.76 pp advantage over Llama-3\.1-8B Guardrails\*\* on the live comparison subset\.",
        "Our ablation study demonstrates that Layer 2 (DeBERTa ONNX with document-chunk scanning) provides the dominant detection signal (86.73% ADR standalone), with Layer 1 adding structurally anomalous detections and the meta-aggregator contributing 12 additional borderline decisions via calibrated probabilistic aggregation. Head-to-head comparisons against commercial guardrails show a **+14.05 pp ADR advantage over Llama Prompt Guard 2** and a **+63.76 pp advantage over Llama-3.1-8B Guardrails** on the live comparison subset, at a mean local pipeline latency of only **600.3 ms**.",
    ),
    # Residual FN count
    (
        r"The prevention policy leaves \*\*42\*\* undetected attacks in the final holdout \(12\.96%\), while the detection policy leaves \*\*41\*\* undetected attacks \(12\.65%\)\.",
        "The pipeline leaves **28** undetected attacks in the final holdout (8.62%).",
    ),
    # Precision-recall discussion
    (
        r"RAG-Shield's final operating point was selected to prioritize \*\*precision\*\* \(99\.30%\) over recall \(87\.04%\)\.",
        "RAG-Shield's final operating point was selected to prioritize **precision** (99.00%) over recall (91.38%).",
    ),
    (
        r"AUC-PR = 0\.9701\) supports",
        "AUC-PR = 0.9723) supports",
    ),
    # Conclusion final sentence (precision/F1)
    (
        r"These results demonstrate that a locally deployed, multi-layered ensemble combining statistical, neural, and behavioral techniques can outperform several commercial guardrails on the evaluated direct, indirect, and evasion-heavy test sets, while keeping false positives low on the sealed final holdout\.",
        "These results demonstrate that a locally deployed, multi-layered ensemble combining statistical, neural, and behavioral techniques can outperform several commercial guardrails on evaluated direct, indirect, and evasion-heavy test sets, while maintaining a precision of **99.00%** and an F1 of **95.04%** on the sealed final holdout.",
    ),
]

changed = 0
for pattern, replacement in replacements:
    new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if n > 0:
        print(f"  OK [{n}x] {pattern[:60]}...")
        content = new_content
        changed += n
    else:
        print(f"  MISS: {pattern[:60]}...")

DRAFT.write_text(content, encoding="utf-8")
print(f"\nDone — {changed} replacements made → paper_draft.md saved.")
