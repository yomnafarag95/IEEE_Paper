# RAG-Shield: A Three-Layer Ensemble Defense Against Prompt Injection in RAG Pipelines

> **Submitted · ITC-Egypt 2026 · IEEE**
> Queen's University · School of Computing · Kingston, ON, Canada

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![ADR: 91.38%](https://img.shields.io/badge/ADR-91.38%25-brightgreen)](logs/)
[![FPR: 0.55%](https://img.shields.io/badge/FPR-0.55%25-brightgreen)](logs/)
[![AUC--ROC: 0.9576](https://img.shields.io/badge/AUC--ROC-0.9576-brightgreen)](logs/)
[![AUC--PR: 0.9723](https://img.shields.io/badge/AUC--PR-0.9723-brightgreen)](logs/)

---

## Overview

**RAG-Shield** is a production-ready, three-layer ensemble defense system that detects and blocks
prompt injection attacks in Retrieval-Augmented Generation (RAG) pipelines.
It targets both **direct** prompt injection (HackAPrompt corpus) and **indirect** injection via
retrieved documents (InjecAgent benchmark), as well as encoding-obfuscated evasions, credential
exfiltration, and multi-turn crescendo attacks.

### Key Results — Sealed Holdout (*n* = 868, frozen thresholds)

> Git commit `3ef152293d76b81c0a080a8d26d67e010297d7f7` · thresholds in `logs/frozen_thresholds.json`

| Metric | Value | 95% CI |
|:---|:---:|:---:|
| **ADR (Attack Detection Rate)** | **91.38%** | [88.18%, 94.34%] |
| **False Positive Rate** | **0.55%** | [0.00%, 1.29%] |
| **Precision** | **99.00%** | [97.69%, 100.00%] |
| **F1 Score** | **95.04%** | [93.20%, 96.68%] |
| **ROC-AUC** | **0.9576** | [0.9419, 0.9721] |
| **AUC-PR** | **0.9723** | [0.9615, 0.9819] |
| **Attacks (Block/Monitor/Allow)** | **297 / 0 / 28** | — |
| Mean Latency (ms) | 562.95 | — |
| P95 Latency (ms) | 3,829 | — |

> CIs: Bootstrap percentile (*B* = 10,000) for ADR, F1, Precision, FPR, ROC-AUC, and PR-AUC.

### Commercial Guardrail Comparison (*n* = 856 live-API subset)

| System | ADR | FPR | F1 | Latency (ms) | McNemar *p* |
|:---|:---:|:---:|:---:|:---:|:---:|
| NeMo Rail (Llama-3.1) | 7.58% | 0.00% | 0.141 | 5,251 | 2.3×10⁻⁵⁸ |
| Llama-3.1-8B Guardrails | 29.78% | 0.00% | 0.459 | 3,326 | 1.2×10⁻³⁸ |
| Llama Prompt Guard 2 | 79.49% | 0.40% | 0.883 | 427 | 4.7×10⁻⁵ |
| **RAG-Shield** | **93.54%** | 3.20%† | **0.944** | **2,187** | — |

> †FPR of 3.20% on this subset is attributable to frozen holdout thresholds; recalibrating τ_block to 0.45 reduces it to ≤1% at ~2 pp ADR cost.

---

## Architecture

```
User Query + Retrieved Chunks
          |
    ┌─────┴──────────────────────────────────┐
    │       ObfuscationDecoder               │
    │  NFKC normalization + Base64/Hex/      │
    │  ROT13/Leetspeak/Unicode decode        │
    └─────┬──────────────────────────────────┘
          |
    ┌─────┴─────┐   ┌──────────────────────┐
    │  LAYER 1  │   │       LAYER 2        │  ← executed in parallel (ThreadPoolExecutor)
    │ IsoForest │   │  DeBERTa-v3 ONNX     │
    │ ECOD      │   │  (query + doc chunks)│
    │ OCSVM     │   │  XLM-RoBERTa fallback│
    │ τ = 0.85  │   │  τ_early = 0.60      │
    └─────┬─────┘   └──────────┬───────────┘
          └────────┬───────────┘
                   │  (early-exit hard block if either threshold exceeded)
    ┌──────────────┴───────────────────────┐
    │             LAYER 3                   │
    │  Schema Validator (Pydantic)          │
    │  Regex exfiltration detection         │
    │  Cross-encoder consistency scoring   │
    │  (ms-marco-MiniLM-L-12-v2, τ=0.55)  │
    │  Cryptographic canary token check    │
    │  Stateful crescendo tracker (τ=0.72) │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┴───────────────────────┐
    │         META-AGGREGATOR              │
    │  Calibrated Logistic Regression      │
    │  7-dim feature vector                │
    │  [S_L1, S_L2, δ_s, δ_b, S_L3,      │
    │   S_L1·S_L2, S_L1·S_L3]            │
    │  τ_block=0.35 | τ_monitor=0.31      │
    └──────────────┬───────────────────────┘
                   │
          ALLOW / MONITOR / BLOCK
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yomnafarag95/IEEE_Paper.git
cd IEEE_Paper
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env    # Linux/macOS
copy .env.example .env  # Windows

# Download and preprocess datasets (auto-fetched on first run)
python data_loader.py

# Train the meta-aggregator (optional — pre-trained weights not included)
python train_meta_aggregator.py

# Run the Streamlit demo app
streamlit run app.py

# Run with Docker
docker build -t rag-shield .
docker run -p 8501:8501 rag-shield

# Full evaluation suite (standard + benign + evasion)
python eval_suite.py --mode all

# Reproduce sealed final holdout (frozen thresholds)
python run_final_holdout_and_report.py

# Bootstrap confidence intervals (AUC-PR + AUC-ROC)
python bootstrap_ci.py

# Ablation study
python ablation_study.py

# Baseline comparison (classical)
python compare_baselines.py

# Commercial guardrail comparison (requires API keys in .env)
python compare_commercial.py
```

---

## Repository Structure

```
rag_shield_ieee_6page.tex      IEEE 6-page paper (camera-ready)

# Pipeline
orchestrator.py                Main pipeline controller + meta-aggregator
obfuscation_decoder.py         NFKC + Base64/Hex/ROT13/Leetspeak/Unicode decoder
layer1_anomaly.py              Isolation Forest + ECOD + OCSVM ensemble
layer2_classifier.py           DeBERTa-v3 ONNX INT8 intent classifier
layer2_multilingual.py         XLM-RoBERTa multilingual fallback
layer3_enhanced.py             Schema validator + exfiltration regex + cross-encoder
layer3_semantic.py             Cross-encoder consistency scoring (base module)
canary_manager.py              Cryptographic canary token injection and monitoring
keyword_detector.py            Keyword blocklist baseline
config.py                      Thresholds, paths, and all pipeline settings

# Training & evaluation
train_meta_aggregator.py       Meta-aggregator training (calibrated LogReg)
fine_tune_l2.py                DeBERTa fine-tuning script
generate_l3_pairs.py           Training pair generation for L3 cross-encoder
eval_suite.py                  Full evaluation suite (standard / benign / evasion)
eval_final_holdout.py          Sealed holdout evaluation with frozen thresholds
eval_stateful.py               Multi-turn / stateful crescendo evaluation
ablation_study.py              Per-layer ablation experiments
compare_baselines.py           Classical baseline comparison
compare_commercial.py          Live API commercial guardrail comparison
run_final_holdout_and_report.py  End-to-end holdout runner + JSON report
run_holdout_ablation_fast.py   Fast ablation over the holdout set

# CI & statistics
bootstrap_ci.py                Bootstrap CIs for ADR/FPR/F1/AUC-ROC (B=10,000)
confidence_intervals.py        Wilson score CIs for ADR/FPR/Precision

# Utilities
data_loader.py                 Dataset download and preprocessing
build_validation_sets.py       Validation / dev set construction
split_helper.py                Deterministic SHA-256 hash-based train/test splitting
split_registry.py              Split membership registry (no-contamination protocol)
freeze_state.py                Threshold freezing (pre-holdout)
quantize_onnx.py               INT8 quantization for DeBERTa ONNX
latency_breakdown.py           Per-layer latency profiling
generate_paper_figures.py      Figure generation (ROC/PR, latency, comparison)
verify_fixes.py                Post-edit verification checks
app.py                         Streamlit interactive demo interface

# Assets
figures/                       Generated paper figures (ROC/PR, latency, comparison)
logs/                          Evaluation results, CI outputs, latency logs
  ablation_results_final_holdout.json          Current headline metrics + ablations
  ablation_results_final_holdout_samples.jsonl Current per-sample audit records
  frozen_thresholds.json       Frozen threshold values (pre-holdout)
  final_holdout_results.json   Historical split-summary artifact
  eval_report_final_holdout_2026-06-16.json  Historical companion artifact
  eval_results.jsonl           Historical per-sample predictions (1,037 entries)
  baseline_comparison.json
  commercial_comparison.json
  bootstrap_ci.json            CIs for ADR/FPR/F1/AUC-ROC
  aucpr_ci.json                CI for AUC-PR [0.9855, 1.0000]
data/                          Datasets (not committed — see data_loader.py)
models/                        Trained weights (not committed — too large for Git)
requirements.txt               Python dependencies
Dockerfile                     Docker container configuration
.env.example                   Environment variable template
```

---

## Ablation Study (*n* = 868)

| Configuration | ADR | FPR | F1 | ROC-AUC | TP / FN |
|:---|:---:|:---:|:---:|:---:|:---:|
| L1 Only | 3.69% | 0.00% | 0.0712 | 0.6631 | 12 / 313 |
| L2 Only | 87.38% | 0.37% | 0.9296 | 0.9440 | 284 / 41 |
| L3 Only | 11.08% | 0.00% | 0.1994 | 0.5523 | 36 / 289 |
| L1 + L2 Union | 87.69% | 0.37% | 0.9314 | 0.9532 | 285 / 40 |
| L1 + L3 Union | 13.85% | 0.00% | 0.2432 | 0.6631 | 45 / 280 |
| L2 + L3 Union | 87.69% | 0.37% | 0.9314 | 0.9440 | 285 / 40 |
| **Full (Pipeline)** | **91.38%** | **0.55%** | **0.9504** | **0.9576** | **297 / 28** |

> Layer 2 dominates: 284 of 325 total attack/evasion samples caught by L2; meta-aggregator contributes 12 unique TPs.
> L1+L3 ROC-AUC equals L1-only (0.6631): L3 changes 33 binary decisions but zero continuous risk scores in this run, so ROC-AUC is unchanged.
> ObfuscationDecoder ablation (*n* = 75 evasions): ADR drops from 93.33% (with decoder) to 57.33% (without decoder), confirming the decoder's critical role.

---

## Classical Baseline Comparison (*n* = 848)

| System | ADR | FPR | F1 | AUC |
|:---|:---:|:---:|:---:|:---:|
| Keyword Blocklist | 4.02% | 0.00% | 0.077 | 0.5259 |
| DeBERTa (L2 only) | 84.77% | 0.00% | 0.918 | 0.9870 |
| **RAG-Shield (Full)** | **98.28%** | **0.00%** | **0.991** | **0.9971** |

> Subset contains 348 attacks + 500 benign; no evasion samples. ADR not directly comparable to sealed holdout.
> Chunk/document scanning plus full orchestration adds +13.51 pp ADR over query-only DeBERTa on this subset.

---

## Evasion Coverage (*n* = 75, 11 families)

Aggregate evasion ADR: **93.33%** (70/75). Per-family results:

| Family | ADR | n |
|:---|:---:|:---:|
| Safety Bypass | 100% | 11 |
| Obfuscated | 100% | 10 |
| Persona Hijack | 100% | 8 |
| Instruction Override | 100% | 9 |
| Refusal Override | 100% | 4 |
| Chain-of-Thought Exfiltration | 100% | 2 |
| Tool Invocation Hijack | 100% | 2 |
| Data Exfiltration | 90.0% | 10 |
| System Prompt Exfiltration | 88.9% | 9 |
| Credential Exfiltration | 85.7% | 7 |
| Malicious Link Injection | 33.3% | 3 |

> The malicious link injection family (lowest ADR) uses URL-embedded payloads that evade the ObfuscationDecoder; targeted regex patterns are planned.
> Evasion set expansion to *n* ≥ 200 is planned for reliable per-family CIs.

---

## Dataset Composition

| Evaluation Set | Purpose | Attacks | Evasions | Benign | Total |
|:---|:---|:---:|:---:|:---:|:---:|
| **Sealed holdout** | Primary results (Tables II–IV) | 250 | 75 | 543 | **868** |
| Classical subset | Baseline comparison | 348 | — | 500 | 848 |
| Live-API subset | Commercial comparison | 313 | 68 | 475 | 856 |

All samples SHA-256 hash-deduplicated against training and development sets.
Splits use deterministic hashing (no random seed dependency) to prevent data leakage.

Attack sources: InjecAgent benchmark, HackAPrompt competition corpus, hand-crafted injection variants.
Benign sources: Multi-domain enterprise QA (HR, finance, IT, medical).

---

## Per-Layer Latency (Final Holdout, *n* = 868)

| Component | Mean (ms) | P95 (ms) |
|:---|:---:|:---:|
| Layer 1 (IForest + ECOD + OCSVM) | 434.11 | 2,351.42 |
| Layer 2 (DeBERTa ONNX, query + docs) | 164.85 | 629.74 |
| **L1 + L2 Wall Clock (parallel)** | **558.31** | **2,631.45** |
| Layer 3 (cross-encoder + canary + regex) | 0.18 | 0.40 |
| Meta-Aggregator (LogReg) | 1.00 | 2.25 |
| **Total Pipeline** | **600.3** | **2,663.8** |

> P95 tail is driven by OCSVM inference on long document chunks (Layer 1 P95 = 2,351 ms).
> Memory footprint: ~2.1 GB (MiniLM 90 MB + DeBERTa 740 MB + XLM-RoBERTa 1.1 GB + Cross-encoder 130 MB).
> INT8 quantization reduces model size 4× with <2% accuracy loss.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# Required only for compare_commercial.py (live API mode)
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-gemini-key-here
GROQ_API_KEY=gsk_your-key-here

# Optional: override default canary token
# RAG_CANARY_TOKEN=your-secret-token
```

---

## Known Limitations and Future Work

| Limitation | Current State | Planned Mitigation |
|:---|:---:|:---|
| Evasion set size | *n* = 75 (11 families) | Expand to *n* ≥ 200 (multilingual obfuscation, homoglyph chaining, semantic paraphrase injection) |
| Cross-lingual coverage | Zero-shot XNLI fallback | Fine-tuned multilingual injection classifier (Arabic, Chinese, Japanese) |
| CPU inference latency | 600 ms mean / 2,664 ms P95 | GPU deployment or ONNX INT8 quantization (→ <200 ms) |
| Canary tokens | Static per-session strings | HMAC-signed per-session tokens |
| Adversarial robustness | Not evaluated under white-box attacks | Adaptive white-box adversaries + adversarial fine-tuning of DeBERTa classifier |
| Commercial subset FPR | 3.20% (frozen thresholds) | Recalibrate τ_block to 0.45 (→ ≤1% FPR, ~2 pp ADR cost) |

---

## Citation

```bibtex
@inproceedings{algendy2026ragshield,
  title     = {RAG-Shield: A Three-Layer Ensemble Defense Against
               Prompt Injection in RAG Pipelines},
  author    = {Algendy, Yasmeen and Algendy, Yomna},
  booktitle = {Proceedings of the International Telecommunications
               Conference (ITC-Egypt 2026)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## Acknowledgments

HuggingFace Transformers · PyOD · scikit-learn · Sentence-Transformers ·
InjecAgent · HackAPrompt · ITC-Egypt 2026 reviewers

---

<div align="center">
<sub>
Queen's University · School of Computing · Kingston, ON, Canada<br>
Submitted · ITC-Egypt 2026 · IEEE
</sub>
</div>
