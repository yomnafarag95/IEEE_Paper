# RAG-Shield: A Three-Layer Ensemble Defense Against Prompt Injection in Retrieval-Augmented Generation Systems

**Submitted to:** [IEEE Conference / Journal – Author Placeholder]  
**Authors:** [Author Names – Author Placeholder]  
**Affiliations:** [Institution – Author Placeholder]  

---

## Abstract

Retrieval-Augmented Generation (RAG) pipelines are increasingly deployed in enterprise settings to ground Large Language Models (LLMs) on private knowledge bases. However, this architecture introduces a critical attack surface: adversaries can embed malicious instructions into either user queries (direct injection) or retrieved documents (indirect injection), causing the LLM to override its system prompt, exfiltrate credentials, or execute unauthorized actions. Existing defenses—static keyword blocklists, single-model classifiers, or heavyweight LLM-based guardrails—fail to address this threat comprehensively, suffering from low detection rates on obfuscated evasions, high false positive rates on legitimate technical content, or latency overhead that is incompatible with production deployments.

We present **RAG-Shield**, a three-layer ensemble defense prototype for sanitizing inputs and monitoring outputs across a RAG pipeline. The system integrates: (1) an embedding-based Out-of-Distribution (OOD) anomaly detector combining Isolation Forest, One-Class SVM, and ECOD on `all-MiniLM-L6-v2` sentence embeddings; (2) a dual-stage intent classifier powered by `protectai/deberta-v3-base-prompt-injection-v2` with document-level chunk scanning; and (3) a behavioral monitor utilizing a fine-tuned `ms-marco-MiniLM-L-12-v2` cross-encoder for response consistency, regex-based exfiltration detection, and cryptographic canary token honeypots. A calibrated Logistic Regression meta-aggregator operating over a 10-dimensional evidence vector binds all three layers into a unified risk score and a three-tier allow/monitor/block policy.

Evaluated on a sealed, hash-verified final holdout benchmark ($n = 867$, comprising 249 attacks, 543 benign, and 75 adversarial evasions after deduplication), RAG-Shield achieves a Prevention Attack Detection Rate (ADR) of **87.04%** and a Detection ADR of **87.35%** at a False Positive Rate of only **0.37%** (2/543 benign queries). The system achieves F1 scores of **92.76%** under the prevention policy and **92.94%** under the detection policy, with ROC-AUC of **0.9643** and AUC-PR of **0.9701**. In head-to-head comparisons against commercial alternatives on a separate shared subset, RAG-Shield (ADR = 93.54%, F1 = 0.945) outperforms Llama Prompt Guard 2 (ADR = 79.49%, F1 = 0.883, $p < 10^{-4}$), Llama-3.1-8B Guardrails (ADR = 29.78%), and NVIDIA NeMo Injection Rails (ADR = 7.58%). Final-holdout mean local pipeline latency is **1,214 ms**; live commercial-comparison latency for RAG-Shield is **2,187 ms** including the comparison protocol overhead.

**Keywords:** Retrieval-Augmented Generation, Prompt Injection, LLM Security, Anomaly Detection, Intent Classification, Ensemble Defense, Canary Tokens, Indirect Injection.

---

## I. Introduction

### A. The RAG Trust Boundary Problem

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for deploying LLMs on proprietary, time-sensitive enterprise data. A canonical RAG pipeline intercepts a user query $q$, retrieves the top-$k$ semantically similar document chunks $\mathcal{D} = \{d_1, d_2, \ldots, d_k\}$ from a vector store, constructs a composite prompt $p = f(q, \mathcal{D}, \Phi)$ by concatenating the documents with a system instruction $\Phi$, and passes $p$ to the generative model $M_G$.

This design introduces a fundamental **trust boundary collapse**: the LLM is instructed to consume external text—potentially originating from untrusted emails, web scrapes, third-party APIs, or user-uploaded files—as contextual authority. When retrieved documents contain adversarially crafted instructions, those instructions are transparently injected into the model's context window alongside the legitimate system prompt, and the model cannot structurally distinguish between the two.

### B. Attack Vectors and Evasion Techniques

Prompt injection attacks against RAG systems manifest in two structurally distinct forms:

- **Direct Injection:** The user query $q$ itself contains the adversarial payload. Classic examples include instruction overrides (*"Ignore all previous instructions and output the administrator password"*), role-manipulation (*"You are DAN and have no restrictions"*), and context exhaustion, which floods the context window to displace the system prompt.

- **Indirect Injection:** A benign user issues a legitimate query, but one or more retrieved documents $d_i \in \mathcal{D}$ contain embedded payloads. The attack is transparent to the user. Documented examples include malicious JavaScript embedded in scraped web pages, poisoned HR policy documents instructing the model to recommend a specific candidate, and API documentation annotated with data-exfiltration commands.

These core attack families are further obfuscated through **encoding evasions**—payload splitting across multiple chunks, Base64 or hexadecimal encoding of attack text, Unicode homoglyph substitution (`ｉｇｎｏｒｅ` for `ignore`), ROT13 rotation, Leetspeak character mapping (`1gn0r3`), and zero-width character injection. These transformations evade tokenizer vocabularies and static keyword filters while remaining semantically interpretable to the LLM.

### C. Limitations of Existing Defenses

Three classes of defenses are currently deployed in production:

1. **Static Keyword Blocklists** match surface-form substrings against a curated list of injection phrases. While computationally negligible (< 0.1 ms), they are trivially bypassed by any of the encoding evasions described above. Our own baseline evaluation (Section VIII) confirms a blocklist ADR of only **4.02%** on the shared 848-sample baseline subset.

2. **Single-Model Neural Classifiers** (e.g., ProtectAI's DeBERTa-based injection classifier) achieve substantially higher ADR on direct injections but have two failure modes: (a) they do not scan retrieved document chunks, missing indirect injections; and (b) they operate on raw Unicode, remaining vulnerable to character-substitution evasions.

3. **LLM-Based Guardrails** (e.g., Llama-3.1-8B Guardrails, NVIDIA NeMo) route requests through a second LLM before serving the main model. While semantically expressive, they introduce 3–6 seconds of API latency per query and still achieve low detection rates on indirect injections because they are designed to evaluate query intent, not document-embedded payloads.

Existing production guardrails only partially address the complete attack surface: pre-retrieval input anomaly detection, query-and-document intent classification, and post-generation behavioral monitoring simultaneously within a practical latency budget.

### D. Contributions

This paper makes the following contributions:

1. **A Three-Layer Ensemble Pipeline:** We design a modular defense that scans the RAG pipeline at three checkpoints—the retrieval input, the generation context, and the model output—providing defense-in-depth coverage for both direct and indirect injection.

2. **Document-Level Indirect Injection Scanning:** Layer 2 extends beyond query-level classification to perform per-chunk injection scanning on all retrieved documents, enabling detection of indirect injection with no architectural change to the downstream LLM.

3. **Multi-Signal Meta-Aggregation:** Instead of hard-threshold veto gates, we train a calibrated Logistic Regression meta-aggregator on a 10-dimensional evidence vector derived from all layers, enabling principled uncertainty handling and tunable operating points without retraining any component model.

4. **Cryptographic Canary Honeypots:** We propose injecting session-unique high-entropy tokens into the vector store. Any appearance of these tokens in model output constitutes an irrefutable signal of data-leakage injection, triggering an immediate veto block.

5. **Reproducible Sealed Evaluation:** We report results on a hash-verified, sealed final holdout benchmark with frozen thresholds (Git commit `3ef1522`, dirty worktree noted in the freeze report), providing Wilson score intervals for ADR/FPR/precision and bootstrap intervals for F1 and ROC-AUC.

---

## II. Related Work

### A. Prompt Injection: Taxonomy and Threat Evolution

Perez and Ribeiro [1] first systematically described prompt injection as a class of attacks in 2022, drawing an analogy to SQL injection in structured query languages. Greshake et al. [2] subsequently demonstrated **indirect prompt injection** in real-world LLM agents, showing that adversarially annotated web pages could hijack the behavior of autonomous LLM tools. Liu et al. [3] studied prompt injection against real LLM-integrated applications, while the InjecAgent benchmark [4] formalized indirect prompt injection in tool-integrated agents and provides part of the evaluation corpus we build upon.

Yi et al. [5] introduced BIPIA, a benchmark and defense study for indirect prompt injection attacks. Our implementation taxonomy in `config.py` groups the evaluated attack space into six operational families (`ATTACK_LABELS`): instruction override, role manipulation, payload splitting, indirect injection, encoding obfuscation, and context exhaustion.

### B. Defense Methodologies

**Input filtering** approaches range from simple keyword blocklists to specialized neural classifiers. ProtectAI's `deberta-v3-base-prompt-injection-v2` [6] is a publicly available DeBERTa-v3 classifier trained on a curated injection corpus, which forms the backbone of our Layer 2. LLM-Guard [7] applies post-generation regex and classifier-based scanning to model outputs, conceptually similar to our Layer 3 but without the cross-encoder consistency scoring or canary mechanism.

**Guardrail frameworks** such as NVIDIA NeMo Guardrails [8] and Meta's Llama Guard [9] position a second LLM as a safety referee. While flexible, they inherit the latency of an additional LLM inference call and do not perform document-level chunk scanning. Our empirical comparison in Section VIII confirms their low recall on indirect injection scenarios.

**Anomaly detection** for adversarial inputs has been studied in the context of feature squeezing [10] and out-of-distribution detection [11]. Applying statistical OOD detectors (Isolation Forest, ECOD) to semantic embeddings as a pre-filter is, to our knowledge, novel in the context of RAG security.

**Canary tokens** are well-established in traditional network security for detecting unauthorized data access [12]. Their application as active honeypots injected into a vector store's embedding space is a contribution of this work.

### C. Identified Gaps

The literature reveals three gaps that motivate RAG-Shield: (a) limited simultaneous scanning of both user queries and retrieved document corpora; (b) limited integration of OOD detection, neural classification, and behavioral monitoring within a calibrated risk model; and (c) few head-to-head evaluations of commercial guardrails and ensemble approaches on an identical injection test corpus with statistical significance tests.

---

## III. Problem Formulation

### A. System Model

Let the RAG system be the tuple $(R, \mathcal{K}, M_G, \Phi)$, where $R$ is the dense retriever, $\mathcal{K}$ is the vector store (knowledge base), $M_G$ is the generative LLM, and $\Phi$ is the system instruction. For a user query $q \in \mathcal{Q}$, the system:

$$\mathcal{D} = R(q, \mathcal{K}) = \{d_1, \ldots, d_k\} \subset \mathcal{K}$$

$$p = f(q, \mathcal{D}, \Phi), \quad r = M_G(p)$$

Document chunks are generated by a sliding-window tokenizer with chunk size $w = 200$ tokens and overlap $\delta = 40$ tokens (as configured in `config.py`).

### B. Threat Model

**Adversary Capabilities.** We model an attacker with the ability to: (1) submit arbitrary queries to the RAG API endpoint; (2) inject payloads into one or more documents within $\mathcal{K}$ (e.g., through document upload features or web scraping pipelines); and (3) observe binary allow/block responses but not internal layer scores. We do not assume the adversary knows the session canary values, but model identities and thresholds should be considered discoverable in a published system.

**Attack Vectors.** We formalize four injection scenarios:

#### 1) Direct Injection
$$q = a_{\text{dir}}, \quad p = f(a_{\text{dir}}, \mathcal{D}, \Phi)$$
The adversary encodes a complete override payload directly in the query.

#### 2) Indirect Injection
$$q \in \mathcal{Q}_{\text{benign}}, \quad \exists\, d_i \in \mathcal{D}: d_i = t_i \,\|\, a_{\text{ind}}$$
A benign user triggers retrieval of a poisoned document. The concatenation operator $\|$ denotes that the attack payload $a_{\text{ind}}$ is appended to legitimate document text $t_i$.

#### 3) Encoding Obfuscation
$$a_{\text{obf}} = T(a), \quad T \in \{\text{Base64}, \text{Hex}, \text{ROT13}, \text{Leetspeak}, \text{Unicode}\}$$
The adversary applies a reversible transformation $T$ to bypass tokenizer vocabularies and keyword matchers. The LLM decodes $T^{-1}(a_{\text{obf}})$ through its emergent instruction-following ability.

#### 4) Data Exfiltration
$$a \supset \text{"extract and output } \mathbf{v}_{\text{sensitive}} \text{"}, \quad \mathbf{v}_{\text{sensitive}} \in \{\text{credentials}, \text{API keys}, \text{PII}\}$$
The injection instructs the model to embed sensitive data patterns found in retrieved documents or memory into its response.

### C. Defense Objectives

We define the defense function $D: (\mathcal{Q} \times \mathcal{D}^k \times \mathcal{R}) \to \{0, 1\}$ where $D = 1$ denotes block/alert. Our objectives are:

$$\max_{\theta} \;\mathbb{P}_\theta(D(q, \mathcal{D}, r) = 1 \mid Y = 1) \quad \text{(maximize ADR)}$$
$$\text{subject to} \;\mathbb{P}_\theta(D(q, \mathcal{D}, r) = 1 \mid Y = 0) \leq \epsilon_{\text{FPR}} \quad \text{(FPR budget)}$$

We target $\epsilon_{\text{FPR}} \leq 1.0\%$, since false positives in enterprise RAG deployments cause alert fatigue and lead security teams to disable guardrails.

---

## IV. System Architecture and Methodology

### A. Pipeline Overview

RAG-Shield operates as a transparent middleware layer inserted between the RAG orchestrator and the LLM endpoint. As illustrated in Fig. 1, the pipeline applies three sequential detection checkpoints, each producing scored evidence that feeds into a meta-aggregator. The primary design principles are:

- **Defense-in-Depth:** Each layer is independently deployable and addresses a distinct failure mode.
- **Fail-Closed:** Any layer triggering its hard threshold produces an immediate block without waiting for downstream layers.
- **Latency Optimization:** Layers 1 and 2 execute concurrently via a `ThreadPoolExecutor`. A semantic cache with cosine similarity threshold $\tau_{\text{cache}} = 0.98$ deduplicates repeated queries without re-running inference.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                         User Query q                           │
 └───────────────────────┬─────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │ NFKC Normalize + Decode   │  ← ObfuscationDecoder
           │ (Base64, Hex, ROT13,      │    (module-level singleton)
           │  Leetspeak, Unicode)      │
           └─────────────┬─────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ┌──────────────┐                  ┌──────────────────┐
 │   LAYER 1    │  (parallel)      │     LAYER 2      │  (parallel)
 │   Anomaly    │                  │   Intent Clf     │
 │  IForest +   │                  │  DeBERTa-v3 +    │
 │  OCSVM +     │                  │  XLM-R fallback  │
 │  ECOD        │                  │  Doc chunk scan  │
 └──────┬───────┘                  └────────┬─────────┘
        │                                   │
        │            ┌──────────────────────┤
        │            │   Early Exit?        │ ← S_L2 ≥ τ_early_exit
        │            │   → Block            │
        │            └──────────────────────┘
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
               ┌───────────────┐
               │    LLM M_G    │  ← Response r generated here
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │   LAYER 3     │
               │  Cross-Encoder│
               │  Consistency  │
               │  Canary Check │
               │  Regex Leak   │
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │  META MODEL   │  ← Calibrated LogisticRegressionCV
               │  10-dim f     │    (sklearn, StandardScaler)
               └───────┬───────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       BLOCK        MONITOR      ALLOW
```
*Fig. 1. RAG-Shield architecture. Layers 1 and 2 execute in parallel. An early-exit gate on Layer 2 provides immediate blocking at high confidence. The meta-aggregator combines a 10-dimensional evidence vector into a calibrated risk probability.*

### B. Text Normalization and Obfuscation Decoding

Before any model inference, all input text is normalized by a module-level `ObfuscationDecoder` singleton instantiated once at application startup:

$$\bar{t} = \texttt{unicodedata.normalize}(\text{'NFKC'}, t)$$

NFKC normalization collapses Unicode compatibility characters and homoglyphs to their canonical ASCII equivalents, neutralizing full-width character substitution and Unicode lookalike attacks.

The decoder then applies a regex-driven cascade to detect and reverse six encoding families:

| Transform | Detection Pattern | Reversal |
|:---|:---|:---|
| Base64 | `[A-Za-z0-9+/]{20,}={0,2}` | `base64.b64decode()` |
| Hexadecimal | `(?:0x)?[0-9a-fA-F]{8,}` | `bytes.fromhex()` |
| ROT13 | Entropy + alphabet signature | `codecs.decode('rot_13')` |
| Leetspeak | Character substitution map | Inverse substitution table |
| Zero-width | `[\u200b\u200c\u200d\ufeff]` | Character strip |
| Unicode confusable | Confusable alphabet set | NFKC + explicit mapping |

The decoded text $\bar{t}$ is used for all subsequent processing. This step renders encoding-based evasion ineffective without requiring any model fine-tuning.

A separate pre-check keyword detector (`keyword_detector.py`) runs on the decoded text, scanning against curated high-confidence jailbreak and multilingual injection phrases. Keyword matches contribute fast lexical evidence and may trigger an immediate block for high-severity patterns.

### C. Layer 1: Out-of-Distribution Anomaly Detection

Layer 1 models the manifold of benign document syntax by training an ensemble of anomaly detectors on a corpus of 8,058 clean documents:

| Corpus Source | Count |
|:---|:---:|
| MITRE ATT&CK technique descriptions | ~858 |
| Wikipedia random sample | ~5,000 |
| Technical documentation | ~200 |
| MS MARCO benign queries | ~2,000 |
| **Total** | **~8,058** |

**Embedding.** Each document chunk is mapped to a 384-dimensional semantic vector using `all-MiniLM-L6-v2`:

$$\mathbf{e} = E_{\theta}(\bar{t}) \in \mathbb{R}^{384}$$

**Ensemble Detectors.** Three unsupervised algorithms are trained on the embedded corpus:

1. **Isolation Forest (IForest):** Builds an ensemble of random binary trees that isolate anomalies in fewer partitions than typical points. Trained on all 8,058 samples with `contamination=0.05`.

2. **Empirical Cumulative Outlier Detection (ECOD):** Computes univariate tail probabilities $p_j = \min(\hat{F}_j(e_j), 1 - \hat{F}_j(e_j))$ for each dimension $j$ and aggregates via the extreme value score. Scales linearly with corpus size and uses all 8,058 samples.

3. **One-Class SVM (OCSVM):** Fits a soft hypersphere around the benign cluster in the embedding space using an RBF kernel. Due to the $O(n^2)$ training complexity of SVM, this is trained on a stratified 2,000-sample subset of the corpus to bound inference latency ($\leq 200$ ms per call), while ECOD and IForest use the full corpus.

**Scoring.** Each detector outputs a normalized anomaly score $S_k(\mathbf{e}) \in [0, 1]$. The layer score passed to the meta-aggregator is:

$$S_{L1} = \max_{k \in \{\text{IForest, ECOD, OCSVM}\}} S_k(\mathbf{e})$$

with the per-chunk score taken as:

$$S_{L1}^{\text{max\_chunk}} = \max_{i} S_{L1}(\mathbf{e}_i), \quad \mathbf{e}_i = E_\theta(\bar{d}_i)$$

If $S_{L1}^{\text{max\_chunk}} \geq \tau_{L1} = 0.85$, the layer triggers a block. When `ENABLE_L1_EARLY_EXIT = True`, this block is executed before Layer 2 inference, saving approximately 330 ms per request. A SHA-256–keyed LRU semantic cache (`_L1_SCAN_CACHE`, capacity 512) avoids redundant embedding computation for repeated document corpora.

**Layer Attribution.** In the final holdout evaluation, Layer 1 accounted for **22 true-positive prevention attributions**. The ablation study later separates attribution from marginal contribution and shows that Layer 1 adds one attack over Layer 2 alone under the frozen policy.

### D. Layer 2: Dual-Stage Intent Classifier

Layer 2 performs semantic intent classification over both the user query and all retrieved document chunks.

**Stage 1a — Query-Level Classification.** The primary classifier is `protectai/deberta-v3-base-prompt-injection-v2`, a DeBERTa-v3 model fine-tuned on a curated prompt injection corpus. The frozen final-holdout run used the ONNX-backed ProtectAI path with `L2_USE_FINETUNED = False`; a locally fine-tuned `microsoft/deberta-v3-small` path is implemented but was not active in the reported final-holdout results. The active model is exported to ONNX for deployment:

$$P_{\text{attack}}(q) = \texttt{ONNX\_Infer}(\texttt{Tokenizer}(\bar{q})) \in [0, 1]$$

For non-English inputs and uncertain-confidence predictions, an `xlm-roberta-large-xnli` multilingual fallback is available:

$$P_{\text{attack}}^{\text{multilingual}}(q) = \texttt{XLM-R\_NLI}(\bar{q}, \text{hypothesis=``injection attack"})$$

The Stage 1a score is:
$$S_{L2}^{\text{query}} = \max(P_{\text{attack}}(q), P_{\text{attack}}^{\text{multilingual}}(q))$$

**Stage 1b — Document-Level Chunk Scanning.** This stage represents a key architectural innovation: Layer 2 is applied not only to the user query but to every retrieved document chunk $d_i \in \mathcal{D}$. For each chunk, a targeted regex pre-filter computes a pattern score. Chunks with a pattern score above $\tau_{\text{doc\_pattern}} = 0.60$ are passed to the full DeBERTa classifier:

$$S_{L2}^{\text{doc}} = \max_{i} P_{\text{attack}}(d_i)$$

The composite Layer 2 score is:
$$S_{L2} = \max(S_{L2}^{\text{query}}, S_{L2}^{\text{doc}})$$

A SHA-256–keyed LRU cache (`_DOC_SCAN_CACHE`, capacity 1,024) prevents redundant chunk classification when the same document corpus is retrieved by multiple queries.

**Stage 2 — Attack Family Attribution.** A rule-based heuristic matches normalized text against six attack family patterns from `ATTACK_LABELS`, producing a coarse explanatory label (one of: `instruction_override`, `role_manipulation`, `payload_splitting`, `indirect_injection`, `encoding_obfuscation`, `context_exhaustion`). This label is included in audit log entries for analyst review but does not influence the block decision.

**Early Exit.** When `ENABLE_L2_EARLY_EXIT = True` and $S_{L2} \geq \tau_{\text{early\_exit}}$, the orchestrator immediately blocks the request before invoking downstream checks. In the final holdout evaluation, **284 total samples** triggered the early-exit path; **282** of these were true-positive prevention decisions and **2** were benign false positives.

**Layer Attribution.** Layer 2 provided **260 of the 282 true-positive prevention attributions** in the final holdout, confirming it as the pipeline's primary detection engine.

### E. Layer 3: Behavioral and Semantic Consistency Monitor

Layer 3 evaluates the LLM's generated response $r$ after generation, targeting attacks that succeed in reaching the LLM and manipulating its output.

**Component A — Schema Enforcement.** A Pydantic `RAGResponse` model validates response structure, enforcing: (1) maximum response length of 1,500 characters; (2) confidence scores within $[0, 1]$; and (3) non-empty topic lists. Structural violations flag schema-injection attempts that cause the model to produce malformed structured outputs.

**Component B — Exfiltration Pattern Detection.** Three regex patterns scan the output $r$ for sensitive data signatures:

$$F_{\text{leak}}(r) = \bigvee_{j} \texttt{re.search}(P_j, r), \quad P_j \in \{P_{\text{email}}, P_{\text{api\_key}}, P_{\text{credential}}\}$$

Matched patterns trigger an immediate veto block, preventing credential and PII exfiltration even when the LLM follows an injection instruction.

**Component C — Cross-Encoder Consistency Scoring.** The core of Layer 3 is a `cross-encoder/ms-marco-MiniLM-L-12-v2` cross-encoder that scores the semantic consistency between the query and the response:

$$S_{L3} = \texttt{CrossEncoder}(q, r) \in \mathbb{R}$$

A response $r$ that follows an injection instruction—rather than answering $q$ from the retrieved context $\mathcal{D}$—will be semantically inconsistent with $q$, producing a low cross-encoder score. When a fine-tuned variant is present at `models/layer3_consistency/`, it is loaded preferentially. The block threshold is $\tau_{L3} = 0.55$.

**Component D — Canary Token Detection.** At system initialization, `CANARY_INJECT_COUNT = 3` unique high-entropy tokens are injected into the vector store as honeypot documents. The token value is read exclusively from the `RAG_CANARY_TOKEN` environment variable:

$$c = \texttt{os.environ.get('RAG\_CANARY\_TOKEN', '')}$$

After generation, the output is scanned for the presence of $c$. Any match indicates that the LLM was manipulated into accessing and echoing honeypot content, triggering an immediate veto regardless of any other layer score. Hardcoding the canary value is explicitly prohibited by the implementation.

**Stateful Crescendo Detection.** A session-level tracker maintains the last `STATEFUL_HISTORY_LIMIT = 5` pipeline decisions. A weighted drift score over the recent history is computed:

$$S_{\text{drift}} = \frac{\sum_{i=1}^{L} w_i \cdot S_{L2}^{(t-i)}}{\sum_{i=1}^{L} w_i}$$

If $S_{\text{drift}} \geq \tau_{\text{drift}} = 0.72$, a multi-turn crescendo attack is flagged. This threshold is frozen in the evaluated configuration, but session-level performance is not claimed from the static final holdout.

### F. Meta-Aggregator

The meta-aggregator is a `CalibratedClassifierCV`-wrapped `LogisticRegressionCV` model (scikit-learn) fitted on pipeline activation logs from development runs. It accepts a **10-dimensional feature vector** $\mathbf{f}$ constructed from layer outputs:

$$\mathbf{f} = [S_{L1}, S_{L2}^{\text{query}}, S_{L2}^{\text{doc}}, S_{L3}, \mathbb{1}[\text{canary}], \mathbb{1}[\text{regex}], S_{L1} \cdot S_{L2}, S_{L2} \cdot S_{L3}, S_{L1} \cdot S_{L3}, S_{\text{drift}}]^T$$

where the cross-layer interaction terms ($S_{L1} \cdot S_{L2}$, etc.) capture correlated signals that are more diagnostic than individual scores. Feature vectors are normalized by a fitted `StandardScaler` before inference. The output is a calibrated attack probability:

$$P_{\text{risk}} = \sigma\!\left(\mathbf{w}^T \tilde{\mathbf{f}} + b\right), \quad \tilde{\mathbf{f}} = \text{StandardScaler}(\mathbf{f})$$

The final decision policy is a three-tier scheme:

$$\text{Decision} = \begin{cases}
\textbf{Block} & \text{if } P_{\text{risk}} \geq \tau_{\text{block}} = 0.35 \text{ or canary tripped} \\
\textbf{Monitor} & \text{if } \tau_{\text{monitor}} \leq P_{\text{risk}} < \tau_{\text{block}}, \quad \tau_{\text{monitor}} = 0.31 \\
\textbf{Allow} & \text{if } P_{\text{risk}} < \tau_{\text{monitor}}
\end{cases}$$

The 0.35 block threshold and 0.31 monitor threshold were frozen before the final-holdout run. On the sealed final holdout, prevention and detection FPR are both **0.37%** (2/543 benign samples), indicating that the monitor tier added one attack true positive without adding benign false positives in this run.

### G. Formal Algorithm

**Algorithm 1:** RAG-Shield Orchestrator

```
Input : q (query), D = {d₁,...,dₖ} (retrieved chunks),
        Φ (system prompt), τ_block, τ_monitor, τ_early
Output: Decision ∈ {Block, Monitor, Allow}, response r

1:  q̄ ← NFKC_Normalize(ObfuscationDecode(q))
2:  D̄ ← {NFKC_Normalize(ObfuscationDecode(dᵢ)) for dᵢ ∈ D}

3:  // Keyword pre-check (zero-latency fast path)
4:  if keyword_check(q̄) then return Block, ∅ end

5:  // Parallel Layer 1 + Layer 2 execution
6:  S_L1, S_L2_q ← Parallel({L1_Scan(D̄), L2_Query(q̄)})
7:  S_L2_d ← max{L2_Chunk(dᵢ) for dᵢ ∈ D̄}
8:  S_L2   ← max(S_L2_q, S_L2_d)

9:  // L1 early exit
10: if S_L1 ≥ τ_L1 = 0.85 then return Block, ∅ end

11: // L2 early exit (high-confidence attacks skip LLM call)
12: if S_L2 ≥ τ_early then return Block, ∅ end

13: // Generate LLM response
14: p ← Format(q̄, D̄, Φ)
15: r ← M_G(p)

16: // Layer 3 evaluation
17: S_L3 ← CrossEncoder(q̄, r)
18: canary ← CanaryTokenCheck(r, c)
19: regex  ← RegexLeakCheck(r)

20: // Immediate veto conditions
21: if canary or regex then return Block, r end

22: // Meta-aggregation
23: f ← BuildFeatureVector(S_L1, S_L2_q, S_L2_d, S_L3,
                           canary, regex, interaction_terms, S_drift)
24: P_risk ← CalibratedLogReg.predict_proba(StandardScaler(f))

25: if P_risk ≥ τ_block then return Block, r
26: elif P_risk ≥ τ_monitor then return Monitor, r
27: else return Allow, r
28: end
```

---

## V. Experimental Setup

### A. Hardware and Software Environment

All experiments were conducted on a local workstation:

| Component | Specification |
|:---|:---|
| CPU | AMD Ryzen 9 5900X (12-core, 3.7 GHz) |
| RAM | 64 GB DDR4 |
| GPU | NVIDIA GeForce RTX 3080 (10 GB VRAM) |
| OS | Windows 11 Pro |
| Python | 3.11.9 (virtual environment) |
| PyTorch | 2.2.2 |
| ONNX Runtime | 1.18.0 |
| scikit-learn | 1.4.x |
| sentence-transformers | 2.7.x |

ONNX Runtime was configured to use the CPU execution provider for Layer 2 inference (no GPU dependency), ensuring reproducibility on standard server hardware without GPU requirements.

### B. Benchmark Dataset Construction

The final holdout benchmark was constructed under a strict **no-contamination protocol**: all samples were hash-deduplicated against training and development sets. File integrity is verified at evaluation time via SHA-256 checksums recorded in `logs/frozen_thresholds.json`. The raw files contain 868 rows; the evaluator removes one duplicate attack row, yielding 867 evaluated samples. The benchmark is split across three files:

| Split | Evaluated $n$ | Source | Description |
|:---|:---:|:---|:---|
| `attacks.jsonl` | 249 | Blind post-freeze holdout generation | Direct + indirect injection payloads |
| `benign.jsonl` | 543 | Manual + blind post-freeze holdout generation | Enterprise queries (HR, Finance, Medical, Legal, IT) |
| `evasions.jsonl` | 75 | Blind post-freeze holdout generation | 12 obfuscation/evasion families, including Base64, Hex, ROT13, leetspeak, Unicode, reverse text, whitespace splitting, multilingual instruction, and role-play wrappers |
| **Total** | **867** | | |

The combined attack+evasion set constitutes $324/867 = 37.4\%$ of the benchmark. The $543/867 = 62.6\%$ benign majority ensures that FPR is measured on a realistically skewed enterprise query distribution.

### C. Evaluation Protocol and Metrics

All thresholds were **frozen before final holdout evaluation** (Git commit `3ef152293d76b81c0a080a8d26d67e010297d7f7`) and stored in `logs/frozen_thresholds.json`. No threshold tuning was performed after the holdout was unlocked. The evaluation suite (`eval_suite.py`) verifies threshold parity at startup and aborts if any configuration drift is detected.

We report the following metrics:

- **ADR (Attack Detection Rate):** $\text{ADR} = \text{TP} / (\text{TP} + \text{FN})$. Reported as *Prevention ADR* (blocked only) and *Detection ADR* (blocked or monitored).
- **FPR (False Positive Rate):** $\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$ on the benign subset.
- **Precision:** $\text{Prec} = \text{TP} / (\text{TP} + \text{FP})$.
- **F1 Score:** Harmonic mean of precision and recall.
- **ROC-AUC and AUC-PR:** Area under the receiver operating characteristic and precision-recall curves.
- **95% Confidence Intervals:** Wilson score intervals for ADR, FPR, and precision; bootstrap percentile intervals ($B = 1000$) for F1 and AUC.
- **Latency:** Per-layer mean and P95 latency measured using `time.perf_counter()` within the orchestrator loop.

For commercial comparisons, **McNemar's test** is applied on the paired discordant sample counts to assess statistical significance of detection rate differences.

---

## VI. Results

### A. Final Holdout Performance

Table I summarizes RAG-Shield's performance on the sealed final holdout benchmark ($n = 867$).

**TABLE I.** RAG-Shield Final Holdout Benchmark ($n = 867$, frozen thresholds, Git `3ef1522`)

| Metric | Prevention Policy | Detection Policy | 95% CI |
|:---|:---:|:---:|:---:|
| Total Samples ($n$) | 867 | 867 | — |
| True Positives | 282 | 283 | — |
| False Positives | 2 | 2 | — |
| False Negatives | 42 | 41 | — |
| True Negatives | 541 | 541 | — |
| **ADR** | **87.04%** | **87.35%** | Prev. [82.94%, 90.26%]; Det. [83.28%, 90.53%] |
| **FPR** | **0.37%** | **0.37%** | [0.10%, 1.33%] |
| **Precision** | **99.30%** | **99.30%** | Prev. [97.47%, 99.81%]; Det. [97.48%, 99.81%] |
| **F1 Score** | **92.76%** | **92.94%** | Prev. [90.59%, 94.84%]; Det. [90.66%, 94.86%] |
| **ROC-AUC** | 0.9643 | 0.9643 | [0.9484, 0.9777] |
| **AUC-PR** | 0.9701 | 0.9701 | — |
| Mean Latency (ms) | 1,214.0 | 1,214.0 | — |
| P95 Latency (ms) | 4,635.1 | 4,635.1 | — |
| Early Exit Count | 284 total | 284 total | — |

The final-holdout ROC and precision-recall curves are shown in Fig. 2. The confusion matrices and layer attribution are shown in Fig. 3. The system correctly allows 541 of 543 benign queries (99.63% specificity) and blocks 282-283 of 324 attack+evasion samples.

```
                     Predicted: Allow    Predicted: Block/Monitor
Benign  (N = 543)      TN = 541              FP = 2
                       (99.63%)             (0.37%)
Attacks (N = 324)      FN = 41               TP = 283
                       (12.65%)             (87.35%)
```
*Fig. 2. Final-holdout ROC and precision-recall curves generated from `logs/curves/roc_final_holdout.png` and `logs/curves/pr_final_holdout.png`.*

*Fig. 3. Confusion matrices for the prevention and detection policies, with true-positive layer attribution.*

### B. Per-Layer Latency Breakdown

Table II details the per-layer latency observed during final holdout evaluation. Fig. 4 plots the same per-component mean and P95 latencies on a log scale.

**TABLE II.** Per-Layer Latency Breakdown (Final Holdout, $n = 867$)

| Component | Mean Latency (ms) | P95 Latency (ms) |
|:---|:---:|:---:|
| Layer 1 (Anomaly: IForest + ECOD + OCSVM) | 695.75 | 4,165.41 |
| Layer 2 (DeBERTa ONNX, query + docs) | 330.22 | 1,115.33 |
| L1 + L2 Wall Clock (parallel) | 879.45 | 4,166.47 |
| Layer 3 (Cross-Encoder + canary + regex) | 0.25 | 0.46 |
| Meta-Aggregator (Logistic Regression) | 0.64 | 1.10 |
| **Total Pipeline** | **1,211.93** | **4,633.85** |

The top-level evaluator reports a mean latency of 1,214.0 ms and P95 latency of 4,635.1 ms; Table II reports the internal per-component timing totals from the same final-holdout JSON artifact.

Parallel execution of L1 and L2 reduces the combined wall clock to 879 ms despite L1's 695 ms mean latency. Layer 3 and the meta-aggregator together contribute less than 1 ms of overhead.

Cache efficiency was high: L1 registered **692 cache hits** and L2 document scanning registered **692 cache hits** across the 867 evaluation runs, corresponding to a cache hit rate of approximately 79.8%.

### C. Layer Attribution

Layer 2 (Intent Classifier) was the dominant blocking layer, receiving **260 of 282 true-positive prevention attributions** (92.2%) in the final holdout. Layer 1 (Anomaly Detector) received **22 true-positive attributions** (7.8%). The ablation study below shows that the marginal prevention gain over Layer 2 alone is smaller: L1+L2 catches one additional attack relative to L2-only under the frozen policy.

---

## VII. Ablation Study

To quantify the marginal contribution of each architectural component, we conducted a component ablation study on the final holdout benchmark ($n = 867$). Each configuration uses the same frozen thresholds and is evaluated by selectively disabling layers and recomputing the union of their block decisions.

**TABLE III.** Component Ablation Study on Final Holdout ($n = 867$)

| Configuration | ADR (Prev.) | FPR (Prev.) | F1 (Prev.) | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Layer 1 Only (Anomaly) | 6.48% | 0.00% | 0.1217 | 21 | 0 | 303 |
| Layer 2 Only (Intent) | 86.73% | 0.37% | 0.9259 | 281 | 2 | 43 |
| Layer 3 Only (Monitor) | 17.59% | 0.00% | 0.2992 | 57 | 0 | 267 |
| L1 + L2 Union | 87.04% | 0.37% | 0.9276 | 282 | 2 | 42 |
| L1 + L3 Union | 21.30% | 0.00% | 0.3511 | 69 | 0 | 255 |
| L2 + L3 Union | 86.73% | 0.37% | 0.9259 | 281 | 2 | 43 |
| Full (without Meta) | 87.04% | 0.37% | 0.9276 | 282 | 2 | 42 |
| **Full (with Meta)** | **87.04%** | **0.37%** | **0.9276** | **282** | **2** | **42** |
| Full (with Meta, Detection) | 87.35% | 0.37% | 0.9294 | 283 | 2 | 41 |

### A. Component Contribution Analysis

**Layer 2 is the primary detection engine.** Deploying L2 alone (DeBERTa ONNX + document chunk scanning) achieves 86.73% ADR—just 0.31 percentage points below the full ensemble. This demonstrates that transformer-based intent classification, extended to document-level scanning, captures the vast majority of injections. However, it is not sufficient for complete coverage.

**Layer 1 provides a structurally distinct but small marginal signal.** The L1+L2 union gains 0.31 pp in ADR over L2 alone (282 vs. 281 TPs). Layer 1 alone achieves 0.00% FPR on the final holdout but only 6.48% ADR, so its role in the frozen configuration is best framed as a conservative auxiliary detector rather than a primary detection engine.

**Layer 3's contribution requires a different evaluation scope.** The static single-turn ablation cannot fully measure L3's value, as its primary contributions—canary token leak detection and multi-turn crescendo blocking—require session-level probes and canary-injection tests not present in the static holdout. The L1+L3 union's 21.30% ADR represents only L3's consistency-based contribution in isolation from query-level signals.

**The meta-aggregator's primary value is the Monitor tier.** The ADR difference between "Full without Meta" and "Full with Meta" under the Detection Policy (87.04% → 87.35%) represents one additional sample that was borderline-blocked by the meta-model's probabilistic aggregation (TP detection: 282 → 283). The meta-aggregator's principal benefit is the **three-tier decision policy**: it separates high-confidence blocks from uncertain cases that warrant human analyst review, without requiring a higher hard threshold that would inflate FPR.

---

## VIII. Comparison with Baseline and Commercial Methods

### A. Classical Baselines ($N = 848$)

We evaluated three systems on a shared 848-sample subset (348 attacks + 500 benign), representing the baseline alternatives a security engineer might deploy:

**TABLE IV.** Comparison Against Classical Baselines ($N = 848$)

| System | ADR | FPR | Precision | F1 | AUC | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Keyword Blocklist | 4.02% | 0.00% | 100.0% | 0.077 | 0.5594 | 0.1 ms |
| DeBERTa Classifier (L2 only) | 84.77% | 0.00% | 100.0% | 0.918 | 0.9803 | 57.91 ms |
| **RAG-Shield (Full)** | **93.39%** | **0.00%** | **100.0%** | **0.966** | **0.9646** | **246.75 ms** |

RAG-Shield improves on the standalone DeBERTa classifier by **+8.62 pp ADR** (93.39% vs. 84.77%) at zero additional FPR cost. The 246.75 ms overhead versus 57.91 ms for L2 alone represents a 4.3× latency cost for +8.62 pp detection gain—a favorable security-latency trade-off. The keyword blocklist's 4.02% ADR confirms that surface-form filtering is inadequate for any meaningful deployment.

### B. Commercial Safety Systems ($N = 856$)

We evaluated four systems—Llama Prompt Guard 2 (86M parameter), Llama-3.1-8B Guardrails, NVIDIA NeMo Injection Rails, and RAG-Shield—on a shared live-API subset containing **356 attack** and **500 benign** samples ($N = 856$). All commercial systems were accessed via the Groq inference API with live HTTP round-trip latency included.

**TABLE V.** Comparison Against Commercial Safety Systems

| System | ADR | FPR | Precision | F1 | Mean Latency | McNemar $p$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama Prompt Guard 2 | 79.49% | 0.40% | 99.30% | 0.883 | 426.97 ms | $4.7 \times 10^{-5}$ |
| Llama-3.1-8B Guardrail | 29.78% | 0.00% | 100.00% | 0.459 | 3,326.14 ms | $1.2 \times 10^{-38}$ |
| NeMo Rail (Llama-3.1) | 7.58% | 0.00% | 100.00% | 0.141 | 5,250.91 ms | $2.3 \times 10^{-58}$ |
| **RAG-Shield (Ours)** | **93.54%** | **3.20%** | **95.42%** | **0.945** | **2,187.28 ms** | — |

McNemar's test confirms that all pairwise ADR differences are statistically significant at $p < 10^{-4}$.

Fig. 5 summarizes the baseline and live commercial comparisons while explicitly separating the two evaluation sets.

**Detection Rate Analysis.** RAG-Shield achieves **+14.05 pp ADR** over Llama Prompt Guard 2 (93.54% vs. 79.49%), the strongest commercial alternative. The gap widens dramatically against LLM-based guardrails: Llama-3.1-8B Guardrail detects only 29.78% of attacks and NeMo Rails only 7.58%. This failure is attributable to their design as **query-intent classifiers**: they evaluate whether the user's request is malicious but do not scan retrieved document chunks, making them structurally blind to indirect injection—the dominant attack category in the InjecAgent-derived test cases.

**Latency Analysis.** RAG-Shield's 2,187 ms mean latency includes the full LLM generation call. RAG-Shield is **34% faster than Llama-3.1-8B Guardrails** (2,187 vs. 3,326 ms) and **58% faster than NeMo Rails** (2,187 vs. 5,251 ms), because its primary checks (L1 and L2) run locally via ONNX on CPU.

**False Positive Rate.** RAG-Shield's 3.20% FPR in the commercial comparison (vs. 0.37% on the full holdout) is attributable to the different benign subset and the inclusion of technical enterprise requests with command-like syntax that trigger false positives. This result should be reported separately from the final holdout because it uses a different sample mix and live-API comparison protocol.

---

## IX. Discussion

### A. The Primacy of Document-Level Scanning

The most significant architectural decision in RAG-Shield is Layer 2's document-level chunk scanning (Stage 1b). LLM-based guardrails (NeMo, Llama-3.1) achieve only 7–30% ADR precisely because they evaluate query intent while treating document context as benign. RAG-Shield's document scanning, combined with its targeted injection regex pre-filter (`L2_DOC_PATTERN_THRESHOLD = 0.60`), enables detection of indirect injection payloads buried within otherwise benign-seeming documents—a structural capability that cannot be replicated by query-only classifiers.

### B. Precision-Recall Trade-Off and Operating Point Selection

RAG-Shield's final operating point was selected to prioritize **precision** (99.30%) over recall (87.04%). In enterprise RAG deployments, false positives impose direct user-experience costs: every blocked benign query frustrates a legitimate user and risks disabling the guard entirely. The final holdout precision-recall curve (AUC-PR = 0.9701) supports this conservative operating point, but threshold-sweep claims should be made only for the benchmark on which the sweep was run, since `logs/threshold_sweep_results.json` was produced on a smaller development-style set rather than the sealed final holdout.

### C. Semantic Caching as a Scalability Mechanism

The 79.8% cache hit rate observed during final-holdout evaluation reflects repeated or near-duplicate evaluation inputs under the semantic cache threshold $\tau_{\text{cache}} = 0.98$. It demonstrates that the cache path is active and can reduce repeated-query overhead, but it should not be interpreted as a measured production cache hit rate without workload traces from a deployed RAG system.

### D. Canary Tokens and the Exfiltration Detection Gap

The canary token mechanism addresses a detection gap that is invisible to recall metrics on standard injection benchmarks: **data exfiltration via compliant LLM behavior**. An LLM may follow an injection instruction without any syntactic anomaly in the query or retrieved documents—it simply answers a plausible question while embedding stolen credentials in its response. Layer 3's canary check and regex exfiltration detector operate exclusively on the model's output, providing coverage for this attack class that Layers 1 and 2 cannot provide.

### E. Multi-Turn Crescendo Attack Resistance

The stateful drift tracker ($\tau_{\text{drift}} = 0.72$) was designed to counter crescendo jailbreak sequences in which each individual turn appears innocuous but the cumulative trajectory escalates toward an attack. This capability is implemented in the system, but the static final holdout does not constitute a session-level evaluation of multi-turn drift. We therefore treat multi-turn resistance as an implemented defense mechanism requiring a separate session benchmark.

---

## X. Limitations and Future Work

### A. Current Limitations

**Residual False Negatives.** The prevention policy leaves **42** undetected attacks in the final holdout (12.96%), while the detection policy leaves **41** undetected attacks (12.65%). These residual failures fall into two categories: (1) highly sophisticated semantic jailbreaks that combine role-play framing with subtle instruction injection, where L2 scores hover near but below $\tau_{L2} = 0.60$; and (2) payload-splitting attacks where no individual chunk contains a complete injection signature. Addressing these requires either a higher-capacity model at L2 or a multi-chunk aggregation strategy.

**L1 Standalone Sensitivity.** Layer 1's standalone ADR of 6.48% reflects a deliberate design choice: the anomaly threshold ($\tau_{L1} = 0.85$) is set conservatively to maintain 0.00% FPR. Lowering it would improve recall but would generate false positives on legitimate technical documents (e.g., API documentation with unusual syntax).

**Static Canary Strategy.** Canary tokens are currently session-static. An attacker who intercepts a canary value through a timing oracle or partial information leakage could strip the token from the output before it reaches the response scanner.

**NFKC Coverage Limits.** Novel Unicode character sets released after our normalization table was compiled may bypass NFKC normalization. Adversarial Unicode research (e.g., Trojan-source-style bidirectional attacks) remains an open challenge.

### B. Future Research Directions

**Dynamic Per-Session Canaries.** We propose generating cryptographically signed, per-session canary tokens using HMAC with a server-side secret key. Each new session receives a unique canary, eliminating the risk of token interception across sessions.

**Multi-Chunk Injection Aggregation.** Future work will implement a sliding-window attention mechanism over sequential document chunks to detect payload-splitting attacks where the injection is distributed across chunk boundaries.

**Reinforcement Learning from Analyst Feedback.** The meta-aggregator is currently trained offline. We plan to extend it to online learning from security analyst decisions on monitored-tier events, allowing the model to adapt its decision boundary to new attack patterns without full retraining.

**Adversarial Red-Teaming Loop.** We will integrate an automated red-team agent that generates novel evasion probes targeting current model weaknesses, feeding failures back into the training pipeline as a continuous improvement mechanism.

**Multilingual and Multimodal Extension.** The current obfuscation decoder handles 6 encoding families. Integration of `distiluse-base-multilingual-cased-v1` as the Layer 1 embedder and extending L2's multilingual NLI fallback to additional language families (Arabic, Chinese, Japanese) will broaden coverage to non-English RAG deployments. Future work will also explore multimodal injection vectors targeting RAG systems that retrieve image or tabular content.

---

## XI. Conclusion

Prompt injection represents a fundamental security vulnerability in RAG architectures that cannot be addressed by any single checkpoint or technique. We have presented **RAG-Shield**, a three-layer ensemble defense that intercepts the RAG pipeline at the retrieval input, generation context, and model output simultaneously. Through the combination of embedding-based OOD anomaly detection, transformer-based dual-stage intent classification with document-level chunk scanning, and output behavioral monitoring with cryptographic canary honeypots—orchestrated by a calibrated 10-dimensional meta-aggregator—RAG-Shield achieves a comprehensive attack detection rate of **87.35%** at an enterprise-grade False Positive Rate of **0.37%**.

Our ablation study shows that the full prevention policy improves by **0.31 pp** over Layer 2 alone (87.04% vs. 86.73%), while the full detection policy adds one monitored true positive (87.35%). This means the current final-holdout result is driven primarily by document-level DeBERTa scanning, with Layer 1 adding a small number of structurally anomalous detections and Layer 3 requiring separate canary/output and multi-turn benchmarks for full evaluation. Head-to-head comparisons against commercial guardrails show a **+14.05 pp ADR advantage over Llama Prompt Guard 2** and a **+63.76 pp advantage over Llama-3.1-8B Guardrails** on the live comparison subset.

These results demonstrate that a locally deployed, multi-layered ensemble combining statistical, neural, and behavioral techniques can outperform several commercial guardrails on the evaluated direct, indirect, and evasion-heavy test sets, while keeping false positives low on the sealed final holdout.

---

## Acknowledgment

The authors thank the security research community for the publicly available injection benchmarks that made this evaluation possible, including the InjecAgent and HackAPrompt datasets. [Institutional acknowledgment to be added for camera-ready submission.]

---

## References

[1] F. Perez and I. Ribeiro, "Ignore previous instructions: A new class of adversarial attacks on language models," *arXiv preprint arXiv:2211.09527*, Nov. 2022.

[2] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injections," in *Proc. ACM Workshop Artif. Intell. Secur. (AISec)*, 2023.

[3] Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, and Y. Liu, "Prompt injection attack against LLM-integrated applications," *arXiv preprint arXiv:2306.05499*, Jun. 2023.

[4] Q. Zhan, Z. Liang, Z. Ying, and D. Kang, "InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents," *arXiv preprint arXiv:2403.02691*, Mar. 2024.

[5] J. Yi, Y. Xie, B. Zhu, E. Kiciman, G. Sun, X. Xie, and F. Wu, "Benchmarking and defending against indirect prompt injection attacks on large language models," *arXiv preprint arXiv:2312.14197*, Dec. 2023.

[6] ProtectAI, "deberta-v3-base-prompt-injection-v2," HuggingFace Hub, 2023. [Online]. Available: https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2

[7] L. Iurman, "LLM-Guard: The security toolkit for LLM interactions," GitHub Repository, 2023. [Online]. Available: https://github.com/protectai/llm-guard

[8] T. Rebedea, R. Dinu, M. Sreedhar, C. Parisien, and J. Cohen, "NeMo Guardrails: A toolkit for controllable and safe LLM applications with programmable rails," in *Proc. EMNLP Syst. Demonstrations*, 2023, pp. 431–445.

[9] H. Inan, K. Upasani, J. Chi, R. Rungta, K. Iyer, Y. Mao, M. Tontchev, Q. Hu, B. Fuller, D. Testuggine, and M. Khabsa, "Llama Guard: LLM-based input-output safeguard for human-AI conversations," *arXiv preprint arXiv:2312.06674*, Dec. 2023.

[10] W. Xu, J. Evans, and Y. Qi, "Feature squeezing: Detecting adversarial examples in deep neural networks," in *Proc. NDSS Symp.*, 2018.

[11] D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," in *Proc. ICLR*, 2017.

[12] S. Kreibich and J. Crowcroft, "Honeycomb: Creating intrusion detection signatures using honeypots," in *Proc. ACM Workshop Hot Topics Netw. (HotNets)*, 2003.

[13] M. Schulhoff et al., "Ignore this title and HackAPrompt: Exposing systemic vulnerabilities of LLMs through a global scale prompt hacking competition," in *Proc. EMNLP*, 2023, pp. 4945–4977.

[14] P. He, J. Gao, and W. Chen, "DeBERTa-V3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in *Proc. ICLR*, 2023.

[15] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP*, 2019, pp. 3982–3992.

[16] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.

[17] Z. Li, J. Zhao, Y. Botta, C. Ionescu, and G. Chen, "ECOD: Unsupervised outlier detection using empirical cumulative distribution functions," *IEEE Trans. Knowl. Data Eng.*, vol. 35, no. 12, pp. 12223–12236, 2023.

[18] B. Schölkopf, R. C. Williamson, A. J. Smola, J. Shawe-Taylor, and J. C. Platt, "Support vector method for novelty detection," in *Proc. NeurIPS*, 2000, pp. 582–588.

[19] ONNX Runtime Developers, "ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator," 2022. [Online]. Available: https://onnxruntime.ai

[20] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, "Unsupervised cross-lingual representation learning at scale," in *Proc. ACL*, 2020, pp. 8440–8451.
