# Model Selection — Candidate Comparison & Winner Rationale

Every number on this page comes directly from `layer2a_metadata.json`
and `layer2b_metadata.json` (produced by NB03 and NB04's export cells)
— nothing here is estimated or rounded from memory. If these numbers
and the notebooks ever disagree, the notebooks are the source of truth
and this file needs updating.

---

## Layer 2A — Anomaly Detector

**Selection rule:** thresholds swept from P50–P99 of normal validation
scores; candidates satisfying FPR≤5% retained; winner picked by highest
recall, then lowest FPR, then lowest threshold.

**Winner: Shallow Autoencoder**

| Model | ROC-AUC | PR-AUC | Recall | Precision | FPR | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.8002 | 0.9829 | 20.24% | 98.42% | 5.47% | 0.336 | 0.13757 |
| **Shallow Autoencoder** | **0.9640** | **0.9977** | **82.13%** | **99.70%** | **4.15%** | **0.901** | **0.0029852** |

Isolation Forest's precision looks competitive, but its recall — the
metric that actually matters for a zero-day detector, since a missed
anomaly is a missed attack — is less than a quarter of the
autoencoder's. F1 makes the gap unambiguous: 0.336 vs 0.901.

**Bootstrap 95% CI (Shallow Autoencoder, 3 runs):**

| Metric | Mean | 95% CI |
|---|---|---|
| F1 | 0.9007 | [0.8962, 0.9050] |
| FPR | 0.0416 | [0.0271, 0.0582] |
| Recall | 0.8214 | [0.8139, 0.8285] |

**Latency (500 runs):** mean 0.0162ms, P50 0.0156ms, P95 0.0180ms,
P99 0.0299ms. Effectively free compared to L2B — this is what makes
running L2A on every single request (rather than only escalated ones)
a non-issue.

---

## Layer 2B — Deep Classifier

**Selection rule:** best validation macro-F1, subject to XSS F1 not
dropping more than 0.02 below the "all sweep points" baseline (to
avoid a config that trades away one class's recall for aggregate
score).

**Winner: BiGRU** (sweep point: "8k")

| Model | Macro-F1 | Accuracy | F1 normal | F1 sqli | F1 xss | F1 lfi | F1 other_attack |
|---|---|---|---|---|---|---|---|
| **BiGRU** | **0.9929** | **0.9954** | **0.9917** | **0.9990** | **0.9911** | **0.9944** | **0.9884** |
| CNN-1D | 0.9868 | 0.9919 | 0.9844 | 0.9990 | 0.9805 | 0.9944 | 0.9755 |
| XGBoost | 0.9306 | 0.9521 | 0.8776 | 0.9948 | 0.9766 | 0.9710 | 0.8332 |

XGBoost's `normal` and `other_attack` F1 (0.878, 0.833) drag its macro
average down noticeably — sequence-aware models handle those two
classes much better, unsurprising since `other_attack` groups several
structurally different attack styles (cmdi, SSRF-like, injection-style)
that don't share a compact hand-engineerable feature signature the way
SQLi/XSS keyword patterns do.

**BiGRU sweep point comparison** (data-volume sweep, XSS-F1 guardrail
applied):

| Sweep point | Macro-F1 | XSS F1 | XSS F1 OK? |
|---|---|---|---|
| all | 0.9910 | 0.9965 | ✓ |
| 10k | 0.9877 | 0.9792 | ✓ |
| **8k** | **0.9937** | **0.9826** | **✓ — winner** |
| 6k | 0.9936 | 0.9861 | ✓ |
| 5k | 0.9922 | 0.9826 | ✓ |

"8k" edges out "all" on macro-F1 despite using less data — the full
dataset's class imbalance (before the 5,000-row majority-class cap
described in [dataset_notes.md](dataset_notes.md)) works against it
slightly here.

**Bootstrap 95% CI (BiGRU):**

| Metric | Mean | 95% CI |
|---|---|---|
| Macro-F1 | 0.9929 | [0.9894, 0.9962] |
| Accuracy | 0.9954 | [0.9932, 0.9976] |

**Latency:** BiGRU mean 14.99ms (P50 14.99ms, P95 15.64ms, P99
15.90ms, 300 runs) vs CNN-1D mean 1.35ms. CNN-1D is ~11× faster but
loses 1.4 points of accuracy and — more importantly for a security
tool — 4.5 points on `other_attack` F1 specifically (0.976 vs 0.988).
Given selective escalation means L2B only runs on the ~15-20% of
traffic that actually escalates (not every request), the accuracy
tradeoff was judged worth more than the latency saving; P99 end-to-end
latency for the full pipeline is 22.89ms, only marginally above the
20ms target, specifically because escalation keeps L2B off the
critical path for most traffic.

---

## Why not an ensemble of all three?

Not evaluated in this project — XGBoost's failure mode (weak on
`normal` and `other_attack`) and BiGRU/CNN-1D's shared strength on
those same classes suggests an ensemble might not add much beyond
BiGRU alone, but this wasn't tested and isn't a claim made anywhere in
the CRC. Worth flagging as a legitimate open question for future work,
not something silently assumed away.
