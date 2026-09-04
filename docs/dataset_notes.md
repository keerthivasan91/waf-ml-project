# Dataset Notes

## Datasets in use

| Dataset | Role | Records |
|---|---|---|
| HttpParamsDataset (Morzeux) | L2B primary — SQLi/XSS/LFI/other_attack examples | 31,067 rows |
| CSIC 2010 | L2A normal-traffic training + L2B normal-class supplement | 61,065 rows |

Both are HTTP-request-level datasets (URL, method, headers, body) —
this matters because the entire feature pipeline
(`ml/feature_engineering/extractor.py`, 29 features) is built around
per-request payload structure, not network-level flow statistics.

## Test-only datasets — never mixed into training

| Dataset | Purpose |
|---|---|
| WAF-A-MoLE | Adversarial robustness evaluation |
| Drift sets (obfuscated LFI, etc.) | NB07's adaptive-retraining drift simulation |

These exist specifically to answer "does this generalize to inputs it
wasn't trained on," so mixing them into training would defeat their
purpose. NB07's drift scenario in particular is what the adaptive
retraining pipeline's recall-recovery claim (9.7% → 100% on the
held-out drift scenario after retraining) is measured against — using
that same data for training would make the claim circular.

## Permanently excluded: CICIDS 2017

CICIDS 2017 was considered early on (it's a large, well-known
benchmark) but excluded permanently. It's **NetFlow-level** data —
packet counts, flow duration, byte counts between IP:port pairs — not
HTTP request payloads. There's no principled way to extract this
project's 29 features (URL structure, payload entropy, JSON/GraphQL
markers, etc.) from flow-level records, so rather than force a partial
or synthetic adaptation, it was dropped from the project entirely.
This is worth stating explicitly in the CRC and anywhere datasets are
listed — CICIDS 2017 appearing in an early draft as an in-use dataset
was a documentation error, not a design decision that was later
reversed.

## Splitting methodology

A single **70/15/15 train/validation/test split** is used across both
L2A and L2B — not separate independent splits per layer. This matters
for two reasons:

1. **Consistency** — L2A's escalation threshold is selected on the
   validation split, then the *same* test split is used for the final
   end-to-end evaluation. If L2A and L2B had different splits, the
   headline numbers (FPR=0.17%, TPR=88.64%) would be measuring the
   pipeline against data that wasn't held out consistently for both
   stages.
2. **Family-aware stratification** (`group_stratified_split()`, NB02)
   — near-duplicate payloads (same attack, different encoding/casing)
   are grouped into "families" before splitting, so a family never
   spans train and test simultaneously. Without this, a model could
   score artificially well on test by having seen a near-identical
   payload during training — inflating recall numbers without
   reflecting genuine generalization.

The test set is touched **exactly once** for the final headline
numbers reported in the CRC — all threshold and multiplier tuning
happens against validation only.

## Class imbalance handling

Majority classes are capped at 5,000 rows, with class weights computed
for `CrossEntropyLoss` to correct for whatever imbalance remains after
capping. **SMOTE is deliberately not used** — interpolating between
two HTTP payloads in feature space (or token space) doesn't produce a
syntactically valid HTTP request; the "synthetic" examples SMOTE would
generate for this data aren't representative of anything a real
attacker or user would actually send, so they'd just add training
noise rather than genuine signal.

## Feature pipeline consistency (training ↔ inference)

`ml/feature_engineering/extractor.py`, `tokenizer.py`, and
`normalizer.py` are used identically at training time (in the
notebooks) and inference time (`app/services/feature_extractor.py`
imports them directly — it's not a reimplementation). The one
non-obvious detail: **domain-stripping** — `strip_to_path_query()`
removes the scheme/host from URLs before feature extraction, so
`http://example.com/search?q=x` and `http://otherhost.com/search?q=x`
produce identical features. This was added specifically because the
protected application's actual hostname shouldn't influence whether a
request looks like an attack — training on absolute URLs would have
made the model partly memorize hostnames that appeared in the training
data rather than learning payload structure.

`scaler_l2a.pkl` (a `StandardScaler`) is fit **once**, on the 70%
train split only, and reused for both L2A and L2B's dense-feature
input — never refit at inference time. A missing or mismatched scaler
was a real bug encountered during development: unscaled features
produce L2A reconstruction-error scores in the 6-10 range instead of
the expected ~0.001-0.01, effectively breaking anomaly detection
silently (every request looks anomalous). See
[deployment.md](deployment.md) for the exact symptom and fix.
