# Architecture — Design Decisions & Rationale

This document covers the *why* behind the pipeline design. For the
request-by-request data flow and the locked formulas themselves, see
the [README's Architecture section](../README.md#architecture) — this
file assumes that as background and goes one level deeper.

---

## 1. Why three layers instead of one model

A single ML model that scores every request would be both slower and
less interpretable than necessary. The three-layer split exists
because each layer solves a different problem at a different cost:

| Layer | Solves | Cost | Coverage |
|---|---|---|---|
| L1 (regex) | Known, exact-match attack signatures | <1ms | High precision, zero generalization |
| L2A (autoencoder) | "Does this look like anything I've seen before?" | ~0.02ms | Generalizes to novel/zero-day payloads |
| L2B (BiGRU) | "If it doesn't look normal, what specifically is it?" | ~15ms | Fine-grained classification, most expensive |

Running L2B on every request would work, but at ~15ms per request
it's the dominant cost in the pipeline (measured mean end-to-end
latency is 3.84ms specifically *because* most traffic never reaches
L2B — see [Selective Escalation](#3-selective-escalation-why-two-thresholds)
below). L1 exists mainly for the pathologically fast case: don't spend
even 0.02ms on a `' OR 1=1--` that a static pattern already recognizes.

## 2. Why L2A is one-class (trained only on normal traffic)

Attack payloads are adversarial by construction — new obfuscation
techniques, encodings, and payload structures appear constantly, and a
supervised classifier trained on known attack examples has no
principled way to generalize to attack *shapes* it has never seen.

A one-class anomaly detector sidesteps this: it only ever learns what
"normal" looks like, so anything sufficiently different — regardless
of whether it resembles a known attack pattern — gets flagged. This is
the mechanism that gives the system its zero-day detection claim, and
it's why L2A's training data deliberately excludes attack traffic
entirely.

**Candidate comparison (NB03):** Isolation Forest was evaluated
alongside the Shallow Autoencoder and lost decisively — Recall 19.87%
vs 82.13%, F1 0.336 vs 0.901, despite similar FPR (~4-5%). The
autoencoder's reconstruction-error signal captures the joint structure
across all 29 features in a way the trees don't. See
[model_selection.md](model_selection.md) for the full comparison table.

## 3. Selective escalation: why two thresholds

L2A produces one number (reconstruction error). Two different
decisions get made from that one number, using two different cutoffs
— and conflating them was a real bug caught during development:

- **L2A's own operating threshold** (`0.0029852`) — where L2A itself
  draws the line between "normal" and "anomalous", chosen to maximize
  recall subject to FPR≤5% on validation. This is what you'd use if
  you cared about L2A's standalone accuracy as reported in Table I.

- **`ESCALATION_THRESHOLD`** (`0.00077472`, P85 of normal validation
  scores) — a *separate*, deliberately lower cutoff used only to
  decide whether L2B runs at all. It's set low enough that ~15% of
  genuinely normal traffic still escalates (and gets correctly
  re-confirmed as normal by L2B), because the cost of under-escalating
  a real attack is much higher than the latency cost of an unnecessary
  L2B call on borderline-normal traffic.

Using L2A's own (higher) threshold as the escalation gate would let
more attacks slip past without ever reaching L2B — that configuration
was tested as the NB06 baseline and scored TPR=69.74%, well below the
tuned 88.64%. The escalation threshold is deliberately more permissive
than L2A's operating threshold *specifically so L2B gets a chance to
catch what L2A alone would miss*.

## 4. Why the L2B multiplier is 90, not 50

The threat-score formula weights L2B's confidence at ×90 versus L2A's
capped ×15 (max 50) contribution. This wasn't an arbitrary choice —
`06-end-to-end-eval (2).ipynb` swept both `L2A_SCORE_MULTIPLIER` and
`L2B_CONF_MULTIPLIER` on the validation set and selected the
combination maximizing recall without pushing FPR above the accepted
band. At ×50, real attacks with high L2B confidence weren't always
crossing the 70-point block threshold on their own; ×90 fixed that
while validation FPR stayed acceptable (0.17% on the final test set).

## 5. Why the 5-class taxonomy folds `cmdi` into `other_attack`

Early iterations trained L2B with `cmdi` as a 6th standalone class.
This was consolidated into `other_attack` for two reasons: (1)
`HttpParamsDataset`'s command-injection examples are a small fraction
of the total attack volume, making per-class metrics for a standalone
`cmdi` class noisy and hard to interpret meaningfully; (2) it keeps
the classifier's output space aligned with what the base papers
(ADL-WAF, Babaey & Faragardi) report, making the comparison in the CRC
more direct. `other_attack` in practice also captures SSRF-style,
deserialization-style, and injection patterns beyond just `cmdi` — see
`test_traffic.py`'s `attacks_other` list for representative examples.

## 6. Why the health-feedback loop doesn't auto-adjust thresholds

NB08 (`08-server-health-feedback-simulation-ipynb.ipynb`) tests exactly
one mechanism: capture → re-score → disagreement → human-verified
feedback. It does **not** test automatic threshold tightening during a
breach or relaxation on recovery. An earlier draft of the manuscript
claimed automatic threshold adjustment as part of this system — that
was a documentation error caught during CRC review, not a deliberate
design choice being walked back. `app/services/health_monitor.py`
implements only what's actually validated; extending it to adaptive
thresholds would need its own experiment before being claimed.

## 7. Why anti-poisoning uses cross-agreement instead of regex

The adaptive retraining pipeline (`app/services/adaptive_retrain.py`)
needs to decide whether a human-verified label is trustworthy before
feeding it back into training — a poisoned or mislabeled sample here
degrades the next model version. The natural first approach —
"does the claimed label match a keyword pattern for that attack
class?" — was tried and rejected: it flagged ~99% of genuinely correct
verified samples as implausible, because most real attack payloads
(especially obfuscated or multi-technique ones) don't match any single
hand-written regex.

The current mechanism instead re-scores the sample with the *current*
L2A/L2B models and checks whether their independent assessment
corroborates the human's verified label — cross-agreement, not pattern
matching. This generalizes across all four attack classes without
needing per-class regex maintenance, at the cost of being only as good
as the current models' judgment (a limitation worth being explicit
about: a sufficiently novel attack that fools both a human reviewer
*and* the current models would still pass this check).

## 8. Known limitation: L2B on out-of-distribution inputs

`HttpParamsDataset` (the primary L2B training source) is built around
requests *with* query parameters. A bare path with no parameters
(`/hello`) was observed scoring `other_attack` at 99%+ confidence in
live testing — a textbook overconfident-on-OOD failure, not a pipeline
bug. This is a training-data coverage gap, not a scoring-formula
issue, and would need param-less normal examples added to the
training set to close.
