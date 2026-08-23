# CRC / Notebook Sync — Change Log

Full-stack app brought in line with the CRC-locked architecture and the
notebooks already in `ml/notebooks/`. Model artifacts (ONNX files) are
NOT included in this pass — see "Still needed" below.

## 1. Feature engineering (INPUT_DIM 25 → 29)
- `ml/feature_engineering/extractor.py` — replaced with the notebook's
  corrected version (pulled directly from `02-feature-engineering.ipynb`
  cell 5): adds `strip_to_path_query()` domain-stripping and 4 new
  JSON/GraphQL structural features (`is_json_body`, `is_graphql_operation`,
  `body_nesting_depth`, `body_key_count`). `INPUT_DIM` is now 29.
- `ml/feature_engineering/tokenizer.py` — replaced with the notebook's
  version (cell 7): now applies the same domain-stripping before encoding,
  so training-time and inference-time text match exactly.
- `ml/feature_engineering/normalizer.py` — checked against the notebook
  (cell 6); byte-for-byte identical, no change needed.

## 2. 5-class taxonomy (cmdi folded into other_attack)
- `app/services/layer2b_deep.py` — `CLASS_NAMES` reduced from 6 classes
  to the CRC's 5: `["normal","sqli","xss","lfi","other_attack"]`, matching
  NB06/NB07/NB08.
- `app/templates/threats.html` — filter loop updated to match (5 classes).
  Left cosmetic-only `cmdi` color entries in `main.css`/`charts.js`/
  `live_logs.js` untouched — harmless, just unused now.

## 3. Threat scoring — CRC Decision 2 locked config
- `app/core/config.py` — added `ESCALATION_THRESHOLD = 0.00077472`,
  `L2A_SCORE_MULTIPLIER = 15.0`, `L2B_CONF_MULTIPLIER = 90.0` (was 50).
  These are the exact validation-selected values from
  `06-end-to-end-eval (2).ipynb` (the canonical `fork-of-06` notebook),
  cells 30–40.
- `app/services/threat_scorer.py` — now reads coefficients from
  `settings` instead of hardcoding them.
- `app/services/layer2a_anomaly.py` — split into `score()` (raw
  reconstruction error) and `infer()` (back-compat bool using L2A's OWN
  trained threshold). The two thresholds — L2A's own operating threshold
  vs. the escalation threshold — are kept explicitly separate; conflating
  them was the root of the old scorer being wrong.
- `app/middleware/waf_middleware.py` — the anomaly gate now compares
  `l2a_score` against `settings.ESCALATION_THRESHOLD` (selective
  escalation) instead of the model's own threshold. This is the
  headline change: most traffic still skips L2B, but the escalation
  point is now the validated P85-of-normal-scores cutoff, not the raw
  L2A operating threshold.

## 4. Server health feedback loop — CRC Decision 1
- `app/middleware/waf_middleware.py` — `_log_and_store()` now captures
  request body text (truncated to 2000 chars) for allow/log decisions
  only. This was previously missing entirely, which made re-auditing
  past traffic impossible.
- `app/services/reaudit.py` (**new**) — shared re-scoring pipeline that
  mirrors the live middleware's decision logic exactly, for use by
  offline/audit re-scoring (not the request-latency path).
- `app/services/health_monitor.py` — `_trigger_audit()` was previously
  called but never defined (dead code). Now implements exactly what
  NB08 demonstrates and nothing more:
  capture recent allow/log traffic → re-score with `reaudit.py` →
  flag disagreements (original allow/log, re-audit says non-normal) →
  push disagreements into `feedback_queue` for human verification.
  **No automatic threshold tightening during breach or relaxation on
  recovery** — matches the CRC manuscript correction, not an
  architecture change.
- `app/db/queries.py` — added `get_recent_allow_log_traffic()` and
  `insert_health_audit()`.
- `app/db/collections.py` — added `health_audit_log()` collection.

## 5. Adaptive retraining anti-poison — cross-agreement, not regex
- `app/services/adaptive_retrain.py` — anti-poison now uses per-IP cap +
  family-diversity cap (URL canonicalization, NB07's approach) +
  **L2A/L2B cross-agreement** (re-score the sample with current models;
  a verified attack label must be corroborated by the models actually
  flagging it as an attack, and vice versa for "normal"). This
  replaces a regex label-plausibility check that would reject too many
  genuinely valid candidates in production use — NB07's own simulation
  notebook uses a regex check tuned narrowly for its one drift scenario
  (obfuscated LFI), which doesn't generalize to the other three attack
  classes; cross-agreement does.

## 6. Bug fixes
- `app/services/model_loader.py` — dummy validation vector was
  `(1, 25)`, now `(1, 29)`. Also referenced non-existent
  `l2b._uses_tokens` (should be the module-level constant
  `l2b.USES_TOKENS`) — fixed.
- `app/api/routes/models.py` — same `l2b._uses_tokens` → `l2b.USES_TOKENS`
  fix in two places; `/api/models/info` now reports both L2A thresholds
  (own vs. escalation) explicitly instead of conflating them.

## 7. Model artifacts — now in place and verified
- `layer2a_best.onnx` (Shallow Autoencoder, self-contained, no external
  data), `layer2b_best.onnx` + `layer2b_bigru.onnx.data` (BiGRU, uses
  external data — the graph references this filename literally, so
  DO NOT rename the `.data` file independently of re-exporting), and
  `scaler_l2a.pkl` (StandardScaler, fit on train split only, loaded
  from the upstream `hiwaf-split-v1` Kaggle dataset) are all in
  `ml/exported_models/`.
- `layer2a_best_threshold.txt` updated to the real retrained value
  (`0.0029852040629296923`) — the old file had a stale pre-CRC number.
- Removed a stale unused `layer2a_best.onnx.data` left over from the
  old pre-CRC model (L2A's current graph is fully self-contained, so
  this file was dead weight).
- Verified end-to-end with real ONNX Runtime inference (not stubs):
  `model_loader.load_all()` — the exact function `main.py`'s lifespan
  calls on boot — passes cleanly. A quick smoke test (benign query,
  SQLi, XSS, path traversal) through the full scaled pipeline
  (`feature_extractor.extract()` → `l2a.score()` → escalation gate →
  `l2b.infer()`) produced correctly-ordered scores (normal < SQLi <
  other_attack < XSS) and 100%-correct L2B labels on all 4 cases.
- Pinned `scikit-learn==1.6.1` in `app/requirements.txt` to match the
  version `scaler_l2a.pkl` was fit with (was unpinned; a version
  mismatch warning appeared with 1.8.0 in this environment).

## 8. Second audit pass — found and fixed a duplicate stale pipeline
- **`app/api/routes/traffic.py`'s `/api/traffic/analyze` endpoint** was a
  second, independent WAF pipeline that never received the CRC 2 escalation
  fix — still used `l2a.infer()`'s `is_anomaly` bool gate and the old
  15/50 multiplier formula. Rewritten to mirror `waf_middleware.py`
  exactly (selective escalation via `settings.ESCALATION_THRESHOLD`,
  `settings.L2A_SCORE_MULTIPLIER`/`L2B_CONF_MULTIPLIER`), and now also
  captures request bodies for allow/log traffic so it feeds the health
  audit correctly, same as the live proxy path.
- **`app/api/routes/dashboard.py`'s `/dashboard/models` page** had a
  duplicate copy of the `l2a._threshold` field-naming bug fixed
  elsewhere in `api/routes/models.py` — same bug, missed because it's a
  separate code block. Fixed, and `app/templates/models.html` updated
  to show both `own_threshold` and `escalation_threshold` distinctly.
- **`app/services/adaptive_retrain.py`** gave no visibility into *why*
  samples were rejected (just a count) — added
  `reject_reason_breakdown` to the persisted `retrain_log` doc, and a
  loud `models_not_loaded` error state instead of silently rejecting
  every sample through a swallowed exception if L2A/L2B aren't loaded
  yet when a retrain cycle runs.

## 9. Made the health-feedback and retrain loops actually demonstrable
- **`dummy_app.py`** — added `POST /simulate/breach` and
  `POST /simulate/recover` so `/health` can report a breach-level
  `error_rate` on demand, since the stock dummy app never returns one
  and the real health-check loop has nothing to react to otherwise.
- **`app/api/routes/health.py`** — added
  `POST /api/health/trigger-audit` to run the capture→re-score→
  disagreement→feedback cycle on demand instead of waiting up to 60s
  for the next monitor tick.
- **`demo_full_loop.py`** (new, repo root) — end-to-end script:
  generates traffic → triggers a health audit → seeds a synthetic
  batch of verified `feedback_queue` samples (clearly marked
  `source: "demo_seed"`, bypassing the human-review UI on purpose,
  for demo speed) → triggers a retrain cycle. Verified working against
  a mocked MongoDB with the real ONNX models loaded: the health audit
  correctly caught a live disagreement (a `1 OR 1=1` SQLi that had
  been allowed, re-audit correctly flagged it), and the retrain cycle
  correctly passed 209/210 synthetic samples through anti-poison and
  transitioned to `status: "queued"` once past `RETRAIN_MIN_SAMPLES`.

## 10. Third bug pass — from your actual runtime error
- **`app/db/queries.py` — `ObjectId` JSON-serialization crash.** Motor's
  `insert_one()` mutates the dict you pass it, injecting a raw
  (non-JSON-serializable) `ObjectId` into `doc["_id"]` **on the same
  object**. `health_monitor._trigger_audit()` built its `report` dict,
  passed it to `insert_health_audit(report)`, then returned that same
  now-mutated dict straight through `POST /api/health/trigger-audit` —
  FastAPI's `jsonable_encoder` can't serialize `ObjectId`, hence the
  500 (`TypeError: 'ObjectId' object is not iterable`). Fixed at the
  source: every `insert_*` helper in `queries.py` now inserts
  `dict(doc)` — a shallow copy — so the caller's original dict is
  never mutated. Audited every other `.insert_one()`/`.insert_many()`
  call site in the app for the same landmine; the only other one that
  returns its doc to a caller (`adaptive_retrain.py`'s `run_doc`) was
  already safe (explicitly overrides `_id` with a `str()` after the
  insert). Reproduced your exact failure ("no eligible allow/log
  traffic to capture in window") against a mocked DB and confirmed
  `jsonable_encoder` now succeeds on both the empty-window and
  disagreement-found branches.
- **`demo_full_loop.py` — wrong path prefix.** The 404s you saw in
  Step 1 were my script's bug, not the app's: `waf_middleware.py` only
  intercepts `/proxy/*` (by design — direct `/api/*` calls bypass the
  WAF pipeline on purpose). The demo script was hitting bare paths
  like `/tienda1/publico/anadir.jsp` with no `/proxy` prefix, so they
  fell through to normal FastAPI routing and 404'd since no such route
  exists in the WAF app itself. Fixed to request `/proxy{path}`,
  matching what your own `test_traffic.py` was already doing correctly.
- Run `demo_full_loop.py` against your actual running stack (real
  MongoDB, real dummy_app, real WAF) — it was only verified against a
  mocked DB in the environment I built this in, which has no MongoDB
  available. The logic is confirmed correct; this is just confirming
  it end-to-end on your machine.
- Everything else from the prior "Still needed" section (full
  quantitative validation against your held-out test set) still applies.
