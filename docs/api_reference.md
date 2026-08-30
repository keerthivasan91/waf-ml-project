# API Reference

Complete reference for every endpoint. All routes are namespaced under
`/api/*` and are bypassed by the WAF's own middleware (see
`bypass_paths` in `app/middleware/waf_middleware.py` and
[architecture.md](architecture.md)) — hitting these does not run
through L1/L2A/L2B, they're the WAF's own management surface.

Interactive Swagger UI is always available at `/api/docs` while the
server is running — this document exists for offline reading and for
the exact field-level detail Swagger's auto-generated schema doesn't
always make obvious (e.g. which fields are optional, valid enum
values).

---

## Traffic — `/api/traffic`

### `POST /api/traffic/analyze`

Runs a single request through the full WAF pipeline — L1 → escalation
gate → L2A → (L2B if escalated) → threat scorer — using the exact same
logic as the live middleware (`app/api/routes/traffic.py` is kept
manually in sync with `waf_middleware.py`; if you change one, change
the other).

**Request body** (`IncomingRequest`):
```json
{
  "url": "/api/products/search?q=1'+OR+1=1",
  "method": "GET",
  "headers": {},
  "body": "",
  "ip": "1.2.3.4"
}
```
| Field | Type | Required | Notes |
|---|---|---|---|
| `url` | string | yes | Path + query, no scheme/host needed (stripped anyway) |
| `method` | string | no | Default `"GET"` |
| `headers` | object | no | Default `{}` |
| `body` | string | no | Default `""` |
| `ip` | string | no | Default `null` — used for logging/rate-limiting context |

**Response** (`ThreatResult`):
```json
{
  "request_id": "a1b2c3d4-...",
  "decision": "block",
  "score": 100,
  "label": "sqli",
  "layer": "L2B",
  "confidence": 0.9982,
  "l2a_score": 1.17078,
  "l2a_contrib": 17.56,
  "l2b_contrib": 89.84,
  "latency_ms": 4.1,
  "timestamp": "2026-08-24T10:15:30.123Z"
}
```
| Field | Type | Notes |
|---|---|---|
| `decision` | string | `"allow"` \| `"log"` \| `"block"` |
| `score` | int | 0–100 |
| `label` | string | `"normal"` \| `"sqli"` \| `"xss"` \| `"lfi"` \| `"other_attack"` \| an L1 rule name (`"sqli_rule"` etc.) if blocked at L1 |
| `layer` | string | `"L1"` \| `"L2A"` \| `"L2B"` — which layer made the final decision |
| `confidence`, `l2a_score`, `l2a_contrib`, `l2b_contrib` | float, nullable | Only populated when the request actually reached that layer |

A request that never escalates (`l2a_score < ESCALATION_THRESHOLD`)
returns `layer: "L2A"`, `label: "normal"`, `decision: "allow"`,
`confidence: 1.0`, with no `l2b_contrib`.

---

## Logs — `/api/logs`

### `GET /api/logs/recent`
| Query param | Type | Default | Notes |
|---|---|---|---|
| `limit` | int | 100 | 1–500 |
| `decision` | string | none | Optional filter: `allow` \| `log` \| `block` |

Returns a list of `RequestLog` documents (most recent first):
```json
[{
  "request_id": "...", "ip": "1.2.3.4", "method": "GET",
  "url": "/api/products?category=electronics", "body_len": 0,
  "decision": "allow", "score": 15, "label": "normal", "layer": "L2A",
  "latency_ms": 3.2, "timestamp": "2026-08-24T10:15:30.123Z"
}]
```

### `GET /api/logs/threats`
| Query param | Type | Default | Notes |
|---|---|---|---|
| `limit` | int | 50 | 1–200 |

Returns recent `threat_events` (block/log decisions only, includes
`l2a_score` and L2B confidence where applicable).

---

## Feedback / Review — `/api/feedback`

### `GET /api/feedback/pending`
| Query param | Type | Default |
|---|---|---|
| `limit` | int | 100 |

Returns `FeedbackItem` documents awaiting human review
(`verified_label: null`) — both borderline `log`-decision traffic and
health-audit-sourced disagreements land here (the latter tagged
`"source": "health_audit"`).

### `POST /api/feedback/review/{request_id}`
```json
{
  "verified_label": "sqli",
  "is_poisoning": false
}
```
| Field | Type | Required | Valid values |
|---|---|---|---|
| `verified_label` | string | yes | `normal` \| `sqli` \| `xss` \| `lfi` \| `other_attack` \| `false_positive` |
| `is_poisoning` | bool | no | Default `false` |

Returns `400` if `verified_label` isn't one of the valid values,
`404` if `request_id` isn't in the feedback queue. On success:
```json
{"status": "ok", "request_id": "...", "verified_label": "sqli"}
```

### `POST /api/feedback/trigger-retrain`
No body. Runs `app/services/adaptive_retrain.py`'s anti-poison
validation over all verified, non-poisoned samples currently in
`feedback_queue`.

**If fewer than `RETRAIN_MIN_SAMPLES` (200) verified samples exist:**
```json
{"status": "skipped", "reason": "insufficient_samples", "n_samples": 42}
```

**If L2A/L2B aren't loaded** (cross-agreement check can't run):
```json
{"status": "error", "reason": "models_not_loaded", "n_samples": 210}
```

**Otherwise:**
```json
{
  "status": "queued",
  "n_raw": 210,
  "n_clean": 205,
  "n_rejected": 5,
  "reject_reason_breakdown": {
    "per_ip_cap_exceeded": 2,
    "cross_agreement_failed_normal_flagged_as_attack": 3
  },
  "note": "Full retraining runs offline in Kaggle/Colab (NB07 pipeline)..."
}
```
This validates and logs the batch — it does **not** actually retrain a
model. Actual retraining runs offline via NB07's pipeline, consuming
this validated batch.

---

## Health — `/api/health`

### `GET /api/health/`
Basic liveness check (pings MongoDB).
```json
{"status": "ok", "db": true}
```

### `GET /api/health/stats`
```json
{
  "requests_24h": 1204, "blocked_24h": 89, "allowed_24h": 1103,
  "block_rate_pct": 7.4, "avg_latency_1h_ms": 4.2, "p99_latency_1h_ms": 21.6
}
```
(Exact field set matches whatever `get_dashboard_stats()` in
`app/db/queries.py` currently returns — check there if fields drift
from this list.)

### `POST /api/health/trigger-audit?error_rate=0.99`
| Query param | Type | Default |
|---|---|---|
| `error_rate` | float | 1.0 |

Manually runs the CRC Decision 1 capture → re-score → disagreement →
feedback cycle immediately, bypassing the normal 60-second monitor
tick and the `ERROR_RATE_THRESHOLD` breach check — `error_rate` here
is just recorded in the report, not compared against the threshold.

**No eligible traffic in the capture window:**
```json
{
  "timestamp": "2026-08-24T10:15:30.123Z",
  "error_rate": 0.99, "error_threshold": 0.1,
  "traffic_window_captured": 0, "feedback_candidates": 0
}
```

**Traffic captured:**
```json
{
  "timestamp": "2026-08-24T10:15:30.123Z",
  "error_rate": 0.99, "error_threshold": 0.1,
  "traffic_window_captured": 47, "capture_pct": 100.0,
  "feedback_candidates": 3
}
```

---

## Models — `/api/models`

### `GET /api/models/info`
```json
{
  "layer2a": {
    "exists": true, "path": "ml/exported_models/layer2a_best.onnx",
    "size_kb": 11.1, "modified": 1755960000.0, "modified_human": "2026-08-23 14:55:00",
    "own_threshold": 0.0029852040629296923,
    "escalation_threshold": 0.00077472,
    "input_name": "input"
  },
  "layer2b": {
    "exists": true, "path": "ml/exported_models/layer2b_best.onnx",
    "size_kb": 12.0, "modified": 1755960000.0, "modified_human": "2026-08-23 14:55:00",
    "input_name": "token_ids", "uses_tokens": true,
    "class_names": ["normal", "sqli", "xss", "lfi", "other_attack"]
  },
  "scaler": {"exists": true, "path": "ml/exported_models/scaler_l2a.pkl", "size_kb": 2.1, ...},
  "threshold_file": {"exists": true, "path": "ml/exported_models/layer2a_best_threshold.txt", ...}
}
```

### `POST /api/models/reload`
No body. Hot-reloads both ONNX models from disk without restarting the
process — use after replacing files in `ml/exported_models/`. Returns
`500` with `{"reload_errors": [...]}` if either model fails to load
(original models stay loaded on failure — this is not atomic across
L2A and L2B individually, though: if L2A reloads successfully but L2B
then fails, L2A has already been swapped).
```json
{"status": "reloaded", "l2a_threshold": 0.0029852040629296923, "l2b_uses_tokens": true}
```

### `GET /api/models/history?limit=20`
Returns the last N `model_versions` documents (hot-reload event log).

---

## Dashboard (SSR pages, not JSON APIs)

`app/api/routes/dashboard.py` serves server-rendered HTML, not part of
the JSON API surface above — listed here for completeness:

| Route | Page |
|---|---|
| `GET /dashboard` | Overview |
| `GET /dashboard/logs` | Live Logs |
| `GET /dashboard/threats` | Threats |
| `GET /dashboard/feedback` | Review Queue |
| `GET /dashboard/models` | Models |
