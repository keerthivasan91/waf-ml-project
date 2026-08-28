<div align="center">

# Hybrid Intelligent Web Application Firewall
### Zero-Day Attack Detection Using Multi-Layer Machine Learning Architecture

**Cambridge Institute of Technology, Bengaluru**
Department of CSE — IoT and Cyber Security including Blockchain
Final Year B.E Project — 2025–26

---

| | |
|---|---|
| **Team** | Keerthi Vasan P · Darshan Gowda C · Santhosh V · Srujan H R |
| **USNs** | 1CD23IC029 · 1CD23IC013 · 1CD23IC049 · 1CD23IC055 |
| **Batch** | 5 |
| **Phase 1** | Jul – Nov 2025 |
| **Phase 2** | Jan – May 2026 |

</div>

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Running Locally (Dev)](#running-locally-dev)
- [Running with Docker](#running-with-docker)
- [Dashboard Pages](#dashboard-pages)
- [API Reference](#api-reference)
- [Testing / Demo Scripts](#testing--demo-scripts)
- [Training the Models](#training-the-models)
- [Locked CRC Results](#locked-crc-results)
- [Known Issues & Caveats](#known-issues--caveats)
- [Team Responsibilities](#team-responsibilities)

---

## What This Project Is

Traditional WAFs rely on static signature-based rules that only detect known attacks. They fail silently against zero-day exploits, obfuscated payloads, and novel attack patterns.

This project builds a **Hybrid Intelligent WAF** that sits as a reverse proxy in front of a web application and runs every incoming HTTP request through three detection layers. The system combines fast rule-based filtering with ML-based anomaly detection and deep classification.

**The adaptive retraining loop is the core novel contribution** — when the protected server's health metrics spike, the system captures recent allow/log traffic, re-scores it against current thresholds, identifies disagreements, and routes them into a human-verified retraining pipeline with anti-poisoning safeguards.

---

## Architecture

```
Internet → [Nginx] → [FastAPI WAF Middleware] → [Protected Web Application]
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              Layer 1      Layer 2A     Layer 2B
           Rule Engine    Anomaly       Deep
           (Regex/Rate)   Detector    Classifier
                 │            │            │
                 └────────────┴────────────┘
                              │
                     Threat Score Engine
                          (0–100)
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
               Allow      Log+Review      Block
               (< 30)      (30–70)      (≥ 70)
                              │
                    Server Health Monitor
                              │
              Capture → Re-score → Disagreement
                              │
                    Adaptive Retraining
```

### Layer 1 — Rule-Based Filter
Regex patterns for SQLi, XSS, LFI, and OS command injection (`sqli_rule`, `xss_rule`, `lfi_rule`, `cmdi_rule`). Drops known attacks in well under 1ms before any ML runs. Rate limiting (`RATE_LIMIT_PER_MIN`) is configured via `slowapi` but not currently wired into the request path — see [Known Issues](#known-issues--caveats).

### Layer 2A — Anomaly Detector
Shallow Autoencoder, one-class, trained **only on normal traffic**. Anything with high reconstruction error deviates from learned normal behaviour — this is what enables zero-day detection. `INPUT_DIM=29` (after domain-stripping the URL and adding 4 JSON/GraphQL structural features). Exported to ONNX, self-contained (no external data file).

**L2A's own operating threshold** (maximize recall subject to FPR≤5% on validation) is a *different* number from the **selective escalation threshold** used for routing — see below. Conflating the two was a real bug caught and fixed during development; they now live in separate config keys.

### Selective Escalation (CRC Decision 2)
Layer 2B is expensive (~15ms vs ~0.02ms for L2A) and only needs to run on requests that could plausibly be attacks. A request escalates to L2B only if:

```
l2a_score >= ESCALATION_THRESHOLD   (0.00077472 — P85 of normal validation L2A scores)
```

Below that, the request is "clearly normal" and is allowed immediately, **never touching L2B**. This threshold is deliberately different from — and lower than — L2A's own operating threshold; by design, roughly 15% of genuinely normal traffic still escalates and gets correctly re-confirmed as normal by L2B, at extra latency cost. That tradeoff is what the locked test numbers below reflect.

### Layer 2B — Deep Classifier
Bidirectional GRU with Bahdanau attention, token-based input (max_len=512), runs **only for escalated requests**. Classifies into the CRC's **5-class taxonomy**: `normal`, `sqli`, `xss`, `lfi`, `other_attack` (`cmdi` is folded into `other_attack`, not a separate class).

### Threat Score Engine (locked formula — do not change without re-running the validation sweep in `06-end-to-end-eval (2).ipynb`)

```
c_L2A = min(50, l2a_score × 15)                          # L2A_SCORE_MULTIPLIER
c_L2B = 0 if label == "normal" else confidence × 90       # L2B_CONF_MULTIPLIER
score = min(100, c_L2A + c_L2B)
```

| Score | Decision | Action |
|---|---|---|
| < 30 | `allow` | Forward to protected app |
| 30–69 | `log` | Forward + add to human review queue |
| ≥ 70 | `block` | Drop request, return 403 |

### Server Health Monitor + Adaptive Retraining (CRC Decision 1)
The monitor pings the protected app's `/health` endpoint every 60 seconds (`HEALTH_CHECK_INTERVAL_SEC`). If reported `error_rate` exceeds `ERROR_RATE_THRESHOLD` (10%), a health audit fires:

1. **Capture** recent allow/log traffic (request bodies are stored specifically to support this — see `app/services/reaudit.py`)
2. **Re-score** each captured request against *current* thresholds and models
3. **Flag disagreements** — original decision was allow/log, but re-audit now says non-normal
4. Push disagreements into the same human-review queue as borderline `log` traffic

This is deliberately limited to exactly what was tested (NB08) — **no automatic threshold tightening during a breach or relaxation on recovery**. That's a manuscript-accuracy decision, not a missing feature.

The retraining cycle (`app/services/adaptive_retrain.py`) requires `RETRAIN_MIN_SAMPLES` (200) verified samples in the review queue, then applies three anti-poisoning safeguards:
- **Per-IP cap** — no single source can flood the batch
- **Family-diversity cap** — URL-canonicalized near-duplicates capped per batch
- **L2A/L2B cross-agreement** — a verified label must be corroborated by re-scoring the sample with the current models, not trusted blindly. (An earlier regex-based label-plausibility check was replaced with this — it rejected ~99% of genuinely valid candidates because most real payloads don't match any single hand-written pattern.)

Actual model retraining itself still runs offline in Kaggle (NB07's pipeline); this service validates and logs the clean batch for that pipeline to consume.

---

## Tech Stack

| Component | Technology |
|---|---|
| Reverse proxy | Nginx |
| WAF backend | FastAPI + Uvicorn (async Python) |
| Anomaly detector (L2A) | Shallow Autoencoder → ONNX Runtime |
| Deep classifier (L2B) | Bidirectional GRU + Bahdanau attention → ONNX Runtime |
| Database | MongoDB (Motor async driver) |
| Dashboard | Jinja2 SSR + Vanilla JS + Canvas charts |
| Training | TensorFlow/Keras · scikit-learn · XGBoost (Kaggle) |
| Feature scaling | scikit-learn `StandardScaler`, fit on train split only |
| Containers | Docker + Docker Compose |
| Datasets | HttpParamsDataset · CSIC 2010 (see [Datasets](#datasets)) |

---

## Project Structure

```
waf-ml-project/
│
├── .env                         # local environment overrides (gitignored)
├── .env.example                 # template — currently empty, see "Create your .env" below
├── docker-compose.yml           # nginx + app + mongodb (no protected-app service — see docs/deployment.md)
├── dummy_app.py                 # protected demo backend (products/orders/cart/etc + /simulate/breach)
├── test_traffic.py              # 100+ request traffic simulation (normal/borderline/attack)
├── demo_full_loop.py            # exercises the health-audit + adaptive-retrain loop end to end
├── README.md
│
├── nginx/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── conf.d/waf.conf
│
├── app/                         # FastAPI WAF application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # lifespan: DB + model loading, router registration
│   │
│   ├── api/routes/
│   │   ├── dashboard.py         # SSR pages: /dashboard, /logs, /threats, /feedback, /models
│   │   ├── traffic.py           # POST /api/traffic/analyze — mirrors waf_middleware.py's logic
│   │   ├── logs.py              # GET /api/logs/recent, /api/logs/threats
│   │   ├── feedback.py          # GET/POST /api/feedback/...
│   │   ├── health.py            # GET /api/health/, /api/health/stats, POST /api/health/trigger-audit
│   │   └── models.py            # GET/POST /api/models/info, /reload, /history
│   │
│   ├── core/
│   │   ├── config.py            # pydantic-settings — ESCALATION_THRESHOLD, multipliers, etc.
│   │   ├── logging.py           # structured logging setup
│   │   └── exceptions.py        # ModelNotLoadedError, DatabaseError handlers
│   │
│   ├── middleware/
│   │   ├── waf_middleware.py    # main interception + pipeline; bypass_paths excludes the
│   │   │                        # WAF's OWN /api/{traffic,health,logs,feedback,models} routes
│   │   │                        # specifically (NOT a blanket "/api" bypass — the protected
│   │   │                        # app's own business routes also live under /api/*)
│   │   ├── rate_limiter.py      # slowapi limiter object (not currently enforced — see caveats)
│   │   └── request_parser.py    # extracts url/method/headers/body/ip
│   │
│   ├── models/schemas/
│   │   ├── request.py           # IncomingRequest
│   │   ├── threat.py            # ThreatResult
│   │   ├── log.py                # RequestLog
│   │   └── feedback.py          # FeedbackItem
│   │
│   ├── services/
│   │   ├── layer1_filter.py     # regex rules: sqli/xss/lfi/cmdi
│   │   ├── layer2a_anomaly.py   # ONNX autoencoder — score() raw, infer() back-compat bool
│   │   ├── layer2b_deep.py      # ONNX GRU classifier — 5-class CLASS_NAMES
│   │   ├── feature_extractor.py # runtime preprocessing incl. scaler.transform()
│   │   ├── threat_scorer.py     # locked-config score + allow/log/block decision
│   │   ├── reaudit.py           # shared re-scoring pipeline (mirrors middleware, used offline)
│   │   ├── health_monitor.py    # async health-check loop + _trigger_audit (CRC Decision 1)
│   │   ├── feedback_classifier.py # auto-labelling heuristics
│   │   └── adaptive_retrain.py  # anti-poisoning batch validation (cross-agreement based)
│   │
│   ├── db/
│   │   ├── mongodb.py           # Motor async client, index creation
│   │   ├── collections.py       # typed collection accessors
│   │   └── queries.py           # reusable async query functions (insert_* copy dicts —
│   │                            # Motor mutates insert_one() args in place with a raw ObjectId)
│   │
│   ├── templates/               # Jinja2 SSR dashboard templates
│   └── static/                  # CSS + JS (charts, live log polling)
│
└── ml/                          # offline training — NOT deployed in app container
    ├── requirements_train.txt
    ├── feature_engineering/
    │   ├── extractor.py         # extract_features(), to_vector() — INPUT_DIM=29
    │   ├── tokenizer.py         # CharTokenizer (max_len=512), domain-stripped
    │   └── normalizer.py        # Normalizer wrapping StandardScaler
    ├── layer2a/, layer2b/       # candidates/, train.py, evaluate.py, export_onnx.py
    ├── exported_models/         # place trained files here (gitignored — see below)
    │   ├── layer2a_best.onnx              # self-contained, no external data
    │   ├── layer2a_best_threshold.txt     # L2A's OWN threshold (not the escalation one)
    │   ├── layer2b_best.onnx
    │   ├── layer2b_bigru.onnx.data        # REQUIRED — the .onnx graph references this exact
    │   │                                  # filename internally; do not rename independently
    │   └── scaler_l2a.pkl                 # StandardScaler, fit on train split only
    └── notebooks/
        ├── 01-data-exploration.ipynb
        ├── 02-feature-engineering.ipynb
        ├── 03-layer2a-experiments.ipynb
        ├── 04-layer2b-experiments.ipynb
        ├── 05_model_comparison.ipynb
        ├── 06-end-to-end-eval.ipynb            # baseline (pre-tuning) config — ablation row
        ├── 06-end-to-end-eval (2).ipynb        # canonical — locked selective-escalation config
        ├── 07-adaptive-retraining-simulation.ipynb
        └── 08-server-health-feedback-simulation-ipynb.ipynb
```

---

## Datasets

| Dataset | Use | Records |
|---|---|---|
| HttpParamsDataset (Morzeux) | L2B primary — SQLi/XSS/LFI/other_attack | 31,067 rows |
| CSIC 2010 | L2A normal training + L2B normal-class supplement | 61,065 rows |
| WAF-A-MoLE | Test-only — adversarial robustness evaluation | — |
| Drift sets (obfuscated LFI etc.) | Test-only — NB07 simulation | — |

**CICIDS 2017 is permanently excluded.** It's network-level NetFlow data (packet/flow statistics), not HTTP payloads — incompatible with this project's request-level feature extraction, so it was dropped entirely rather than partially adapted.

A single **70/15/15 family-aware split** (`group_stratified_split()`, NB02) is shared across L2A and L2B — no separate splits per layer, to keep validation-selected thresholds and reported metrics consistent across the whole pipeline.

---

## Running Locally (Dev)

### Prerequisites
- Python 3.11+
- MongoDB running locally
- Two terminal windows (protected app + WAF), both using the **same** virtualenv

### Step 1 — Clone and set up environment

```powershell
git clone <repo-url>
cd waf-ml-project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r app/requirements.txt
```

> Run every script (`uvicorn`, `test_traffic.py`, `demo_full_loop.py`) from this **same** activated venv. A different venv won't have `motor`/`onnxruntime`/etc. installed and will fail with confusing `ModuleNotFoundError`s.

### Step 2 — Create your `.env` file

`.env.example` is currently an empty template — create `.env` directly with:

```env
APP_NAME=WAF-ML
APP_VERSION=1.0.0
DEBUG=True

MONGO_URI=mongodb://localhost:27017
MONGO_DB=waf_dev

L2A_ONNX_PATH=ml/exported_models/layer2a_best.onnx
L2A_THRESHOLD_PATH=ml/exported_models/layer2a_best_threshold.txt
L2B_ONNX_PATH=ml/exported_models/layer2b_best.onnx
SCALER_PATH=ml/exported_models/scaler_l2a.pkl

ESCALATION_THRESHOLD=0.00077472
L2A_SCORE_MULTIPLIER=15.0
L2B_CONF_MULTIPLIER=90.0
SCORE_LOG_THRESHOLD=30
SCORE_BLOCK_THRESHOLD=70

RATE_LIMIT_PER_MIN=100
PROTECTED_APP_URL=http://127.0.0.1:5000
HEALTH_CHECK_INTERVAL_SEC=60
ERROR_RATE_THRESHOLD=0.10
RETRAIN_MIN_SAMPLES=200
HEALTH_CAPTURE_PCT=100.0
```

> **Important:** `MONGO_URI` must be `localhost` (not `mongodb`) when running outside Docker. `PROTECTED_APP_URL` must be `http://127.0.0.1:5000` for local dev, `http://webapp:5000` inside Docker Compose.

### Step 3 — Place trained model files

```
ml/exported_models/
├── layer2a_best.onnx
├── layer2a_best_threshold.txt
├── layer2b_best.onnx
├── layer2b_bigru.onnx.data     # required by the L2B graph — see note above
└── scaler_l2a.pkl
```

> **sklearn version:** `scaler_l2a.pkl` is pickled with a specific sklearn version (`scikit-learn==1.6.1` as of the current locked models). Mismatched versions load with a warning and may scale incorrectly — pin the same version in `app/requirements.txt`.

### Step 4 — Start MongoDB

```powershell
mongosh --eval "db.adminCommand('ping')"   # should return { ok: 1 }
```

### Step 5 — Start both servers

**Terminal 1 — Protected app:**
```powershell
uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

**Terminal 2 — WAF:**
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Successful startup looks like:
```
INFO | waf | Starting WAF-ML v1.0.0
INFO | waf | MongoDB connected → waf_dev
INFO | waf | L2A loaded | input=input | own_threshold=0.00299 | escalation_threshold=0.00077472
INFO | waf | L2B loaded | input=token_ids | uses_tokens=True
INFO | waf | All ML models loaded successfully
INFO | waf | Health monitor started (interval=60s)
INFO | waf | WAF ready ◈
```

### Step 6 — Open the dashboard

```
http://127.0.0.1:8000/dashboard
```

### Step 7 — Simulate traffic

```powershell
python test_traffic.py
```

Sends 100+ requests (normal / borderline / attack, spanning all 4 non-normal classes) directly to `http://127.0.0.1:8000/...` — **no `/proxy` prefix**; the middleware forwards path + query 1:1 to the protected app, except its own bypassed `/api/*` management routes. Check the dashboard to see logs, threats, and the review queue populate.

To exercise the health-audit and adaptive-retrain loop specifically:
```powershell
python demo_full_loop.py
```

---

## Running with Docker

### Prerequisites
- Docker Desktop + Docker Compose

### Step 1 — Environment
```bash
cp .env.example .env
# fill in the same keys as above, but:
# MONGO_URI=mongodb://mongodb:27017
# PROTECTED_APP_URL=http://webapp:5000
```

### Step 2 — Model files
Same as [Step 3 above](#step-3--place-trained-model-files).

### Step 3 — Build and start
```bash
docker-compose up --build
```

### Step 4 — Access

| URL | Description |
|---|---|
| `http://localhost/dashboard` | Main dashboard (via Nginx) |
| `http://localhost/...` | Proxied application traffic (goes through the WAF) |
| `http://localhost:8000/api/docs` | FastAPI Swagger UI |

### Useful Docker commands
```bash
docker-compose logs -f app
docker-compose restart app
docker-compose down
docker-compose down -v   # also wipes the MongoDB volume
```

---

## Dashboard Pages

| URL | Page | Description |
|---|---|---|
| `/dashboard` | Overview | 24h stats, attack breakdown, recent threats |
| `/dashboard/logs` | Live Logs | Real-time request log with filter by decision |
| `/dashboard/threats` | Threats | All blocked/flagged events with attack type cards |
| `/dashboard/feedback` | Review Queue | Human labelling interface for borderline + health-audit items |
| `/dashboard/models` | Models | ONNX model metadata (own + escalation thresholds), hot reload button |
| `/api/docs` | API Docs | Swagger UI for all REST endpoints |

---

## API Reference

### Traffic Analysis
```
POST /api/traffic/analyze
```
Runs a single request through the full WAF pipeline (identical logic to the live middleware).
```json
{
  "url": "/api/products/search?q=test",
  "method": "GET",
  "headers": {},
  "body": "",
  "ip": "1.2.3.4"
}
```
Response:
```json
{
  "request_id": "uuid",
  "decision": "allow",
  "score": 12,
  "label": "normal",
  "layer": "L2A",
  "l2a_score": 0.04231,
  "latency_ms": 4.2
}
```

### Logs
```
GET /api/logs/recent?limit=100&decision=block
GET /api/logs/threats?limit=50
```

### Feedback / Review
```
GET  /api/feedback/pending?limit=100
POST /api/feedback/review/{request_id}
     Body: { "verified_label": "sqli", "is_poisoning": false }
POST /api/feedback/trigger-retrain
```
Valid labels: `normal`, `sqli`, `xss`, `lfi`, `other_attack`, `false_positive`

### Health
```
GET  /api/health/
GET  /api/health/stats
POST /api/health/trigger-audit?error_rate=0.99
```
`trigger-audit` runs the capture → re-score → disagreement → feedback cycle immediately instead of waiting for the next 60s monitor tick — useful for demoing or testing CRC Decision 1 without needing a real backend outage.

### Models
```
GET  /api/models/info
POST /api/models/reload
GET  /api/models/history
```

---

## Testing / Demo Scripts

| Script | Purpose |
|---|---|
| `test_traffic.py` | 104 requests across normal / borderline / sqli / xss / lfi / other_attack, ~0.6s apart, with 429 backoff |
| `demo_full_loop.py` | Sends traffic → triggers a health audit → seeds synthetic verified feedback → triggers a retrain cycle, so you can watch the full CRC Decision 1 + adaptive retrain gate fire without waiting on real volume or a real outage |

Both scripts target `dummy_app.py`'s actual routes (`/api/products`, `/api/orders`, `/api/cart`, `/api/files/*`, `/api/system/*`, `/api/admin/*`, `/api/contact`, `/api/users/*`) — not the old CSIC-style `/tienda1/publico/*.jsp` paths from earlier iterations of this project.

---

## Training the Models

Training runs offline on Kaggle. Run notebooks in order — NB01 through NB08:

```
01-data-exploration.ipynb              → dataset distribution
02-feature-engineering.ipynb           → 29-dim feature pipeline + domain-stripping (writes
                                          extractor.py / tokenizer.py / normalizer.py)
03-layer2a-experiments.ipynb           → Isolation Forest vs Shallow Autoencoder, pick winner
04-layer2b-experiments.ipynb           → XGBoost vs CNN-1D vs BiGRU, pick winner
05_model_comparison.ipynb              → side-by-side metrics table
06-end-to-end-eval (2).ipynb           → canonical — locked selective-escalation config
07-adaptive-retraining-simulation.ipynb → anti-poisoning mechanism, drift scenario
08-server-health-feedback-simulation-ipynb.ipynb → capture/re-score/disagreement evaluation
```

```bash
cd ml
pip install -r requirements_train.txt
```

After training, copy outputs to `ml/exported_models/` (see [Step 3](#step-3--place-trained-model-files) above for the exact file list).

---

## MongoDB Collections

| Collection | Stores |
|---|---|
| `request_logs` | Every request: URL, method, score, decision, latency; `body` captured for allow/log only |
| `threat_events` | Blocked/logged requests with L2A score and L2B confidence |
| `feedback_queue` | Score 30–69 requests + health-audit disagreements pending human review |
| `model_versions` | Hot reload events with threshold and model path |
| `health_snapshots` | Periodic health check results from the protected app |
| `health_audit_log` | CRC Decision 1 audit reports: traffic captured, disagreements found |
| `retrain_log` | Retrain trigger history: `n_raw`/`n_clean`/`n_rejected` + `reject_reason_breakdown` |

```js
mongosh waf_dev

db.request_logs.countDocuments({decision: "block"})
db.feedback_queue.find({verified_label: null}).limit(5)
db.health_audit_log.find().sort({timestamp: -1}).limit(5)
db.retrain_log.find().sort({timestamp: -1}).limit(1)

// reset for a clean demo run
db.request_logs.drop(); db.threat_events.drop(); db.feedback_queue.drop()
```

---

## Locked CRC Results

These are the actual validation-selected, test-set-evaluated numbers from `06-end-to-end-eval (2).ipynb` — the deployed config matches these exactly, not generic targets.

**Layer 2A (Shallow Autoencoder, standalone):**

| Metric | Value |
|---|---|
| Recall | 82.13% |
| FPR | 4.15% |
| ROC-AUC | 0.964 |
| Own threshold | 0.0029852 |

**Layer 2B (BiGRU, standalone):** Macro-F1 = 0.9929, Accuracy = 0.9954

**End-to-end (selective escalation, headline result):**

| Metric | Value |
|---|---|
| FPR | 0.17% |
| TPR (block-only) | 88.64% |
| SQLi block rate | 99.95% |
| XSS block rate | 98.21% |
| other_attack/CMDi block rate | 76.73% |
| LFI block rate | 45.41% |
| Mean latency | 3.84ms |
| P99 latency | 22.89ms (known limitation — marginally above the 20ms target) |

The original NB06 baseline config (`L2B_CONF_MULTIPLIER=50`, escalating on L2A's own threshold, TPR=69.74%) is reported as the pre-tuning ablation row, not the primary result.

---

## Known Issues & Caveats

**`middleware.rate_limiter`'s `limiter` object is configured but not enforced.** `app/main.py` registers a `RateLimitExceeded` handler and sets `app.state.limiter`, but never adds `SlowAPIMiddleware` or a per-route `@limiter.limit(...)` decorator. `RATE_LIMIT_PER_MIN` currently has no effect — you won't see 429s regardless of request rate. `test_traffic.py` still handles 429 with backoff so it's correct once this gets wired up.

**L2B can be confidently wrong on out-of-distribution inputs.** A bare path with no query parameters (e.g. `/hello`) was observed scoring `other_attack` at 99%+ confidence and getting blocked. HttpParamsDataset is built around requests *with* parameters, so param-less paths are likely rare-to-absent in training — a classic overconfident-on-OOD failure mode, not a pipeline bug. Worth augmenting L2B's training set with param-less normal examples before the next retrain if this matters for your deployment.

**`bypass_paths` in `waf_middleware.py` must name the WAF's own routes specifically, never blanket-bypass `/api`.** The protected application's own business routes (`dummy_app.py`'s `/api/products`, `/api/orders`, etc.) intentionally share the `/api` prefix with the WAF's internal management routes (`/api/traffic`, `/api/health`, `/api/logs`, `/api/feedback`, `/api/models`). Bypassing all of `/api` would silently disable WAF protection for the entire backend.

**Model calibration mismatch.** If L2A reconstruction errors are very high (6–10+ range) for normal traffic, the scaler most likely isn't being applied, or the sklearn version differs between training and runtime. Correctly scaled normal traffic should score well under 1.0.

**`dummy_app.py` must be running separately.** The WAF returns 502 Bad Gateway for any allowed request if it can't reach port 5000.

**`--reload` watches the whole project directory.** Saving `test_traffic.py` or `demo_full_loop.py` while `uvicorn --reload` is running will restart the WAF mid-test. Either drop `--reload` during traffic testing, or add `--reload-exclude test_traffic.py --reload-exclude demo_full_loop.py`.

**Run test/demo scripts from the same venv as the server.** `motor`, `onnxruntime`, etc. are only installed in the server's environment — a different terminal with a different venv activated will fail with `ModuleNotFoundError`.

---

## Team Responsibilities

| Member | Primary area |
|---|---|
| Keerthi Vasan P | FastAPI backend, middleware pipeline, MongoDB integration, deployment |
| Darshan Gowda C | Layer 2A training, feature engineering, ONNX export |
| Santhosh V | Layer 2B training, threat scorer, explainability |
| Srujan H R | Nginx config, Docker setup, dashboard UI, testing |

---

## Base Papers

| Paper | Authors | Contribution used |
|---|---|---|
| *Adaptive Dual-Layer WAF (ADL-WAF)* | Sameh & Selim | Dual-layer ML architecture concept |
| *Detecting Zero-Day Web Attacks with LSTM, GRU, and Stacked Autoencoders* | Babaey & Faragardi (Computers, MDPI 2025) | One-class autoencoder for zero-day detection |

---

*Cambridge Institute of Technology, Bengaluru · Dept. of CSE (IoT & Cyber Security) · B.E. Final Year Project 2025–26*
