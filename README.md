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
- [Training the Models](#training-the-models)
- [Evaluation Targets](#evaluation-targets)
- [Known Issues & Caveats](#known-issues--caveats)
- [Team Responsibilities](#team-responsibilities)

---

## What This Project Is

Traditional WAFs rely on static signature-based rules that only detect known attacks. They fail silently against zero-day exploits, obfuscated payloads, and novel attack patterns.

This project builds a **Hybrid Intelligent WAF** that sits as a reverse proxy in front of a web application and runs every incoming HTTP request through three detection layers. The system combines fast rule-based filtering with ML-based anomaly detection and deep classification.

**The adaptive retraining loop is the core novel contribution** — when the protected server's health metrics spike, the system pulls borderline-scored requests for human review and triggers a retraining cycle with anti-poisoning safeguards.

---

## Architecture

```
Internet → [Nginx] → [FastAPI WAF Middleware] → [Web Application]
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
               Allow      Log+Alert      Block
               (< 30)      (30–70)      (> 70)
                              │
                    Server Health Monitor
                              │
                    Feedback + Re-audit
                              │
                    Adaptive Retraining
```

### Layer 1 — Rule-Based Filter
Regex patterns for SQLi, XSS, LFI, and OS command injection. Rate limiter at 100 req/min per IP. Drops known attacks in < 0.1ms before any ML runs.

### Layer 2A — Anomaly Detector
One-class autoencoder trained **only on normal traffic**. Anything deviating from learned normal behaviour is flagged — this is what enables zero-day detection. Exported to ONNX for ~1–2ms inference.

**Threat score contribution:**
```
L2A contribution = min(50, reconstruction_error × 15)
```

### Layer 2B — Deep Classifier
Bidirectional GRU that runs **only when L2A flags an anomaly**. Classifies into: `normal`, `sqli`, `xss`, `lfi`, `cmdi`, `other_attack`. Exported to ONNX for ~15–20ms inference.

**Threat score contribution:**
```
L2B contribution = attack_confidence × 50   (0 if class = normal)
threat_score     = L2A_contrib + L2B_contrib   (capped at 100)
```

### Threat Score Engine
| Score | Decision | Action |
|---|---|---|
| < 30 | `allow` | Forward to web app |
| 30–70 | `log` | Log + add to human review queue |
| > 70 | `block` | Drop request, return 403 |

### Server Health Monitor + Adaptive Retraining
The monitor pings the protected app's `/health` endpoint every 60 seconds. If error rate exceeds 10%, borderline requests are pulled for re-audit. The retraining cycle includes anti-poisoning safeguards: per-IP caps, L1 re-scan, and minimum sample thresholds.

---

## Tech Stack

| Component | Technology |
|---|---|
| Reverse proxy | Nginx |
| WAF backend | FastAPI + Uvicorn (async Python) |
| Anomaly detector (L2A) | Shallow Autoencoder → ONNX Runtime |
| Deep classifier (L2B) | Bidirectional GRU → ONNX Runtime |
| Database | MongoDB (Motor async driver) |
| Dashboard | Jinja2 SSR + Vanilla JS + Canvas charts |
| Training | PyTorch · scikit-learn · XGBoost |
| Experiment tracking | MLflow |
| Containers | Docker + Docker Compose |
| Datasets | CSIC 2010 · HttpParamsDataset · PayloadBox |

---

## Project Structure

```
waf-ml-project/
│
├── .env                         # local environment overrides (gitignored)
├── .env.example                 # template — copy this to .env
├── docker-compose.yml           # nginx + fastapi + mongodb
├── dummy_app.py                 # lightweight protected app for local dev/testing
├── test_traffic.py              # traffic simulation script
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
│   │   ├── traffic.py           # POST /api/traffic/analyze
│   │   ├── logs.py              # GET /api/logs/recent, /api/logs/threats
│   │   ├── feedback.py          # GET/POST /api/feedback/...
│   │   ├── health.py            # GET /api/health/, /api/health/stats
│   │   └── models.py            # GET/POST /api/models/info, /reload, /history
│   │
│   ├── core/
│   │   ├── config.py            # pydantic-settings, loads from .env
│   │   ├── logging.py           # structured logging setup
│   │   └── exceptions.py        # ModelNotLoadedError, DatabaseError handlers
│   │
│   ├── middleware/
│   │   ├── waf_middleware.py    # main proxy interception + pipeline
│   │   ├── rate_limiter.py      # slowapi limiter
│   │   └── request_parser.py    # extracts url/method/headers/body/ip
│   │
│   ├── models/schemas/
│   │   ├── request.py           # IncomingRequest
│   │   ├── threat.py            # ThreatResult
│   │   ├── log.py               # RequestLog
│   │   └── feedback.py          # FeedbackItem
│   │
│   ├── services/
│   │   ├── layer1_filter.py     # regex rules: sqli/xss/lfi/cmdi
│   │   ├── layer2a_anomaly.py   # ONNX autoencoder inference
│   │   ├── layer2b_deep.py      # ONNX GRU classifier inference
│   │   ├── feature_extractor.py # runtime preprocessing (must match training)
│   │   ├── threat_scorer.py     # 0–100 score + allow/log/block decision
│   │   ├── health_monitor.py    # async health check loop
│   │   ├── feedback_classifier.py # auto-labelling heuristics
│   │   └── adaptive_retrain.py  # anti-poisoning retraining pipeline
│   │
│   ├── db/
│   │   ├── mongodb.py           # Motor async client, index creation
│   │   ├── collections.py       # typed collection accessors
│   │   └── queries.py           # reusable async query functions
│   │
│   ├── templates/               # Jinja2 SSR dashboard templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── logs.html
│   │   ├── threats.html
│   │   ├── feedback.html
│   │   ├── models.html
│   │   └── partials/
│   │       ├── nav.html
│   │       ├── threat_card.html
│   │       └── log_row.html
│   │
│   └── static/
│       ├── css/main.css         # industrial/terminal dark theme
│       └── js/
│           ├── main.js          # nav highlight, stat animations
│           ├── charts.js        # Canvas 2D: sparkline, donut, latency bars
│           └── live_logs.js     # polling /api/logs/recent every 5s
│
└── ml/                          # offline training — NOT deployed in app container
    ├── requirements_train.txt
    ├── feature_engineering/
    │   ├── extractor.py         # extract_features(), to_vector()
    │   ├── tokenizer.py         # CharTokenizer (max_len=512)
    │   └── normalizer.py        # Normalizer wrapping StandardScaler
    ├── layer2a/
    │   ├── candidates/
    │   │   ├── isolation_forest.py
    │   │   └── autoencoder_shallow.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── export_onnx.py
    ├── layer2b/
    │   ├── candidates/
    │   │   ├── xgboost_model.py
    │   │   ├── cnn_1d.py
    │   │   └── gru.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── export_onnx.py
    ├── exported_models/         # ← place trained files here (gitignored)
    │   ├── layer2a_best.onnx
    │   ├── layer2a_best_threshold.txt
    │   ├── layer2b_best.onnx
    │   └── scaler_l2a.pkl
    └── notebooks/
        ├── 01_data_exploration.ipynb
        ├── 02_feature_engineering.ipynb
        ├── 03_layer2a_experiments.ipynb
        ├── 04_layer2b_experiments.ipynb
        ├── 05_model_comparison.ipynb
        └── 06_end_to_end_eval.ipynb
```

---

## Datasets

| Dataset | Use | Records |
|---|---|---|
| CSIC 2010 (Kaggle) | L2A normal training + L2B full | 61,000 HTTP requests |
| HttpParamsDataset (Morzeux) | L2B primary — all 4 attack types | ~12,000 payloads |
| PayloadBox SQLi list | L2B SQLi augmentation | 6,100+ payloads |
| PayloadBox XSS list | L2B XSS augmentation | 7,800+ payloads |
| PayloadBox CMDi list | L2B CMDi augmentation | 3,700+ payloads |
| PayloadBox LFI list | L2B LFI augmentation | 628+ payloads |
| CICIDS 2017 BENIGN | L2A normal traffic pool | 2.8M+ records |

**Class imbalance:** Cap majority classes at 5,000 rows + compute class weights for `CrossEntropyLoss`. SMOTE is not used — interpolating between HTTP payloads produces syntactically invalid text.

---

## Running Locally (Dev)

### Prerequisites

- Python 3.11+
- MongoDB running locally
- Two terminal windows

### Step 1 — Clone and set up environment

```powershell
git clone <repo-url>
cd waf-ml-project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r app/requirements.txt
```

### Step 2 — Create your .env file

Copy the example and edit:

```powershell
copy .env.example .env
```

Your `.env` should contain:

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

SCORE_LOG_THRESHOLD=30
SCORE_BLOCK_THRESHOLD=70
RATE_LIMIT_PER_MIN=100
PROTECTED_APP_URL=http://127.0.0.1:5000
HEALTH_CHECK_INTERVAL_SEC=60
ERROR_RATE_THRESHOLD=0.10
RETRAIN_MIN_SAMPLES=200
```

> **Important:** `MONGO_URI` must be `localhost` (not `mongodb`) when running outside Docker. `PROTECTED_APP_URL` must be `http://127.0.0.1:5000` for local dev.

### Step 3 — Place trained model files

Download the exported models and place them at:

```
ml/exported_models/
├── layer2a_best.onnx
├── layer2a_best_threshold.txt
├── layer2b_best.onnx
└── scaler_l2a.pkl
```

> **sklearn version:** The scaler was pickled with a specific sklearn version. Check the version used during training and match it:
> ```powershell
> pip install scikit-learn==1.6.1   # adjust to match training env
> ```

### Step 4 — Start MongoDB

```powershell
# Verify MongoDB is running
mongosh --eval "db.adminCommand('ping')"
# Should return: { ok: 1 }
```

If not running, start it via Windows Services or MongoDB Compass.

### Step 5 — Start both servers

**Terminal 1 — Protected app (dummy backend):**
```powershell
cd G:\path\to\waf-ml-project
.venv\Scripts\activate
uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

**Terminal 2 — WAF:**
```powershell
cd G:\path\to\waf-ml-project
.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Successful startup looks like:
```
INFO  | waf | Starting WAF-ML v1.0.0
INFO  | waf | MongoDB connected → waf_dev
INFO  | waf | L2A loaded | input=features | threshold=0.08141
INFO  | waf | L2B loaded | input=token_ids | uses_tokens=True
INFO  | waf | All ML models loaded successfully
INFO  | waf | Health monitor started (interval=60s)
INFO  | waf | WAF ready ◈
```

### Step 6 — Open the dashboard

```
http://127.0.0.1:8000/dashboard
```

### Step 7 — Simulate traffic (optional)

```powershell
python test_traffic.py
```

This sends normal, borderline, and attack requests through the WAF proxy at `http://127.0.0.1:8000/proxy/...`. Check the dashboard to see logs, threats, and review queue populate.

---

## Running with Docker

### Prerequisites

- Docker Desktop
- Docker Compose

### Step 1 — Set up environment

```bash
cp .env.example .env
# Edit .env — leave MONGO_URI=mongodb://mongodb:27017 and PROTECTED_APP_URL=http://webapp:5000
# These are the Docker service hostnames, not localhost
```

### Step 2 — Place model files

```
ml/exported_models/
├── layer2a_best.onnx
├── layer2a_best_threshold.txt
├── layer2b_best.onnx
└── scaler_l2a.pkl
```

### Step 3 — Build and start

```bash
docker-compose up --build
```

### Step 4 — Access

| URL | Description |
|---|---|
| `http://localhost/dashboard` | Main dashboard (via Nginx) |
| `http://localhost/proxy/...` | Proxied traffic (goes through WAF) |
| `http://localhost:8000/api/docs` | FastAPI Swagger UI |

### Useful Docker commands

```bash
# View WAF logs
docker-compose logs -f fastapi

# Restart just the WAF (after code changes)
docker-compose restart fastapi

# Stop everything
docker-compose down

# Stop and wipe the MongoDB volume
docker-compose down -v
```

---

## Dashboard Pages

| URL | Page | Description |
|---|---|---|
| `/dashboard` | Overview | 24h stats, attack breakdown, recent threats |
| `/dashboard/logs` | Live Logs | Real-time request log with filter by decision |
| `/dashboard/threats` | Threats | All blocked/flagged events with attack type cards |
| `/dashboard/feedback` | Review Queue | Human labelling interface for borderline requests |
| `/dashboard/models` | Models | ONNX model metadata + hot reload button |
| `/api/docs` | API Docs | Swagger UI for all REST endpoints |

---

## API Reference

### Traffic Analysis

```
POST /api/traffic/analyze
```
Run a single request through the full WAF pipeline.

```json
{
  "url": "/tienda1/publico/buscar.jsp?texto=test",
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

### Models

```
GET  /api/models/info
POST /api/models/reload
GET  /api/models/history
```

### Health

```
GET /api/health/
GET /api/health/stats
```

---

## Training the Models

Training runs offline in Colab or Kaggle notebooks. Run notebooks in order:

```
01_data_exploration.ipynb    → understand dataset distribution
02_feature_engineering.ipynb → build and validate feature pipeline
03_layer2a_experiments.ipynb → train Isolation Forest + Autoencoder, pick winner
04_layer2b_experiments.ipynb → train XGBoost + CNN + GRU, pick winner
05_model_comparison.ipynb    → side-by-side metrics table
06_end_to_end_eval.ipynb     → full pipeline evaluation vs base paper
```

Install training dependencies:
```bash
cd ml
pip install -r requirements_train.txt
```

After training, copy outputs to `ml/exported_models/`:
- `layer2a_best.onnx`
- `layer2a_best_threshold.txt` — single float, the reconstruction error cutoff
- `layer2b_best.onnx`
- `scaler_l2a.pkl` — StandardScaler fitted on normal training data

> **Critical:** The sklearn version used to save `scaler_l2a.pkl` must match the version installed in the runtime environment, or you will get `InconsistentVersionWarning` and incorrect scaling. Pin the version in both environments.

---

## MongoDB Collections

| Collection | Stores |
|---|---|
| `request_logs` | Every proxied request: URL, method, score, decision, latency |
| `threat_events` | Blocked/logged requests with L2A score and L2B confidence |
| `feedback_queue` | Score 30–70 requests pending human review |
| `model_versions` | Hot reload events with threshold and model path |
| `health_snapshots` | Periodic health check results from protected app |
| `retrain_log` | History of retraining triggers with sample counts |

Useful mongosh commands for debugging:

```js
// Connect
mongosh waf_dev

// Count decisions
db.request_logs.countDocuments({decision: "block"})
db.request_logs.countDocuments({decision: "log"})
db.request_logs.countDocuments({decision: "allow"})

// View pending review items
db.feedback_queue.find({verified_label: null}).limit(5)

// Clear bad documents missing url field
db.threat_events.deleteMany({url: {$exists: false}})

// Drop all logs to start fresh
db.request_logs.drop()
db.threat_events.drop()
db.feedback_queue.drop()
```

---

## Evaluation Targets

| Metric | Target | Reference |
|---|---|---|
| L2A detection rate (TPR) | > 95% | Base paper 2 benchmark |
| L2A false positive rate | < 5% | Base paper 2 (0.2% FPR achieved) |
| L2A inference latency | < 2ms | Architecture requirement |
| L2B macro F1 | > 97% | Base paper 1 (99.88% accuracy) |
| L2B per-class F1 (all classes) | > 90% | Ensures no attack type is missed |
| L2B inference latency | < 20ms | Architecture requirement |
| Zero-day detection rate | > 90% | Primary research claim |

---

## Known Issues & Caveats

**Model calibration mismatch**  
If L2A reconstruction errors are very high (3–150 range) for normal traffic, the scaler sklearn version likely differs between training and runtime. Fix: pin `scikit-learn` to the same version used during training in both environments.

**L1 false positives**  
The `&&` and `||` CMDI regex patterns can match legitimate URL-encoded query strings. The current `layer1_filter.py` regexes are intentionally conservative — tighten them if you see too many false positives from normal traffic.

**feedback_queue population**  
Items only appear in the review queue when a request scores 30–70 (`decision=log`). With a miscalibrated model that scores everything high, all requests become `block` and nothing reaches the review queue. Fix the scaler first.

**dummy_app must be running**  
When running locally, `dummy_app.py` must be started separately in its own terminal before sending test traffic. The WAF will return 502 Bad Gateway for any allowed request if it can't reach port 5000.

**`--reload` watches all files**  
Uvicorn's `--reload` flag watches the entire project directory. Saving `test_traffic.py` will restart the WAF server mid-test. Either disable `--reload` during traffic testing, or add `--reload-exclude test_traffic.py` to the uvicorn command.

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
| *Detecting Zero-Day Web Attacks with LSTM, GRU, and Stacked Autoencoders* | Babaey & Faragardi (Computers, MDPI 2025) | One-class autoencoder for zero-day detection; CSIC 2012 benchmark |

---

*Cambridge Institute of Technology, Bengaluru · Dept. of CSE (IoT & Cyber Security) · B.E. Final Year Project 2025–26*