\<div *align*="center">

**# Hybrid Intelligent Web Application Firewall**

**### Zero-Day Attack Detection Using Multi-Layer Machine Learning Architecture**

**\*\*Cambridge Institute of Technology, Bengaluru\*\***  

Department of CSE — IoT and Cyber Security including Blockchain  

Final Year B.E Project — 2025–26

**---**

\| | |

\|---|---|

\| **\*\*Team\*\*** | Keerthi Vasan P · Darshan Gowda C · Santhosh V · Srujan H R |

\| **\*\*USNs\*\*** | 1CD23IC029 · 1CD23IC013 · 1CD23IC049 · 1CD23IC055 |

\| **\*\*Batch\*\*** | 5 |

\| **\*\*Phase 1\*\*** | Jul – Nov 2025 |

\| **\*\*Phase 2\*\*** | Jan – May 2026 |

\</div>

**---**

**## Table of Contents**

\- [What This Project Is]\(#what-this-project-is)

\- [Architecture]\(#architecture)

\- [Tech Stack]\(#tech-stack)

\- [Project Structure]\(#project-structure)

\- [Datasets]\(#datasets)

\- [Running Locally (Dev)]\(#running-locally-dev)

\- [Running with Docker]\(#running-with-docker)

\- [Dashboard Pages]\(#dashboard-pages)

\- [API Reference]\(#api-reference)

\- [Training the Models]\(#training-the-models)

\- [Evaluation Targets]\(#evaluation-targets)

\- [Known Issues & Caveats]\(#known-issues--caveats)

\- [Team Responsibilities]\(#team-responsibilities)

**---**

**## What This Project Is**

Traditional WAFs rely on static signature-based rules that only detect known attacks. They fail silently against zero-day exploits, obfuscated payloads, and novel attack patterns.

This project builds a **\*\*Hybrid Intelligent WAF\*\*** that sits as a reverse proxy in front of a web application and runs every incoming HTTP request through three detection layers. The system combines fast rule-based filtering with ML-based anomaly detection and deep classification.

**\*\*The adaptive retraining loop is the core novel contribution\*\*** — when the protected server's health metrics spike, the system pulls borderline-scored requests for human review and triggers a retraining cycle with anti-poisoning safeguards.

**---**

**## Architecture**

\`\`\`

Internet → [Nginx] → [FastAPI WAF Middleware] → [Web Application]

                              │

                 ┌────────────┼────────────┐

                 ▼            ▼            ▼

              Layer 1      Layer 2A     Layer 2B

           Rule Engine    Anomaly       Deep

           (Regex/Rate)   Detector    Classifier

                 │            │            │

                 └────────────┴────────────┘

                              │

                     Threat Score Engine

                          (0–100)

                              │

                 ┌────────────┼────────────┐

                 ▼            ▼            ▼

               Allow      Log+Alert      Block

               (< 30)      (30–70)      (> 70)

                              │

                    Server Health Monitor

                              │

                    Feedback + Re-audit

                              │

                    Adaptive Retraining

\`\`\`

**### Layer 1 — Rule-Based Filter**

Regex patterns for SQLi, XSS, LFI, and OS command injection. Rate limiter at 100 req/min per IP. Drops known attacks in < 0.1ms before any ML runs.

**### Layer 2A — Anomaly Detector**

One-class autoencoder trained **\*\*only on normal traffic\*\***. Anything deviating from learned normal behaviour is flagged — this is what enables zero-day detection. Exported to ONNX for \~1–2ms inference.

**\*\*Threat score contribution:\*\***

\`\`\`

L2A contribution = min(50, reconstruction\_error × 15)

\`\`\`

**### Layer 2B — Deep Classifier**

Bidirectional GRU that runs **\*\*only when L2A flags an anomaly\*\***. Classifies into: \`normal\`, \`sqli\`, \`xss\`, \`lfi\`, \`cmdi\`, \`other\_attack\`. Exported to ONNX for \~15–20ms inference.

**\*\*Threat score contribution:\*\***

\`\`\`

L2B contribution = attack\_confidence × 50   (0 if class = normal)

threat\_score     = L2A\_contrib + L2B\_contrib   (capped at 100)

\`\`\`

**### Threat Score Engine**

\| Score | Decision | Action |

\|---|---|---|

\| < 30 | \`allow\` | Forward to web app |

\| 30–70 | \`log\` | Log + add to human review queue |

\| > 70 | \`block\` | Drop request, return 403 |

**### Server Health Monitor + Adaptive Retraining**

The monitor pings the protected app's \`/health\` endpoint every 60 seconds. If error rate exceeds 10%, borderline requests are pulled for re-audit. The retraining cycle includes anti-poisoning safeguards: per-IP caps, L1 re-scan, and minimum sample thresholds.

**---**

**## Tech Stack**

\| Component | Technology |

\|---|---|

\| Reverse proxy | Nginx |

\| WAF backend | FastAPI + Uvicorn (async Python) |

\| Anomaly detector (L2A) | Shallow Autoencoder → ONNX Runtime |

\| Deep classifier (L2B) | Bidirectional GRU → ONNX Runtime |

\| Database | MongoDB (Motor async driver) |

\| Dashboard | Jinja2 SSR + Vanilla JS + Canvas charts |

\| Training | PyTorch · scikit-learn · XGBoost |

\| Experiment tracking | MLflow |

\| Containers | Docker + Docker Compose |

\| Datasets | CSIC 2010 · HttpParamsDataset · PayloadBox |

**---**

**## Project Structure**

\`\`\`

waf-ml-project/

│

├── .env                         # local environment overrides (gitignored)

├── .env.example                 # template — copy this to .env

├── docker-compose.yml           # nginx + fastapi + mongodb

├── dummy\_app.py                 # lightweight protected app for local dev/testing

├── test\_traffic.py              # traffic simulation script

├── README.md

│

├── nginx/

│   ├── Dockerfile

│   ├── nginx.conf

│   └── conf.d/waf.conf

│

├── app/                         # FastAPI WAF application

│   ├── Dockerfile

│   ├── requirements.txt

│   ├── main.py                  # lifespan: DB + model loading, router registration

│   │

│   ├── api/routes/

│   │   ├── dashboard.py         # SSR pages: /dashboard, /logs, /threats, /feedback, /models

│   │   ├── traffic.py           # POST /api/traffic/analyze

│   │   ├── logs.py              # GET /api/logs/recent, /api/logs/threats

│   │   ├── feedback.py          # GET/POST /api/feedback/...

│   │   ├── health.py            # GET /api/health/, /api/health/stats

│   │   └── models.py            # GET/POST /api/models/info, /reload, /history

│   │

│   ├── core/

│   │   ├── config.py            # pydantic-settings, loads from .env

│   │   ├── logging.py           # structured logging setup

│   │   └── exceptions.py        # ModelNotLoadedError, DatabaseError handlers

│   │

│   ├── middleware/

│   │   ├── waf\_middleware.py    # main proxy interception + pipeline

│   │   ├── rate\_limiter.py      # slowapi limiter

│   │   └── request\_parser.py    # extracts url/method/headers/body/ip

│   │

│   ├── models/schemas/

│   │   ├── request.py           # IncomingRequest

│   │   ├── threat.py            # ThreatResult

│   │   ├── log.py               # RequestLog

│   │   └── feedback.py          # FeedbackItem

│   │

│   ├── services/

│   │   ├── layer1\_filter.py     # regex rules: sqli/xss/lfi/cmdi

│   │   ├── layer2a\_anomaly.py   # ONNX autoencoder inference

│   │   ├── layer2b\_deep.py      # ONNX GRU classifier inference

│   │   ├── feature\_extractor.py # runtime preprocessing (must match training)

│   │   ├── threat\_scorer.py     # 0–100 score + allow/log/block decision

│   │   ├── health\_monitor.py    # async health check loop

│   │   ├── feedback\_classifier.py # auto-labelling heuristics

│   │   └── adaptive\_retrain.py  # anti-poisoning retraining pipeline

│   │

│   ├── db/

│   │   ├── mongodb.py           # Motor async client, index creation

│   │   ├── collections.py       # typed collection accessors

│   │   └── queries.py           # reusable async query functions

│   │

│   ├── templates/               # Jinja2 SSR dashboard templates

│   │   ├── base.html

│   │   ├── dashboard.html

│   │   ├── logs.html

│   │   ├── threats.html

│   │   ├── feedback.html

│   │   ├── models.html

│   │   └── partials/

│   │       ├── nav.html

│   │       ├── threat\_card.html

│   │       └── log\_row\.html

│   │

│   └── static/

│       ├── css/main.css         # industrial/terminal dark theme

│       └── js/

│           ├── main.js          # nav highlight, stat animations

│           ├── charts.js        # Canvas 2D: sparkline, donut, latency bars

│           └── live\_logs.js     # polling /api/logs/recent every 5s

│

└── ml/                          # offline training — NOT deployed in app container

    ├── requirements\_train.txt

    ├── feature\_engineering/

    │   ├── extractor.py         # extract\_features(), to\_vector()

    │   ├── tokenizer.py         # CharTokenizer (max\_len=512)

    │   └── normalizer.py        # Normalizer wrapping StandardScaler

    ├── layer2a/

    │   ├── candidates/

    │   │   ├── isolation\_forest.py

    │   │   └── autoencoder\_shallow\.py

    │   ├── train.py

    │   ├── evaluate.py

    │   └── export\_onnx.py

    ├── layer2b/

    │   ├── candidates/

    │   │   ├── xgboost\_model.py

    │   │   ├── cnn\_1d.py

    │   │   └── gru.py

    │   ├── train.py

    │   ├── evaluate.py

    │   └── export\_onnx.py

    ├── exported\_models/         # ← place trained files here (gitignored)

    │   ├── layer2a\_best.onnx

    │   ├── layer2a\_best\_threshold.txt

    │   ├── layer2b\_best.onnx

    │   └── scaler\_l2a.pkl

    └── notebooks/

        ├── 01\_data\_exploration.ipynb

        ├── 02\_feature\_engineering.ipynb

        ├── 03\_layer2a\_experiments.ipynb

        ├── 04\_layer2b\_experiments.ipynb

        ├── 05\_model\_comparison.ipynb

        └── 06\_end\_to\_end\_eval.ipynb

\`\`\`

**---**

**## Datasets**

\| Dataset | Use | Records |

\|---|---|---|

\| CSIC 2010 (Kaggle) | L2A normal training + L2B full | 61,000 HTTP requests |

\| HttpParamsDataset (Morzeux) | L2B primary — all 4 attack types | \~12,000 payloads |

\| PayloadBox SQLi list | L2B SQLi augmentation | 6,100+ payloads |

\| PayloadBox XSS list | L2B XSS augmentation | 7,800+ payloads |

\| PayloadBox CMDi list | L2B CMDi augmentation | 3,700+ payloads |

\| PayloadBox LFI list | L2B LFI augmentation | 628+ payloads |

\| CICIDS 2017 BENIGN | L2A normal traffic pool | 2.8M+ records |

**\*\*Class imbalance:\*\*** Cap majority classes at 5,000 rows + compute class weights for \`CrossEntropyLoss\`. SMOTE is not used — interpolating between HTTP payloads produces syntactically invalid text.

**---**

**## Running Locally (Dev)

The current development and live-demo setup runs **without Docker**.

### Local architecture

| Component | Address | Purpose |
|---|---|---|
| WAF | `http://127.0.0.1:8000` | Receives, analyzes, logs/blocks, and forwards traffic |
| Protected app | `http://127.0.0.1:5000` | Dummy FastAPI backend |
| MongoDB | `mongodb://localhost:27017` | Request logs, threats, feedback, health, and model data |

**Important:** For a live demo, send traffic to **port 8000**. Port 5000 is the backend and bypasses the WAF.

### Prerequisites

- Python 3.11+
- MongoDB running locally
- Two terminal windows
- Exported L2A/L2B model files

### Step 1 — Clone and set up environment

```powershell
git clone <repo-url>
cd waf-ml-project

python -m venv .venv
.venv\Scripts\activate

pip install -r app/requirements.txt
```

### Step 2 — Create your `.env` file

```powershell
copy .env.example .env
```

For local development, use:

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

> **Important:** When running outside Docker, `MONGO_URI` must use `localhost` and `PROTECTED_APP_URL` must use `http://127.0.0.1:5000`. Do not use the Docker hostname `webapp:5000`.

### Step 3 — Place trained model files

```text
ml/exported_models/

├── layer2a_best.onnx
├── layer2a_best_threshold.txt
├── layer2b_best.onnx
└── scaler_l2a.pkl
```

> **sklearn version:** The scaler was pickled with a specific sklearn version. Match the runtime version to the training environment.

```powershell
pip install scikit-learn==1.6.1
```

Adjust the version if your training environment used a different version.

### Step 4 — Start MongoDB

```powershell
mongosh --eval "db.adminCommand('ping')"
```

Expected result:

```text
{ ok: 1 }
```

If MongoDB is not running, start it through Windows Services or MongoDB Compass.

### Step 5 — Start the protected application

**Terminal 1:**

```powershell
cd G:\path\to\waf-ml-project
.venv\Scripts\activate

uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

The backend should show:

```text
Uvicorn running on http://127.0.0.1:5000
```

### Step 6 — Start the WAF

**Terminal 2:**

```powershell
cd G:\path\to\waf-ml-project
.venv\Scripts\activate

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Successful startup should include:

```text
INFO | waf | Starting WAF-ML v1.0.0
INFO | waf | MongoDB connected → waf_dev
INFO | waf | L2A loaded
INFO | waf | L2B loaded
INFO | waf | All ML models loaded successfully
INFO | waf | Health monitor started
INFO | waf | WAF ready ◈
```

### Step 7 — Use clean WAF URLs

The WAF now intercepts normal application paths directly. There is **no `/proxy` prefix** in the live-demo URL.

Example:

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

The request flow is:

```text
Browser
   ↓
127.0.0.1:8000/api/products
   ↓
WAF Middleware
   ↓
L1 → L2A → L2B → Threat Scorer
   ↓
ALLOW / LOG / BLOCK
   ↓
127.0.0.1:5000/api/products
   ↓
dummy_app.py
```

The WAF analyzes only the **path + query**, for example:

```text
/api/products?category=electronics&page=1
```

It does not pass the absolute browser URL to the ML pipeline.

### Step 8 — Test live traffic

Normal request:

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

Search request:

```text
http://127.0.0.1:8000/api/products/search?q=laptop
```

Another normal request:

```text
http://127.0.0.1:8000/search?q=products
```

Attack test:

```text
http://127.0.0.1:8000/api/products/search?q=1+OR+1=1
```

The attack request should be analyzed by the WAF before it can reach `dummy_app.py`.

### Step 9 — Run the automated traffic test

```powershell
python test_traffic.py
```

`test_traffic.py` should target:

```text
http://127.0.0.1:8000
```

rather than port `5000`.

### Step 10 — Open the dashboard

```text
http://127.0.0.1:8000/dashboard
```

Useful pages:

```text
http://127.0.0.1:8000/dashboard
http://127.0.0.1:8000/dashboard/logs
http://127.0.0.1:8000/dashboard/threats
http://127.0.0.1:8000/dashboard/feedback
http://127.0.0.1:8000/dashboard/models
```

### Important live-demo rule

Do **not** use:

```text
http://127.0.0.1:5000/...
```

for WAF testing.

Port `5000` is the protected backend. Requests sent directly there will return the backend response but will **not create WAF request logs or pass through L1/L2A/L2B**.

Always use:

```text
http://127.0.0.1:8000/...
```

for traffic that should be protected.

## Running with Docker**

**### Prerequisites**

\- Docker Desktop

\- Docker Compose

**### Step 1 — Set up environment**

\`\`\`bash

cp .env.example .env

\# Edit .env — leave MONGO\_URI=mongodb://mongodb:27017 and PROTECTED\_APP\_URL=http\://webapp:5000

\# These are the Docker service hostnames, not localhost

\`\`\`

**### Step 2 — Place model files**

\`\`\`

ml/exported\_models/

├── layer2a\_best.onnx

├── layer2a\_best\_threshold.txt

├── layer2b\_best.onnx

└── scaler\_l2a.pkl

\`\`\`

**### Step 3 — Build and start**

\`\`\`bash

docker-compose up --build

\`\`\`

**### Step 4 — Access**

\| URL | Description |

\|---|---|

\| \`http\://localhost/dashboard\` | Main dashboard (via Nginx) |

\| \`http\://localhost/proxy/...\` | Proxied traffic (goes through WAF) |

\| \`http\://localhost:8000/api/docs\` | FastAPI Swagger UI |

**### Useful Docker commands**

\`\`\`bash

\# View WAF logs

docker-compose logs -f fastapi

\# Restart just the WAF (after code changes)

docker-compose restart fastapi

\# Stop everything

docker-compose down

\# Stop and wipe the MongoDB volume

docker-compose down -v

\`\`\`

**---**

**## Dashboard Pages**

\| URL | Page | Description |

\|---|---|---|

\| \`/dashboard\` | Overview | 24h stats, attack breakdown, recent threats |

\| \`/dashboard/logs\` | Live Logs | Real-time request log with filter by decision |

\| \`/dashboard/threats\` | Threats | All blocked/flagged events with attack type cards |

\| \`/dashboard/feedback\` | Review Queue | Human labelling interface for borderline requests |

\| \`/dashboard/models\` | Models | ONNX model metadata + hot reload button |

\| \`/api/docs\` | API Docs | Swagger UI for all REST endpoints |

**---**

**## API Reference**

**### Traffic Analysis**

\`\`\`

POST /api/traffic/analyze

\`\`\`

Run a single request through the full WAF pipeline.

\`\`\`json

{

  "url": "/tienda1/publico/buscar.jsp?texto=test",

  "method": "GET",

  "headers": {},

  "body": "",

  "ip": "1.2.3.4"

}

\`\`\`

Response:

\`\`\`json

{

  "request\_id": "uuid",

  "decision": "allow",

  "score": 12,

  "label": "normal",

  "layer": "L2A",

  "l2a\_score": 0.04231,

  "latency\_ms": 4.2

}

\`\`\`

**### Logs**

\`\`\`

GET /api/logs/recent?limit=100&decision=block

GET /api/logs/threats?limit=50

\`\`\`

**### Feedback / Review**

\`\`\`

GET  /api/feedback/pending?limit=100

POST /api/feedback/review/{request\_id}

     Body: { "verified\_label": "sqli", "is\_poisoning": false }

POST /api/feedback/trigger-retrain

\`\`\`

Valid labels: \`normal\`, \`sqli\`, \`xss\`, \`lfi\`, \`other\_attack\`, \`false\_positive\`

**### Models**

\`\`\`

GET  /api/models/info

POST /api/models/reload

GET  /api/models/history

\`\`\`

**### Health**

\`\`\`

GET /api/health/

GET /api/health/stats

\`\`\`

**---**

**## Training the Models**

Training runs offline in Colab or Kaggle notebooks. Run notebooks in order:

\`\`\`

01\_data\_exploration.ipynb    → understand dataset distribution

02\_feature\_engineering.ipynb → build and validate feature pipeline

03\_layer2a\_experiments.ipynb → train Isolation Forest + Autoencoder, pick winner

04\_layer2b\_experiments.ipynb → train XGBoost + CNN + GRU, pick winner

05\_model\_comparison.ipynb    → side-by-side metrics table

06\_end\_to\_end\_eval.ipynb     → full pipeline evaluation vs base paper

\`\`\`

Install training dependencies:

\`\`\`bash

cd ml

pip install -r requirements\_train.txt

\`\`\`

After training, copy outputs to \`ml/exported\_models/\`:

\- \`layer2a\_best.onnx\`

\- \`layer2a\_best\_threshold.txt\` — single float, the reconstruction error cutoff

\- \`layer2b\_best.onnx\`

\- \`scaler\_l2a.pkl\` — StandardScaler fitted on normal training data

\> **\*\*Critical:\*\*** The sklearn version used to save \`scaler\_l2a.pkl\` must match the version installed in the runtime environment, or you will get \`InconsistentVersionWarning\` and incorrect scaling. Pin the version in both environments.

**---**

**## MongoDB Collections**

\| Collection | Stores |

\|---|---|

\| \`request\_logs\` | Every proxied request: URL, method, score, decision, latency |

\| \`threat\_events\` | Blocked/logged requests with L2A score and L2B confidence |

\| \`feedback\_queue\` | Score 30–70 requests pending human review |

\| \`model\_versions\` | Hot reload events with threshold and model path |

\| \`health\_snapshots\` | Periodic health check results from protected app |

\| \`retrain\_log\` | History of retraining triggers with sample counts |

Useful mongosh commands for debugging:

\`\`\`js

// Connect

mongosh waf\_dev

// Count decisions

db.request\_logs.countDocuments({decision: "block"})

db.request\_logs.countDocuments({decision: "log"})

db.request\_logs.countDocuments({decision: "allow"})

// View pending review items

db.feedback\_queue.find({verified\_label: null}).limit(5)

// Clear bad documents missing url field

db.threat\_events.deleteMany({url: {$exists: false}})

// Drop all logs to start fresh

db.request\_logs.drop()

db.threat\_events.drop()

db.feedback\_queue.drop()

\`\`\`

**---**

**## Evaluation Targets**

\| Metric | Target | Reference |

\|---|---|---|

\| L2A detection rate (TPR) | > 95% | Base paper 2 benchmark |

\| L2A false positive rate | < 5% | Base paper 2 (0.2% FPR achieved) |

\| L2A inference latency | < 2ms | Architecture requirement |

\| L2B macro F1 | > 97% | Base paper 1 (99.88% accuracy) |

\| L2B per-class F1 (all classes) | > 90% | Ensures no attack type is missed |

\| L2B inference latency | < 20ms | Architecture requirement |

\| Zero-day detection rate | > 90% | Primary research claim |

**---**

**## Live Demo Troubleshooting

### `{"detail":"Not Found"}` on port 8000

Make sure the requested path is supported by the protected backend and that the WAF middleware is intercepting application traffic.

Use:

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

Do not use the old `/proxy/...` format in the current local setup.

### Browser returns backend JSON but WAF logs are empty

You probably opened port `5000` directly.

Wrong:

```text
http://127.0.0.1:5000/api/products
```

Correct:

```text
http://127.0.0.1:8000/api/products
```

### WAF returns `502 Bad Gateway`

The protected application is probably not running.

Start:

```powershell
uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

### Normal traffic is unexpectedly blocked

Check the WAF terminal for:

```text
WAF DEBUG
WAF DECISION
l2a_score
label
confidence
```

A very high L2A score on clearly normal traffic can indicate model/scaler calibration problems. Do not simply increase thresholds without evaluating the model and feature pipeline.

## Known Issues & Caveats**

**\*\*Model calibration mismatch\*\***  

If L2A reconstruction errors are very high (3–150 range) for normal traffic, the scaler sklearn version likely differs between training and runtime. Fix: pin \`scikit-learn\` to the same version used during training in both environments.

**\*\*L1 false positives\*\***  

The \`&&\` and \`||\` CMDI regex patterns can match legitimate URL-encoded query strings. The current \`layer1\_filter.py\` regexes are intentionally conservative — tighten them if you see too many false positives from normal traffic.

**\*\*feedback\_queue population\*\***  

Items only appear in the review queue when a request scores 30–70 (\`decision=log\`). With a miscalibrated model that scores everything high, all requests become \`block\` and nothing reaches the review queue. Fix the scaler first.

**\*\*dummy\_app must be running\*\***  

When running locally, \`dummy\_app.py\` must be started separately in its own terminal before sending test traffic. The WAF will return 502 Bad Gateway for any allowed request if it can't reach port 5000.

**\*\*\`--reload\` watches all files\*\***  

Uvicorn's \`--reload\` flag watches the entire project directory. Saving \`test\_traffic.py\` will restart the WAF server mid-test. Either disable \`--reload\` during traffic testing, or add \`--reload-exclude test\_traffic.py\` to the uvicorn command.

**---**

**## Team Responsibilities**

\| Member | Primary area |

\|---|---|

\| Keerthi Vasan P | FastAPI backend, middleware pipeline, MongoDB integration, deployment |

\| Darshan Gowda C | Layer 2A training, feature engineering, ONNX export |

\| Santhosh V | Layer 2B training, threat scorer, explainability |

\| Srujan H R | Nginx config, Docker setup, dashboard UI, testing |

**---**

**## Base Papers**

\| Paper | Authors | Contribution used |

\|---|---|---|

\| *\*Adaptive Dual-Layer WAF (ADL-WAF)\** | Sameh & Selim | Dual-layer ML architecture concept |

\| *\*Detecting Zero-Day Web Attacks with LSTM, GRU, and Stacked Autoencoders\** | Babaey & Faragardi (Computers, MDPI 2025) | One-class autoencoder for zero-day detection; CSIC 2012 benchmark |

**---**

*\*Cambridge Institute of Technology, Bengaluru · Dept. of CSE (IoT & Cyber Security) · B.E. Final Year Project 2025–26\**