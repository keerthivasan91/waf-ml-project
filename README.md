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

## What this project is

Traditional WAFs rely on static signature-based rules that only detect known attacks. They fail silently against zero-day exploits, obfuscated payloads, and novel attack patterns.

This project builds a **Hybrid Intelligent WAF** that sits as a reverse proxy in front of a web application and runs every incoming HTTP request through three detection layers:

```
Internet → Nginx → FastAPI WAF Middleware → Web Application
                        │
              ┌─────────┼──────────┐
              ▼         ▼          ▼
           Layer 1   Layer 2A   Layer 2B
           Rules    Anomaly     Deep
           Engine   Detector   Analyser
              │         │          │
              └─────────┴──────────┘
                        │
                 Threat Score Engine
                  (0–100 scoring)
                        │
              ┌─────────┼──────────┐
              ▼         ▼          ▼
            Allow    Log+Alert   Block
            (<30)    (30–70)     (>70)
                        │
               Server Health Monitor
                        │
               Feedback + Re-audit
                        │
               Adaptive Retraining
```

**The adaptive retraining loop is the core novel contribution** — neither base paper implemented this. When the protected server's health metrics spike (high error rate, latency, CPU), the system pulls the last 5 minutes of allowed traffic, flags borderline-scored requests for human review, and triggers a retraining cycle with anti-poisoning safeguards.

---

## Base papers

| Paper | Authors | Key contribution used |
|---|---|---|
| *Adaptive Dual-Layer WAF (ADL-WAF)* | Sameh & Selim | Dual-layer ML architecture concept |
| *Detecting Zero-Day Web Attacks with LSTM, GRU, and Stacked Autoencoders* | Babaey & Faragardi (Computers, MDPI 2025) | One-class autoencoder for zero-day detection; CSIC 2012 benchmark |

---

## Architecture layers

### Layer 1 — Classic filter
Rule-based engine with regex patterns for SQLi, XSS, LFI, and OS command injection. Rate limiter (sliding window, 100 req/min per IP). Drops known attacks in < 0.1ms before any ML runs.

### Layer 2A — Anomaly detector
One-class model trained **only on normal traffic**. No attack labels required. Anything that deviates from learned normal behaviour is flagged as anomalous. This is what enables zero-day detection — the model has never seen the attack before but it doesn't look normal.

Two candidates trained and compared:
- Isolation Forest (sklearn)
- Shallow Autoencoder — `Input(20) → 64 → 32 → 16 → 32 → 64 → Output(20)` (PyTorch)

Selection criterion: lowest FPR ≤ 5%, then highest TPR. Winner exported to ONNX.

### Layer 2B — Deep analysis
Multi-class classifier that only runs when Layer 2A flags an anomaly. Classifies the attack type: SQLi, XSS, LFI, or OS Command Injection. Provides confidence scores used by the threat scorer.

Three candidates trained and compared:
- XGBoost (numeric features, no GPU, baseline)
- 1D CNN with multi-scale kernels [3, 5, 7] (character sequences)
- Bidirectional GRU with Bahdanau attention (character sequences) — same architecture as base paper 2

Selection criterion: highest macro F1 with per-class F1 ≥ 0.90 on all attack classes.

### Threat Score Engine
Combines L2A reconstruction error and L2B attack confidence into a 0–100 score:

```
L2A contribution = min(50, recon_error × 15)
L2B contribution = attack_confidence × 50   (0 if class=normal)
threat_score     = L2A_contrib + L2B_contrib
```

- Score < 30  → **Allow** (forward to web app)
- Score 30–70 → **Log + Alert** (human review queue)
- Score > 70  → **Block** (drop request, IP ban)

### Server Health Monitor + Adaptive Retraining
The protected web app posts health metrics (error rate, latency p99, CPU%) to `/api/health`. If any metric breaches a threshold, the system:
1. Pulls the last 5 minutes of allowed traffic from MongoDB
2. Flags requests with score > 15 (borderline-allowed) for re-audit
3. Runs anti-poisoning checks (rate per IP, Layer 1 re-scan, score sanity)
4. Queues verified samples for the next retraining cycle

---

## Tech stack

| Component | Technology | Why |
|---|---|---|
| Reverse proxy | Nginx | Industry standard, TLS termination |
| Backend / WAF core | FastAPI (Python) | Async, Pydantic, native Jinja2 SSR |
| Layer 1 | Python rule engine | Custom regex + rate limiter |
| Layer 2A | Isolation Forest / Autoencoder → ONNX | One-class, ~1–2ms inference |
| Layer 2B | XGBoost / CNN / GRU → ONNX | Multi-class, ~15–20ms inference |
| Database | MongoDB (Motor async) | Document-shaped logs, flexible schema |
| Frontend | HTML + Jinja2 SSR | No framework overhead |
| Containers | Docker + Docker Compose | nginx + fastapi + mongodb |
| Training | PyTorch · scikit-learn · XGBoost | Offline, Colab/Kaggle notebooks |
| Model export | ONNX + ONNX Runtime | 3–5× faster than native PyTorch |
| Experiment tracking | MLflow | Compare runs, log winner |
| Datasets | CSIC 2010 · HttpParamsDataset · PayloadBox | Covers all 4 attack types |

---

## Project structure

```
waf-ml-project/
│
├── docker-compose.yml           # nginx + fastapi + mongodb
├── .env.example                 # environment variable template
├── README.md
│
├── nginx/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── conf.d/waf.conf          # proxy_pass → fastapi:8000
│
├── app/                         # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # lifespan: load ONNX + connect DB
│   ├── api/routes/              # traffic · logs · feedback · health · dashboard
│   ├── core/                    # config · security · logging · exceptions
│   ├── middleware/              # waf_middleware · rate_limiter · request_parser
│   ├── models/schemas/          # request · threat · feedback · log
│   ├── services/                # layer1 · layer2a · layer2b · scorer · health · retrain
│   ├── db/                      # mongodb · collections · queries
│   ├── templates/               # Jinja2 SSR dashboard pages
│   └── static/                  # CSS + vanilla JS
│
└── ml/                          # offline training — NOT in app container
    ├── requirements_train.txt
    ├── data/
    │   ├── raw/                 # original dataset files (gitignored)
    │   ├── processed/           # parsed CSVs, class weights
    │   └── splits/              # train/val/test .npy arrays
    ├── feature_engineering/     # extractor · tokenizer · normalizer
    ├── layer2a/
    │   ├── candidates/
    │   │   ├── isolation_forest.py
    │   │   └── autoencoder_shallow.py
    │   ├── train.py · evaluate.py · export_onnx.py
    ├── layer2b/
    │   ├── candidates/
    │   │   ├── xgboost_model.py
    │   │   ├── cnn_1d.py
    │   │   └── gru.py
    │   ├── train.py · evaluate.py · export_onnx.py
    ├── evaluation/              # metrics · compare_models · benchmark
    ├── exported_models/         # *.onnx + scaler + threshold (gitignored)
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

| Dataset | Use | Records | Source |
|---|---|---|---|
| CSIC 2010 (Kaggle mirror) | L2A normal training + L2B full | 61,000 HTTP requests | `ispangler/csic-2010-web-application-attacks` |
| HttpParamsDataset (Morzeux) | L2B primary — all 4 attack types | ~12,000 payloads | `github.com/Morzeux/HttpParamsDataset` |
| PayloadBox SQLi list | L2B SQLi augmentation | 6,100+ payloads | `payloadbox/sql-injection-payload-list` |
| PayloadBox XSS list | L2B XSS augmentation | 7,800+ payloads | `payloadbox/xss-payload-list` |
| PayloadBox CMDi list | L2B CMDi augmentation | 3,700+ payloads | `payloadbox/command-injection-payload-list` |
| PayloadBox LFI list | L2B LFI augmentation | 628+ payloads | `payloadbox/rfi-lfi-payload-list` |
| CICIDS 2017 (BENIGN only) | L2A normal traffic pool | 2.8M+ records | `chethuhn/network-intrusion-dataset` |
| WAF-A-MoLE | Zero-day adversarial testing only | Generative | `AvalZ/WAF-A-MoLE` |
| OWASP ModSecurity 30-day | Adaptive retraining demo | 30-day production logs | `zenodo.org/records/11075382` |

### Class imbalance handling

Raw class distribution before balancing (approximate):

| Class | Raw count | After cap (5,000) | Weight |
|---|---|---|---|
| normal | ~72,000 | 5,000 | 0.20 |
| sqli | ~17,000 | 5,000 | 0.20 |
| xss | ~8,300 | 5,000 | 0.20 |
| lfi | ~918 | 918 | 1.09 |
| other_attack | ~3,789 | 3,789 | 0.26 |

Strategy: cap majority classes at 5,000 rows + compute class weights for `CrossEntropyLoss`. SMOTE is not used — interpolating between HTTP payloads produces syntactically invalid text that degrades model quality.

---

## MongoDB collections

| Collection | Stores |
|---|---|
| `request_logs` | every request: URL, method, L1/L2A/L2B result, score, decision, latency |
| `threat_events` | blocked/alerted requests with score breakdown and explanation |
| `feedback_queue` | score 30–70 requests pending human review |
| `model_versions` | trained model metadata: accuracy, FPR, export path, active flag |
| `health_snapshots` | periodic server metrics: error rate, latency, CPU, memory |
| `retrain_log` | history of retraining runs: trigger, samples used, delta accuracy |

---

## Getting started

```bash
# 1. Clone the repo
git clone <repo-url>
cd waf-ml-project

# 2. Copy env template and edit if needed
cp .env.example .env

# 3. Place trained ONNX models (download from shared Drive after training)
# ml/exported_models/layer2a_best.onnx
# ml/exported_models/layer2a_best_threshold.txt
# ml/exported_models/layer2b_best.onnx
# ml/exported_models/scaler_l2a.pkl

# 4. Start all services
docker-compose up --build

# Dashboard → http://localhost/dashboard
```

Training (run notebooks in order on Colab/Kaggle):

```bash
cd ml
pip install -r requirements_train.txt
jupyter notebook
# Run: 01 → 02 → 03 → 04 → 05 → 06
```

Tests:

```bash
pytest tests/ -v
```

---

## Objectives and TODO

### Phase 1 — Odd semester 2025 (Jul – Nov)

- [ ] Literature review — both base papers + 5 related works
- [ ] Download and preprocess CSIC 2010 dataset (notebook 01)
- [ ] Build feature extraction pipeline — 20 numeric features (notebook 02)
- [ ] Train Layer 2A candidates; export winner to ONNX (notebook 03)
- [ ] Validate L2A: FPR < 5%, TPR > 95%, latency < 2ms
- [ ] Implement Layer 1 rule engine with SQLi, XSS, LFI, CMDi rules
- [ ] Submit synopsis document
- [ ] **Phase-1 review demo**: Layer 1 + Layer 2A working end-to-end

### Phase 2 — Even semester 2026 (Jan – May)

- [ ] Train Layer 2B candidates; export winner to ONNX (notebook 04)
- [ ] Validate L2B: macro F1 > 97%, per-class F1 > 90%, latency < 20ms
- [ ] Wire full pipeline: Nginx → FastAPI → L1 → L2A → L2B → threat scorer
- [ ] Set up Docker Compose with all three services
- [ ] Implement MongoDB logging across all 6 collections
- [ ] Build Threat Score Engine with 0–100 scoring and explanations
- [ ] Add explainability output (GRU attention map or SHAP for XGBoost)
- [ ] Implement Server Health Monitor
- [ ] Build Feedback Classification with anti-poisoning safeguards
- [ ] Implement Adaptive Retraining pipeline
- [ ] Build SSR dashboard: live logs, threat timeline, model metrics, review queue
- [ ] Run full evaluation vs base paper benchmarks on CSIC 2010
- [ ] Write final project report
- [ ] **Final viva demo**: live attack simulation against running system

### Stretch goals

- [ ] SHAP-based per-request explanation page in dashboard
- [ ] Email/webhook alerting for score > 70 events
- [ ] CICIDS 2018 dataset for broader attack coverage

---

## Evaluation targets

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

## Key design decisions

**Why one-class learning for L2A?** Training only on normal traffic means the model generalises to any deviation — including zero-day attacks never seen before. Core insight from base paper 2.

**Why ONNX Runtime for inference?** 3–5× faster than native PyTorch. Both models load once at FastAPI startup as singleton `InferenceSession` objects — never reloaded per request. This is what makes sub-20ms latency achievable on CPU.

**Why cap + class weights instead of SMOTE?** SMOTE interpolates between feature vectors. For numeric tabular data this is valid. For raw HTTP payloads and character sequences, interpolating between `' OR 1=1--` and `<script>alert(1)</script>` produces syntactically invalid strings that confuse the model rather than helping it generalise.

**Why MongoDB over PostgreSQL for logs?** Request log documents are heterogeneous — different attacks have different fields, the feature vector changes as models evolve, and you want fast appends under high traffic. A document store fits better than a rigid relational schema.

**Why train 3 candidates for L2B?** XGBoost gives a strong classical baseline with built-in feature importance (explainability). CNN captures local n-gram patterns. GRU captures long-range dependencies. The comparison table — accuracy, F1, latency — becomes a key result in the report.

---

## Team responsibilities

| Member | Primary area |
|---|---|
| Keerthi Vasan P | FastAPI backend, middleware pipeline, MongoDB integration |
| Darshan Gowda C | Layer 2A training, feature engineering, ONNX export |
| Santhosh V | Layer 2B training, threat scorer, explainability |
| Srujan H R | Nginx config, Docker setup, dashboard UI, testing |

---

*Cambridge Institute of Technology, Bengaluru · Dept. of CSE (IoT & Cyber Security) · B.E. Final Year Project 2025–26*
