# WAF-ML — Live Demo Guide

## 1. Demo Objective

Demonstrate the Hybrid Intelligent Web Application Firewall protecting a FastAPI application in real time.

The demonstration should show:

1. Normal requests are allowed.
2. Suspicious/borderline requests are analyzed by the ML pipeline.
3. Attack requests are blocked.
4. Requests are logged in MongoDB.
5. The dashboard displays live request/threat information.
6. The protected backend receives allowed traffic but does not receive blocked traffic.

---

# 2. Current Local Architecture

This demo runs **without Docker**.

```text
                 Browser
                    │
                    │ http://127.0.0.1:8000/...
                    ▼
        ┌─────────────────────────┐
        │       WAF :8000         │
        │                         │
        │  L1 Rule-Based Filter   │
        │          ↓              │
        │  L2A Anomaly Detector   │
        │          ↓              │
        │  L2B Deep Classifier    │
        │          ↓              │
        │    Threat Scorer        │
        └────────────┬────────────┘
                     │
              ALLOW / LOG
                     │
                     ▼
        ┌─────────────────────────┐
        │ Protected App :5000     │
        │      dummy_app.py       │
        └─────────────────────────┘
                     │
                     ▼
                 MongoDB
```

### Important

For the demo, always use:

```text
http://127.0.0.1:8000
```

Port `8000` is the WAF.

Do **not** use port `5000` for WAF testing:

```text
http://127.0.0.1:5000
```

Port `5000` is the protected backend and bypasses the WAF.

---

# 3. Before Starting the Demo

Make sure MongoDB is running.

Check:

```powershell
mongosh --eval "db.adminCommand('ping')"
```

Expected:

```text
{ ok: 1 }
```

Make sure the local configuration points to:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=waf_dev
PROTECTED_APP_URL=http://127.0.0.1:5000
```

---

# 4. Start the Protected Application

## Terminal 1

```powershell
cd G:\classroom\waf-ml-project
.venv\Scripts\activate

uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

Expected:

```text
Uvicorn running on http://127.0.0.1:5000
```

Keep this terminal visible.

This terminal demonstrates whether a request actually reaches the protected backend.

---

# 5. Start the WAF

## Terminal 2

```powershell
cd G:\classroom\waf-ml-project
.venv\Scripts\activate

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for:

```text
WAF ready ◈
```

Also verify that startup reports:

```text
MongoDB connected → waf_dev
L2A loaded
L2B loaded
All ML models loaded successfully
Health monitor started
```

---

# 6. Open the Dashboard

Open:

```text
http://127.0.0.1:8000/dashboard
```

Keep the dashboard open in a browser tab.

Useful dashboard pages:

```text
http://127.0.0.1:8000/dashboard
http://127.0.0.1:8000/dashboard/logs
http://127.0.0.1:8000/dashboard/threats
http://127.0.0.1:8000/dashboard/feedback
http://127.0.0.1:8000/dashboard/models
```

### Dashboard rule

The dashboard and its static resources must remain WAF-internal.

The ML middleware should bypass:

```text
/dashboard
/static
/health
/docs
/redoc
/openapi.json
/favicon.ico
```

Only protected application traffic should enter L1 → L2A → L2B.

---

# 7. Live Demo — Part A: Normal Traffic

Start with legitimate requests.

## Normal Request 1 — Products

Open:

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

Expected:

```text
HTTP 200
```

The WAF terminal should show the request being analyzed.

The backend terminal should show:

```text
GET /api/products?category=electronics&page=1
```

### Explain

> "The client sends the request to port 8000. The request first passes through the WAF. If it is considered safe, the WAF forwards it to the protected application running on port 5000."

---

## Normal Request 2 — Product Search

```text
http://127.0.0.1:8000/api/products/search?q=laptop
```

Expected:

```text
HTTP 200
```

---

## Normal Request 3 — Search

```text
http://127.0.0.1:8000/search?q=products
```

Expected:

```text
HTTP 200
```

---

## Normal Request 4 — Product Category

```text
http://127.0.0.1:8000/api/products?category=books&page=2
```

Expected:

```text
HTTP 200
```

---

# 8. Important ML Demonstration Point

The WAF should analyze:

```text
/api/products/search?q=laptop
```

not:

```text
http://127.0.0.1:8000/api/products/search?q=laptop
```

The absolute base URL is not part of the ML input.

This keeps the model focused on the actual request path and query.

---

# 9. Live Demo — Part B: Borderline / Suspicious Traffic

These requests are useful to demonstrate that the WAF does not depend only on obvious attack strings.

Run these one at a time.

## Borderline 1 — SQL Keyword

```text
http://127.0.0.1:8000/api/products/search?q=select+name+from+users
```

## Borderline 2 — HTML

```text
http://127.0.0.1:8000/api/products/search?q=%3Cb%3Etest%3C%2Fb%3E
```

## Borderline 3 — JavaScript-like Input

```text
http://127.0.0.1:8000/api/products/search?q=javascript%3Avoid%280%29
```

## Borderline 4 — Path Traversal Pattern

```text
http://127.0.0.1:8000/api/products/search?q=..%2F..%2Fconfig
```

## Borderline 5 — SQL Boolean Pattern

```text
http://127.0.0.1:8000/api/products/search?q=1+OR+1%3D1
```

### What to show

After each request, look at:

```text
L1
L2A score
Escalation
L2B label
Confidence
Threat score
Decision
```

Do not assume that every borderline request must have the same decision. The actual model output shown during the demo is the result to present.

---

# 10. Live Demo — Part C: Attack Traffic

Use these as controlled demonstration inputs against the local dummy application.

## Attack 1 — SQL Injection

```text
http://127.0.0.1:8000/api/products/search?q=1%27+UNION+SELECT+username+FROM+users
```

Expected behavior:

```text
WAF analyzes request
        ↓
Threat detected
        ↓
BLOCK
        ↓
HTTP 403
```

---

## Attack 2 — XSS

```text
http://127.0.0.1:8000/api/products/search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E
```

Expected:

```text
HTTP 403
```

if classified as malicious by the configured WAF pipeline.

---

## Attack 3 — Path Traversal

```text
http://127.0.0.1:8000/api/products/search?q=..%2F..%2F..%2Fetc%2Fpasswd
```

Expected:

```text
HTTP 403
```

if detected by L1 or the ML layers.

---

## Attack 4 — Authentication SQL Injection

```text
http://127.0.0.1:8000/api/login?username=admin%27--%26password=test
```

Use this only if the current dummy application exposes `/api/login`.

If the endpoint does not exist, use the existing search endpoint instead:

```text
http://127.0.0.1:8000/api/products/search?q=admin%27-- 
```

---

# 11. Best Attack to Demonstrate L1

For a clearly recognizable rule-based attack, use:

```text
http://127.0.0.1:8000/api/products/search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E
```

Then point out the WAF terminal.

You want to show something similar to:

```text
WAF L1 BLOCK
```

Explain:

> "This request can be rejected immediately by the first layer without requiring deeper ML analysis."

---

# 12. Best Attack to Demonstrate L2A → L2B

Use a suspicious request that is not necessarily caught immediately by L1:

```text
http://127.0.0.1:8000/api/products/search?q=1+OR+1%3D1
```

Look for:

```text
l2a_score=...
escalated=True
label=...
confidence=...
score=...
decision=...
```

Explain:

> "L2A acts as the anomaly detector. Only sufficiently suspicious traffic is escalated to the deeper L2B classifier."

---

# 13. Show the Backend Protection

This is an important visual part of the demo.

### Send a normal request

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

Show Terminal 1:

```text
GET /api/products?... 200 OK
```

This proves the WAF allowed and forwarded the request.

### Send an attack

```text
http://127.0.0.1:8000/api/products/search?q=1+OR+1%3D1
```

If the WAF blocks it, show that the request does **not** appear as a successful request in the backend terminal.

Explain:

> "The important security property is that malicious traffic is stopped before it reaches the protected application."

---

# 14. Show MongoDB Logging

Open:

```text
http://127.0.0.1:8000/dashboard/logs
```

Show:

- request URL/path
- method
- decision
- score
- layer
- latency
- timestamp

Then open:

```text
http://127.0.0.1:8000/dashboard/threats
```

Show detected threats.

Then:

```text
http://127.0.0.1:8000/dashboard/feedback
```

Show feedback/review records if the current decision flow generated them.

---

# 15. Verify MongoDB Directly

In another terminal:

```powershell
mongosh
```

Then:

```javascript
use waf_dev
show collections
```

Expected collections include:

```text
feedback_queue
health_audit_log
health_snapshots
model_versions
request_logs
retrain_log
threat_events
```

Check request logs:

```javascript
db.request_logs.countDocuments()
```

Check threats:

```javascript
db.threat_events.countDocuments()
```

Check recent requests:

```javascript
db.request_logs.find().sort({timestamp:-1}).limit(5)
```

---

# 16. Health Monitoring Demonstration

The dummy backend contains a controlled health simulation.

To simulate degraded backend health:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri "http://127.0.0.1:5000/simulate/breach?error_rate=0.15"
```

Expected response:

```json
{
  "status": "breach_simulated",
  "error_rate": 0.15
}
```

The WAF health monitor checks the protected application periodically.

Wait for the configured health-monitor interval.

Then inspect:

```text
http://127.0.0.1:8000/dashboard
```

and MongoDB:

```javascript
use waf_dev
db.health_snapshots.find().sort({timestamp:-1}).limit(5)
```

To restore normal health:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri "http://127.0.0.1:5000/simulate/recover"
```

Expected:

```json
{
  "status": "recovered"
}
```

---

# 17. Optional Automated Traffic Demo

If time is limited, run:

```powershell
python test_traffic.py
```

Make sure the script targets:

```text
http://127.0.0.1:8000
```

not:

```text
http://127.0.0.1:5000
```

Then open:

```text
http://127.0.0.1:8000/dashboard
```

and show the generated traffic.

---

# 18. Recommended 10-Minute Demo Sequence

## Minute 1 — Architecture

Show:

```text
Client
  ↓
WAF :8000
  ↓
L1 → L2A → L2B
  ↓
Threat Scorer
  ↓
Protected App :5000
```

Explain that the WAF sits between the client and application.

---

## Minute 2 — Normal Traffic

Open:

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

Then:

```text
http://127.0.0.1:8000/api/products/search?q=laptop
```

Show `200 OK`.

---

## Minute 3 — Dashboard

Open:

```text
http://127.0.0.1:8000/dashboard
```

Show that the requests have appeared in the logs.

---

## Minutes 4–6 — Attacks

Run:

```text
/api/products/search?q=1+OR+1%3D1
```

```text
/api/products/search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E
```

```text
/api/products/search?q=..%2F..%2F..%2Fetc%2Fpasswd
```

Show:

```text
403
```

and the corresponding WAF terminal decision.

---

## Minutes 7–8 — ML Pipeline

Show:

```text
L2A score
Escalation
L2B label
Confidence
Threat score
Decision
```

Explain selective escalation.

---

## Minute 9 — MongoDB / Feedback

Show:

```text
/dashboard/logs
/dashboard/threats
/dashboard/feedback
```

---

## Minute 10 — Health Feedback

Run:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri "http://127.0.0.1:5000/simulate/breach?error_rate=0.15"
```

Explain that backend health is part of the adaptive feedback loop.

---

# 19. URLs to Keep Ready in Browser Tabs

### Dashboard

```text
http://127.0.0.1:8000/dashboard
```

### Normal

```text
http://127.0.0.1:8000/api/products?category=electronics&page=1
```

### Normal Search

```text
http://127.0.0.1:8000/api/products/search?q=laptop
```

### SQL Injection

```text
http://127.0.0.1:8000/api/products/search?q=1+OR+1%3D1
```

### XSS

```text
http://127.0.0.1:8000/api/products/search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E
```

### Path Traversal

```text
http://127.0.0.1:8000/api/products/search?q=..%2F..%2F..%2Fetc%2Fpasswd
```

### SQL UNION

```text
http://127.0.0.1:8000/api/products/search?q=1%27+UNION+SELECT+username+FROM+users
```

---

# 20. What NOT to Do During the Demo

Do not test protected traffic directly against:

```text
http://127.0.0.1:5000
```

Do not use the old local URL format:

```text
http://127.0.0.1:8000/proxy/...
```

Do not refresh the dashboard through the ML pipeline.

Do not clear MongoDB immediately before the demo unless you have already verified the application after clearing it.

Do not change L1/L2A/L2B thresholds immediately before the presentation.

Do not claim that a particular attack is always caught by a particular layer unless the live output actually shows that layer.

---

# 21. If Something Goes Wrong

## Dashboard has no CSS

Check that the middleware bypasses:

```text
/static
/dashboard
```

Restart the WAF:

```powershell
CTRL+C

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## `{"detail":"Not Found"}`

Check that:

1. The WAF is running on `8000`.
2. The backend is running on `5000`.
3. The requested endpoint exists in `dummy_app.py`.
4. You are using the current clean URL without `/proxy`.

---

## `502 Bad Gateway`

Start the protected application:

```powershell
uvicorn dummy_app:app --host 127.0.0.1 --port 5000
```

---

## Request gives 200 but no WAF log

You probably sent the request to:

```text
127.0.0.1:5000
```

Use:

```text
127.0.0.1:8000
```

---

## Normal traffic is blocked

Check the WAF terminal for:

```text
l2a_score
escalated
label
confidence
score
decision
```

Do not silently change the threshold. Record the output first because this may indicate an L2A/L2B calibration or feature-extraction issue.

---

# 22. Short Viva Explanation

### What is the main idea?

> "The system is a hybrid intelligent WAF that combines a fast rule-based Layer 1 filter with machine-learning based anomaly detection and deep classification. L2A identifies suspicious traffic and selectively escalates it to L2B for deeper analysis."

### Why two ML layers?

> "L2A provides lightweight anomaly detection and selective escalation, while L2B performs deeper classification only for suspicious requests. This avoids applying the expensive classifier to every request."

### What happens to a normal request?

```text
Request
  ↓
L1
  ↓
L2A
  ↓
Low anomaly
  ↓
ALLOW
  ↓
Protected application
```

### What happens to suspicious traffic?

```text
Request
  ↓
L1
  ↓
L2A
  ↓
High anomaly
  ↓
L2B
  ↓
Threat Score
  ↓
BLOCK / LOG / ALLOW
```

### What makes the demo architecture different from directly calling the backend?

> "The client communicates with the WAF endpoint on port 8000. The protected application is isolated behind it on port 5000. Therefore, a request must pass through the WAF before reaching the application."

---

# 23. Final Demo Checklist

Before leaving for the presentation:

- [ ] MongoDB is running.
- [ ] `.venv` works.
- [ ] Model files exist.
- [ ] `dummy_app.py` starts on port 5000.
- [ ] WAF starts on port 8000.
- [ ] WAF reports `WAF ready ◈`.
- [ ] Dashboard loads with CSS/JS.
- [ ] Normal request returns `200`.
- [ ] Normal request appears in WAF logs.
- [ ] Attack request is analyzed.
- [ ] At least one attack produces `403`.
- [ ] Threat appears in dashboard.
- [ ] MongoDB contains request logs.
- [ ] `test_traffic.py` points to port 8000.
- [ ] Health simulation works.
- [ ] Browser tabs with demo URLs are prepared.
- [ ] Do not send demo traffic directly to port 5000.

