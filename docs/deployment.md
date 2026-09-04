# Deployment

Covers what the local-dev instructions in the [README](../README.md#running-with-docker)
don't: the actual `docker-compose.yml`/`nginx.conf` setup, gaps found
in it, and production considerations.

---

## Current Docker Compose topology

```yaml
services:
  nginx:    # :80 → reverse proxy in front of everything
  app:      # the WAF itself (FastAPI), built from ./app
  mongodb:  # mongo:7, persisted via mongo_data volume
```

**There is no protected-application service in `docker-compose.yml`.**
`dummy_app.py` is never containerized or started by Compose — only the
WAF (`app`), `nginx`, and `mongodb` are defined. This means
`docker-compose up --build` alone gives you a running WAF with nothing
behind it to protect. Every allowed request will hit `PROTECTED_APP_URL`
and get a connection failure (502), because nothing is listening there
inside the Compose network.

**Fix, option A — add `dummy_app.py` as a service** (recommended for demo/dev parity with local runs):
```yaml
  webapp:
    build:
      context: .
      dockerfile: Dockerfile.dummy   # needs adding — a minimal Dockerfile
                                      # that copies dummy_app.py and runs
                                      # `uvicorn dummy_app:app --host 0.0.0.0 --port 5000`
    restart: unless-stopped
```
and set `PROTECTED_APP_URL=http://webapp:5000` in `.env` for the Docker
profile (the app's `config.py` default is currently
`http://127.0.0.1:5000`, correct for local dev but wrong inside
Compose's network namespace — `127.0.0.1` inside the `app` container
doesn't reach a sibling container).

**Fix, option B — point at a real application** you already have a
Dockerfile/image for, setting `PROTECTED_APP_URL` to that service's
in-network hostname:port.

Until one of these is done, Docker Compose deployment is WAF-only —
fine for verifying the WAF boots and the dashboard loads, not fine for
demonstrating end-to-end request protection.

---

## `nginx/conf.d/waf.conf` is stale relative to the current middleware

```nginx
location / {
  proxy_pass http://app:8000;
  ...
}

location /proxy/ {
  proxy_pass http://app:8000/proxy/;
  ...
}
```

The `/proxy/` block is a leftover from an earlier middleware design
that stripped a `/proxy` prefix before forwarding. The current
`app/middleware/waf_middleware.py` forwards `request.url.path` as-is
with no prefix stripping (see [Architecture](architecture.md) and the
main README) — so this block is now redundant at best, and at worst
sends anything under `/proxy/*` through nginx to the WAF as a literal
path segment `/proxy/...`, which the WAF then tries to forward to the
protected app's `/proxy/...` (which won't exist on a normal backend).

**Fix:** remove the `/proxy/` location block entirely — the catch-all
`location /` block already forwards everything to the WAF correctly:
```nginx
server {
  listen 80;
  server_name _;

  location / {
    proxy_pass         http://app:8000;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_read_timeout 30s;
  }
}
```

---

## Model volume mount is partially redundant

```yaml
volumes:
  - ./ml/exported_models:/app/models:ro   # ONNX + scaler + threshold
  - ./ml:/app/ml:ro                        # feature_engineering package
```

`config.py`'s model paths (`L2A_ONNX_PATH`, `SCALER_PATH`, etc.) are
relative paths like `ml/exported_models/layer2a_best.onnx`. With
`WORKDIR /app` inside the container, that resolves to
`/app/ml/exported_models/layer2a_best.onnx` — which the **second**
mount (`./ml:/app/ml:ro`) already provides, since `./ml` includes
`./ml/exported_models` as a subdirectory. The first mount
(`./ml/exported_models:/app/models:ro`) puts the same files at
`/app/models`, a path nothing in the codebase currently references.

Not breaking anything today, just dead weight — safe to remove the
first line if you want a cleaner compose file, or leave it as a
convenience mount if you plan to reference `/app/models` from
somewhere else later.

---

## Environment variables reference

| Variable | Local dev value | Docker value | Notes |
|---|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | `mongodb://mongodb:27017` | Compose service name, not `localhost` |
| `MONGO_DB` | `waf_dev` | `waf_db` | `docker-compose.yml`'s `MONGO_INITDB_DATABASE=waf_db` must match |
| `PROTECTED_APP_URL` | `http://127.0.0.1:5000` | `http://webapp:5000` *(once added — see above)* | Compose service name, not `127.0.0.1` |
| `L2A_ONNX_PATH` etc. | `ml/exported_models/...` | same | Relative to `WORKDIR`, resolved via the `./ml:/app/ml:ro` mount |

---

## Health checks

`docker-compose.yml`'s `app` service has:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/health/"]
  interval: 30s
  timeout: 5s
  retries: 3
```
This checks the WAF's own liveness (`GET /api/health/` — DB ping),
not the protected application's health. The WAF's own internal health
monitor (`app/services/health_monitor.py`, CRC Decision 1) is a
separate, independent mechanism that polls whatever `PROTECTED_APP_URL`
points to every `HEALTH_CHECK_INTERVAL_SEC` — the two aren't related
and shouldn't be confused with each other when debugging.

---

## Production considerations (not yet implemented)

These are gaps worth being explicit about rather than silently
assuming are handled — none of this is currently in the codebase:

- **TLS** — nginx currently only listens on port 80. A production
  deployment needs a TLS-terminating config (or a TLS-terminating load
  balancer in front of nginx).
- **Secrets** — `.env` is gitignored, but nothing currently encrypts
  or rotates `MONGO_URI` credentials if you add authentication to
  MongoDB (`docker-compose.yml`'s `mongodb` service currently has none
  configured — fine for local dev, not fine for anything internet-facing).
- **Horizontal scaling** — `app`'s Dockerfile runs `uvicorn` with
  `--workers 1`. Multiple workers/replicas would need to share model
  state correctly (ONNX Runtime sessions are loaded per-process at
  startup via `model_loader.load_all()` — this should work fine with
  multiple workers since each loads its own session, but hasn't been
  load-tested).
- **Rate limiting** — see the README's Known Issues: `slowapi`'s
  limiter is configured but not actually enforced anywhere yet.

None of these block a local demo or a controlled Docker deployment for
evaluation purposes — they matter once this sits in front of real,
adversarial internet traffic.
