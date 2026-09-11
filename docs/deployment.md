# Deployment

## Docker Compose topology

The demo stack contains four services:

```text
Internet
   |
 nginx:80
   |
 app:8000  (Hybrid Intelligent WAF)
   |
 protected-app:5000  (demo backend)
   |
 mongodb:27017
```

`protected-app` is built from `Dockerfile.protected` and runs `dummy_app.py`. The WAF reaches it through the Compose service name `protected-app`, not `127.0.0.1`.

## Start the complete stack

From the repository root:

```bash
docker compose up --build
```

The startup order is:

1. MongoDB becomes healthy.
2. The protected demo application starts.
3. The WAF starts and validates the required ONNX/scaler/threshold artifacts.
4. Nginx starts after the WAF health check passes.

Open the dashboard at `http://localhost/dashboard`.

## Runtime model files

The WAF uses the repository's `ml` directory directly through a read-only mount:

```text
./ml:/app/ml:ro
```

The configured paths therefore resolve to `/app/ml/exported_models/...` inside the WAF container. The startup check requires:

- `layer2a_best.onnx`
- `layer2a_best_threshold.txt`
- `layer2b_best.onnx`
- `scaler_l2a.pkl`

If any required artifact is missing or a model cannot be loaded, the WAF fails startup rather than running without ML protection.

## Request flow

All client traffic is sent to Nginx and then to the WAF. The WAF middleware performs the Layer 1 → Layer 2A → selective Layer 2B → threat scoring decision. Allowed traffic is forwarded to `http://protected-app:5000`.

There is intentionally no separate `/proxy/` Nginx route. The catch-all `/` route sends all traffic through the WAF so the middleware remains the single interception point.

## Health monitoring

The WAF's `/api/health/` endpoint checks MongoDB connectivity. Separately, the health monitor polls the protected application's `/health` endpoint using `PROTECTED_APP_URL` and triggers the tested capture → re-score → disagreement → feedback workflow when the configured error-rate threshold is exceeded.

The Compose health checks verify service readiness; they do not replace the WAF's application-health feedback loop.

## Local development

For non-Docker development, keep the default protected-app URL as:

```env
PROTECTED_APP_URL=http://127.0.0.1:5000
```

Run `dummy_app.py` on port 5000 and the WAF on port 8000 in the same local environment. For Docker Compose, the compose file explicitly overrides the URL to:

```env
PROTECTED_APP_URL=http://protected-app:5000
```

## Production limitations

This Compose configuration is intended for controlled development/demo use. It does not provide TLS termination, production secret management, MongoDB authentication, or horizontally scaled model workers. Those should be added before internet-facing deployment.
