"""
demo_full_loop.py — exercises CRC Decision 1 + adaptive retraining
end-to-end against a running WAF instance.

Workflow:
    1. Generate normal traffic through the live WAF
    2. Manually trigger the CRC Decision 1 health audit
    3. Seed deterministic synthetic verified feedback
    4. Trigger the online anti-poisoning / retraining-batch preparation
    5. Start native local retraining
    6. Poll the local retraining job until completion

Prerequisites:
    - WAF running:
        uvicorn app.main:app --port 8000

    - dummy_app.py running on port 5000
    - MongoDB reachable at settings.MONGO_URI
    - Local retraining artifacts available if Step 5 is executed
    - Training dependencies installed:
        pip install -r requirements-training.txt

Usage:
    python demo_full_loop.py

IMPORTANT:
    This is a DEV/DEMO script.

    The synthetic feedback inserted here intentionally bypasses the human
    review UI so the minimum-sample and anti-poisoning gates can be
    demonstrated without collecting hundreds of manual reviews.

    The final local-retraining step may promote a new model when
    LOCAL_RETRAIN_AUTO_PROMOTE=True.
"""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, ".")

WAF_URL = "http://127.0.0.1:8000"

# Set to False when you only want to prepare the batch and inspect it.
RUN_LOCAL_RETRAIN = True

LOCAL_RETRAIN_POLL_SECONDS = 3
LOCAL_RETRAIN_MAX_WAIT_SECONDS = 300


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

async def step1_generate_traffic():
    import httpx

    print("=" * 72)
    print("STEP 1: generating mixed traffic through the live WAF")
    print("=" * 72)

    # These are ordinary application requests. No /proxy prefix is used.
    reqs = [
        ("GET", "/api/products"),
        ("GET", "/api/products?category=electronics"),
        ("GET", "/api/products/search?q=coffee+maker"),
        ("GET", "/api/users/profile?user_id=42"),
        ("POST", "/api/users/login?username=john&password=hello123"),
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        for method, path in reqs:
            try:
                r = await client.request(
                    method,
                    f"{WAF_URL}{path}",
                )

                decision = r.headers.get("X-WAF-Decision", "?")
                score = r.headers.get("X-WAF-Score", "?")

                print(
                    f"  {r.status_code:<3} "
                    f"decision={decision:<6} "
                    f"score={score:<5} "
                    f"{method} {path}"
                )

            except Exception as exc:
                print(f"  FAILED {method} {path}: {exc}")

    print()


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

async def step2_trigger_audit():
    import httpx

    print("=" * 72)
    print("STEP 2: manually triggering CRC Decision 1 health audit")
    print("=" * 72)

    print(
        "Normally this runs from the health-monitor loop when the error-rate "
        "threshold is breached."
    )
    print(
        "This endpoint invokes the same audit function directly so the demo "
        "does not need to wait for the 60-second monitor tick."
    )
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                f"{WAF_URL}/api/health/trigger-audit",
                params={"error_rate": 0.99},
            )

            print("  HTTP status:", r.status_code)

            try:
                print("  audit result:")
                print(" ", r.json())
            except Exception:
                print(" ", r.text)

        except Exception as exc:
            print("  FAILED:", exc)

    print()


# ---------------------------------------------------------------------------
# Helpers for Step 3
# ---------------------------------------------------------------------------

def build_safe_templates():
    """
    These are traffic patterns that were already included in the project's
    normal-traffic demo and are intended to be ordinary application traffic.
    """
    return [
        "/api/products?category=electronics&page=2&limit=5",
        "/api/products/search?q=coffee+maker",
        "/api/users/profile?user_id=42",
        "/api/products?category=books&page=1&limit=20",
    ]


async def find_currently_normal_templates():
    """
    Check the CURRENT WAF models before generating the synthetic clean set.

    This makes the demo more robust than blindly assuming that a historical
    URL is still classified as normal after a model change.
    """
    from app.services.reaudit import reaudit

    candidates = build_safe_templates()
    usable = []

    print("  Pre-checking candidate normal request templates:")

    for url in candidates:
        try:
            result = reaudit(url, "GET", "")

            print(
                f"    {url:<70} "
                f"label={result.get('label')} "
                f"decision={result.get('decision')} "
                f"L2A={result.get('l2a_score')}"
            )

            if (
                result.get("label") == "normal"
                and result.get("decision") == "allow"
            ):
                usable.append(url)

        except Exception as exc:
            print(f"    FAILED {url}: {exc}")

    return usable


def build_synthetic_samples(clean_templates, n_clean=230):
    """
    Build:
        - enough clean samples to cross RETRAIN_MIN_SAMPLES
        - a per-IP-cap attack on the clean population
        - a family-diversity violation
        - a Layer-1 rejection
        - an intentional cross-agreement mismatch

    The clean samples use unique nonces so each URL becomes a different
    canonical family.
    """
    samples = []

    # ------------------------------------------------------------------
    # A. Clean verified samples
    # ------------------------------------------------------------------

    for i in range(n_clean):
        base_url = clean_templates[i % len(clean_templates)]

        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}_demo_nonce={i}"

        samples.append(
            {
                "request_id": str(uuid.uuid4()),
                "ip": f"10.20.{(i // 250) % 200}.{(i % 250) + 1}",
                "method": "GET",
                "url": url,
                "body": "",
                "decision": "allow",
                "score": 1,
                "label": "normal",
                "layer": "L2B",
                "latency_ms": 4.0,
                "timestamp": datetime.utcnow() - timedelta(seconds=i),
                "verified_label": "normal",
                "poisoning_flag": False,
                "auto_classified": False,
                "source": "demo_seed_clean",
            }
        )

    # ------------------------------------------------------------------
    # B. Per-IP cap demonstration
    #
    # 25 samples from one IP.
    # The configured cap is 20, so some are intentionally rejected.
    # ------------------------------------------------------------------

    for i in range(25):
        base_url = clean_templates[i % len(clean_templates)]
        separator = "&" if "?" in base_url else "?"

        samples.append(
            {
                "request_id": str(uuid.uuid4()),
                "ip": "10.99.99.99",
                "method": "GET",
                "url": f"{base_url}{separator}_ip_cap={i}",
                "body": "",
                "decision": "allow",
                "score": 1,
                "label": "normal",
                "layer": "L2B",
                "latency_ms": 4.0,
                "timestamp": datetime.utcnow(),
                "verified_label": "normal",
                "poisoning_flag": False,
                "auto_classified": False,
                "source": "demo_seed_ip_cap",
            }
        )

    # ------------------------------------------------------------------
    # C. Family-diversity cap demonstration
    #
    # The exact same URL appears 5 times.
    # MAX_FAMILY_PER_BATCH = 3, therefore these are intentionally rejected.
    # ------------------------------------------------------------------

    family_url = "/api/products?category=electronics&page=1&limit=5"

    for _ in range(5):
        samples.append(
            {
                "request_id": str(uuid.uuid4()),
                "ip": "10.88.88.1",
                "method": "GET",
                "url": family_url,
                "body": "",
                "decision": "allow",
                "score": 1,
                "label": "normal",
                "layer": "L2B",
                "latency_ms": 4.0,
                "timestamp": datetime.utcnow(),
                "verified_label": "normal",
                "poisoning_flag": False,
                "auto_classified": False,
                "source": "demo_seed_family_cap",
            }
        )

    # ------------------------------------------------------------------
    # D. Layer-1 rejection demonstration
    #
    # Human label says SQLi, but L1 should reject this sample immediately
    # rather than treating it as useful adaptive-training data.
    # ------------------------------------------------------------------

    samples.append(
        {
            "request_id": str(uuid.uuid4()),
            "ip": "10.77.77.1",
            "method": "GET",
            "url": "/api/products/search?q=' OR 1=1 --",
            "body": "",
            "decision": "block",
            "score": 90,
            "label": "sqli",
            "layer": "L1",
            "latency_ms": 2.0,
            "timestamp": datetime.utcnow(),
            "verified_label": "sqli",
            "poisoning_flag": False,
            "auto_classified": False,
            "source": "demo_seed_l1_rejection",
        }
    )

    # ------------------------------------------------------------------
    # E. Cross-agreement rejection demonstration
    #
    # Deliberately claim an ordinary request is SQLi. Current L2A/L2B
    # should disagree with this verified label.
    # ------------------------------------------------------------------

    mismatch_url = clean_templates[0]

    samples.append(
        {
            "request_id": str(uuid.uuid4()),
            "ip": "10.66.66.1",
            "method": "GET",
            "url": mismatch_url,
            "body": "",
            "decision": "block",
            "score": 85,
            "label": "sqli",
            "layer": "L2B",
            "latency_ms": 4.0,
            "timestamp": datetime.utcnow(),
            "verified_label": "sqli",
            "poisoning_flag": False,
            "auto_classified": False,
            "source": "demo_seed_cross_agreement",
        }
    )

    return samples


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

async def step3_seed_verified_feedback():
    from app.db.mongodb import connect_db, close_db, get_db
    from app.core.config import settings

    print("=" * 72)
    print("STEP 3: seeding synthetic verified feedback")
    print("=" * 72)

    # The demo is a separate process from the running WAF server.
    # Load its own inference sessions before using reaudit().
    ensure_models_loaded()

    await connect_db()
    db = get_db()

    try:
        clean_templates = await find_currently_normal_templates()

        if not clean_templates:
            print()
            print(
                "ERROR: none of the demo normal-request templates currently "
                "re-audit as normal/allow."
            )
            return False

        target_clean = settings.RETRAIN_MIN_SAMPLES + 30

        samples = build_synthetic_samples(
            clean_templates,
            n_clean=target_clean,
        )

        await db.feedback_queue.insert_many(samples)

        print()
        print(f"  inserted raw samples: {len(samples)}")
        print(f"  intended clean samples: {target_clean}")
        print(f"  RETRAIN_MIN_SAMPLES: {settings.RETRAIN_MIN_SAMPLES}")

        return True

    finally:
        await close_db()


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

async def step4_prepare_retrain_batch():
    import httpx

    print("=" * 72)
    print("STEP 4: running online anti-poisoning + batch preparation")
    print("=" * 72)

    print(
        "This endpoint now performs the online validation stage:"
        "\n  feedback_queue"
        "\n    -> per-IP cap"
        "\n    -> family-diversity cap"
        "\n    -> Layer-1 rejection"
        "\n    -> current L2A/L2B cross-agreement"
        "\n    -> RETRAIN_MIN_SAMPLES gate"
        "\n    -> retrain_batches"
    )
    print()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            r = await client.post(
                f"{WAF_URL}/api/feedback/trigger-retrain"
            )

            print("  HTTP status:", r.status_code)

            try:
                result = r.json()
                print("  retrain preparation result:")
                print(" ", result)
            except Exception:
                print(" ", r.text)
                return None

            return result

        except Exception as exc:
            print("  FAILED:", exc)
            return None

    print()


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------

async def step5_start_local_retrain():
    import httpx

    if not RUN_LOCAL_RETRAIN:
        print("=" * 72)
        print("STEP 5: local retraining skipped")
        print("=" * 72)
        print("RUN_LOCAL_RETRAIN=False")
        print()
        return

    print("=" * 72)
    print("STEP 5: starting native local retraining")
    print("=" * 72)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                f"{WAF_URL}/api/feedback/local-retrain/start"
            )

            print("  HTTP status:", r.status_code)

            try:
                start_result = r.json()
                print("  start result:")
                print(" ", start_result)
            except Exception:
                print(" ", r.text)

            if r.status_code not in (200, 202):
                print()
                print(
                    "Local retraining was not started. "
                    "The validated batch may still be available in:"
                )
                print(
                    f"  {WAF_URL}/api/feedback/retrain-batches/latest"
                )
                print()
                return

        except Exception as exc:
            print("  FAILED to start local retraining:", exc)
            return

    print()
    print("  Polling local retraining status...")

    elapsed = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < LOCAL_RETRAIN_MAX_WAIT_SECONDS:
            await asyncio.sleep(LOCAL_RETRAIN_POLL_SECONDS)
            elapsed += LOCAL_RETRAIN_POLL_SECONDS

            try:
                r = await client.get(
                    f"{WAF_URL}/api/feedback/local-retrain/status"
                )

                if r.status_code != 200:
                    print(
                        f"  status request failed: HTTP {r.status_code}"
                    )
                    continue

                status = r.json()

                state = status.get("status", "unknown")

                print(
                    f"  [{elapsed:>3}s] "
                    f"status={state}"
                )

                if state in {
                    "completed",
                    "deployed",
                    "failed",
                    "error",
                }:
                    print()
                    print("  final local retraining status:")
                    print(" ", status)
                    return

            except Exception as exc:
                print("  status polling error:", exc)

    print()
    print(
        f"  Local retraining did not finish within "
        f"{LOCAL_RETRAIN_MAX_WAIT_SECONDS}s."
    )
    print(
        f"  Check: {WAF_URL}/api/feedback/local-retrain/status"
    )
    print()


def ensure_models_loaded():
    """
    The demo runs in a separate Python process from Uvicorn, so it must
    load its own ONNX inference sessions before calling reaudit().
    """
    import app.services.layer2a_anomaly as l2a
    import app.services.layer2b_deep as l2b

    if l2a._sess is None:
        print("  Loading L2A model for demo re-audit...")
        l2a.load()

    if l2b._sess is None:
        print("  Loading L2B model for demo re-audit...")
        l2b.load()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print()
    print("=" * 72)
    print("WAF-ML FULL FEEDBACK / ADAPTIVE-RETRAINING DEMO")
    print("=" * 72)
    print(f"WAF: {WAF_URL}")
    print()

    await step1_generate_traffic()

    await step2_trigger_audit()

    seeded = await step3_seed_verified_feedback()

    if not seeded:
        print("Demo stopped before retraining-batch preparation.")
        return

    prepared = await step4_prepare_retrain_batch()

    if not prepared:
        print("Demo stopped after batch-preparation failure.")
        return

    if prepared.get("status") != "queued":
        print()
        print(
            "The online gate did not produce a queued retraining batch."
        )
        print(
            "This is useful diagnostic output — inspect n_clean and "
            "reject_reason_breakdown above."
        )
        return

    await step5_start_local_retrain()

    print()
    print("=" * 72)
    print("DEMO COMPLETE")
    print("=" * 72)
    print()
    print("Useful endpoints:")
    print(f"  Review queue:        {WAF_URL}/dashboard/feedback")
    print(f"  Latest batch:        {WAF_URL}/api/feedback/retrain-batches/latest")
    print(
        f"  Batch export:       "
        f"{WAF_URL}/api/feedback/retrain-batches/latest/export"
    )
    print(
        f"  Retrain status:     "
        f"{WAF_URL}/api/feedback/local-retrain/status"
    )
    print()
    print("Useful MongoDB collections:")
    print("  feedback_queue")
    print("  health_audit_log")
    print("  retrain_log")
    print("  retrain_batches")
    print()


if __name__ == "__main__":
    asyncio.run(main())