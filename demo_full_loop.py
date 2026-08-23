"""
demo_full_loop.py — exercises CRC Decision 1 (server health feedback) and
the adaptive retraining gate end to end, against a running WAF instance.

Prerequisites:
    - The WAF app is running (e.g. `uvicorn app.main:app --port 8000`)
    - dummy_app.py is running on port 5000 (the PROTECTED_APP_URL)
    - MongoDB is reachable at settings.MONGO_URI

Usage:
    python demo_full_loop.py

This does NOT require waiting for the 60s health-monitor tick — it calls
the manual /api/health/trigger-audit endpoint added for exactly this
purpose. It also seeds a synthetic batch of verified feedback_queue
samples so you can see the anti-poison gate (per-IP cap, family-diversity
cap, L2A/L2B cross-agreement) and RETRAIN_MIN_SAMPLES threshold actually
fire, without needing hundreds of real human reviews first.
"""
import asyncio
import random
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, ".")

WAF_URL = "http://127.0.0.1:8000"


async def step1_generate_traffic():
    import httpx
    print("=== Step 1: generating mixed traffic through the live WAF ===")
    reqs = [
        ("/tienda1/publico/anadir.jsp?id=1&nombre=laptop", "normal"),
        ("/tienda1/publico/buscar.jsp?texto=zapatos+rojos", "normal"),
        ("/tienda1/publico/productos.jsp?categoria=electronics", "normal"),
        ("/tienda1/publico/login.jsp?usuario=john&password=hello123", "normal"),
        ("/search?q=products", "normal"),
    ]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for path, _ in reqs:
            try:
                r = await client.get(f"{WAF_URL}/proxy{path}")
                print(f"  {r.status_code}  {path}")
            except Exception as e:
                print(f"  FAILED {path}: {e}")
    print()


async def step2_trigger_audit():
    import httpx
    print("=== Step 2: manually triggering the health audit (CRC Decision 1) ===")
    print("(normally fires automatically when error_rate >= ERROR_RATE_THRESHOLD;")
    print(" this hits the same _trigger_audit() function directly for demo speed)\n")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{WAF_URL}/api/health/trigger-audit", params={"error_rate": 0.99})
        print("  audit result:", r.json())
    print()


async def step3_seed_verified_feedback():
    """
    Insert synthetic VERIFIED feedback_queue samples directly into MongoDB
    (bypassing the human-review UI) so run_retrain_cycle()'s
    RETRAIN_MIN_SAMPLES=200 gate and anti-poison logic can actually be
    observed without needing 200 real manual reviews first.

    Clearly a demo/dev tool — in production every verified_label comes
    from a real human reviewer via /api/feedback/review/{request_id}.
    """
    from app.db.mongodb import connect_db, close_db, get_db
    from app.core.config import settings

    print("=== Step 3: seeding synthetic verified feedback (demo only) ===")
    await connect_db()
    db = get_db()

    n_target = settings.RETRAIN_MIN_SAMPLES + 20  # comfortably over the gate
    samples = []
    attack_urls = [
        "/tienda1/publico/buscar.jsp?texto=1+OR+1=1",
        "/tienda1/publico/usuarios.jsp?nombre=admin' OR '1'='1",
        "/tienda1/publico/ver.jsp?file=../../etc/passwd",
        "/tienda1/publico/comentarios.jsp?texto=<script>alert(1)</script>",
    ]
    normal_urls = [
        "/tienda1/publico/productos.jsp?categoria=shoes",
        "/tienda1/publico/buscar.jsp?texto=zapatillas",
        "/tienda1/publico/anadir.jsp?id=2&nombre=mouse",
    ]

    for i in range(n_target):
        is_attack = i % 3 == 0
        base_url = random.choice(attack_urls if is_attack else normal_urls)
        # vary IP and a query-string nonce so family-diversity/per-IP caps
        # don't reject the whole batch outright
        url = f"{base_url}&_nonce={i}"
        samples.append({
            "request_id": str(uuid.uuid4()),
            "ip": f"10.0.{i % 40}.{i % 250}",
            "method": "GET",
            "url": url,
            "body": "",
            "decision": "block" if is_attack else "log",
            "score": 85 if is_attack else 35,
            "label": "sqli" if is_attack else "normal",
            "layer": "L2B",
            "latency_ms": 5.0,
            "timestamp": datetime.utcnow() - timedelta(minutes=i),
            "verified_label": ("sqli" if is_attack else "normal"),
            "poisoning_flag": False,
            "auto_classified": False,
            "source": "demo_seed",
        })

    await db.feedback_queue.insert_many(samples)
    print(f"  inserted {len(samples)} synthetic verified samples "
          f"(RETRAIN_MIN_SAMPLES={settings.RETRAIN_MIN_SAMPLES})")
    await close_db()
    print()


async def step4_trigger_retrain():
    import httpx
    print("=== Step 4: triggering the adaptive retrain cycle ===")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{WAF_URL}/api/feedback/trigger-retrain")
        print("  retrain result:", r.json())
    print()


async def main():
    await step1_generate_traffic()
    await step2_trigger_audit()
    await step3_seed_verified_feedback()
    await step4_trigger_retrain()
    print("Done. Check:")
    print(f"  {WAF_URL}/dashboard/feedback   (pending human review)")
    print(f"  MongoDB health_audit_log        (capture/disagreement report)")
    print(f"  MongoDB retrain_log             (anti-poison breakdown, n_clean/n_rejected)")


if __name__ == "__main__":
    asyncio.run(main())
