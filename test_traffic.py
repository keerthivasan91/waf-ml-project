# test_traffic.py — run from project root
#
# Sends 100+ varied requests through the live WAF (port 8000) to the
# protected demo backend (dummy_app.py, port 5000). No /proxy prefix —
# the middleware forwards path + query 1:1, except its own bypassed
# routes (see bypass_paths in app/middleware/waf_middleware.py).
#
# NOTE: as of this script, app/main.py registers `limiter` and a
# RateLimitExceeded handler but never actually adds SlowAPIMiddleware
# or a per-route @limiter.limit(...) decorator — so RATE_LIMIT_PER_MIN
# isn't currently enforced anywhere, and you likely won't see 429s no
# matter how fast you send requests. The delay + retry below are kept
# anyway: harmless if the limiter stays unwired, and correct once it's
# actually turned on.
import time
import requests

BASE = "http://127.0.0.1:8000"
DELAY_SEC = 0.6          # ~100/min-safe spacing if/when rate limiting is enforced
MAX_RETRIES = 3

def send(method, path, **kwargs):
    for attempt in range(MAX_RETRIES):
        r = requests.request(method, BASE + path, timeout=10, **kwargs)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2))
            print(f"  429 rate-limited on {path[:60]} — backing off {wait}s")
            time.sleep(wait)
            continue
        return r
    return r  # give up after MAX_RETRIES, return last response anyway


# ============================================================
# NORMAL — ordinary use of every route (score < 30, expect allow)
# ============================================================
normal = [
    ("GET",  "/api/products"),
    ("GET",  "/api/products?category=electronics"),
    ("GET",  "/api/products?category=electronics&page=2&limit=5"),
    ("GET",  "/api/products/search?q=laptop"),
    ("GET",  "/api/products/search?q=wireless+mouse"),
    ("GET",  "/api/products/search?q=headphones"),
    ("GET",  "/api/products/101"),
    ("GET",  "/api/products/102"),
    ("GET",  "/api/users/profile?user_id=1"),
    ("GET",  "/api/users/profile?user_id=42"),
    ("GET",  "/api/orders"),
    ("GET",  "/api/orders?user_id=1&status=delivered"),
    ("GET",  "/api/orders/details?order_id=5001"),
    ("GET",  "/api/cart"),
    ("GET",  "/api/cart?user_id=7"),
    ("GET",  "/api/reviews"),
    ("GET",  "/api/reviews?product_id=101&sort=recent"),
    ("GET",  "/api/reviews?product_id=102&sort=top"),
    ("GET",  "/api/contact?subject=Question&message=When+will+my+order+ship"),
    ("GET",  "/api/contact?subject=Feedback&message=Great+service+thanks"),
    ("GET",  "/api/files/view?path=readme.txt"),
    ("GET",  "/api/files/download?file=invoice.pdf"),
    ("GET",  "/api/system/check?value=status"),
    ("GET",  "/api/admin/dashboard"),
    ("GET",  "/api/admin/users?search=john&page=1"),
    ("GET",  "/hello"),
    ("GET",  "/"),
    ("POST", "/api/users/login", {"params": {"username": "john", "password": "hello123"}}),
    ("POST", "/api/users/login", {"params": {"username": "maria", "password": "correcthorsebattery"}}),
    ("GET",  "/api/products?category=books&page=1&limit=20"),
    ("GET",  "/api/products/search?q=coffee+maker"),
    ("GET",  "/api/products/search?q=run