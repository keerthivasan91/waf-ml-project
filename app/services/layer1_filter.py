"""app/services/layer1.py — rule-based filter (Layer 1)"""
import re
from app.core.logging import logger

SQLI_RE = re.compile(
    r"union\s+.*\bselect\b|or\s+\d+=\d+|drop\s+table|--|sleep\s*\(|benchmark\s*\(",
    re.I)
XSS_RE = re.compile(
    r"<\s*script|onerror\s*=|javascript\s*:|alert\s*\(|svg.*onload",
    re.I)
LFI_RE = re.compile(
    r"\.\./|\.\.[/\\]|/etc/passwd|boot\.ini|/proc/self",
    re.I)
CMDI_RE = re.compile(
    r";\s*(ls|cat|id|whoami|wget|curl|bash|sh)\b|&&|\|\|",
    re.I)

_RULES = [
    (SQLI_RE, "sqli_rule"),
    (XSS_RE,  "xss_rule"),
    (LFI_RE,  "lfi_rule"),
    (CMDI_RE, "cmdi_rule"),
]

def check(url: str, body: str) -> tuple[bool, str]:
    """Returns (blocked: bool, reason: str)"""
    text = url + " " + body
    for pattern, label in _RULES:
        if pattern.search(text):
            logger.debug("L1 block: %s on %s", label, url[:80])
            return True, label
    return False, ""