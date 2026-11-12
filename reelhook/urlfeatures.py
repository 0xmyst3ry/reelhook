"""Lexical and structural features extracted from a URL (no network access)."""
import math
import re
from urllib.parse import urlparse, parse_qsl

BRANDS = {
    "paypal": "paypal.com", "google": "google.com", "microsoft": "microsoft.com",
    "apple": "apple.com", "amazon": "amazon.com", "netflix": "netflix.com",
    "facebook": "facebook.com", "instagram": "instagram.com", "github": "github.com",
    "dropbox": "dropbox.com", "linkedin": "linkedin.com", "chase": "chase.com",
    "wellsfargo": "wellsfargo.com", "binance": "binance.com", "coinbase": "coinbase.com",
}
KEYWORDS = ("login", "signin", "verify", "secure", "update", "account", "confirm",
            "wallet", "banking", "password", "suspend", "unlock", "billing", "recover")
BAD_TLDS = {"zip", "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "work", "support", "rest", "icu", "buzz"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly", "rb.gy"}
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_HOMOGLYPHS = [("rn", "m"), ("vv", "w"), ("cl", "d"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("|", "l")]

FEATURE_NAMES = [
    "url_length", "host_length", "dots", "host_hyphens", "has_at", "is_ip", "digit_ratio",
    "host_entropy", "subdomains", "bad_tld", "keyword_hits", "is_https", "punycode",
    "brand_misuse", "brand_typosquat", "odd_port", "path_depth", "query_params",
    "double_slash_path", "shortener", "long_hex_token",
]


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in (s.count(ch) for ch in set(s)))


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def skeleton(s: str) -> str:
    """Collapse look-alike characters so 'paypa1' and 'rnicrosoft' reveal their targets."""
    s = s.lower()
    for a, b in _HOMOGLYPHS:
        s = s.replace(a, b)
    return s


def registered_domain(host: str) -> str:
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _brand_signals(host: str, path: str) -> tuple[int, int]:
    """Return (misuse, typosquat) flags."""
    reg = registered_domain(host)
    reg_name = reg.split(".")[0]
    misuse = typo = 0
    haystack = host + path.lower()
    for brand, real in BRANDS.items():
        if reg == real:
            continue
        if brand in haystack:
            misuse = 1
        for tok in re.split(r"[.\-_]", host):
            if tok == brand:
                continue
            sk = skeleton(tok)
            if sk == brand or (len(brand) >= 5 and tok != brand and levenshtein(tok, brand) == 1):
                typo = 1
        if reg_name != brand and (skeleton(reg_name) == brand):
            typo = 1
    return misuse, typo


def parse(url: str):
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def extract(url: str) -> dict:
    u = parse(url)
    host = (u.hostname or "").lower()
    path = u.path or ""
    labels = host.split(".") if host else []
    is_ip = int(bool(_IP.match(host)))
    misuse, typo = _brand_signals(host, path) if not is_ip else (0, 0)
    tld = labels[-1] if labels else ""
    full = url.lower()
    digits = sum(c.isdigit() for c in host)
    return {
        "url_length": min(len(url), 200) / 100,
        "host_length": min(len(host), 80) / 40,
        "dots": min(host.count("."), 8),
        "host_hyphens": min(host.count("-"), 6),
        "has_at": int("@" in u.netloc),
        "is_ip": is_ip,
        "digit_ratio": digits / max(len(host), 1),
        "host_entropy": entropy(host) / 4,
        "subdomains": max(len(labels) - 2, 0) if not is_ip else 0,
        "bad_tld": int(tld in BAD_TLDS),
        "keyword_hits": min(sum(k in full for k in KEYWORDS), 5),
        "is_https": int(u.scheme == "https"),
        "punycode": int("xn--" in host),
        "brand_misuse": misuse,
        "brand_typosquat": typo,
        "odd_port": int(u.port not in (None, 80, 443)) if _safe_port(u) else 1,
        "path_depth": min(path.count("/"), 8),
        "query_params": min(len(parse_qsl(u.query)), 8),
        "double_slash_path": int("//" in path),
        "shortener": int(host in SHORTENERS),
        "long_hex_token": int(bool(re.search(r"[0-9a-f]{24,}", path.lower()))),
    }


def _safe_port(u) -> bool:
    try:
        u.port
        return True
    except ValueError:
        return False


def vector(url: str) -> list[float]:
    f = extract(url)
    return [float(f[n]) for n in FEATURE_NAMES]
