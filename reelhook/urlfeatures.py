"""Lexical, structural and brand-impersonation features extracted from a URL (no network access)."""
import math
import re
import unicodedata
from urllib.parse import parse_qsl, urlparse

# brand -> registered domains that legitimately belong to it
BRANDS = {
    "paypal": {"paypal.com", "paypal.me"}, "google": {"google.com", "gmail.com", "youtube.com", "googleusercontent.com",
                                                      "gstatic.com", "google.co.uk", "google.de", "google.fr"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com", "azure.com",
                  "windows.net", "office365.com", "sharepoint.com", "bing.com", "msauth.net", "msftauth.net", "s-microsoft.com",
                  "microsoftonline-p.com", "msecnd.net"},
    "apple": {"apple.com", "icloud.com"}, "amazon": {"amazon.com", "amazonaws.com", "amazon.co.uk", "amazon.de",
                                                     "amazon.in", "amazon.fr", "amazon.ca", "cloudfront.net"},
    "netflix": {"netflix.com"}, "facebook": {"facebook.com", "fb.com", "meta.com"}, "instagram": {"instagram.com"},
    "github": {"github.com", "github.io", "githubusercontent.com"}, "dropbox": {"dropbox.com"},
    "linkedin": {"linkedin.com"}, "chase": {"chase.com"}, "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"}, "citibank": {"citi.com", "citibank.com"}, "binance": {"binance.com"},
    "coinbase": {"coinbase.com"}, "dhl": {"dhl.com", "dhl.de"}, "usps": {"usps.com"}, "fedex": {"fedex.com"},
    "docusign": {"docusign.com", "docusign.net"}, "adobe": {"adobe.com"}, "zoom": {"zoom.us"},
    "whatsapp": {"whatsapp.com", "whatsapp.net"}, "telegram": {"telegram.org", "t.me"},
    "steam": {"steampowered.com", "steamcommunity.com"}, "roblox": {"roblox.com"}, "twitter": {"twitter.com", "x.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk"}, "barclays": {"barclays.co.uk", "barclays.com"}, "santander": {"santander.com", "santander.co.uk"},
    "americanexpress": {"americanexpress.com", "amex.com"}, "yahoo": {"yahoo.com"}, "dhlexpress": {"dhl.com"},
    "outlook": {"outlook.com", "live.com", "office.com"}, "office365": {"office.com", "office365.com"},
    "metamask": {"metamask.io"}, "ebay": {"ebay.com", "ebay.co.uk"}, "spotify": {"spotify.com"}, "stripe": {"stripe.com"},
}
KEYWORDS = ("login", "signin", "verify", "secure", "update", "account", "confirm", "wallet", "banking", "password",
            "suspend", "unlock", "billing", "recover", "support", "alert", "authenticate", "validate", "limited")
BAD_TLDS = {"zip", "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "work", "support", "rest", "icu", "buzz", "monster",
            "cyou", "sbs", "live", "cam", "country", "kim", "loan", "men", "party", "review", "stream", "info"}
FREE_HOSTS = {"pages.dev", "web.app", "firebaseapp.com", "weebly.com", "blogspot.com", "sites.google.com", "netlify.app",
              "vercel.app", "workers.dev", "wixsite.com", "glitch.me", "herokuapp.com", "ngrok.io", "ngrok-free.app",
              "000webhostapp.com", "godaddysites.com", "square.site", "webflow.io", "framer.app", "r2.dev", "repl.co",
              "myshopify.com", "infinityfreeapp.com", "rf.gd", "epizy.com"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "s.id"}
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$|^\[[0-9a-f:]+\]$", re.I)
_KIT_PATH = [re.compile(p) for p in [r"webscr", r"cmd=_", r"/signin\.php", r"/login\.php", r"/verify\.php", r"\.php\?.*(email|session|id)=",
                                     r"/wp-(content|includes|admin)/.*(login|verify|secure)", r"/~[a-z0-9_]+/", r"/update[-_]?(info|billing|account)",
                                     r"/(myaccount|webmail|owa)/", r"[?&]e?mail=", r"/auth/(login|verify)"]]
# characters that look like Latin letters; maps a "skeleton" back to the brand a human would read
_CONFUSABLE = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
               "ɡ": "g", "һ": "h", "к": "k", "ӏ": "l", "ο": "o", "ν": "v", "α": "a", "ρ": "p", "ι": "i", "ϲ": "c", "ѵ": "v",
               "ɑ": "a", "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j", "ᴋ": "k",
               "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r", "ꜱ": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ᴢ": "z"}
_LEET = [("rn", "m"), ("vv", "w"), ("cl", "d"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("|", "l")]

FEATURE_NAMES = [
    "url_length", "host_length", "dots", "host_hyphens", "has_at", "is_ip", "digit_ratio", "host_entropy", "subdomains",
    "bad_tld", "kw_host", "kw_path", "is_https", "punycode", "brand_misuse", "brand_typosquat", "brand_confusable",
    "brand_combosquat", "tld_swap", "free_hosting", "free_host_brand", "mixed_script", "nonascii_host", "odd_port",
    "path_depth", "query_params", "double_slash_path", "shortener", "long_hex_token", "dga_like", "query_has_email",
    "kit_path", "host_tokens",
]


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in (s.count(ch) for ch in set(s)))


def osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: edits, inserts, deletes and adjacent swaps each cost 1."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


levenshtein = osa_distance      # backwards-compatible name


def skeleton(s: str) -> str:
    """Collapse look-alike characters so 'paypa1', 'rnicrosoft' and Cyrillic 'раураl' reveal their targets."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(_CONFUSABLE.get(c, c) for c in s)
    for a, b in _LEET:
        s = s.replace(a, b)
    return s


def decode_idna(host: str) -> str:
    out = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                label = label.encode("ascii").decode("idna")
            except (UnicodeError, ValueError):
                pass
        out.append(label)
    return ".".join(out)


def scripts_in(label: str) -> set:
    s = set()
    for ch in label:
        if ch.isalpha():
            try:
                s.add(unicodedata.name(ch).split()[0])
            except ValueError:
                s.add("UNKNOWN")
    return s


def registered_domain(host: str) -> str:
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    two = ".".join(labels[-2:])
    if two in FREE_HOSTS:
        return two
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _tokens(s: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def brand_signals(host: str, path: str = "") -> dict:
    """Flags for brand misuse, typosquatting, confusables, combosquatting and TLD swaps."""
    reg = registered_domain(host)
    reg_label = reg.split(".")[0]
    toks = _tokens(host)
    ptoks = _tokens(path)
    out = {"misuse": 0, "typo": 0, "confusable": 0, "combo": 0, "tld_swap": 0}
    for brand, official in BRANDS.items():
        if reg in official:
            continue
        if brand in toks or brand in ptoks:
            out["misuse"] = 1
        if reg_label == brand:
            out["tld_swap"] = 1
        elif brand in reg_label and len(brand) >= 4:
            out["combo"] = 1             # brand plus extra words in the registered name: paypal-security-center.com
        for tok in toks + [reg_label]:
            if tok == brand:
                continue
            if len(brand) >= 5 and (tok.startswith(brand) or tok.endswith(brand)) and len(tok) <= len(brand) + 12:
                out["combo"] = 1
            if skeleton(tok) == brand and len(brand) >= 4:
                out["confusable"] = 1
            if len(brand) >= 5 and abs(len(tok) - len(brand)) <= 1 and osa_distance(tok, brand) == 1:
                out["typo"] = 1
    return out


def parse(url: str):
    if "://" not in url:
        url = "http://" + url
    try:
        return urlparse(url)
    except ValueError:                    # e.g. an unterminated IPv6 bracket in hostile input
        return urlparse(re.sub(r"[\[\]]", "", url))


def _port_ok(u):
    try:
        return u.port
    except ValueError:
        return -1


def extract(url: str) -> dict:
    u = parse(url)
    raw_host = (u.hostname or "").lower()
    host = decode_idna(raw_host)
    path = u.path or ""
    labels = host.split(".") if host else []
    is_ip = int(bool(_IP.match(raw_host)))
    brand = brand_signals(host, path) if not is_ip else {"misuse": 0, "typo": 0, "confusable": 0, "combo": 0, "tld_swap": 0}
    reg = registered_domain(host)
    tld = labels[-1] if labels else ""
    low_url = url.lower()
    digits = sum(c.isdigit() for c in host)
    longest = max(labels[:-1] or labels, key=len, default="")
    letters = sum(c.isalpha() for c in longest) or 1
    transitions = sum(1 for a, b in zip(longest, longest[1:]) if a.isdigit() != b.isdigit())
    mixed = int(any(len(scripts_in(l)) > 1 for l in labels))
    port = _port_ok(u)
    free = int(reg in FREE_HOSTS)
    return {
        "url_length": min(len(url), 200) / 100,
        "host_length": min(len(host), 80) / 40,
        "dots": min(host.count("."), 8),
        "host_hyphens": min(host.count("-"), 6),
        "has_at": int("@" in u.netloc),
        "is_ip": is_ip,
        "digit_ratio": digits / max(len(host), 1),
        "host_entropy": entropy(host) / 4,
        "subdomains": max(len(labels) - 2 - (1 if reg.count(".") == 2 else 0), 0) if not is_ip else 0,
        "bad_tld": int(tld in BAD_TLDS),
        "kw_host": min(sum(k in host for k in KEYWORDS), 4),
        "kw_path": min(sum(k in (path + "?" + (u.query or "")).lower() for k in KEYWORDS), 4),
        "is_https": int(u.scheme == "https"),
        "punycode": int("xn--" in raw_host),
        "brand_misuse": brand["misuse"],
        "brand_typosquat": brand["typo"],
        "brand_confusable": brand["confusable"],
        "brand_combosquat": brand["combo"],
        "tld_swap": brand["tld_swap"],
        "free_hosting": free,
        "free_host_brand": int(free and (brand["misuse"] or brand["combo"] or brand["typo"])),
        "mixed_script": mixed,
        "nonascii_host": int(any(ord(c) > 127 for c in host)),
        "odd_port": int(port not in (None, 80, 443)),
        "path_depth": min(path.count("/"), 8),
        "query_params": min(len(parse_qsl(u.query)), 8),
        "double_slash_path": int("//" in path),
        "shortener": int(host in SHORTENERS),
        "long_hex_token": int(bool(re.search(r"[0-9a-f]{24,}", path.lower()))),
        "dga_like": int(len(longest) >= 10 and entropy(longest) > 3.2 and transitions >= 3 and not is_ip),
        "query_has_email": int(bool(re.search(r"[?&=][\w.+-]+(@|%40)[\w-]+\.", low_url))),
        "kit_path": min(sum(bool(p.search(low_url)) for p in _KIT_PATH), 3),
        "host_tokens": min(len(_tokens(host)), 10),
    }


def vector(url: str) -> list:
    f = extract(url)
    return [float(f[n]) for n in FEATURE_NAMES]
