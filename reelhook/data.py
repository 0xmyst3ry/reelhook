"""Synthetic URL corpus generator, plus a CSV loader for real datasets.

The generator mixes easy and deliberately hard cases (clean-looking phishing,
messy-looking legitimate sites) so the model has to learn trade-offs, not a
single giveaway feature. For production use, train on a real feed via load_csv().
"""
import csv
import random

from .urlfeatures import BRANDS, BAD_TLDS, KEYWORDS, SHORTENERS

_WORDS = ["blue", "river", "stone", "cloud", "pixel", "garden", "north", "swift", "maple", "orbit",
          "harbor", "lumen", "atlas", "cedar", "nimbus", "forge", "tidal", "quartz", "meadow", "ember"]
_PATHS = ["blog", "products", "about", "docs", "news", "help", "pricing", "contact", "careers", "shop"]
_TLDS = ["com", "org", "net", "io", "co", "dev", "app", "edu"]


def _word(r):
    return r.choice(_WORDS)


def _typo(brand, r):
    mode = r.choice(["swap", "digit", "drop", "double", "rn"])
    i = r.randrange(1, len(brand) - 1)
    if mode == "swap":
        return brand[:i] + brand[i + 1] + brand[i] + brand[i + 2:] if i + 2 <= len(brand) else brand + "x"
    if mode == "digit":
        return brand.replace("o", "0", 1).replace("l", "1", 1) if ("o" in brand or "l" in brand) else brand + "1"
    if mode == "drop":
        return brand[:i] + brand[i + 1:]
    if mode == "double":
        return brand[:i] + brand[i] + brand[i:]
    return brand.replace("m", "rn", 1) if "m" in brand else brand + "n"


def benign(r: random.Random) -> str:
    kind = r.random()
    if kind < 0.35:
        brand = r.choice(list(BRANDS))
        sub = r.choice(["www.", "", "accounts.", "mail.", "support.", "login."])
        path = r.choice(["", "/", f"/{r.choice(_PATHS)}", "/signin", "/account/settings", "/login?next=/home"])
        return f"https://{sub}{BRANDS[brand]}{path}"
    host = f"{_word(r)}{r.choice(['', '-', ''])}{_word(r)}.{r.choice(_TLDS)}"
    if r.random() < 0.15:
        host = f"{r.choice(['app','docs','blog','shop'])}.{host}"
    scheme = "https" if r.random() < 0.9 else "http"
    path = "/" + "/".join(r.choice(_PATHS) for _ in range(r.randint(0, 3)))
    if r.random() < 0.2:
        path += f"?id={r.randint(1, 9999)}&ref={_word(r)}"
    if r.random() < 0.08:
        path = "/account/" + r.choice(["login", "verify-email", "update-profile"])
    return f"{scheme}://{host}{path}"


def phishing(r: random.Random) -> str:
    kind = r.random()
    brand = r.choice(list(BRANDS))
    kw = r.choice(KEYWORDS)
    tld = r.choice(sorted(BAD_TLDS) + ["com", "net", "info"])
    if kind < 0.18:      # typosquat
        host = f"{_typo(brand, r)}.{r.choice(['com', 'net', tld])}"
        return f"{r.choice(['http', 'https'])}://{host}/{kw}"
    if kind < 0.40:      # brand as subdomain / path of an unrelated domain
        host = f"{brand}.{kw}-{_word(r)}.{tld}"
        return f"{r.choice(['http', 'https'])}://{host}/{r.choice(['', kw, 'auth/' + kw])}"
    if kind < 0.55:      # raw IP
        ip = ".".join(str(r.randint(1, 254)) for _ in range(4))
        port = r.choice(["", ":8080", ":8443"])
        return f"http://{ip}{port}/{brand}/{kw}.php?id={r.randint(1000, 99999)}"
    if kind < 0.68:      # credentials-in-URL trick
        return f"https://{brand}.com@{_word(r)}-{kw}.{tld}/{kw}"
    if kind < 0.78:      # shortener
        return f"https://{r.choice(sorted(SHORTENERS))}/{''.join(r.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(7))}"
    if kind < 0.86:      # punycode
        return f"https://xn--{_word(r)}-{r.randint(10, 99)}a.{tld}/{brand}/{kw}"
    # hard: clean-looking, no giveaway
    return f"https://{_word(r)}{_word(r)}.{r.choice(['com', 'net', 'org'])}/{kw}/{r.randint(100, 999)}"


def generate(n: int, seed: int = 1, phish_ratio: float = 0.5):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        if r.random() < phish_ratio:
            rows.append((phishing(r), 1))
        else:
            rows.append((benign(r), 0))
    return rows


def load_csv(path: str):
    """CSV with columns: url,label (1 = phishing)."""
    with open(path, newline="") as f:
        return [(row["url"], int(row["label"])) for row in csv.DictReader(f)]
