"""Synthetic URL corpus generator, plus a CSV loader for real datasets.

The generator mixes easy and deliberately hard cases (clean-looking phishing,
messy-looking legitimate sites) so models have to learn trade-offs, not one giveaway.
It is NOT a substitute for real data: train on a real feed via load_csv() for production.
"""
import csv
import random

from .urlfeatures import BAD_TLDS, BRANDS, FREE_HOSTS, KEYWORDS, SHORTENERS

_WORDS = ["blue", "river", "stone", "cloud", "pixel", "garden", "north", "swift", "maple", "orbit", "harbor", "lumen",
          "atlas", "cedar", "nimbus", "forge", "tidal", "quartz", "meadow", "ember", "falcon", "willow", "copper", "summit",
          "lantern", "harvest", "anchor", "prairie", "velvet", "granite"]
_PATHS = ["blog", "products", "about", "docs", "news", "help", "pricing", "contact", "careers", "shop", "events", "team",
          "guides", "faq", "press", "downloads", "support"]
_TLDS = ["com", "org", "net", "io", "co", "dev", "app", "edu", "co.uk", "de"]
_BRAND_LIST = sorted(BRANDS)
_CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}
_FREE = sorted(FREE_HOSTS - {"sites.google.com", "blogspot.com"})


def _word(r):
    return r.choice(_WORDS)


def _real(brand, r):
    return r.choice(sorted(BRANDS[brand]))


def _typo(brand, r):
    mode = r.choice(["swap", "digit", "drop", "double", "rn", "replace"])
    i = r.randrange(1, len(brand) - 1)
    if mode == "swap":
        return brand[:i] + brand[i + 1] + brand[i] + brand[i + 2:]
    if mode == "digit":
        out = brand.replace("o", "0", 1).replace("l", "1", 1)
        return out if out != brand else brand + "1"
    if mode == "drop":
        return brand[:i] + brand[i + 1:]
    if mode == "double":
        return brand[:i] + brand[i] + brand[i:]
    if mode == "replace":
        return brand[:i] + r.choice("aeiou") + brand[i + 1:]
    return brand.replace("m", "rn", 1) if "m" in brand else brand + "n"


def _cyrillic(brand, r):
    swaps = [k for k in _CYR if k in brand]
    if not swaps:
        return brand
    out = brand
    for k in r.sample(swaps, min(len(swaps), r.randint(1, 3))):
        out = out.replace(k, _CYR[k], 1)
    return out


def _kw(r):
    return r.choice(KEYWORDS)


def benign(r: random.Random) -> str:
    k = r.random()
    if k < 0.20:                                  # legitimate brand pages, login pages included
        brand = r.choice(_BRAND_LIST)
        sub = r.choice(["www.", "", "accounts.", "mail.", "support.", "login.", "secure.", "my.", "help."])
        path = r.choice(["", "/", f"/{r.choice(_PATHS)}", "/signin", "/login", "/account/settings", "/login?next=/home",
                         "/security/verify-device", "/billing/update-payment", "/help/password-reset"])
        return f"https://{sub}{_real(brand, r)}{path}"
    if k < 0.28:                                  # big cloud / CDN URLs with long random paths
        host = r.choice(["d1x7q9k2m3.cloudfront.net", "s3.amazonaws.com", "storage.googleapis.com", "cdn.jsdelivr.net",
                         "raw.githubusercontent.com", "docs.google.com", "drive.google.com", "onedrive.live.com"])
        return f"https://{host}/{''.join(r.choice('abcdef0123456789') for _ in range(r.randint(12, 30)))}/{_word(r)}.{r.choice(['pdf','png','js','css','zip'])}"
    if k < 0.34:                                  # personal / small projects on free hosts
        return f"https://{_word(r)}-{_word(r)}.{r.choice(['github.io', 'netlify.app', 'vercel.app', 'pages.dev'])}/{r.choice(['', _word(r), 'docs/' + _word(r)])}"
    if k < 0.38:                                  # internal / dev addresses
        return f"http://{r.choice(['localhost', '127.0.0.1', '192.168.1.%d' % r.randint(2, 250), '10.0.%d.%d' % (r.randint(0, 9), r.randint(1, 250))])}:{r.choice([3000, 8000, 8080, 5173])}/{r.choice(['', 'admin', 'login', 'api/v1/status'])}"
    if k < 0.50:                                  # institutions, news, forums, blogs, marketplaces
        kind = r.randrange(7)
        slug = "-".join(_word(r) for _ in range(r.randint(2, 5)))
        if kind == 0:
            return f"https://www.{_word(r)}.{r.choice(['gov', 'gov.uk', 'edu', 'ac.uk', 'mil'])}/{r.choice(['', 'about', 'admissions', 'services/' + _word(r), 'flu/index.htm', 'topics/' + slug])}"
        if kind == 1:
            return f"https://www.{_word(r)}{_word(r)}.com/{r.randint(2019, 2025)}/{r.randint(1, 12):02d}/{r.randint(1, 28):02d}/{slug}.html"
        if kind == 2:
            return f"https://{r.choice(['medium.com', 'substack.com', 'dev.to', 'hashnode.dev'])}/@{_word(r)}/{slug}-{''.join(r.choice('0123456789abcdef') for _ in range(12))}"
        if kind == 3:
            return f"https://{r.choice(['news.ycombinator.com/item?id=', 'www.reddit.com/r/' + _word(r) + '/comments/', 'stackoverflow.com/questions/'])}{r.randint(1000000, 99999999)}"
        if kind == 4:
            return f"https://www.{r.choice(['etsy', 'ebay', 'walmart', 'target', 'ikea'])}.com/{r.choice(['listing', 'itm', 'p', 'ip'])}/{r.randint(10**8, 10**10)}/{slug}"
        if kind == 5:
            return f"http://192.168.{r.randint(0, 9)}.{r.randint(1, 250)}/{r.choice(['login.html', 'admin', 'setup.cgi', 'index.html'])}"
        return f"https://{r.choice(['support', 'accounts', 'my', 'portal', 'help'])}.{_word(r)}{_word(r)}.{r.choice(_TLDS)}/{r.choice(['signin', 'login', 'account', 'password-reset', 'billing'])}"
    host = f"{_word(r)}{r.choice(['', '-', ''])}{_word(r)}.{r.choice(_TLDS)}"
    if r.random() < 0.2:
        host = f"{r.choice(['app', 'docs', 'blog', 'shop', 'api', 'status', 'eu-west-1.console'])}.{host}"
    scheme = "https" if r.random() < 0.9 else "http"
    path = "/" + "/".join(r.choice(_PATHS) for _ in range(r.randint(0, 3)))
    if r.random() < 0.25:
        path += f"?id={r.randint(1, 99999)}&ref={_word(r)}&utm_source={_word(r)}"
    if r.random() < 0.10:
        path = "/account/" + r.choice(["login", "verify-email", "update-profile", "billing", "signin"])
    return f"{scheme}://{host}{path}"


def phishing(r: random.Random) -> str:
    k = r.random()
    brand = r.choice(_BRAND_LIST)
    kw = _kw(r)
    tld = r.choice(sorted(BAD_TLDS) + ["com", "net", "info", "online", "site", "co"])
    scheme = r.choice(["http", "https"])
    if k < 0.12:                                  # typosquat
        return f"{scheme}://{_typo(brand, r)}.{r.choice(['com', 'net', tld])}/{kw}"
    if k < 0.24:                                  # brand as a subdomain of an unrelated domain
        return f"{scheme}://{brand}.{kw}-{_word(r)}.{tld}/{r.choice(['', kw, 'auth/' + kw])}"
    if k < 0.36:                                  # combosquatting: brand + words
        glue = r.choice(["-", "", "-"])
        return f"{scheme}://{brand}{glue}{kw}{r.choice(['', '-' + _word(r), '-center', '-team'])}.{tld}/{r.choice([kw, 'account/' + kw, ''])}"
    if k < 0.46:                                  # raw IP and kit path
        ip = ".".join(str(r.randint(1, 254)) for _ in range(4))
        return f"http://{ip}{r.choice(['', ':8080', ':8443'])}/{r.choice(['~' + brand + '/', brand + '/', 'wp-content/'])}{kw}.php?{r.choice(['id', 'session', 'email'])}={r.randint(1000, 99999)}"
    if k < 0.54:                                  # user@host trick
        return f"https://{brand}.com@{_word(r)}-{kw}.{tld}/{kw}"
    if k < 0.62:                                  # shortener
        return f"https://{r.choice(sorted(SHORTENERS))}/{''.join(r.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(7))}"
    if k < 0.70:                                  # punycode / unicode homoglyph
        word = _cyrillic(brand, r)
        if r.random() < 0.5:
            return f"https://{word}.{r.choice(['com', 'net', tld])}/{kw}"
        return f"https://xn--{_word(r)}-{r.randint(10, 99)}a.{tld}/{brand}/{kw}"
    if k < 0.80:                                  # abuse of free hosting
        return f"https://{brand}-{kw}{r.choice(['', '-' + _word(r)])}.{r.choice(_FREE)}/{r.choice(['', kw, 'login.html'])}"
    if k < 0.86:                                  # TLD swap
        return f"https://www.{brand}.{r.choice(['co', 'net', 'org', 'info', 'us', 'cc'])}/{kw}"
    if k < 0.91:                                  # email-harvest kit parameters
        return f"https://{_word(r)}{r.randint(10, 999)}.{tld}/{r.choice(['verify', 'secure', 'doc'])}/index.html?email={_word(r)}%40{_word(r)}.com&id={r.randint(1000, 9999)}"
    # hard: clean-looking, no giveaway (the model should sometimes miss these)
    return f"https://{_word(r)}{_word(r)}.{r.choice(['com', 'net', 'org'])}/{r.choice(['doc', 'file', 'shared', 'view'])}/{r.randint(10000, 99999)}"


def generate(n: int, seed: int = 1, phish_ratio: float = 0.5):
    r = random.Random(seed)
    return [(phishing(r), 1) if r.random() < phish_ratio else (benign(r), 0) for _ in range(n)]


def load_csv(path: str):
    """CSV with columns: url,label (1 = phishing)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["url"], int(row["label"])) for row in csv.DictReader(f)]
