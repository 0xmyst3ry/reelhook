"""Email triage: header forensics, content cues, attachments, and every link run through the URL model."""
import email
import re
from dataclasses import dataclass, field
from email import policy
from email.utils import parseaddr
from html.parser import HTMLParser

from .htmlscan import Signal
from .urlfeatures import BRANDS, registered_domain

_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
_DOMAIN_TEXT = re.compile(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})", re.I)
RISKY_EXT = {"html", "htm", "shtml", "iso", "img", "lnk", "js", "jse", "vbs", "vbe", "exe", "scr", "bat", "cmd", "com", "msi",
             "jar", "docm", "xlsm", "pptm", "one", "hta", "ps1", "svg", "chm"}
DOC_EXT = {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "jpg", "jpeg", "png", "zip"}
URGENCY = [r"\burgent(ly)?\b", r"\bimmediately\b", r"within \d+ (hours?|days?)", r"\bsuspend(ed)?\b", r"\bterminated?\b",
           r"final (notice|warning)", r"account (will be )?(closed|locked|limited|disabled)", r"\bunusual (sign|activity)",
           r"\baction required\b", r"\blast chance\b", r"expires? (today|soon)"]
CREDENTIAL = [r"verify your (account|identity|information|email)", r"confirm your (password|identity|account|details)",
              r"update your (billing|payment|account|password)", r"re-?enter your", r"(login|sign.?in) (to|here|below)",
              r"click (here|the link|below) to (verify|confirm|update|restore|unlock)", r"security (alert|check)",
              r"reset your password", r"\b(gift card|wire transfer|bitcoin|crypto)\b"]
GENERIC_GREETING = re.compile(r"\bdear (customer|user|client|member|valued|account holder)\b", re.I)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self._a, self.text = [], None, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._a = [dict(attrs).get("href", ""), ""]

    def handle_endtag(self, tag):
        if tag == "a" and self._a:
            self.links.append(tuple(self._a))
            self._a = None

    def handle_data(self, data):
        self.text.append(data)
        if self._a is not None:
            self._a[1] += data


@dataclass
class EmailReport:
    score: float
    verdict: str
    signals: list = field(default_factory=list)
    urls: list = field(default_factory=list)       # [(url, Report)]

    def to_dict(self):
        return {"score": round(self.score, 3), "verdict": self.verdict,
                "reasons": [f"[{s.id}] {s.note}" for s in self.signals],
                "urls": [{"url": u, "verdict": r.verdict, "score": round(r.score, 3)} for u, r in self.urls]}


def _domain(addr: str) -> str:
    return parseaddr(addr)[1].rsplit("@", 1)[-1].lower() if "@" in parseaddr(addr)[1] else ""


def _body(msg):
    text, html = [], []
    for part in msg.walk():
        ct = part.get_content_type()
        if part.get_filename():
            continue
        if ct == "text/plain":
            text.append(part.get_content())
        elif ct == "text/html":
            html.append(part.get_content())
    return "\n".join(text), "\n".join(html)


def analyze_email(raw: str | bytes, check_url) -> EmailReport:
    """`check_url(url) -> Report` is injected so this module stays independent of the model code."""
    msg = email.message_from_bytes(raw, policy=policy.default) if isinstance(raw, bytes) else email.message_from_string(raw, policy=policy.default)
    sig: list[Signal] = []
    name, addr = parseaddr(str(msg.get("From", "")))
    from_dom = _domain(addr)
    from_reg = registered_domain(from_dom) if from_dom else ""

    shown = f"{name} {addr}".lower()
    for brand, official in BRANDS.items():
        if re.search(rf"\b{brand}\b", name.lower()) and from_reg and from_reg not in official:
            sig.append(Signal("E01", 0.60, f"Sender name says '{brand}' but mail comes from {from_reg}"))
            break
    reply_dom = _domain(str(msg.get("Reply-To", "")))
    if reply_dom and from_reg and registered_domain(reply_dom) != from_reg:
        sig.append(Signal("E02", 0.35, f"Reply-To goes to {registered_domain(reply_dom)}, not the sender's {from_reg}"))
    rp = _domain(str(msg.get("Return-Path", "")))
    if rp and from_reg and registered_domain(rp) != from_reg:
        sig.append(Signal("E03", 0.20, f"Envelope sender {registered_domain(rp)} differs from the visible From domain"))

    auth = " ".join(str(h) for h in msg.get_all("Authentication-Results", [])).lower()
    spf = re.search(r"spf=(\w+)", auth)
    dkim = re.search(r"dkim=(\w+)", auth)
    dmarc = re.search(r"dmarc=(\w+)", auth)
    if dmarc and dmarc.group(1) == "fail":
        sig.append(Signal("E04", 0.50, "DMARC failed"))
    if spf and spf.group(1) in ("fail", "softfail"):
        sig.append(Signal("E05", 0.35, f"SPF {spf.group(1)}: sending server is not authorised for this domain"))
    if dkim and dkim.group(1) == "fail":
        sig.append(Signal("E06", 0.35, "DKIM signature failed"))
    if not auth.strip():
        sig.append(Signal("E07", 0.10, "No authentication results recorded"))

    text, html = _body(msg)
    subject = str(msg.get("Subject", ""))
    blob = f"{subject}\n{text}\n{re.sub(r'<[^>]+>', ' ', html)}".lower()
    urgency = sum(bool(re.search(p, blob)) for p in URGENCY)
    cred = sum(bool(re.search(p, blob)) for p in CREDENTIAL)
    if urgency >= 2:
        sig.append(Signal("E08", 0.30, f"Pressure language ({urgency} urgency cues)"))
    if cred >= 1:
        sig.append(Signal("E09", 0.40 if cred == 1 else 0.6, f"Asks you to verify/confirm/update credentials ({cred} cues)"))
    if GENERIC_GREETING.search(blob):
        sig.append(Signal("E10", 0.15, "Generic greeting instead of your name"))

    for part in msg.iter_attachments():
        fn = (part.get_filename() or "").lower()
        exts = fn.split(".")[1:]
        if len(exts) >= 2 and exts[-2] in DOC_EXT and exts[-1] in RISKY_EXT:
            sig.append(Signal("E11", 0.85, f"Double-extension attachment '{fn}'"))
        elif exts and exts[-1] in RISKY_EXT:
            sig.append(Signal("E12", 0.50, f"Risky attachment type '.{exts[-1]}' ({fn})"))

    parser = _Links()
    parser.feed(html)
    for href, label in parser.links:
        m = _DOMAIN_TEXT.search(label.strip())
        hh = (re.match(r"https?://([^/:?#]+)", href.strip(), re.I) or [None, ""])[1].lower()
        if m and hh and registered_domain(m.group(1).lower()) != registered_domain(hh):
            sig.append(Signal("E13", 0.65, f"Link text shows {m.group(1)} but goes to {hh}"))
            break

    urls = list(dict.fromkeys([h for h, _ in parser.links if h.lower().startswith("http")] + _URL.findall(text)))[:15]
    scored = [(u, check_url(u)) for u in urls]
    risky = [(u, r) for u, r in scored if r.verdict != "safe"]
    if risky:
        u, r = max(risky, key=lambda t: t[1].score)
        sig.append(Signal("E14", min(0.9, r.score), f"Link looks {r.verdict}: {u[:70]}"))

    miss = 1.0
    for s in sig:
        miss *= 1 - s.weight
    score = 1 - miss
    verdict = "phishing" if score >= 0.75 else "suspicious" if score >= 0.45 else "safe"
    return EmailReport(score, verdict, sorted(sig, key=lambda s: -s.weight), scored)
