"""Page-content analysis: spots credential-harvesting and brand-cloning tricks in saved HTML."""
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

from .urlfeatures import BRANDS, registered_domain, skeleton


@dataclass
class Signal:
    id: str
    weight: float
    note: str


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = []          # {"action", "inputs": [(type, name)], "password": bool}
        self.links = []          # (href, text)
        self.iframes = []
        self.resources = []      # absolute URLs of img/script/link/css
        self.scripts = []
        self.metas = []
        self.title = ""
        self.visible = []
        self._in_title = False
        self._in_a = None
        self._in_script = False
        self._form_stack = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            f = {"action": a.get("action", ""), "inputs": [], "password": False}
            self.forms.append(f)
            self._form_stack.append(f)
        elif tag == "input":
            kind = a.get("type", "text").lower()
            if self._form_stack:
                self._form_stack[-1]["inputs"].append((kind, (a.get("name", "") + " " + a.get("id", "") + " " + a.get("placeholder", "")).lower()))
                if kind == "password":
                    self._form_stack[-1]["password"] = True
        elif tag == "a":
            self._in_a = [a.get("href", ""), ""]
        elif tag == "iframe":
            self.iframes.append(a)
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            self.metas.append(a)
        elif tag == "script":
            self._in_script = True
        if tag in ("img", "script", "link", "source", "video", "audio"):
            src = a.get("src") or (a.get("href") if tag == "link" else None)
            if src:
                self.resources.append(src)

    def handle_endtag(self, tag):
        if tag == "form" and self._form_stack:
            self._form_stack.pop()
        elif tag == "a" and self._in_a:
            self.links.append(tuple(self._in_a))
            self._in_a = None
        elif tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_script:
            if len(data) > 20:
                self.scripts.append(data)
        elif data.strip():
            self.visible.append(data.strip())
        if self._in_a is not None:
            self._in_a[1] += data


def _host(url: str) -> str:
    url = url.strip()
    if url.startswith("//"):
        url = "http:" + url
    if not re.match(r"^[a-z][a-z0-9+.-]*:", url, re.I):
        return ""
    return (urlparse(url).hostname or "").lower()


_HIDDEN = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|width\s*[:=]\s*\"?0|height\s*[:=]\s*\"?0")
_DOMAIN_TEXT = re.compile(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})", re.I)
_SENSITIVE = {"card": re.compile(r"card|cc[-_ ]?num|cvv|cvc|expir"), "ssn": re.compile(r"ssn|social.?security|sin\b"),
              "otp": re.compile(r"otp|one.?time|2fa|verification.?code|sms.?code"), "pin": re.compile(r"\bpin\b"),
              "seed": re.compile(r"seed.?phrase|recovery.?phrase|mnemonic|private.?key")}


def _brand_of(text: str):
    t = text.lower()
    sk = skeleton(t)                       # catches look-alike spellings in the title too
    for brand in BRANDS:
        if re.search(rf"\b{brand}\b", t) or re.search(rf"\b{brand}\b", sk):
            return brand
    return None


def analyze(html: str, page_url: str) -> list[Signal]:
    c = _Collector()
    c.feed(html)
    page_host = _host(page_url if "://" in page_url else "http://" + page_url)
    page_reg = registered_domain(page_host)
    out: list[Signal] = []

    for f in c.forms:
        kinds = " ".join(f"{k} {n}" for k, n in f["inputs"])
        harvested = [name for name, rx in _SENSITIVE.items() if rx.search(kinds)]
        if harvested:
            out.append(Signal("H11", 0.45 if len(harvested) == 1 else 0.7,
                              f"Form asks for sensitive data ({', '.join(harvested)})"))
        if not f["password"] and not harvested:
            continue
        action = f["action"].strip()
        ah = _host(action)
        if action.lower().startswith("data:"):
            out.append(Signal("H12", 0.7, "Form submits to an inline data: URL"))
        elif action in ("", "#") or action.lower().startswith("javascript:"):
            out.append(Signal("H01", 0.30, "Credential form with no real submit target"))
        elif action.lower().startswith("mailto:"):
            out.append(Signal("H03", 0.85, "Credential form submits by email"))
        elif ah and registered_domain(ah) != page_reg:
            out.append(Signal("H02", 0.80, f"Credential form posts to a different domain ({ah})"))
        elif page_url.startswith("http://"):
            out.append(Signal("H04", 0.40, "Credentials collected over plain HTTP"))

    for href, text in c.links:
        m = _DOMAIN_TEXT.search(text.strip())
        hh = _host(href)
        if m and hh and registered_domain(m.group(1).lower()) != registered_domain(hh):
            out.append(Signal("H05", 0.65, f"Link text shows {m.group(1)} but goes to {hh}"))
            break

    for fr in c.iframes:
        style = fr.get("style", "") + " " + " ".join(f"{k}={v}" for k, v in fr.items() if k in ("width", "height"))
        if _HIDDEN.search(style):
            out.append(Signal("H06", 0.55, "Hidden iframe"))
            break

    brand = _brand_of(c.title)
    if brand and page_reg not in BRANDS[brand]:
        out.append(Signal("H07", 0.60, f"Page title claims '{brand}' but domain is {page_reg}"))

    # brand cloning: page borrows most of its images/scripts/styles from a brand's real servers
    hosts = [h for h in (_host(r) for r in c.resources) if h]
    if hosts:
        for b, official in BRANDS.items():
            borrowed = sum(registered_domain(h) in official for h in hosts)
            if borrowed >= 3 and borrowed / len(hosts) >= 0.5 and page_reg not in official:
                out.append(Signal("H13", 0.75, f"{borrowed}/{len(hosts)} page resources load from {b}'s servers (cloned page)"))
                break
        external = sum(registered_domain(h) != page_reg for h in hosts)
        if len(hosts) >= 6 and external / len(hosts) > 0.9 and any(f["password"] for f in c.forms):
            out.append(Signal("H14", 0.25, "Login page loads almost everything from other domains"))

    blob = " ".join(c.scripts).lower()
    if "contextmenu" in blob and "return false" in blob:
        out.append(Signal("H08", 0.35, "Disables right-click (hides page source)"))
    for m in c.metas:
        if m.get("http-equiv", "").lower() == "refresh" and "url=" in m.get("content", "").lower():
            out.append(Signal("H09", 0.35, "Meta-refresh redirect"))
            break
    if re.search(r"\b(atob|unescape|eval)\s*\(|fromcharcode", blob):
        out.append(Signal("H10", 0.40, "Obfuscated script (eval/atob/unescape/fromCharCode)"))
    if re.search(r"(window\.)?location(\.href)?\s*=\s*['\"]https?://", blob) and c.forms:
        out.append(Signal("H15", 0.30, "Script redirects the visitor to another site"))
    return out
