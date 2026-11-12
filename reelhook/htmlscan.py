"""Page-content analysis: spots credential-harvesting tricks in saved HTML."""
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

from .urlfeatures import BRANDS, registered_domain


@dataclass
class Signal:
    id: str
    weight: float
    note: str


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = []          # {"action": str, "password": bool}
        self.links = []          # (href, text)
        self.iframes = []        # attrs dicts
        self.scripts = []
        self.metas = []
        self.title = ""
        self._in_title = False
        self._in_a = None
        self._form_stack = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            f = {"action": a.get("action", ""), "password": False}
            self.forms.append(f)
            self._form_stack.append(f)
        elif tag == "input" and a.get("type", "").lower() == "password" and self._form_stack:
            self._form_stack[-1]["password"] = True
        elif tag == "a":
            self._in_a = [a.get("href", ""), ""]
        elif tag == "iframe":
            self.iframes.append(a)
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            self.metas.append(a)

    def handle_endtag(self, tag):
        if tag == "form" and self._form_stack:
            self._form_stack.pop()
        elif tag == "a" and self._in_a:
            self.links.append(tuple(self._in_a))
            self._in_a = None
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_a is not None:
            self._in_a[1] += data
        self.scripts.append(data) if len(data) > 20 else None


def _host(url: str) -> str:
    if url.startswith("//"):
        url = "http:" + url
    return (urlparse(url).hostname or "").lower()


_HIDDEN = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|width\s*[:=]\s*\"?0|height\s*[:=]\s*\"?0")
_DOMAIN_TEXT = re.compile(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})", re.I)


def analyze(html: str, page_url: str) -> list[Signal]:
    c = _Collector()
    c.feed(html)
    page_host = _host(page_url if "://" in page_url else "http://" + page_url)
    page_reg = registered_domain(page_host)
    out: list[Signal] = []

    for f in c.forms:
        if not f["password"]:
            continue
        action = f["action"].strip()
        ah = _host(action)
        if action in ("", "#") or action.lower().startswith("javascript:"):
            out.append(Signal("H01", 0.30, "Password form with no real submit target"))
        elif ah and registered_domain(ah) != page_reg:
            out.append(Signal("H02", 0.80, f"Password form posts to a different domain ({ah})"))
        elif action.lower().startswith("mailto:"):
            out.append(Signal("H03", 0.85, "Password form submits by email"))
        elif page_url.startswith("http://"):
            out.append(Signal("H04", 0.40, "Password collected over plain HTTP"))
            break

    for href, text in c.links:
        m = _DOMAIN_TEXT.search(text.strip())
        hh = _host(href)
        if m and hh and registered_domain(m.group(1).lower()) != registered_domain(hh):
            out.append(Signal("H05", 0.65, f"Link text shows {m.group(1)} but goes to {hh}"))
            break

    for fr in c.iframes:
        style = (fr.get("style", "") + " " + " ".join(f'{k}={v}' for k, v in fr.items() if k in ("width", "height")))
        if _HIDDEN.search(style):
            out.append(Signal("H06", 0.55, "Hidden iframe"))
            break

    title = c.title.lower()
    for brand, real in BRANDS.items():
        if brand in title and page_reg != real:
            out.append(Signal("H07", 0.60, f"Page title claims '{brand}' but domain is {page_reg}"))
            break

    blob = " ".join(c.scripts).lower()
    if "contextmenu" in blob and "return false" in blob:
        out.append(Signal("H08", 0.35, "Disables right-click (hides page source)"))
    for m in c.metas:
        if m.get("http-equiv", "").lower() == "refresh" and "url=" in m.get("content", "").lower():
            out.append(Signal("H09", 0.35, "Meta-refresh redirect"))
            break
    if re.search(r"\b(atob|unescape|eval)\s*\(", blob):
        out.append(Signal("H10", 0.40, "Obfuscated script (eval/atob/unescape)"))
    return out
