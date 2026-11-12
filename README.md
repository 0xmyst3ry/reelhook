# Reelhook

A zero-dependency phishing detector that judges **the URL and the page behind it**, and tells you *why*.

## What it looks at

**URL layer** (21 features, scored by a logistic regression written from scratch)
- Brand impersonation: typosquats (`paypa1.com`), homoglyphs (`rnicrosoft`), brands hiding in subdomains or paths of unrelated domains
- Structure: `user@host` tricks, raw IPs, punycode, odd ports, URL shorteners, risky TLDs, entropy, hex tokens
- Lexical: login/verify/secure keywords, length, subdomain depth

**Page layer** (optional, pass saved HTML)
- Password forms that post to a different domain, to `mailto:`, or nowhere
- Link text that shows one domain but points to another
- Hidden iframes, brand names in `<title>` that don't match the domain, right-click blocking, meta-refresh redirects, `eval/atob` obfuscation

Signals combine with noisy-OR into a score → `safe` / `suspicious` / `phishing`, with a human-readable reason list.

## Usage

```bash
python -m reelhook check "http://paypal.secure-login.xyz/verify/account"
python -m reelhook check https://quietmeadow.com/account --html saved_page.html --json
python -m reelhook eval            # train + held-out metrics (precision/recall/F1/AUC)
python -m reelhook eval --train feed.csv --test holdout.csv --save model.json
```

Exit codes: `0` safe, `1` suspicious, `2` phishing.

```python
from reelhook import check
report = check(url, html=page_html)
print(report.verdict, report.reasons)
```

## Honest note on data

The bundled model trains on a **synthetic** corpus (includes deliberately hard cases such as clean-looking phishing URLs and messy legitimate ones). The metrics from `eval` therefore describe the synthetic distribution, not the internet. For real use, train on a labeled feed (`url,label` CSV) via `--train`.

## Tests

```bash
python -m unittest discover tests
```

## License

MIT
