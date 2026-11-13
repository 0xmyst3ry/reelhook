# Reelhook

**Phishing detection for URLs, the pages behind them, and emails**, with explanations, not just a score.

Zero dependencies, Python 3.10+. The gradient-boosted trees, the logistic regression and the n-gram model are all implemented from scratch.

## What it looks at

**URL layer** (33 features + a raw-string model)
- *Brand impersonation*: ~40 brands with their legitimate domains. Typosquats (`paypa1.com`, `arnazon.com`), **homoglyphs** (Cyrillic `раураl.com`, `xn--` punycode decoded and checked for mixed scripts), combosquatting (`paypal-security-center.com`), TLD swaps (`paypal.co`), brands buried in subdomains/paths, free-hosting abuse (`netflix-billing.web.app`).
- *Structure*: `user@host` tricks, raw IPs, odd ports, shorteners, risky TLDs, entropy / DGA-like labels, kit paths (`/webscr`, `?email=`), emails in query strings.

**Page layer** (pass the saved HTML)
- Credential forms posting to another domain, to `mailto:` or an inline `data:` URL; forms harvesting card/SSN/OTP/seed-phrase fields.
- **Brand cloning**: most images/scripts/styles loaded from the brand's real servers while the domain isn't the brand's.
- Link text that shows one domain but goes to another, brand names in `<title>` (look-alike spellings included), hidden iframes, right-click blocking, meta/JS redirects, `eval/atob/fromCharCode` obfuscation.

**Email layer** (`.eml`)
- Display-name spoofing, Reply-To / Return-Path mismatches, **SPF / DKIM / DMARC** results, urgency and credential-request language, generic greetings, risky or double-extension attachments (`invoice.pdf.exe`), link-text mismatches, and **every link run through the URL models**.

## Models

| model | what it is |
|---|---|
| logistic regression | standardised features, SGD, L2 |
| **gradient-boosted trees** | Newton boosting on logistic loss, histogram splits, depth-3 trees; per-prediction **Saabas attributions** say which features pushed the score |
| char n-gram | hashed 3-5-grams of the URL, sparse SGD logistic regression |
| ensemble | mean of the three; alert thresholds are set from held-out data at 5% and 1% false-positive rate |

Signals from the page/email layers combine with the model score by noisy-OR into `safe` / `suspicious` / `phishing`.

## Try it

```bash
python -m reelhook check "http://paypal.secure-login.xyz/verify/account"
python -m reelhook check https://ms-login-portal.top/common/oauth2 --html examples/phish_page.html
python -m reelhook email examples/phish_email.eml
python -m reelhook eval                    # model comparison + realistic hold-out
python -m reelhook train                   # retrain the shipped model (add --train feed.csv for real data)
python -m reelhook serve                   # POST /check {"url","html"}, POST /email {"raw"}
```

Exit codes: `0` safe, `1` suspicious, `2` phishing.

```
PHISHING  score=1.00  (url models 0.89: lr 0.76, gbdt 0.98, ngram 0.95)
  - bad tld (+2.2)
  - kw host (+2.1)
  - path depth (+0.5)
  - [H11] Form asks for sensitive data (otp)
  - [H02] Credential form posts to a different domain (collector.evil-kit.top)
  - [H05] Link text shows www.microsoft.com but goes to evil-kit.top
  - [H07] Page title claims 'microsoft' but domain is ms-login-portal.top
  - [H13] 5/5 page resources load from microsoft's servers (cloned page)
  - [H08] Disables right-click (hides page source)
  - [H10] Obfuscated script (eval/atob/unescape/fromCharCode)
```

```
PHISHING  score=1.00
  - [E14] Link looks phishing: http://paypa1-security.top/verify/account?email=victim%40example.com
  - [E11] Double-extension attachment 'invoice.pdf.exe'
  - [E13] Link text shows www.paypal.com but goes to paypa1-security.top
  - [E01] Sender name says 'paypal' but mail comes from paypa1-security.top
  - [E04] DMARC failed
  - [E09] Asks you to verify/confirm/update credentials (1 cues)
  - [E02] Reply-To goes to gmail-recover.xyz, not the sender's paypa1-security.top
  - [E05] SPF fail: sending server is not authorised for this domain
  - [E06] DKIM signature failed
  - [E08] Pressure language (6 urgency cues)
  - [E03] Envelope sender bulk-sender.ru differs from the visible From domain
  - [E10] Generic greeting instead of your name
  link: phishing   http://paypa1-security.top/verify/account?email=victim%40example.com
```

## Evaluation (and why there are two test sets)

`python -m reelhook eval`:

```
A. held-out synthetic data (same generator as training): 506 phishing / 494 legitimate
model                 AUC  TPR@1%FPR  precision  recall     F1
logistic (feat)     0.989      0.895      1.000   0.891  0.943
GBDT (feat)         0.996      0.891      0.954   0.990  0.972
char n-gram         1.000      0.994      0.996   0.988  0.992
ensemble            1.000      0.996      1.000   0.992  0.996

B. hand-written realistic URLs (different author, different style): 64 phishing / 75 legitimate
model                 AUC  TPR@1%FPR  precision  recall     F1
logistic (feat)     0.957      0.797      0.981   0.797  0.879
GBDT (feat)         0.968      0.781      0.911   0.797  0.850
char n-gram         0.952      0.641      0.744   0.906  0.817
ensemble            0.993      0.797      0.981   0.812  0.889

5-fold CV AUC on training corpus: lr 0.992 (min 0.989), gbdt 0.996 (min 0.994)
```

- **Set A** comes from the same generator as the training data. The n-gram model scores a perfect 1.000 there, which is exactly what *memorising a generator* looks like.
- **Set B** is 139 URLs written by hand in a different style (real login pages of big brands, long CDN links, internal IPs, versus typosquats, homoglyphs, free-host abuse and clean-looking domains). Here the n-gram model drops to 0.952 AUC, the feature models hold up better, and **the ensemble is best (0.993 AUC)**.

With the shipped thresholds, on Set B:

```
shipped thresholds suspicious>=0.35, phishing>=0.60
  flagged (suspicious+): recall 61/64, false alarms 3/75
  blocked (phishing):    recall 51/64, false alarms 0/75
```

## Honest limits

- **The training data is synthetic.** The hand-written Set B is the only realistic check, it is small, and the author of the generator and of Set B is the same person. Treat the numbers as optimistic. After a first evaluation showed the n-gram model over-flagging legitimate government/news/blog URLs, the generator's benign coverage was widened; Set B was not otherwise used for training or tuning.
- For production, retrain on a real labelled feed: `python -m reelhook train --train feed.csv` (CSV `url,label`).
- Clean-looking phishing on fresh generic domains, and abuse of legitimate sites (`sites.google.com`, document-sharing pages), are the main misses. The page and email layers exist to catch what the URL alone cannot.
- No live fetching, no redirect following, no WHOIS/age/certificate data; it analyses what you give it.
- Brand list and legitimate-domain sets are finite and need maintenance.

## Tests

```bash
python -m unittest discover tests     # 40 tests; includes a quality gate on the hand-written hold-out
```

Covers the homoglyph/IDN logic, the tree learner (learns an XOR a linear model cannot; attributions sum exactly to the decision), metrics, page and email signals, malformed-input robustness, the shipped model's quality floor, and the HTTP service.

## License

MIT
