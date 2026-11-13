import json
import os
import random
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from reelhook import check, check_email
from reelhook.data import generate, load_csv
from reelhook.ensemble import MODEL_PATH, Ensemble, default_model
from reelhook.evaluation import HOLDOUT
from reelhook.gbdt import GBDT
from reelhook.htmlscan import analyze
from reelhook.model import LogisticRegression, auc, metrics, threshold_at_fpr, tpr_at_fpr
from reelhook.ngram import NgramModel, featurize
from reelhook.server import Handler
from reelhook.urlfeatures import (FEATURE_NAMES, decode_idna, extract, osa_distance, registered_domain, scripts_in,
                                  skeleton, vector)

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def flags(url):
    f = extract(url)
    return {k for k in ("brand_misuse", "brand_typosquat", "brand_confusable", "brand_combosquat", "tld_swap",
                        "free_hosting", "free_host_brand", "mixed_script", "nonascii_host", "punycode", "has_at", "is_ip",
                        "shortener") if f[k]}


class URLFeatureTests(unittest.TestCase):
    def test_osa_counts_transpositions(self):
        self.assertEqual(osa_distance("paypal", "paypla"), 1)
        self.assertEqual(osa_distance("paypal", "paypa1"), 1)
        self.assertEqual(osa_distance("abc", "abc"), 0)
        self.assertEqual(osa_distance("kitten", "sitting"), 3)

    def test_skeleton_unmasks_homoglyphs(self):
        self.assertEqual(skeleton("rnicrosoft"), "microsoft")
        self.assertEqual(skeleton("paypa1"), "paypal")
        self.assertEqual(skeleton("раураl"), "paypal")         # Cyrillic а, р, у, а

    def test_idna_and_scripts(self):
        self.assertEqual(decode_idna("xn--bcher-kva.example"), "bücher.example")
        self.assertEqual(scripts_in("раураl"), {"CYRILLIC", "LATIN"})
        self.assertEqual(scripts_in("paypal"), {"LATIN"})

    def test_registered_domain(self):
        self.assertEqual(registered_domain("a.b.example.co.uk"), "example.co.uk")
        self.assertEqual(registered_domain("login.paypal.com"), "paypal.com")
        self.assertEqual(registered_domain("evil.pages.dev"), "pages.dev")

    def test_official_domains_raise_no_brand_flags(self):
        for u in ["https://www.paypal.com/signin", "https://login.microsoftonline.com/common/oauth2",
                  "https://amazon.co.uk/gp/css/order-history", "https://github.com/user/repo",
                  "https://groups.google.com/forum", "https://us02web.zoom.us/j/123", "https://janedoe.github.io/x"]:
            self.assertFalse(flags(u) & {"brand_misuse", "brand_typosquat", "brand_confusable", "brand_combosquat", "tld_swap"}, u)

    def test_brand_attacks(self):
        self.assertTrue({"brand_typosquat", "brand_confusable"} <= flags("http://paypa1.com/login"))
        self.assertIn("brand_misuse", flags("https://paypal.com.account-verify.tk/signin"))
        self.assertIn("brand_combosquat", flags("https://paypal-security-center.com/verify"))
        self.assertTrue({"brand_confusable", "mixed_script", "nonascii_host"} <= flags("https://раураl.com/signin"))
        self.assertTrue({"punycode", "nonascii_host"} <= flags("https://xn--pypal-4ve.com/"))
        self.assertIn("tld_swap", flags("https://www.paypal.co/"))
        self.assertTrue({"free_hosting", "free_host_brand"} <= flags("https://netflix-billing-update.web.app/"))
        self.assertIn("brand_confusable", flags("https://arnazon.com/"))

    def test_structure_flags(self):
        self.assertIn("has_at", flags("https://paypal.com@evil.xyz/x"))
        self.assertIn("is_ip", flags("http://192.168.1.5:8080/a"))
        self.assertIn("shortener", flags("https://bit.ly/abc123"))

    def test_malformed_urls_do_not_crash(self):
        for u in ["", "http://", "://x", "http://[::1", "http://a:99999999/", "%%%", "javascript:alert(1)", "http://" + "a" * 5000]:
            self.assertEqual(len(vector(u)), len(FEATURE_NAMES), u)

    def test_vector_matches_names(self):
        self.assertEqual(len(vector("https://example.com")), len(FEATURE_NAMES))


class GBDTTests(unittest.TestCase):
    def xor_data(self, n, seed):
        r = random.Random(seed)
        X = [[r.random(), r.random(), r.random()] for _ in range(n)]
        y = [int((x[0] > 0.5) != (x[1] > 0.5)) for x in X]
        return X, y

    def test_learns_interaction_that_logistic_cannot(self):
        X, y = self.xor_data(1200, 1)
        Xt, yt = self.xor_data(400, 2)
        gb = GBDT(["a", "b", "c"], n_trees=60, depth=3).fit(X, y)
        lr = LogisticRegression(["a", "b", "c"]).fit(X, y, epochs=30)
        self.assertGreater(auc(yt, [gb.predict_proba(x) for x in Xt]), 0.95)
        self.assertLess(auc(yt, [lr.predict_proba(x) for x in Xt]), 0.65)

    def test_attributions_sum_to_decision(self):
        X, y = self.xor_data(600, 3)
        gb = GBDT(["a", "b", "c"], n_trees=25).fit(X, y)
        x = X[7]
        roots = sum(t[0] if len(t) == 1 else t[2] for t in gb.trees)
        total = gb.base + gb.lr * roots + sum(c for _, c in gb.explain(x, top=10))
        self.assertAlmostEqual(total, gb.decision(x), places=6)

    def test_explanation_names_the_informative_features(self):
        X, y = self.xor_data(800, 4)
        gb = GBDT(["a", "b", "c"], n_trees=40).fit(X, y)
        self.assertEqual({n for n, _ in gb.explain([0.9, 0.1, 0.5], top=2)}, {"a", "b"})

    def test_roundtrip_and_determinism(self):
        X, y = self.xor_data(300, 5)
        a = GBDT(["a", "b", "c"], n_trees=10, seed=2).fit(X, y)
        b = GBDT(["a", "b", "c"], n_trees=10, seed=2).fit(X, y)
        c = GBDT.from_dict(json.loads(json.dumps(a.to_dict())))
        for x in X[:20]:
            self.assertEqual(a.predict_proba(x), b.predict_proba(x))
            self.assertAlmostEqual(a.predict_proba(x), c.predict_proba(x))

    def test_constant_feature_is_harmless(self):
        gb = GBDT(["k"], n_trees=5).fit([[1.0]] * 40, [0, 1] * 20)
        self.assertAlmostEqual(gb.predict_proba([1.0]), 0.5, delta=0.05)


class NgramTests(unittest.TestCase):
    def test_features_normalised(self):
        x = featurize("https://paypal-security.top/verify")
        self.assertAlmostEqual(sum(v * v for v in x.values()), 1.0)

    def test_separates_simple_corpus(self):
        train = generate(800, seed=4)
        m = NgramModel().fit(train, epochs=6)
        test = generate(300, seed=5)
        self.assertGreater(auc([l for _, l in test], [m.predict_proba(u) for u, _ in test]), 0.95)


class MetricTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(auc([0, 1, 0, 1], [0.5] * 4), 0.5)

    def test_tpr_at_fpr_and_threshold(self):
        y = [0] * 100 + [1] * 10
        p = [i / 100 for i in range(100)] + [0.995] * 8 + [0.3, 0.2]
        self.assertAlmostEqual(tpr_at_fpr(y, p, 0.01), 0.8)
        self.assertAlmostEqual(threshold_at_fpr(y, p, 0.01), 0.98)

    def test_metrics_confusion(self):
        m = metrics([1, 1, 0, 0], [0.9, 0.2, 0.6, 0.1])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))


class HTMLTests(unittest.TestCase):
    def ids(self, html, url="https://evil.top/"):
        return {s.id for s in analyze(html, url)}

    def test_cross_domain_credential_form(self):
        self.assertIn("H02", self.ids('<form action="https://c.evil.xyz/p"><input type="password"></form>'))

    def test_same_domain_form_clean(self):
        self.assertEqual(self.ids('<form action="/login"><input type="password"></form>', "https://example.com/"), set())

    def test_link_and_title_deception(self):
        ids = self.ids('<title>PayPal - Log in</title><a href="http://evil.top/x">www.paypal.com</a>')
        self.assertTrue({"H05", "H07"} <= ids)

    def test_lookalike_title(self):
        self.assertIn("H07", self.ids("<title>Paypa1 Login</title>"))

    def test_hidden_iframe_and_scripts(self):
        self.assertIn("H06", self.ids('<iframe src="http://x.top" style="display:none"></iframe>'))
        self.assertIn("H10", self.ids("<script>var a = eval(atob('YWxlcnQoMSk='));</script>"))
        self.assertIn("H08", self.ids("<script>document.oncontextmenu = function(){ return false; }; /* contextmenu */ </script>"))

    def test_sensitive_fields_and_data_action(self):
        html = '<form action="data:text/html,x"><input name="cardnumber"><input name="cvv"></form>'
        self.assertTrue({"H11", "H12"} <= self.ids(html))

    def test_brand_resource_cloning(self):
        imgs = "".join(f'<img src="https://www.paypal.com/i{i}.png">' for i in range(4)) + '<img src="https://evil.top/x.png">'
        self.assertIn("H13", self.ids(imgs))
        self.assertNotIn("H13", self.ids(imgs, "https://www.paypal.com/"))

    def test_plain_page_is_clean(self):
        self.assertEqual(self.ids("<title>My blog</title><p>Hello</p><a href='/about'>about</a>"), set())

    def test_example_page(self):
        html = open(os.path.join(EX, "phish_page.html")).read()
        self.assertTrue({"H02", "H05", "H07", "H08", "H10", "H11", "H13"} <= self.ids(html, "https://ms-login-portal.top/x"))


class EmailTests(unittest.TestCase):
    def load(self, name):
        return open(os.path.join(EX, name), "rb").read()

    def test_phishing_email(self):
        r = check_email(self.load("phish_email.eml"))
        self.assertEqual(r.verdict, "phishing")
        ids = {s.id for s in r.signals}
        self.assertTrue({"E01", "E02", "E04", "E05", "E06", "E08", "E09", "E11", "E13", "E14"} <= ids, ids)

    def test_legit_email(self):
        r = check_email(self.load("legit_email.eml"))
        self.assertEqual(r.verdict, "safe")

    def test_single_weak_signal_is_not_enough(self):
        raw = "From: Bob <bob@example.com>\nSubject: lunch\nAuthentication-Results: x; spf=pass; dkim=pass; dmarc=pass\n\nDear customer, see you at noon."
        self.assertEqual(check_email(raw).verdict, "safe")

    def test_risky_attachment_alone_is_flagged(self):
        raw = ("From: a@example.com\nSubject: doc\nAuthentication-Results: x; spf=pass; dkim=pass; dmarc=pass\n"
               "MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=b\n\n--b\nContent-Type: text/plain\n\nsee attached\n"
               "--b\nContent-Type: application/octet-stream; name=x.iso\nContent-Disposition: attachment; filename=x.iso\n\nAAAA\n--b--\n")
        self.assertIn("E12", {s.id for s in check_email(raw).signals})

    def test_garbage_input_does_not_crash(self):
        for raw in ["", "not an email at all", b"\xff\xfe\x00junk", "From:\n\n"]:
            self.assertIn(check_email(raw).verdict, ("safe", "suspicious", "phishing"))


class ShippedModelTests(unittest.TestCase):
    def test_model_file_loads(self):
        self.assertTrue(os.path.exists(MODEL_PATH))
        m = Ensemble.load()
        self.assertTrue(0 < m.thresholds["suspicious"] < m.thresholds["phishing"] < 1)

    def test_obvious_cases(self):
        self.assertEqual(check("https://www.github.com/login").verdict, "safe")
        self.assertEqual(check("http://paypal.secure-login.xyz/verify/account").verdict, "phishing")
        self.assertNotEqual(check("https://раураl.com/signin").verdict, "safe")

    def test_html_escalates(self):
        url = "https://quietmeadow.com/account/42"
        html = '<title>Microsoft sign in</title><form action="https://evil.top/c"><input type="password"></form>'
        self.assertGreater(check(url, html).score, check(url).score)

    def test_quality_floor_on_handwritten_holdout(self):
        """Different author and style from the training generator; guards against shipping a worse model."""
        rows = load_csv(HOLDOUT)
        y = [l for _, l in rows]
        m = default_model()
        p = [m.predict_proba(u) for u, _ in rows]
        self.assertGreater(auc(y, p), 0.97)
        self.assertGreater(tpr_at_fpr(y, p, 0.01), 0.70)
        flagged = [(q >= m.thresholds["suspicious"]) for q in p]
        fpr = sum(f for f, t in zip(flagged, y) if t == 0) / y.count(0)
        recall = sum(f for f, t in zip(flagged, y) if t == 1) / y.count(1)
        self.assertLess(fpr, 0.12)
        self.assertGreater(recall, 0.88)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_check(self):
        code, body = self.post("/check", {"url": "http://paypa1-secure.top/login"})
        self.assertEqual((code, body["verdict"]), (200, "phishing"))

    def test_email(self):
        code, body = self.post("/email", {"raw": open(os.path.join(EX, "phish_email.eml")).read()})
        self.assertEqual((code, body["verdict"]), (200, "phishing"))

    def test_bad_requests(self):
        self.assertEqual(self.post("/check", {"nope": 1})[0], 400)
        self.assertEqual(self.post("/check", {"url": 5})[0], 400)
        self.assertEqual(self.post("/x", {"url": "a"})[0], 404)


if __name__ == "__main__":
    unittest.main()
