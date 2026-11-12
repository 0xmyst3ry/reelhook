import unittest

from reelhook import check
from reelhook.data import generate
from reelhook.htmlscan import analyze
from reelhook.model import LogisticRegression, auc, metrics
from reelhook.urlfeatures import FEATURE_NAMES, extract, levenshtein, registered_domain, skeleton, vector


class URLFeatureTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("paypal", "paypa1"), 1)
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_skeleton_unmasks_homoglyphs(self):
        self.assertEqual(skeleton("rnicrosoft"), "microsoft")
        self.assertEqual(skeleton("paypa1"), "paypal")

    def test_registered_domain(self):
        self.assertEqual(registered_domain("a.b.example.co.uk"), "example.co.uk")
        self.assertEqual(registered_domain("login.paypal.com"), "paypal.com")

    def test_typosquat_and_misuse(self):
        self.assertEqual(extract("http://paypa1.com/login")["brand_typosquat"], 1)
        self.assertEqual(extract("http://paypal.secure-login.xyz/")["brand_misuse"], 1)
        self.assertEqual(extract("https://www.paypal.com/signin")["brand_misuse"], 0)
        self.assertEqual(extract("https://www.paypal.com/signin")["brand_typosquat"], 0)

    def test_at_and_ip(self):
        self.assertEqual(extract("https://paypal.com@evil.xyz/x")["has_at"], 1)
        self.assertEqual(extract("http://192.168.1.5:8080/a")["is_ip"], 1)

    def test_vector_length(self):
        self.assertEqual(len(vector("https://example.com")), len(FEATURE_NAMES))


class HTMLTests(unittest.TestCase):
    def test_cross_domain_password_form(self):
        html = '<form action="https://collector.evil.xyz/p"><input type="password"></form>'
        ids = [s.id for s in analyze(html, "http://paypa1-login.top/")]
        self.assertIn("H02", ids)

    def test_same_domain_form_clean(self):
        html = '<form action="/login"><input type="password"></form>'
        self.assertEqual(analyze(html, "https://example.com/"), [])

    def test_link_text_mismatch_and_title(self):
        html = '<title>PayPal - Log in</title><a href="http://evil.top/x">www.paypal.com</a>'
        ids = {s.id for s in analyze(html, "https://evil.top/")}
        self.assertTrue({"H05", "H07"} <= ids)

    def test_hidden_iframe(self):
        html = '<iframe src="http://x.top" style="display:none"></iframe>'
        self.assertIn("H06", [s.id for s in analyze(html, "https://a.com")])


class ModelTests(unittest.TestCase):
    def test_auc_perfect_and_random(self):
        self.assertEqual(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_generalises_to_heldout(self):
        train, test = generate(2000, seed=1), generate(600, seed=42)
        m = LogisticRegression(FEATURE_NAMES).fit([vector(u) for u, _ in train], [l for _, l in train], epochs=40)
        res = metrics([l for _, l in test], [m.predict_proba(vector(u)) for u, _ in test])
        self.assertGreater(res["auc"], 0.95)
        self.assertGreater(res["recall"], 0.85)


class EndToEndTests(unittest.TestCase):
    def test_known_good(self):
        self.assertEqual(check("https://www.github.com/login").verdict, "safe")

    def test_obvious_phish(self):
        self.assertEqual(check("http://paypal.secure-login.xyz/verify/account").verdict, "phishing")

    def test_html_escalates(self):
        url = "https://quietmeadow.com/account/42"
        html = '<title>Microsoft sign in</title><form action="https://evil.top/c"><input type="password"></form>'
        self.assertGreater(check(url, html).score, check(url).score)


if __name__ == "__main__":
    unittest.main()
