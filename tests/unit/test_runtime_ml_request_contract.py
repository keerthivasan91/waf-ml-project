import unittest

import numpy as np

from app.services.feature_extractor import extract, normalize_request_for_ml


class TestRuntimeMLRequestContract(unittest.TestCase):
    def test_browser_headers_are_stripped_for_ml(self):
        request = {
            "url": "/api/products?category=electronics&page=1",
            "method": "GET",
            "headers": {
                "user-agent": "Mozilla/5.0",
                "accept": "application/json",
                "referer": "http://127.0.0.1:8000/",
                "cookie": "session=abc",
            },
            "body": "",
            "ip": "127.0.0.1",
        }

        normalized = normalize_request_for_ml(request)

        self.assertEqual(normalized["headers"], {})
        self.assertEqual(normalized["url"], request["url"])
        self.assertEqual(normalized["method"], request["method"])
        self.assertEqual(normalized["body"], request["body"])

    def test_runtime_features_ignore_browser_headers(self):
        base = {
            "url": "/api/products?category=electronics&page=1",
            "method": "GET",
            "body": "",
        }

        with_headers = {
            **base,
            "headers": {
                "user-agent": "Mozilla/5.0",
                "accept": "application/json",
                "accept-language": "en-US,en;q=0.9",
                "referer": "http://127.0.0.1:8000/products",
                "cookie": "session=abc",
                "sec-fetch-site": "same-origin",
            },
        }
        without_headers = {**base, "headers": {}}

        fvec_a, tokens_a = extract(with_headers)
        fvec_b, tokens_b = extract(without_headers)

        np.testing.assert_array_equal(fvec_a, fvec_b)
        np.testing.assert_array_equal(tokens_a, tokens_b)

    def test_ip_is_excluded_from_ml_request(self):
        normalized = normalize_request_for_ml({
            "url": "/api/products",
            "method": "GET",
            "headers": {"user-agent": "browser"},
            "body": "",
            "ip": "127.0.0.1",
        })

        self.assertNotIn("ip", normalized)


if __name__ == "__main__":
    unittest.main()
