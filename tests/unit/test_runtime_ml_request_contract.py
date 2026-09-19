import unittest


from app.services.feature_extractor import normalize_request_for_ml


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
