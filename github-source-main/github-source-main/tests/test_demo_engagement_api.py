from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from canteen_checkout.engagement import EngagementStore


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apps" / "01_demo_checkout_app.py"
SPEC = importlib.util.spec_from_file_location("demo_checkout_app_api_test", SCRIPT)
app = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(app)


class DemoEngagementApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        app.ENGAGEMENT_STORE = EngagementStore(root / "engagement.sqlite3", phone_key=b"api-test-secret-key-32-bytes-long")
        app.BILLS_DIR = root / "bills"
        app.CROPPED_DISHES_DIR = root / "crops"
        self.server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.DemoHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        app.ENGAGEMENT_STORE = None
        self.temp.cleanup()

    def post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def get(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def run_forced_bill(self, customer_id: str, count: int, voucher_id: str | None = None) -> dict:
        image_path = app.list_demo_images()[0]["path"]
        regions = [
            {"name": f"rice_{index}", "x": 0, "y": 0, "w": 48, "h": 48, "label": "com_trang"}
            for index in range(count)
        ]
        return self.post(
            "/api/run",
            {
                "image_path": image_path,
                "regions": regions,
                "customer_id": customer_id,
                "voucher_id": voucher_id,
                "model_path": "models/missing-test-model.pt",
            },
        )

    def test_complete_customer_voucher_and_rating_flow(self) -> None:
        created = self.post("/api/customers", {"phone": "0912345678"})
        customer = created["customer"]
        lookup = self.post("/api/customers/lookup", {"phone": "+84 912 345 678"})
        self.assertTrue(created["created"])
        self.assertTrue(lookup["found"])
        self.assertEqual(lookup["customer"]["id"], customer["id"])

        draft = self.run_forced_bill(customer["id"], 10)
        self.assertEqual(draft["status"], "draft")
        self.assertEqual(draft["gross_total_vnd"], 100_000)
        paid = self.post("/api/checkout/confirm", {"bill_id": draft["bill_id"]})
        paid_again = self.post("/api/checkout/confirm", {"bill_id": draft["bill_id"]})
        self.assertEqual(paid["earned_points"], 10)
        self.assertEqual(paid_again["customer"]["points_balance"], 10)

        issued = self.post(
            "/api/vouchers",
            {
                "customer_id": customer["id"],
                "source_bill_id": draft["bill_id"],
                "class_name": "com_trang",
            },
        )
        self.assertEqual(issued["customer"]["points_balance"], 0)
        self.assertEqual(len(issued["customer"]["active_vouchers"]), 1)
        self.assertEqual(issued["customer"]["active_vouchers"][0]["id"], issued["voucher"]["id"])

        redeemed = self.run_forced_bill(customer["id"], 1, issued["voucher"]["id"])
        self.assertEqual(redeemed["discount_vnd"], 10_000)
        self.assertEqual(redeemed["net_total_vnd"], 0)
        redeemed_paid = self.post("/api/checkout/confirm", {"bill_id": redeemed["bill_id"]})
        self.assertEqual(redeemed_paid["voucher"]["status"], "consumed")

        rated = self.post(
            "/api/ratings",
            {
                "bill_item_id": redeemed_paid["items"][0]["bill_item_id"],
                "customer_id": customer["id"],
                "stars": 5,
                "comment": "Rất ngon",
            },
        )
        summary = self.get("/api/ratings/summary")
        self.assertEqual(rated["summary"], {"average": 5.0, "count": 1})
        self.assertEqual(summary["summaries"]["com_trang"], {"average": 5.0, "count": 1})


if __name__ == "__main__":
    unittest.main()
