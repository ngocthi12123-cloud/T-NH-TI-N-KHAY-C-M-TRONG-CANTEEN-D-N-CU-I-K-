from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from canteen_checkout.engagement import EngagementError, EngagementStore, normalize_vietnamese_phone


def item(class_name: str = "com_trang", *, total: int = 10_000, ignored: bool = False, uncertain: bool = False) -> dict:
    return {
        "class_name": class_name,
        "display_name": class_name,
        "price_vnd": total,
        "base_price_vnd": total,
        "extra_price_vnd": 0,
        "ignored": ignored,
        "uncertain": uncertain,
        "crop_path": "outputs/crops/item.jpg",
    }


class EngagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = EngagementStore(self.root / "engagement.sqlite3", phone_key=b"unit-test-secret-key-32-bytes!!")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create_draft(
        self,
        bill_id: str,
        items: list[dict],
        *,
        customer_id: str | None = None,
        voucher_id: str | None = None,
    ) -> dict:
        return self.store.create_draft(
            bill_id=bill_id,
            bill_path=str(self.root / f"{bill_id}.json"),
            payload={"created_at": "2026-06-19T00:00:00", "items": items},
            customer_id=customer_id,
            voucher_id=voucher_id,
        )

    def paid_member_bill(self, bill_id: str = "earn-bill", total: int = 100_000) -> tuple[dict, dict]:
        customer, _ = self.store.create_customer("0912 345 678")
        self.create_draft(bill_id, [item(total=total)], customer_id=customer["id"])
        return customer, self.store.confirm_bill(bill_id)

    def test_phone_normalization_and_private_storage(self) -> None:
        self.assertEqual(normalize_vietnamese_phone("+84 912-345-678"), "0912345678")
        with self.assertRaises(EngagementError):
            normalize_vietnamese_phone("123")

        first, created = self.store.create_customer("0912345678")
        second, created_again = self.store.create_customer("+84 912 345 678")
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["phone_masked"], "******5678")
        self.assertNotIn(b"0912345678", self.store.db_path.read_bytes())

    def test_confirm_is_idempotent_and_earns_from_net_total(self) -> None:
        customer, paid = self.paid_member_bill()
        paid_again = self.store.confirm_bill("earn-bill")
        refreshed = self.store.customer(customer["id"])

        self.assertEqual(paid["status"], "paid")
        self.assertEqual(paid["earned_points"], 10)
        self.assertEqual(paid_again["earned_points"], 10)
        self.assertEqual(refreshed["points_balance"], 10)
        with closing(sqlite3.connect(self.store.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM loyalty_ledger").fetchone()[0], 1)

    def test_issue_and_consume_voucher_on_later_matching_bill(self) -> None:
        customer, _ = self.paid_member_bill()
        issued = self.store.issue_voucher(
            customer_id=customer["id"],
            source_bill_id="earn-bill",
            class_name="com_trang",
            display_name="Cơm trắng",
            points_cost=10,
            discount_vnd=10_000,
        )
        voucher = issued["voucher"]
        self.assertEqual(issued["customer"]["points_balance"], 0)
        with self.assertRaisesRegex(EngagementError, "đã dùng quyền đổi voucher"):
            self.store.issue_voucher(
                customer_id=customer["id"],
                source_bill_id="earn-bill",
                class_name="com_trang",
                display_name="Cơm trắng",
                points_cost=10,
                discount_vnd=10_000,
            )

        voucher_item = item(total=15_000)
        voucher_item["base_price_vnd"] = 10_000
        voucher_item["extra_price_vnd"] = 5_000
        draft = self.create_draft(
            "redeem-bill",
            [voucher_item],
            customer_id=customer["id"],
            voucher_id=voucher["id"],
        )
        self.assertEqual(draft["gross_total_vnd"], 15_000)
        self.assertEqual(draft["discount_vnd"], 10_000)
        self.assertEqual(draft["net_total_vnd"], 5_000)
        self.assertEqual(draft["items"][0]["final_price_vnd"], 5_000)

        paid = self.store.confirm_bill("redeem-bill")
        paid_again = self.store.confirm_bill("redeem-bill")
        self.assertEqual(paid["voucher"]["status"], "consumed")
        self.assertEqual(paid_again["voucher"]["status"], "consumed")
        self.assertEqual(paid["earned_points"], 0)
        self.assertEqual(self.store.customer(customer["id"])["active_vouchers"], [])

    def test_voucher_rules_reject_insufficient_points_and_wrong_dish(self) -> None:
        customer, _ = self.paid_member_bill(total=10_000)
        with self.assertRaisesRegex(EngagementError, "Không đủ điểm"):
            self.store.issue_voucher(
                customer_id=customer["id"],
                source_bill_id="earn-bill",
                class_name="suon_nuong",
                display_name="Sườn nướng",
                points_cost=30,
                discount_vnd=30_000,
            )
        self.assertEqual(self.store.customer(customer["id"])["points_balance"], 1)

        rich_customer, _ = self.store.create_customer("0987654321")
        self.create_draft("rich-bill", [item(total=100_000)], customer_id=rich_customer["id"])
        self.store.confirm_bill("rich-bill")
        issued = self.store.issue_voucher(
            customer_id=rich_customer["id"],
            source_bill_id="rich-bill",
            class_name="com_trang",
            display_name="Cơm trắng",
            points_cost=10,
            discount_vnd=10_000,
        )
        with self.assertRaisesRegex(EngagementError, "không có món phù hợp"):
            self.create_draft(
                "wrong-dish",
                [item("suon_nuong", total=30_000)],
                customer_id=rich_customer["id"],
                voucher_id=issued["voucher"]["id"],
            )

    def test_rating_requires_paid_rateable_item_and_updates_summary(self) -> None:
        draft = self.create_draft("rating-bill", [item()])
        item_id = draft["items"][0]["bill_item_id"]
        with self.assertRaisesRegex(EngagementError, "sau khi thanh toán"):
            self.store.save_rating(bill_item_id=item_id, stars=5)

        self.store.confirm_bill("rating-bill")
        saved = self.store.save_rating(bill_item_id=item_id, stars=5, comment="Ngon")
        updated = self.store.save_rating(bill_item_id=item_id, stars=3, comment="Ổn")
        self.assertEqual(saved["summary"], {"average": 5.0, "count": 1})
        self.assertEqual(updated["summary"], {"average": 3.0, "count": 1})
        self.assertEqual(self.store.rating_summaries()["com_trang"], {"average": 3.0, "count": 1})

        ignored = self.create_draft("ignored-bill", [item(ignored=True)])
        self.store.confirm_bill("ignored-bill")
        with self.assertRaisesRegex(EngagementError, "không đủ điều kiện"):
            self.store.save_rating(bill_item_id=ignored["items"][0]["bill_item_id"], stars=4)

        with self.assertRaises(EngagementError):
            self.store.save_rating(bill_item_id=item_id, stars=6)
        with self.assertRaises(EngagementError):
            self.store.save_rating(bill_item_id=item_id, stars=4, comment="x" * 501)

    def test_bill_payload_never_contains_phone(self) -> None:
        customer, _ = self.store.create_customer("0901234567")
        draft = self.create_draft("private-bill", [item()], customer_id=customer["id"])
        self.assertNotIn("0901234567", json.dumps(draft, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
