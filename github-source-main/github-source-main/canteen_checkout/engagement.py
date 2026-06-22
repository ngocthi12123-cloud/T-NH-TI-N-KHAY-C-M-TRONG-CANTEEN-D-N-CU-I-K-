from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ENGAGEMENT_DB_PATH, PHONE_HMAC_KEY_PATH


POINT_EARNING_VND = 10_000
MAX_RATING_COMMENT = 500


class EngagementError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_request", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_vietnamese_phone(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 11 and digits.startswith("84"):
        digits = "0" + digits[2:]
    if len(digits) != 10 or not digits.startswith("0"):
        raise EngagementError("Số điện thoại phải có 10 chữ số và bắt đầu bằng 0.", code="invalid_phone")
    return digits


def phone_digest(normalized_phone: str, key: bytes) -> str:
    return hmac.new(key, normalized_phone.encode("ascii"), hashlib.sha256).hexdigest()


def _load_or_create_phone_key(path: Path) -> bytes:
    configured = os.environ.get("CANTEEN_PHONE_HMAC_KEY")
    if configured:
        key = configured.encode("utf-8")
        if len(key) < 16:
            raise EngagementError("CANTEEN_PHONE_HMAC_KEY must contain at least 16 bytes.", code="invalid_phone_key")
        return key

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(secrets.token_bytes(32))
    except FileExistsError:
        pass
    key = path.read_bytes()
    if len(key) < 16:
        raise EngagementError(f"Phone HMAC key is invalid: {path}", code="invalid_phone_key")
    return key


class EngagementStore:
    def __init__(
        self,
        db_path: Path = ENGAGEMENT_DB_PATH,
        *,
        phone_key: bytes | None = None,
        phone_key_path: Path = PHONE_HMAC_KEY_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.phone_key = phone_key or _load_or_create_phone_key(Path(phone_key_path))
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY,
                    phone_hmac TEXT NOT NULL UNIQUE,
                    phone_last4 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bills (
                    id TEXT PRIMARY KEY,
                    bill_path TEXT NOT NULL,
                    customer_id TEXT REFERENCES customers(id),
                    voucher_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('draft', 'paid')),
                    gross_total_vnd INTEGER NOT NULL CHECK(gross_total_vnd >= 0),
                    discount_vnd INTEGER NOT NULL CHECK(discount_vnd >= 0),
                    net_total_vnd INTEGER NOT NULL CHECK(net_total_vnd >= 0),
                    earned_points INTEGER NOT NULL DEFAULT 0 CHECK(earned_points >= 0),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    paid_at TEXT
                );

                CREATE TABLE IF NOT EXISTS bill_items (
                    id TEXT PRIMARY KEY,
                    bill_id TEXT NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
                    line_index INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    base_price_vnd INTEGER NOT NULL,
                    extra_price_vnd INTEGER NOT NULL,
                    total_price_vnd INTEGER NOT NULL,
                    ignored INTEGER NOT NULL,
                    uncertain INTEGER NOT NULL,
                    crop_path TEXT NOT NULL,
                    UNIQUE(bill_id, line_index)
                );

                CREATE TABLE IF NOT EXISTS loyalty_ledger (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(id),
                    bill_id TEXT REFERENCES bills(id),
                    voucher_id TEXT,
                    points_delta INTEGER NOT NULL,
                    reason TEXT NOT NULL CHECK(reason IN ('earn', 'voucher_issue')),
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vouchers (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(id),
                    source_bill_id TEXT NOT NULL UNIQUE REFERENCES bills(id),
                    class_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    points_cost INTEGER NOT NULL CHECK(points_cost > 0),
                    discount_vnd INTEGER NOT NULL CHECK(discount_vnd > 0),
                    status TEXT NOT NULL CHECK(status IN ('active', 'consumed')),
                    issued_at TEXT NOT NULL,
                    consumed_at TEXT,
                    consumed_bill_id TEXT REFERENCES bills(id)
                );

                CREATE TABLE IF NOT EXISTS ratings (
                    id TEXT PRIMARY KEY,
                    bill_item_id TEXT NOT NULL UNIQUE REFERENCES bill_items(id),
                    customer_id TEXT REFERENCES customers(id),
                    class_name TEXT NOT NULL,
                    stars INTEGER NOT NULL CHECK(stars BETWEEN 1 AND 5),
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ledger_customer ON loyalty_ledger(customer_id);
                CREATE INDEX IF NOT EXISTS idx_vouchers_customer_status ON vouchers(customer_id, status);
                CREATE INDEX IF NOT EXISTS idx_ratings_class ON ratings(class_name);
                PRAGMA user_version = 1;
                """
            )

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _balance(connection: sqlite3.Connection, customer_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(SUM(points_delta), 0) AS balance FROM loyalty_ledger WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        return int(row["balance"])

    @staticmethod
    def _voucher_view(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "class_name": row["class_name"],
            "display_name": row["display_name"],
            "points_cost": int(row["points_cost"]),
            "discount_vnd": int(row["discount_vnd"]),
            "status": row["status"],
            "issued_at": row["issued_at"],
        }

    def _customer_view(self, connection: sqlite3.Connection, customer_id: str) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if row is None:
            raise EngagementError("Không tìm thấy thành viên.", code="customer_not_found", status=404)
        vouchers = connection.execute(
            "SELECT * FROM vouchers WHERE customer_id = ? AND status = 'active' ORDER BY issued_at, id",
            (customer_id,),
        ).fetchall()
        return {
            "id": row["id"],
            "phone_masked": f"******{row['phone_last4']}",
            "points_balance": self._balance(connection, customer_id),
            "active_vouchers": [self._voucher_view(voucher) for voucher in vouchers],
            "created_at": row["created_at"],
        }

    def lookup_customer(self, phone: str) -> dict[str, Any] | None:
        normalized = normalize_vietnamese_phone(phone)
        digest = phone_digest(normalized, self.phone_key)
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT id FROM customers WHERE phone_hmac = ?", (digest,)).fetchone()
            return self._customer_view(connection, row["id"]) if row else None

    def create_customer(self, phone: str) -> tuple[dict[str, Any], bool]:
        normalized = normalize_vietnamese_phone(phone)
        digest = phone_digest(normalized, self.phone_key)
        with closing(self.connect()) as connection:
            self._begin(connection)
            try:
                row = connection.execute("SELECT id FROM customers WHERE phone_hmac = ?", (digest,)).fetchone()
                created = row is None
                customer_id = row["id"] if row else uuid.uuid4().hex
                if created:
                    connection.execute(
                        "INSERT INTO customers(id, phone_hmac, phone_last4, created_at) VALUES (?, ?, ?, ?)",
                        (customer_id, digest, normalized[-4:], utc_now()),
                    )
                customer = self._customer_view(connection, customer_id)
                connection.commit()
                return customer, created
            except Exception:
                connection.rollback()
                raise

    def customer(self, customer_id: str) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            return self._customer_view(connection, customer_id)

    @staticmethod
    def _rating_summary_map(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        rows = connection.execute(
            "SELECT class_name, ROUND(AVG(stars), 2) AS average, COUNT(*) AS count FROM ratings GROUP BY class_name"
        ).fetchall()
        return {
            row["class_name"]: {"average": float(row["average"]), "count": int(row["count"])}
            for row in rows
        }

    def rating_summaries(self) -> dict[str, dict[str, Any]]:
        with closing(self.connect()) as connection:
            return self._rating_summary_map(connection)

    def _decorate_ratings(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        summaries = self._rating_summary_map(connection)
        items = payload.get("items", [])
        item_ids = [item.get("bill_item_id") for item in items if item.get("bill_item_id")]
        ratings: dict[str, sqlite3.Row] = {}
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            rows = connection.execute(
                f"SELECT bill_item_id, stars, comment, updated_at FROM ratings WHERE bill_item_id IN ({placeholders})",
                item_ids,
            ).fetchall()
            ratings = {row["bill_item_id"]: row for row in rows}
        for item in items:
            item["rating_summary"] = summaries.get(item.get("class_name"), {"average": 0.0, "count": 0})
            rating = ratings.get(item.get("bill_item_id"))
            item["rating"] = (
                {
                    "stars": int(rating["stars"]),
                    "comment": rating["comment"],
                    "updated_at": rating["updated_at"],
                }
                if rating
                else None
            )

    def create_draft(
        self,
        *,
        bill_id: str,
        bill_path: str,
        payload: dict[str, Any],
        customer_id: str | None = None,
        voucher_id: str | None = None,
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(payload, ensure_ascii=False))
        items = result.get("items", [])
        created_at = result.get("created_at") or utc_now()
        with closing(self.connect()) as connection:
            self._begin(connection)
            try:
                customer = self._customer_view(connection, customer_id) if customer_id else None
                voucher_row = None
                discount = 0
                discounted_item_id = None
                if voucher_id:
                    if not customer_id:
                        raise EngagementError("Phải chọn thành viên trước khi dùng voucher.", code="voucher_requires_customer")
                    voucher_row = connection.execute("SELECT * FROM vouchers WHERE id = ?", (voucher_id,)).fetchone()
                    if voucher_row is None or voucher_row["customer_id"] != customer_id:
                        raise EngagementError("Voucher không thuộc thành viên này.", code="voucher_not_found", status=404)
                    if voucher_row["status"] != "active":
                        raise EngagementError("Voucher đã được sử dụng.", code="voucher_consumed", status=409)

                gross_total = 0
                rows_to_insert: list[tuple[Any, ...]] = []
                for index, item in enumerate(items):
                    item_id = uuid.uuid4().hex
                    item["bill_item_id"] = item_id
                    total_price = max(0, int(item.get("price_vnd") or 0))
                    base_price = max(0, int(item.get("base_price_vnd") or 0))
                    extra_price = max(0, int(item.get("extra_price_vnd") or 0))
                    item_discount = 0
                    if (
                        voucher_row is not None
                        and discounted_item_id is None
                        and not item.get("ignored")
                        and not item.get("uncertain")
                        and item.get("class_name") == voucher_row["class_name"]
                        and base_price > 0
                    ):
                        item_discount = min(base_price, int(voucher_row["discount_vnd"]))
                        discounted_item_id = item_id
                        discount = item_discount
                    item["discount_vnd"] = item_discount
                    item["final_price_vnd"] = max(0, total_price - item_discount)
                    gross_total += total_price
                    rows_to_insert.append(
                        (
                            item_id,
                            bill_id,
                            index,
                            str(item.get("class_name") or "unknown"),
                            str(item.get("display_name") or item.get("class_name") or "unknown"),
                            base_price,
                            extra_price,
                            total_price,
                            int(bool(item.get("ignored"))),
                            int(bool(item.get("uncertain"))),
                            str(item.get("crop_path") or ""),
                        )
                    )
                if voucher_row is not None and discounted_item_id is None:
                    raise EngagementError("Bill không có món phù hợp với voucher.", code="voucher_dish_missing", status=409)

                net_total = max(0, gross_total - discount)
                result.update(
                    {
                        "bill_id": bill_id,
                        "status": "draft",
                        "customer_id": customer_id,
                        "customer": customer,
                        "voucher_id": voucher_id,
                        "voucher": self._voucher_view(voucher_row) if voucher_row else None,
                        "gross_total_vnd": gross_total,
                        "discount_vnd": discount,
                        "net_total_vnd": net_total,
                        "total_vnd": net_total,
                        "earned_points": 0,
                        "paid_at": None,
                    }
                )
                self._decorate_ratings(connection, result)
                connection.execute(
                    """
                    INSERT INTO bills(
                        id, bill_path, customer_id, voucher_id, status, gross_total_vnd,
                        discount_vnd, net_total_vnd, earned_points, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        bill_id,
                        bill_path,
                        customer_id,
                        voucher_id,
                        gross_total,
                        discount,
                        net_total,
                        json.dumps(result, ensure_ascii=False),
                        created_at,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO bill_items(
                        id, bill_id, line_index, class_name, display_name, base_price_vnd,
                        extra_price_vnd, total_price_vnd, ignored, uncertain, crop_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def get_bill(self, bill_id: str) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT payload_json FROM bills WHERE id = ?", (bill_id,)).fetchone()
            if row is None:
                raise EngagementError("Không tìm thấy bill.", code="bill_not_found", status=404)
            payload = json.loads(row["payload_json"])
            self._decorate_ratings(connection, payload)
            return payload

    def confirm_bill(self, bill_id: str) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            self._begin(connection)
            try:
                row = connection.execute("SELECT * FROM bills WHERE id = ?", (bill_id,)).fetchone()
                if row is None:
                    raise EngagementError("Không tìm thấy bill.", code="bill_not_found", status=404)
                payload = json.loads(row["payload_json"])
                if row["status"] == "paid":
                    self._decorate_ratings(connection, payload)
                    connection.commit()
                    return payload

                paid_at = utc_now()
                voucher_view = payload.get("voucher")
                if row["voucher_id"]:
                    voucher = connection.execute("SELECT * FROM vouchers WHERE id = ?", (row["voucher_id"],)).fetchone()
                    if voucher is None or voucher["status"] != "active":
                        raise EngagementError("Voucher không còn khả dụng.", code="voucher_consumed", status=409)
                    if voucher["customer_id"] != row["customer_id"]:
                        raise EngagementError("Voucher không thuộc thành viên của bill.", code="voucher_not_found", status=409)
                    connection.execute(
                        "UPDATE vouchers SET status = 'consumed', consumed_at = ?, consumed_bill_id = ? WHERE id = ?",
                        (paid_at, bill_id, row["voucher_id"]),
                    )
                    voucher_view = self._voucher_view(voucher)
                    voucher_view["status"] = "consumed"

                earned_points = int(row["net_total_vnd"]) // POINT_EARNING_VND if row["customer_id"] else 0
                if earned_points:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO loyalty_ledger(
                            id, customer_id, bill_id, voucher_id, points_delta, reason, idempotency_key, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'earn', ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            row["customer_id"],
                            bill_id,
                            row["voucher_id"],
                            earned_points,
                            f"bill:{bill_id}:earn",
                            paid_at,
                        ),
                    )
                customer = self._customer_view(connection, row["customer_id"]) if row["customer_id"] else None
                payload.update(
                    {
                        "status": "paid",
                        "paid_at": paid_at,
                        "earned_points": earned_points,
                        "customer": customer,
                        "voucher": voucher_view,
                    }
                )
                self._decorate_ratings(connection, payload)
                connection.execute(
                    "UPDATE bills SET status = 'paid', earned_points = ?, paid_at = ?, payload_json = ? WHERE id = ?",
                    (earned_points, paid_at, json.dumps(payload, ensure_ascii=False), bill_id),
                )
                connection.commit()
                return payload
            except Exception:
                connection.rollback()
                raise

    def issue_voucher(
        self,
        *,
        customer_id: str,
        source_bill_id: str,
        class_name: str,
        display_name: str,
        points_cost: int,
        discount_vnd: int,
    ) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            self._begin(connection)
            try:
                bill = connection.execute("SELECT * FROM bills WHERE id = ?", (source_bill_id,)).fetchone()
                if bill is None or bill["status"] != "paid":
                    raise EngagementError("Chỉ có thể đổi voucher sau khi thanh toán.", code="bill_not_paid", status=409)
                if bill["customer_id"] != customer_id:
                    raise EngagementError("Bill không thuộc thành viên này.", code="customer_mismatch", status=409)
                existing = connection.execute(
                    "SELECT id FROM vouchers WHERE source_bill_id = ?", (source_bill_id,)
                ).fetchone()
                if existing:
                    raise EngagementError("Bill này đã dùng quyền đổi voucher.", code="voucher_already_issued", status=409)
                balance = self._balance(connection, customer_id)
                if balance < points_cost:
                    raise EngagementError(
                        f"Không đủ điểm: cần {points_cost}, hiện có {balance}.",
                        code="insufficient_points",
                        status=409,
                    )
                voucher_id = uuid.uuid4().hex
                issued_at = utc_now()
                connection.execute(
                    """
                    INSERT INTO vouchers(
                        id, customer_id, source_bill_id, class_name, display_name,
                        points_cost, discount_vnd, status, issued_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        voucher_id,
                        customer_id,
                        source_bill_id,
                        class_name,
                        display_name,
                        points_cost,
                        discount_vnd,
                        issued_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO loyalty_ledger(
                        id, customer_id, bill_id, voucher_id, points_delta, reason, idempotency_key, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'voucher_issue', ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        customer_id,
                        source_bill_id,
                        voucher_id,
                        -points_cost,
                        f"bill:{source_bill_id}:voucher",
                        issued_at,
                    ),
                )
                voucher = connection.execute("SELECT * FROM vouchers WHERE id = ?", (voucher_id,)).fetchone()
                customer = self._customer_view(connection, customer_id)
                connection.commit()
                return {"voucher": self._voucher_view(voucher), "customer": customer}
            except Exception:
                connection.rollback()
                raise

    def save_rating(
        self,
        *,
        bill_item_id: str,
        stars: int,
        comment: str = "",
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(stars, bool) or not isinstance(stars, int) or not 1 <= stars <= 5:
            raise EngagementError("Đánh giá phải từ 1 đến 5 sao.", code="invalid_rating")
        clean_comment = str(comment or "").strip()
        if len(clean_comment) > MAX_RATING_COMMENT:
            raise EngagementError(
                f"Nhận xét không được vượt quá {MAX_RATING_COMMENT} ký tự.", code="comment_too_long"
            )

        with closing(self.connect()) as connection:
            self._begin(connection)
            try:
                row = connection.execute(
                    """
                    SELECT i.*, b.status AS bill_status, b.customer_id AS bill_customer_id
                    FROM bill_items i JOIN bills b ON b.id = i.bill_id WHERE i.id = ?
                    """,
                    (bill_item_id,),
                ).fetchone()
                if row is None:
                    raise EngagementError("Không tìm thấy món trong bill.", code="bill_item_not_found", status=404)
                if row["bill_status"] != "paid":
                    raise EngagementError("Chỉ đánh giá sau khi thanh toán.", code="bill_not_paid", status=409)
                if row["ignored"] or row["uncertain"] or int(row["total_price_vnd"]) <= 0:
                    raise EngagementError("Món này không đủ điều kiện đánh giá.", code="item_not_rateable", status=409)
                if customer_id and customer_id != row["bill_customer_id"]:
                    raise EngagementError("Thành viên không khớp với bill.", code="customer_mismatch", status=409)
                now = utc_now()
                existing = connection.execute("SELECT id, created_at FROM ratings WHERE bill_item_id = ?", (bill_item_id,)).fetchone()
                rating_id = existing["id"] if existing else uuid.uuid4().hex
                created_at = existing["created_at"] if existing else now
                connection.execute(
                    """
                    INSERT INTO ratings(id, bill_item_id, customer_id, class_name, stars, comment, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bill_item_id) DO UPDATE SET
                        customer_id = excluded.customer_id,
                        stars = excluded.stars,
                        comment = excluded.comment,
                        updated_at = excluded.updated_at
                    """,
                    (rating_id, bill_item_id, customer_id, row["class_name"], stars, clean_comment, created_at, now),
                )
                summary = self._rating_summary_map(connection)[row["class_name"]]
                connection.commit()
                return {
                    "rating": {
                        "id": rating_id,
                        "bill_item_id": bill_item_id,
                        "stars": stars,
                        "comment": clean_comment,
                        "updated_at": now,
                    },
                    "class_name": row["class_name"],
                    "summary": summary,
                }
            except Exception:
                connection.rollback()
                raise
