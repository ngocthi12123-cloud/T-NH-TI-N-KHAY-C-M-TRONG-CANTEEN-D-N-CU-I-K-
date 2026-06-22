from __future__ import annotations

import unittest

from canteen_checkout.io_utils import PriceRow
from canteen_checkout.pricing import EXTRA_EGG_PRICE_VND, THIT_KHO_TRUNG_CLASS, dish_price


def price_row(class_name: str, price_vnd: int) -> PriceRow:
    return PriceRow(
        class_name=class_name,
        display_name=class_name,
        price_vnd=price_vnd,
        reward_points=1,
    )


class PricingTests(unittest.TestCase):
    def test_uncertain_known_dish_is_still_billed(self) -> None:
        prices = {"com_trang": price_row("com_trang", 10_000)}

        result = dish_price("com_trang", prices, uncertain=True)

        self.assertEqual(result.total_price_vnd, 10_000)

    def test_unknown_dish_is_not_billed(self) -> None:
        result = dish_price("unknown", {}, uncertain=True)

        self.assertEqual(result.total_price_vnd, 0)

    def test_uncertain_thit_kho_trung_keeps_extra_egg_price(self) -> None:
        prices = {THIT_KHO_TRUNG_CLASS: price_row(THIT_KHO_TRUNG_CLASS, 30_000)}

        result = dish_price(
            THIT_KHO_TRUNG_CLASS,
            prices,
            uncertain=True,
            egg_count=3,
        )

        self.assertEqual(result.base_price_vnd, 30_000)
        self.assertEqual(result.extra_price_vnd, 2 * EXTRA_EGG_PRICE_VND)
        self.assertEqual(result.total_price_vnd, 30_000 + 2 * EXTRA_EGG_PRICE_VND)


if __name__ == "__main__":
    unittest.main()
