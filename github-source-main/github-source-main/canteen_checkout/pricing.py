from __future__ import annotations

from dataclasses import dataclass

from .io_utils import PriceRow


THIT_KHO_TRUNG_CLASS = "thit_kho_trung"
INCLUDED_EGGS = 1
EXTRA_EGG_PRICE_VND = 6000


@dataclass(frozen=True)
class BillPrice:
    base_price_vnd: int
    extra_price_vnd: int
    total_price_vnd: int
    egg_count: int | None = None


def dish_price(
    class_name: str,
    prices: dict[str, PriceRow],
    *,
    uncertain: bool = False,
    egg_count: int | None = None,
) -> BillPrice:
    # Keep uncertain predictions visibly flagged, but bill the model's best
    # known class instead of silently turning the item into a zero-price dish.
    _ = uncertain
    if class_name not in prices:
        return BillPrice(base_price_vnd=0, extra_price_vnd=0, total_price_vnd=0)

    base_price = prices[class_name].price_vnd
    if class_name != THIT_KHO_TRUNG_CLASS:
        return BillPrice(base_price_vnd=base_price, extra_price_vnd=0, total_price_vnd=base_price)

    safe_egg_count = max(INCLUDED_EGGS, int(egg_count or INCLUDED_EGGS))
    extra_price = max(0, safe_egg_count - INCLUDED_EGGS) * EXTRA_EGG_PRICE_VND
    return BillPrice(
        base_price_vnd=base_price,
        extra_price_vnd=extra_price,
        total_price_vnd=base_price + extra_price,
        egg_count=safe_egg_count,
    )
