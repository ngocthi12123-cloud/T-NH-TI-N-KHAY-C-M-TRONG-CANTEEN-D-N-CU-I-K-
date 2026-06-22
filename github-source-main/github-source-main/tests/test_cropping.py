from __future__ import annotations

import unittest

from canteen_checkout.cropping import five_compartment_template


class CroppingTests(unittest.TestCase):
    def test_official_landscape_template_at_reference_size(self) -> None:
        regions = five_compartment_template(1920, 1080)
        self.assertEqual([region.name for region in regions], ["top_left", "top_right", "bottom_left", "bottom_center", "bottom_right"])
        self.assertEqual(
            [(region.x, region.y, region.w, region.h) for region in regions],
            [(330, 74, 520, 480), (1029, 70, 430, 500), (274, 549, 424, 420), (700, 574, 370, 380), (1080, 579, 380, 399)],
        )

    def test_official_landscape_template_scales(self) -> None:
        large = five_compartment_template(1920, 1080)
        small = five_compartment_template(960, 540)
        for left, right in zip(large, small):
            self.assertLessEqual(abs(left.x / 2 - right.x), 1)
            self.assertLessEqual(abs(left.y / 2 - right.y), 1)
            self.assertLessEqual(abs(left.w / 2 - right.w), 1)
            self.assertLessEqual(abs(left.h / 2 - right.h), 1)


if __name__ == "__main__":
    unittest.main()
