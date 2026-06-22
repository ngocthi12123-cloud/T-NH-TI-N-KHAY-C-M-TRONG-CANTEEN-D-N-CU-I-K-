from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from canteen_checkout.config import CLASSIFICATION_DIR, CROPPED_DISHES_DIR, DISH_CLASSES
from canteen_checkout.cropping import (
    crop_regions,
    five_compartment_template,
    load_regions,
    save_regions,
    select_regions_interactive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop dish regions from a tray image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--regions-json", type=Path, default=None)
    parser.add_argument("--save-regions", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--add-to-dataset",
        choices=["train", "val", "test"],
        default=None,
        help="Copy labeled crops into data/classification/<split>/<label>/.",
    )
    args = parser.parse_args()

    image_path = args.image
    out_dir = args.out_dir or (CROPPED_DISHES_DIR / image_path.stem)

    if args.interactive:
        regions = select_regions_interactive(image_path)
    elif args.regions_json:
        regions = load_regions(args.regions_json)
    else:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]
        regions = five_compartment_template(w, h)

    if args.save_regions:
        save_regions(args.save_regions, regions)
        print(f"Saved regions to {args.save_regions}")

    outputs = crop_regions(image_path, regions, out_dir)
    print(f"Cropped {len(outputs)} regions to {out_dir}")
    for path in outputs:
        print(path)

    if args.add_to_dataset:
        copied = 0
        for path, region in zip(outputs, regions):
            if not region.label:
                continue
            if region.label not in DISH_CLASSES:
                raise ValueError(f"Unknown label in region JSON: {region.label}")
            target_dir = CLASSIFICATION_DIR / args.add_to_dataset / region.label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{image_path.stem}_{path.name}"
            shutil.copy2(path, target)
            copied += 1
        print(f"Copied {copied} labeled crops into {CLASSIFICATION_DIR / args.add_to_dataset}")


if __name__ == "__main__":
    main()
