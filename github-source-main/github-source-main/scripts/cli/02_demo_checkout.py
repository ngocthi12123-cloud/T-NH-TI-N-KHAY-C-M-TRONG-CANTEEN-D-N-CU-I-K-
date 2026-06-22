from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import torch
from PIL import Image

from canteen_checkout.config import BILLS_DIR, CROPPED_DISHES_DIR, DEFAULT_MODEL_PATH
from canteen_checkout.cropping import crop_regions, five_compartment_template, load_regions
from canteen_checkout.io_utils import load_prices
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device
from canteen_checkout.pricing import THIT_KHO_TRUNG_CLASS, dish_price


@torch.no_grad()
def predict_crop(model, class_names: list[str], image_size: int, path: Path, device: torch.device) -> tuple[str, float]:
    image = Image.open(path).convert("RGB")
    tensor = eval_transforms(image_size)(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, index = torch.max(probabilities, dim=0)
    return class_names[int(index)], float(confidence.cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN-only canteen checkout on a tray image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--regions-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    height, width = image.shape[:2]
    regions = load_regions(args.regions_json) if args.regions_json else five_compartment_template(width, height)

    device = resolve_device()
    model, class_names, image_size, _ = load_checkpoint(args.model, device)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_paths = crop_regions(args.image, regions, CROPPED_DISHES_DIR / f"{args.image.stem}_{run_id}")
    prices = load_prices()

    items: list[dict[str, object]] = []
    total = 0
    for crop_path, region in zip(crop_paths, regions):
        if region.label:
            class_name, confidence = region.label, 1.0
        else:
            class_name, confidence = predict_crop(model, class_names, image_size, crop_path, device)
        uncertain = confidence < args.threshold
        price = dish_price(
            class_name,
            prices,
            uncertain=uncertain,
            egg_count=1 if class_name == THIT_KHO_TRUNG_CLASS else None,
        )
        total += price.total_price_vnd
        items.append(
            {
                "crop_path": str(crop_path),
                "region_name": region.name,
                "class_name": class_name,
                "display_name": prices[class_name].display_name if class_name in prices else class_name,
                "confidence": round(confidence, 4),
                "uncertain": uncertain,
                "price_vnd": price.total_price_vnd,
            }
        )

    bill = {
        "image_path": str(args.image),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model),
        "region_source": "json" if args.regions_json else "template",
        "threshold": args.threshold,
        "items": items,
        "total_vnd": total,
    }
    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    bill_path = BILLS_DIR / f"{args.image.stem}_{run_id}_bill.json"
    bill_path.write_text(json.dumps(bill, indent=2, ensure_ascii=False), encoding="utf-8")

    for index, item in enumerate(items, 1):
        marker = " (uncertain)" if item["uncertain"] else ""
        print(f"{index}. {item['display_name']} - {item['price_vnd']:,} VND - conf={item['confidence']:.2f}{marker}")
    print(f"Total: {total:,} VND")
    print(f"Bill JSON: {bill_path}")


if __name__ == "__main__":
    main()
