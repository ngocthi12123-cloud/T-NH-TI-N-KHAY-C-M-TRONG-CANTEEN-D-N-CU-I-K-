from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import (
    BILLS_DIR,
    CLASSIFICATION_DIR,
    CLASS_NAMES_JSON,
    CROPPED_DISHES_DIR,
    DATA_DIR,
    DEMO_TRAYS_DIR,
    DISH_CLASSES,
    DOWNLOADS_DIR,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_CANDIDATES_DIR,
    PRICES_CSV,
    RAW_TEACHER_TRAYS_DIR,
    REPORTS_DIR,
    REJECTED_CANDIDATES_DIR,
    SCRAPED_CANDIDATES_DIR,
    TEMP_TEACHER_CROPS_DIR,
)


@dataclass(frozen=True)
class PriceRow:
    class_name: str
    display_name: str
    price_vnd: int
    reward_points: int
    note: str = ""


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_TEACHER_TRAYS_DIR,
        DEMO_TRAYS_DIR,
        CLASSIFICATION_DIR,
        DOWNLOADS_DIR,
        SCRAPED_CANDIDATES_DIR,
        PROCESSED_CANDIDATES_DIR,
        REJECTED_CANDIDATES_DIR,
        TEMP_TEACHER_CROPS_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        CROPPED_DISHES_DIR,
        BILLS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for class_name in DISH_CLASSES:
            (CLASSIFICATION_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def load_prices(path: Path = PRICES_CSV) -> dict[str, PriceRow]:
    rows: dict[str, PriceRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            price_vnd = int(row["price_vnd"])
            rows[row["class_name"]] = PriceRow(
                class_name=row["class_name"],
                display_name=row["display_name"],
                price_vnd=price_vnd,
                reward_points=int(row.get("reward_points") or max(1, (price_vnd + 999) // 1000)),
                note=row.get("note", ""),
            )
    return rows


def save_class_names(path: Path = CLASS_NAMES_JSON, class_names: Iterable[str] = DISH_CLASSES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(class_names), indent=2, ensure_ascii=False), encoding="utf-8")


def load_class_names(path: Path = CLASS_NAMES_JSON) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prices_and_classes() -> list[str]:
    prices = load_prices()
    issues: list[str] = []
    for class_name in DISH_CLASSES:
        if class_name not in prices:
            issues.append(f"Missing price row for {class_name}")
    extra_prices = sorted(set(prices) - set(DISH_CLASSES))
    for class_name in extra_prices:
        issues.append(f"Price row not in model classes: {class_name}")
    return issues


def copy_teacher_images(source_dir: Path, limit_demo: int = 6) -> tuple[int, int]:
    """Copy teacher tray images into raw and demo folders.

    Returns (raw_count, demo_count). Existing files are left in place.
    """
    raw_images = list_images(source_dir)
    copied_raw = 0
    copied_demo = 0
    for image_path in raw_images:
        rel = image_path.relative_to(source_dir)
        dst = RAW_TEACHER_TRAYS_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(image_path, dst)
            copied_raw += 1
    for image_path in raw_images[:limit_demo]:
        dst = DEMO_TRAYS_DIR / image_path.name
        if not dst.exists():
            shutil.copy2(image_path, dst)
            copied_demo += 1
    return copied_raw, copied_demo
