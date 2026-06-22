from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, DOWNLOADS_DIR, PROJECT_ROOT, REPORTS_DIR, REVIEWED_DIR
from canteen_checkout.data_quality import assess_image, hamming_distance_hex, normalize_image
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    train_weight: int
    priority: int


@dataclass(frozen=True)
class ImageItem:
    source_name: str
    class_name: str
    path: Path
    phash: str
    sha256: str
    train_weight: int
    priority: int


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def latest_merge_processed() -> Path:
    root = DOWNLOADS_DIR / "merge_batches"
    candidates = sorted(
        (p / "processed" for p in root.glob("merge_*") if (p / "processed").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No processed merge folders found in {root}")
    return candidates[0]


def latest_external_reviewed() -> Path:
    if REVIEWED_DIR.exists():
        return REVIEWED_DIR
    root = DOWNLOADS_DIR / "external_staging"
    candidates = sorted(
        (p / "reviewed" for p in root.glob("external_*") if (p / "reviewed").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No reviewed external staging folders found in {root}")
    return candidates[0]


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def is_near_duplicate(phash: str, seen: list[str], threshold: int) -> bool:
    return any(hamming_distance_hex(phash, old) <= threshold for old in seen)


def collect_candidate_items(specs: list[SourceSpec]) -> tuple[list[ImageItem], list[dict[str, str]]]:
    items: list[ImageItem] = []
    rows: list[dict[str, str]] = []

    for class_name in DISH_CLASSES:
        for spec in specs:
            class_dir = spec.root / class_name
            for path in list_images(class_dir):
                image, metrics, reasons = assess_image(path)
                if metrics is None:
                    rows.append(
                        {
                            "status": "rejected",
                            "reason": ";".join(reasons) or "invalid_image",
                            "source": spec.name,
                            "class_name": class_name,
                            "source_path": relative_or_absolute(path),
                            "target_split": "",
                            "target_path": "",
                            "repeat_index": "",
                            "phash": "",
                        }
                    )
                    continue
                items.append(
                    ImageItem(
                        source_name=spec.name,
                        class_name=class_name,
                        path=path,
                        phash=metrics.phash,
                        sha256=metrics.sha256,
                        train_weight=max(1, spec.train_weight),
                        priority=spec.priority,
                    )
                )
    return items, rows


def collect_items(specs: list[SourceSpec], duplicate_hamming: int, cross_class_hamming: int) -> tuple[dict[str, list[ImageItem]], list[dict[str, str]]]:
    candidate_items, rows = collect_candidate_items(specs)
    excluded_ids: set[int] = set()

    by_sha: dict[str, list[ImageItem]] = defaultdict(list)
    for item in candidate_items:
        by_sha[item.sha256].append(item)
    for sha_items in by_sha.values():
        if len({item.class_name for item in sha_items}) <= 1:
            continue
        for item in sha_items:
            excluded_ids.add(id(item))
            rows.append(
                {
                    "status": "skipped",
                    "reason": "exact_cross_class_conflict",
                    "source": item.source_name,
                    "class_name": item.class_name,
                    "source_path": relative_or_absolute(item.path),
                    "target_split": "",
                    "target_path": "",
                    "repeat_index": "",
                    "phash": item.phash,
                }
            )

    sorted_items = sorted(candidate_items, key=lambda item: (item.sha256, item.path.as_posix()))
    for idx, item in enumerate(sorted_items):
        if id(item) in excluded_ids:
            continue
        for other in sorted_items[idx + 1 :]:
            if id(other) in excluded_ids:
                continue
            if item.class_name == other.class_name:
                continue
            distance = hamming_distance_hex(item.phash, other.phash)
            if distance <= cross_class_hamming:
                for conflict_item in [item, other]:
                    excluded_ids.add(id(conflict_item))
                    rows.append(
                        {
                            "status": "skipped",
                            "reason": f"near_cross_class_conflict_hamming_{distance}",
                            "source": conflict_item.source_name,
                            "class_name": conflict_item.class_name,
                            "source_path": relative_or_absolute(conflict_item.path),
                            "target_split": "",
                            "target_path": "",
                            "repeat_index": "",
                            "phash": conflict_item.phash,
                        }
                    )

    by_class: dict[str, list[ImageItem]] = defaultdict(list)
    seen_by_class: dict[str, list[str]] = defaultdict(list)
    for item in sorted(candidate_items, key=lambda item: (item.class_name, item.sha256, item.path.as_posix())):
        if id(item) in excluded_ids:
            continue
        seen = seen_by_class[item.class_name]
        if is_near_duplicate(item.phash, seen, duplicate_hamming):
            rows.append(
                {
                    "status": "skipped",
                    "reason": "near_duplicate_same_class",
                    "source": item.source_name,
                    "class_name": item.class_name,
                    "source_path": relative_or_absolute(item.path),
                    "target_split": "",
                    "target_path": "",
                    "repeat_index": "",
                    "phash": item.phash,
                }
            )
            continue
        seen.append(item.phash)
        by_class[item.class_name].append(item)
    return by_class, rows


def split_items(
    items: list[ImageItem],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[ImageItem]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    total_ratio = train_ratio + val_ratio + test_ratio
    val_count = int(round(n * val_ratio / total_ratio))
    test_count = int(round(n * test_ratio / total_ratio))
    if n >= 3:
        if val_ratio > 0:
            val_count = max(1, val_count)
        if test_ratio > 0:
            test_count = max(1, test_count)
    while val_count + test_count >= n and (val_count > 0 or test_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    train_count = n - val_count - test_count
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def cap_reviewed_items(
    by_class: dict[str, list[ImageItem]],
    *,
    max_reviewed_per_old: float,
    min_reviewed_per_class: int,
    seed: int,
) -> tuple[dict[str, list[ImageItem]], list[dict[str, str]]]:
    if max_reviewed_per_old <= 0:
        return by_class, []

    rng = random.Random(seed)
    capped: dict[str, list[ImageItem]] = {}
    rows: list[dict[str, str]] = []
    for class_name, items in by_class.items():
        old_items = [item for item in items if item.source_name == "old"]
        reviewed_items = [item for item in items if item.source_name == "reviewed"]
        other_items = [item for item in items if item.source_name not in {"old", "reviewed"}]
        reviewed_limit = max(min_reviewed_per_class, int(round(len(old_items) * max_reviewed_per_old)))
        if len(reviewed_items) <= reviewed_limit:
            capped[class_name] = items
            continue
        shuffled = reviewed_items[:]
        rng.shuffle(shuffled)
        kept_reviewed = shuffled[:reviewed_limit]
        skipped_reviewed = shuffled[reviewed_limit:]
        capped[class_name] = old_items + kept_reviewed + other_items
        for item in skipped_reviewed:
            rows.append(
                {
                    "status": "skipped",
                    "reason": "reviewed_cap",
                    "source": item.source_name,
                    "class_name": class_name,
                    "source_path": relative_or_absolute(item.path),
                    "target_split": "",
                    "target_path": "",
                    "repeat_index": "",
                    "phash": item.phash,
                }
            )
    return capped, rows


def clear_classification_root(root: Path, *, clear_all: bool = False) -> None:
    managed_prefixes = ("old_", "reviewed_")
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if clear_all and split_dir.exists():
            shutil.rmtree(split_dir)
            continue
        if split_dir.exists():
            for path in split_dir.rglob("*"):
                if path.is_file() and path.name.startswith(managed_prefixes):
                    path.unlink()
    for split in ["train", "val", "test"]:
        for class_name in DISH_CLASSES:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_item(
    item: ImageItem,
    target_dir: Path,
    *,
    repeat_index: int,
    image_size: int,
    mode: str,
    dry_run: bool,
) -> Path:
    suffix = f"r{repeat_index:02d}" if repeat_index else "r00"
    filename = f"{item.source_name}_{item.sha256[:10]}_{suffix}.jpg"
    target = target_dir / filename
    if dry_run:
        return target
    image, _, _ = assess_image(item.path)
    if image is None:
        raise ValueError(f"Cannot reopen image: {item.path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_image(image, image_size=image_size, mode=mode)
    normalized.save(target, format="JPEG", quality=92, optimize=True)
    return target


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "reason",
        "source",
        "class_name",
        "source_path",
        "target_split",
        "target_path",
        "repeat_index",
        "phash",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted data/classification from old candidates and reviewed images.")
    parser.add_argument("--old-source", type=Path, default=None, help="Old curated candidate source. Defaults to latest merge_*/processed.")
    parser.add_argument("--reviewed-source", type=Path, default=None, help="New reviewed source. Defaults to latest external_*/reviewed.")
    parser.add_argument("--out", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--old-weight", type=int, default=1)
    parser.add_argument("--reviewed-weight", type=int, default=1)
    parser.add_argument("--max-reviewed-per-old", type=float, default=0.0, help="Cap reviewed unique images to N times old unique images per class. Use 0 to disable.")
    parser.add_argument("--min-reviewed-per-class", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--duplicate-hamming", type=int, default=8)
    parser.add_argument("--cross-class-hamming", type=int, default=4, help="Skip near-duplicate images that appear under different classes.")
    parser.add_argument("--clear", action="store_true", help="Clear only files generated by this script before writing. User-added files are preserved.")
    parser.add_argument("--clear-all", action="store_true", help="Danger: remove all existing train/val/test files before writing.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "weighted_classification_dataset_report.csv")
    args = parser.parse_args()

    if args.old_source is not None:
        old_source = args.old_source
    else:
        try:
            old_source = latest_merge_processed()
        except FileNotFoundError:
            old_source = PROJECT_ROOT / "data" / "archive" / "empty_old_source"
    reviewed_source = args.reviewed_source or latest_external_reviewed()
    specs = [
        SourceSpec("old", old_source, args.old_weight, priority=1),
        SourceSpec("reviewed", reviewed_source, args.reviewed_weight, priority=1),
    ]

    print("Sources:")
    for spec in specs:
        print(f"- {spec.name}: {spec.root} (train_weight={spec.train_weight})")
    print("Output:", args.out)
    print("Dry run:", args.dry_run)

    by_class, report_rows = collect_items(specs, args.duplicate_hamming, args.cross_class_hamming)
    by_class, cap_rows = cap_reviewed_items(
        by_class,
        max_reviewed_per_old=args.max_reviewed_per_old,
        min_reviewed_per_class=args.min_reviewed_per_class,
        seed=args.seed,
    )
    report_rows.extend(cap_rows)
    summary: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out": relative_or_absolute(args.out),
        "sources": [
            {
                "name": spec.name,
                "root": relative_or_absolute(spec.root),
                "train_weight": spec.train_weight,
                "priority": spec.priority,
            }
            for spec in specs
        ],
        "duplicate_hamming": args.duplicate_hamming,
        "cross_class_hamming": args.cross_class_hamming,
        "max_reviewed_per_old": args.max_reviewed_per_old,
        "min_reviewed_per_class": args.min_reviewed_per_class,
        "classes": {},
    }

    if args.clear_all and not args.clear:
        parser.error("--clear-all requires --clear")

    if not args.dry_run and args.clear:
        clear_classification_root(args.out, clear_all=args.clear_all)

    total_written = 0
    for class_name in DISH_CLASSES:
        split_map = split_items(
            by_class[class_name],
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        class_summary: dict[str, object] = {"unique": len(by_class[class_name]), "splits": {}, "sources": {}}
        source_counter = Counter(item.source_name for item in by_class[class_name])
        class_summary["sources"] = dict(source_counter)

        for split_name, split_items_list in split_map.items():
            split_written = 0
            split_unique = len(split_items_list)
            for item in split_items_list:
                repeats = item.train_weight if split_name == "train" else 1
                for repeat_index in range(repeats):
                    target_dir = args.out / split_name / class_name
                    target = copy_item(
                        item,
                        target_dir,
                        repeat_index=repeat_index,
                        image_size=args.image_size,
                        mode=args.mode,
                        dry_run=args.dry_run,
                    )
                    split_written += 1
                    total_written += 1
                    report_rows.append(
                        {
                            "status": "written" if not args.dry_run else "would_write",
                            "reason": "",
                            "source": item.source_name,
                            "class_name": class_name,
                            "source_path": relative_or_absolute(item.path),
                            "target_split": split_name,
                            "target_path": relative_or_absolute(target),
                            "repeat_index": str(repeat_index),
                            "phash": item.phash,
                        }
                    )
            class_summary["splits"][split_name] = {"unique": split_unique, "written": split_written}
        summary["classes"][class_name] = class_summary
        print(f"{class_name}: unique={class_summary['unique']} sources={class_summary['sources']} splits={class_summary['splits']}")

    if not args.dry_run:
        write_csv(args.report, report_rows)
        summary_path = args.report.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report: {args.report}")
        print(f"Summary: {summary_path}")
    print(f"Total {'would write' if args.dry_run else 'written'}: {total_written}")


if __name__ == "__main__":
    main()
