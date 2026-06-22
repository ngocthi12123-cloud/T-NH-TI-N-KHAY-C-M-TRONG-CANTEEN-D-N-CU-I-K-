from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.data_quality import assess_image, hamming_distance_hex
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


REPORT_FIELDS = [
    "conflict_type",
    "distance",
    "sha256",
    "phash_a",
    "phash_b",
    "split_a",
    "class_a",
    "source_a",
    "path_a",
    "split_b",
    "class_b",
    "source_b",
    "path_b",
    "action",
    "quarantine_path_a",
    "quarantine_path_b",
]


@dataclass(frozen=True)
class AuditItem:
    split: str
    class_name: str
    source: str
    path: Path
    rel_path: str
    sha256: str
    phash: str
    blur_score: float
    brightness: float


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def source_from_name(path: Path) -> str:
    if path.name.startswith("old_"):
        return "old"
    if path.name.startswith("reviewed_"):
        return "reviewed"
    return "unmanaged"


def split_class_for_path(root: Path, path: Path) -> tuple[str, str] | None:
    rel = path.resolve().relative_to(root.resolve())
    parts = rel.parts
    if len(parts) >= 3 and parts[0] in {"train", "val", "test"} and parts[1] in DISH_CLASSES:
        return parts[0], parts[1]
    if len(parts) >= 2 and parts[0] in DISH_CLASSES:
        return "", parts[0]
    return None


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_items(root: Path) -> tuple[list[AuditItem], list[dict[str, str]]]:
    items: list[AuditItem] = []
    invalid_rows: list[dict[str, str]] = []
    for path in list_images(root):
        split_class = split_class_for_path(root, path)
        if split_class is None:
            continue
        split, class_name = split_class
        _, metrics, reasons = assess_image(path)
        if metrics is None:
            invalid_rows.append(
                {
                    "conflict_type": "invalid_image",
                    "distance": "",
                    "sha256": "",
                    "phash_a": "",
                    "phash_b": "",
                    "split_a": split,
                    "class_a": class_name,
                    "source_a": source_from_name(path),
                    "path_a": relative_or_absolute(path),
                    "split_b": "",
                    "class_b": "",
                    "source_b": "",
                    "path_b": "",
                    "action": ";".join(reasons) or "invalid_image",
                    "quarantine_path_a": "",
                    "quarantine_path_b": "",
                }
            )
            continue
        items.append(
            AuditItem(
                split=split,
                class_name=class_name,
                source=source_from_name(path),
                path=path,
                rel_path=relative_or_absolute(path),
                sha256=metrics.sha256,
                phash=metrics.phash,
                blur_score=metrics.blur_score,
                brightness=metrics.brightness,
            )
        )
    return items, invalid_rows


def row_for_pair(conflict_type: str, a: AuditItem, b: AuditItem, *, distance: str = "", action: str = "") -> dict[str, str]:
    return {
        "conflict_type": conflict_type,
        "distance": distance,
        "sha256": a.sha256 if a.sha256 == b.sha256 else "",
        "phash_a": a.phash,
        "phash_b": b.phash,
        "split_a": a.split,
        "class_a": a.class_name,
        "source_a": a.source,
        "path_a": a.rel_path,
        "split_b": b.split,
        "class_b": b.class_name,
        "source_b": b.source,
        "path_b": b.rel_path,
        "action": action,
        "quarantine_path_a": "",
        "quarantine_path_b": "",
    }


def conflict_rows(items: list[AuditItem], phash_threshold: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_sha: dict[str, list[AuditItem]] = defaultdict(list)
    for item in items:
        by_sha[item.sha256].append(item)

    exact_pairs: set[tuple[str, str]] = set()
    for group in by_sha.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if a.class_name != b.class_name:
                rows.append(row_for_pair("exact_cross_class", a, b, action="review_or_quarantine"))
                exact_pairs.add(tuple(sorted((a.rel_path, b.rel_path))))
            elif a.split != b.split:
                rows.append(row_for_pair("exact_cross_split_same_class", a, b, action="review"))
                exact_pairs.add(tuple(sorted((a.rel_path, b.rel_path))))

    for a, b in combinations(items, 2):
        if a.class_name == b.class_name:
            continue
        key = tuple(sorted((a.rel_path, b.rel_path)))
        if key in exact_pairs:
            continue
        distance = hamming_distance_hex(a.phash, b.phash)
        if distance <= phash_threshold:
            rows.append(row_for_pair("near_cross_class", a, b, distance=str(distance), action="review_or_quarantine"))
    return rows


def safe_folder_name(*parts: str) -> str:
    return "__".join(part.replace("/", "_").replace("\\", "_").replace(" ", "_") for part in parts if part)


def unique_destination(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    idx = 1
    while True:
        candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def quarantine_rows(rows: list[dict[str, str]], quarantine_root: Path, dry_run: bool) -> list[dict[str, str]]:
    moved: dict[str, str] = {}
    out_rows: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        if row["conflict_type"] not in {"exact_cross_class", "near_cross_class"}:
            out_rows.append(row)
            continue
        pair = safe_folder_name(row["class_a"], "vs", row["class_b"])
        target_dir = quarantine_root / "label_conflicts" / pair
        for side in ["a", "b"]:
            path_key = f"path_{side}"
            quarantine_key = f"quarantine_path_{side}"
            source_path = PROJECT_ROOT / row[path_key] if not Path(row[path_key]).is_absolute() else Path(row[path_key])
            resolved = str(source_path.resolve())
            if resolved in moved:
                row[quarantine_key] = moved[resolved]
                continue
            target = unique_destination(target_dir, source_path.name)
            row[quarantine_key] = relative_or_absolute(target)
            moved[resolved] = row[quarantine_key]
            if not dry_run and source_path.exists():
                shutil.move(str(source_path), str(target))
        row["action"] = "quarantined" if not dry_run else "would_quarantine"
        out_rows.append(row)
    return out_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit classification data for invalid images and label conflicts.")
    parser.add_argument("--root", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "dataset_conflicts.csv")
    parser.add_argument("--summary", type=Path, default=REPORTS_DIR / "dataset_conflicts.summary.json")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--quarantine-root", type=Path, default=PROJECT_ROOT / "data" / "quarantine")
    parser.add_argument("--quarantine", action="store_true", help="Move cross-class conflicts into data/quarantine.")
    parser.add_argument("--dry-run", action="store_true", help="Only report quarantine targets when used with --quarantine.")
    args = parser.parse_args()

    items, invalid_rows = load_items(args.root)
    rows = invalid_rows + conflict_rows(items, args.phash_threshold)
    if args.quarantine:
        rows = quarantine_rows(rows, args.quarantine_root, args.dry_run)

    write_csv(args.out, rows)
    counts = Counter(row["conflict_type"] for row in rows)
    source_counts = Counter(item.source for item in items)
    class_counts = Counter(item.class_name for item in items)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": relative_or_absolute(args.root),
        "images": len(items),
        "invalid_images": len(invalid_rows),
        "phash_threshold": args.phash_threshold,
        "quarantine": args.quarantine,
        "dry_run": args.dry_run,
        "conflict_counts": dict(counts),
        "source_counts": dict(source_counts),
        "class_counts": dict(class_counts),
        "report": relative_or_absolute(args.out),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
