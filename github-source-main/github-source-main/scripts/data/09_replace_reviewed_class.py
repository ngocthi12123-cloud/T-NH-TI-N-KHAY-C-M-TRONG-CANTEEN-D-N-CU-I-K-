from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import ARCHIVE_DIR, DISH_CLASSES, PROJECT_ROOT, REPORTS_DIR, REVIEWED_DIR
from canteen_checkout.data_quality import (
    assess_image,
    hamming_distance_hex,
    normalize_image,
    perceptual_hash,
    quality_reasons,
)
from canteen_checkout.io_utils import list_images


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def normalized_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def near_duplicate(phash: str, seen: list[tuple[str, str]], threshold: int) -> str | None:
    for old_hash, old_path in seen:
        distance = hamming_distance_hex(phash, old_hash)
        if distance <= threshold:
            return f"{distance}:{old_path}"
    return None


def index_other_classes(
    reviewed_root: Path,
    target_class: str,
    image_size: int,
) -> tuple[set[str], list[tuple[str, str]]]:
    digests: set[str] = set()
    phashes: list[tuple[str, str]] = []
    for class_name in DISH_CLASSES:
        if class_name == target_class:
            continue
        for path in list_images(reviewed_root / class_name):
            image, _metrics, _reasons = assess_image(path)
            if image is None:
                continue
            normalized = normalize_image(image, image_size=image_size, mode="pad")
            digests.add(normalized_digest(normalized))
            phashes.append((perceptual_hash(normalized), relative_or_absolute(path)))
    return digests, phashes


def write_contact_sheets(paths: list[Path], output_dir: Path, page_size: int = 48) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    columns = 6
    thumb_size = 150
    label_height = 22
    rows = page_size // columns
    for page_index, start in enumerate(range(0, len(paths), page_size), start=1):
        page_paths = paths[start : start + page_size]
        canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(page_paths):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            col = index % columns
            row = index // columns
            x = col * thumb_size + (thumb_size - image.width) // 2
            y = row * (thumb_size + label_height) + (thumb_size - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((col * thumb_size + 4, row * (thumb_size + label_height) + thumb_size + 3), path.stem, fill="black")
        output = output_dir / f"accepted_{page_index:02d}.jpg"
        canvas.save(output, "JPEG", quality=88, optimize=True)
        produced.append(output)
    return produced


def write_reports(rows: list[dict[str, str]], summary: dict[str, object], report_stem: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"{report_stem}.csv"
    fields = ["status", "reason", "source_path", "target_path", "width", "height", "blur_score", "phash"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = REPORTS_DIR / f"{report_stem}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and transactionally replace one reviewed classification class.")
    parser.add_argument("--class-name", required=True, choices=DISH_CLASSES)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reviewed-root", type=Path, default=REVIEWED_DIR)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--cross-class-phash-threshold", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=180)
    parser.add_argument("--min-blur-score", type=float, default=20.0)
    parser.add_argument(
        "--exclude-list",
        type=Path,
        help="Optional UTF-8 file containing one source filename per line to reject after manual review.",
    )
    parser.add_argument("--apply", action="store_true", help="Swap the staged class into reviewed after validation.")
    parser.add_argument("--report-stem", default="reviewed_class_replacement")
    args = parser.parse_args()

    source = args.source.resolve()
    source_paths = list_images(source)
    if not source_paths:
        raise SystemExit(f"No images found in {source}")
    manual_excludes: set[str] = set()
    if args.exclude_list:
        manual_excludes = {
            line.strip()
            for line in args.exclude_list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = PROJECT_ROOT / "data" / "staging" / "reviewed_replacements" / f"{args.class_name}_{timestamp}"
    staging.mkdir(parents=True, exist_ok=False)

    cross_digests, cross_phashes = index_other_classes(args.reviewed_root, args.class_name, args.image_size)
    seen_digests: set[str] = set()
    seen_phashes: list[tuple[str, str]] = []
    accepted_paths: list[Path] = []
    rows: list[dict[str, str]] = []

    for source_path in source_paths:
        image, metrics, errors = assess_image(source_path)
        reasons = list(errors)
        if source_path.name in manual_excludes:
            reasons.append("manual_review_reject")
        if metrics is not None:
            reasons.extend(
                quality_reasons(
                    metrics,
                    min_size=args.min_size,
                    min_blur_score=args.min_blur_score,
                )
            )

        normalized = None
        phash = ""
        digest = ""
        if image is not None and not reasons:
            normalized = normalize_image(image, image_size=args.image_size, mode="pad")
            phash = perceptual_hash(normalized)
            digest = normalized_digest(normalized)
            if digest in seen_digests:
                reasons.append("duplicate_pixels_batch")
            else:
                duplicate = near_duplicate(phash, seen_phashes, args.phash_threshold)
                if duplicate:
                    reasons.append(f"duplicate_phash_batch:{duplicate}")
            if digest in cross_digests:
                reasons.append("duplicate_pixels_cross_class")
            else:
                conflict = near_duplicate(phash, cross_phashes, args.cross_class_phash_threshold)
                if conflict:
                    reasons.append(f"duplicate_phash_cross_class:{conflict}")

        target_path = ""
        if normalized is not None and not reasons:
            target = staging / f"{args.class_name}_{len(accepted_paths) + 1:04d}.jpg"
            normalized.save(target, "JPEG", quality=92, optimize=True)
            target_path = relative_or_absolute(target)
            accepted_paths.append(target)
            seen_digests.add(digest)
            seen_phashes.append((phash, relative_or_absolute(source_path)))

        rows.append(
            {
                "status": "accepted" if not reasons else "rejected",
                "reason": ";".join(reasons),
                "source_path": str(source_path),
                "target_path": target_path,
                "width": str(metrics.width) if metrics else "",
                "height": str(metrics.height) if metrics else "",
                "blur_score": f"{metrics.blur_score:.2f}" if metrics else "",
                "phash": phash,
            }
        )

    if not accepted_paths:
        shutil.rmtree(staging)
        raise SystemExit("No images passed validation; reviewed data was not changed.")

    sheet_dir = REPORTS_DIR / f"{args.report_stem}_contact_sheets"
    sheets = write_contact_sheets(accepted_paths, sheet_dir)
    backup_path: Path | None = None
    target_dir = args.reviewed_root / args.class_name
    if args.apply:
        backup_path = ARCHIVE_DIR / f"reviewed_{args.class_name}_before_desktop_{timestamp}"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            target_dir.replace(backup_path)
        try:
            staging.replace(target_dir)
        except Exception:
            if backup_path.exists() and not target_dir.exists():
                backup_path.replace(target_dir)
            raise

    reason_counts = Counter(reason.split(":", 1)[0] for row in rows for reason in row["reason"].split(";") if reason)
    summary: dict[str, object] = {
        "class_name": args.class_name,
        "source": str(source),
        "source_images": len(source_paths),
        "accepted": len(accepted_paths),
        "rejected": len(source_paths) - len(accepted_paths),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "dedupe_after_preprocessing": True,
        "phash_threshold": args.phash_threshold,
        "cross_class_phash_threshold": args.cross_class_phash_threshold,
        "image_size": args.image_size,
        "manual_exclude_count": len(manual_excludes),
        "applied": args.apply,
        "staging_or_target": relative_or_absolute(target_dir if args.apply else staging),
        "backup": relative_or_absolute(backup_path) if backup_path else None,
        "contact_sheets": [relative_or_absolute(path) for path in sheets],
    }
    write_reports(rows, summary, args.report_stem)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
