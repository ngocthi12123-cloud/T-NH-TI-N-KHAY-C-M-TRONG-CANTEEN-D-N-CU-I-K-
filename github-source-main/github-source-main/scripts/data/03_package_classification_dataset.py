from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import CLASSIFICATION_DIR, DISH_CLASSES, OUTPUTS_DIR, PROJECT_ROOT
from canteen_checkout.io_utils import IMAGE_EXTENSIONS


EXPECTED_SPLITS = ["train", "val", "test"]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_and_class(root: Path, path: Path) -> tuple[str, str]:
    rel = path.resolve().relative_to(root.resolve())
    if len(rel.parts) < 3:
        raise ValueError(f"Expected split/class/file layout, got: {relative_or_absolute(path)}")
    return rel.parts[0], rel.parts[1]


def validate_layout(root: Path, allow_missing: bool) -> None:
    missing: list[str] = []
    for split in EXPECTED_SPLITS:
        for class_name in DISH_CLASSES:
            class_dir = root / split / class_name
            if not class_dir.exists() or not list_images(class_dir):
                missing.append(f"{split}/{class_name}")
    if missing and not allow_missing:
        raise SystemExit("Missing or empty class folders: " + ", ".join(missing))


def build_manifest(source: Path, files: list[Path]) -> dict[str, object]:
    split_counts = {split: {class_name: 0 for class_name in DISH_CLASSES} for split in EXPECTED_SPLITS}
    class_totals = {class_name: 0 for class_name in DISH_CLASSES}
    file_rows: list[dict[str, object]] = []
    total_size = 0

    for path in files:
        split, class_name = split_and_class(source, path)
        size = path.stat().st_size
        total_size += size
        if split in split_counts and class_name in split_counts[split]:
            split_counts[split][class_name] += 1
        if class_name in class_totals:
            class_totals[class_name] += 1
        rel = path.resolve().relative_to(source.resolve()).as_posix()
        file_rows.append(
            {
                "path": f"classification/{rel}",
                "split": split,
                "class_name": class_name,
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": relative_or_absolute(source),
        "layout_root": "classification",
        "total_files": len(files),
        "total_size_bytes": total_size,
        "split_counts": split_counts,
        "class_totals": class_totals,
        "files": file_rows,
    }


def write_zip(source: Path, files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            rel = path.resolve().relative_to(source.resolve()).as_posix()
            archive.write(path, f"classification/{rel}")


def verify_zip(output: Path, manifest: dict[str, object]) -> None:
    expected = sorted(row["path"] for row in manifest["files"])  # type: ignore[index]
    with zipfile.ZipFile(output, "r") as archive:
        actual = sorted(name for name in archive.namelist() if not name.endswith("/"))
        bad = archive.testzip()
    if bad:
        raise SystemExit(f"Zip verification failed at entry: {bad}")
    if actual != expected:
        raise SystemExit("Zip verification failed: archive entries do not match manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package data/classification for Colab/Kaggle/Drive training.")
    parser.add_argument("--source", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "cloud" / "classification.zip")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing split/class folders for smoke datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect and print manifest summary; do not write zip.")
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    validate_layout(source, allow_missing=args.allow_missing)
    files = list_images(source)
    if not files:
        raise SystemExit(f"No images found under {relative_or_absolute(source)}")

    manifest = build_manifest(source, files)
    manifest["archive"] = relative_or_absolute(output)

    if not args.dry_run:
        write_zip(source, files, output)
        verify_zip(output, manifest)
        manifest["archive_size_bytes"] = output.stat().st_size
        manifest["archive_sha256"] = sha256_file(output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["manifest"] = relative_or_absolute(manifest_path)

    summary = {
        "source": manifest["source"],
        "archive": manifest["archive"],
        "manifest": relative_or_absolute(manifest_path),
        "dry_run": args.dry_run,
        "total_files": manifest["total_files"],
        "total_size_mb": round(float(manifest["total_size_bytes"]) / (1024 * 1024), 2),
        "archive_size_mb": round(float(manifest.get("archive_size_bytes", 0)) / (1024 * 1024), 2),
        "archive_sha256": manifest.get("archive_sha256", ""),
        "split_counts": manifest["split_counts"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
