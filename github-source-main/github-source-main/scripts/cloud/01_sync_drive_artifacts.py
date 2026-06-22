from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import string
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, REPORTS_DIR


MODEL_FILES = [
    "dish_classifier.pt",
    "class_names.json",
]

PACKAGE_FILES = [
    "classification.zip",
    "classification.manifest.json",
]

PROJECT_FILES = [
    "00_colab_kaggle_workflow.ipynb",
    "README.md",
    "data/README.md",
    "scripts/README.md",
    "requirements.txt",
    "prices.csv",
]

RUN_KINDS = ("classifier",)


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_equal(source: Path, target: Path) -> bool:
    if not source.exists() or not target.exists():
        return False
    if source.stat().st_size != target.stat().st_size:
        return False
    return sha256_file(source) == sha256_file(target)


def candidate_drive_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("CANTEEN_DRIVE_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:/")
        candidates.extend(
            [
                drive / "My Drive" / "canteen_checkout",
                drive / "MyDrive" / "canteen_checkout",
                drive / "Drive của tôi" / "canteen_checkout",
            ]
        )

    home = Path.home()
    candidates.extend(
        [
            home / "Google Drive" / "My Drive" / "canteen_checkout",
            home / "Google Drive" / "canteen_checkout",
            home / "My Drive" / "canteen_checkout",
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_drive_root(value: str | None) -> Path:
    if value:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Drive root not found: {root}")
        return root

    for candidate in candidate_drive_roots():
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(str(path) for path in candidate_drive_roots()[:12])
    raise FileNotFoundError(
        "Could not auto-detect Google Drive canteen folder. "
        "Install Google Drive for desktop or pass --drive-root.\n"
        f"Checked examples:\n{checked}"
    )


def newest_path(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_drive_file(drive_root: Path, filename: str) -> Path | None:
    canonical = drive_root / "models" / filename
    if canonical.exists():
        return canonical
    runs_dir = drive_root / "runs"
    if not runs_dir.exists():
        return None
    return newest_path(list(runs_dir.rglob(filename)))


def latest_model_run(drive_root: Path, run_kind: str) -> Path | None:
    runs_dir = drive_root / "runs" / run_kind
    if not runs_dir.exists():
        return None
    runs = [path for path in runs_dir.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.name) if runs else None


def report_files_in_run(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    suffixes = {".txt", ".json", ".csv", ".png", ".jpg", ".jpeg"}
    keywords = ("report", "history", "result", "confusion", "curve", "labels", "pred")
    files = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        name = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if "reports" in parts or any(keyword in name for keyword in keywords):
            files.append(path.relative_to(run_dir).as_posix())
    return sorted(files)


def copy_file(source: Path, target: Path, *, apply: bool, overwrite: bool) -> dict[str, object]:
    row = {
        "action": "copy_file",
        "source": str(source),
        "target": str(target),
        "status": "planned",
        "bytes": source.stat().st_size if source.exists() else 0,
    }
    if not source.exists():
        row["status"] = "missing_source"
        return row
    if target.exists() and files_equal(source, target):
        row["status"] = "same"
        return row
    if target.exists() and not overwrite:
        row["status"] = "target_exists"
        return row
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row["status"] = "copied"
    return row


def copy_tree(source: Path, target: Path, *, apply: bool) -> dict[str, object]:
    file_count = sum(1 for path in source.rglob("*") if path.is_file()) if source.exists() else 0
    row = {
        "action": "copy_tree",
        "source": str(source),
        "target": str(target),
        "status": "planned",
        "files": file_count,
    }
    if not source.exists():
        row["status"] = "missing_source"
        return row
    if apply:
        shutil.copytree(source, target, dirs_exist_ok=True)
        row["status"] = "copied"
    return row


def push_packages(drive_root: Path, *, apply: bool, overwrite: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in PACKAGE_FILES:
        source = OUTPUTS_DIR / "cloud" / filename
        target = drive_root / "datasets" / filename
        rows.append(copy_file(source, target, apply=apply, overwrite=overwrite))
    return rows


def push_models(drive_root: Path, *, apply: bool, overwrite: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in MODEL_FILES:
        source = MODELS_DIR / filename
        target = drive_root / "models" / filename
        rows.append(copy_file(source, target, apply=apply, overwrite=overwrite))
    return rows


def push_project_files(drive_root: Path, *, apply: bool, overwrite: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in PROJECT_FILES:
        source = PROJECT_ROOT / filename
        target = drive_root / "project_files" / filename
        rows.append(copy_file(source, target, apply=apply, overwrite=overwrite))
    return rows


def pull_models(drive_root: Path, *, apply: bool, overwrite: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in MODEL_FILES:
        source = find_drive_file(drive_root, filename)
        if source is None:
            rows.append(
                {
                    "action": "pull_model",
                    "source": "",
                    "target": relative_or_absolute(MODELS_DIR / filename),
                    "status": "missing_source",
                    "filename": filename,
                }
            )
            continue
        rows.append(copy_file(source, MODELS_DIR / filename, apply=apply, overwrite=overwrite))
    return rows


def pull_latest_runs(drive_root: Path, *, apply: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_kind in RUN_KINDS:
        run_dir = latest_model_run(drive_root, run_kind)
        target_root = OUTPUTS_DIR / "cloud" / "drive_runs" / run_kind
        if run_dir is None:
            rows.append(
                {
                    "action": "pull_latest_run",
                    "source": "",
                    "target": relative_or_absolute(target_root),
                    "status": "missing_source",
                    "run_kind": run_kind,
                }
            )
            continue
        target = target_root / run_dir.name
        row = copy_tree(run_dir, target, apply=apply)
        row["run_kind"] = run_kind
        row["run_name"] = run_dir.name
        row["report_files"] = report_files_in_run(run_dir)
        rows.append(row)
    return rows


def status_payload(drive_root: Path) -> dict[str, object]:
    drive_models = {
        filename: str(find_drive_file(drive_root, filename) or "")
        for filename in MODEL_FILES
    }
    local_models = {
        filename: relative_or_absolute(MODELS_DIR / filename) if (MODELS_DIR / filename).exists() else ""
        for filename in MODEL_FILES
    }
    packages = {
        filename: {
            "local": relative_or_absolute(OUTPUTS_DIR / "cloud" / filename)
            if (OUTPUTS_DIR / "cloud" / filename).exists()
            else "",
            "drive": str(drive_root / "datasets" / filename)
            if (drive_root / "datasets" / filename).exists()
            else "",
        }
        for filename in PACKAGE_FILES
    }
    project_files = {
        filename: {
            "local": relative_or_absolute(PROJECT_ROOT / filename)
            if (PROJECT_ROOT / filename).exists()
            else "",
            "drive": str(drive_root / "project_files" / filename)
            if (drive_root / "project_files" / filename).exists()
            else "",
        }
        for filename in PROJECT_FILES
    }
    latest_runs = {run_kind: latest_model_run(drive_root, run_kind) for run_kind in RUN_KINDS}
    return {
        "drive_root": str(drive_root),
        "drive_models": drive_models,
        "local_models": local_models,
        "packages": packages,
        "project_files": project_files,
        "latest_drive_runs": {
            run_kind: {
                "path": str(run_dir) if run_dir else "",
                "report_files": report_files_in_run(run_dir) if run_dir else [],
            }
            for run_kind, run_dir in latest_runs.items()
        },
    }


def requested_operations(args: argparse.Namespace) -> dict[str, bool]:
    push_inputs = args.push_inputs or args.publish or args.all
    pull_results = args.pull_results or args.pull or args.all
    return {
        "push_packages": push_inputs or args.push_packages,
        "push_project_files": push_inputs or args.push_project_files,
        "push_models": args.push_models,
        "pull_models": pull_results or args.pull_models,
        "pull_latest_runs": pull_results or args.pull_latest_runs or args.pull_latest_run,
    }


def write_log(payload: dict[str, object]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"drive_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Canteen Checkout artifacts between local project and Google Drive for desktop.")
    parser.add_argument("--drive-root", default=None, help="Path such as G:\\My Drive\\canteen_checkout. Defaults to auto-detect.")
    parser.add_argument("--push-packages", action="store_true", help="Copy outputs/cloud/*.zip and manifests to Drive datasets/.")
    parser.add_argument("--push-models", action="store_true", help="Copy local models/*.pt and class_names.json to Drive models/.")
    parser.add_argument("--push-project-files", action="store_true", help="Copy the main notebook, README, requirements, and prices.csv to Drive project_files/.")
    parser.add_argument("--push-inputs", action="store_true", help="Copy dataset packages and project files to Drive; never copies local models.")
    parser.add_argument("--publish", action="store_true", help="Deprecated alias for --push-inputs.")
    parser.add_argument("--pull-results", action="store_true", help="Copy canonical models and the newest run for each model from Drive to local.")
    parser.add_argument("--pull", action="store_true", help="Alias for --pull-results.")
    parser.add_argument("--pull-models", action="store_true", help="Copy classifier model artifacts from Drive to local models/.")
    parser.add_argument("--pull-latest-runs", action="store_true", help="Copy the newest classifier run from Drive.")
    parser.add_argument("--pull-latest-run", action="store_true", help="Deprecated alias for --pull-latest-runs.")
    parser.add_argument("--all", action="store_true", help="Run --push-inputs and --pull-results.")
    parser.add_argument("--status", action="store_true", help="Print Drive/local artifact status.")
    parser.add_argument("--apply", action="store_true", help="Actually copy files. Without this, the script only prints a dry-run plan.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing targets.")
    args = parser.parse_args()

    drive_root = resolve_drive_root(args.drive_root)
    overwrite = not args.no_overwrite
    actions: list[dict[str, object]] = []
    operations = requested_operations(args)
    if args.publish:
        print("warning: --publish is deprecated; using the safe --push-inputs behavior (models are not pushed).", file=sys.stderr)

    if args.status or not any(
        [
            args.push_packages,
            args.push_models,
            args.push_project_files,
            args.push_inputs,
            args.publish,
            args.pull_results,
            args.pull,
            args.pull_models,
            args.pull_latest_runs,
            args.pull_latest_run,
            args.all,
        ]
    ):
        actions.append({"action": "status", "status": "ok", **status_payload(drive_root)})

    if operations["push_packages"]:
        actions.extend(push_packages(drive_root, apply=args.apply, overwrite=overwrite))
    if operations["push_models"]:
        actions.extend(push_models(drive_root, apply=args.apply, overwrite=overwrite))
    if operations["push_project_files"]:
        actions.extend(push_project_files(drive_root, apply=args.apply, overwrite=overwrite))
    if operations["pull_models"]:
        actions.extend(pull_models(drive_root, apply=args.apply, overwrite=overwrite))
    if operations["pull_latest_runs"]:
        actions.extend(pull_latest_runs(drive_root, apply=args.apply))

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "drive_root": str(drive_root),
        "dry_run": not args.apply,
        "actions": actions,
    }
    log_path = write_log(payload)
    payload["log_path"] = relative_or_absolute(log_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
