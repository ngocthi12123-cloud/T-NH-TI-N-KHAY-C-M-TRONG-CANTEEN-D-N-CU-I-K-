from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import Image

from canteen_checkout.config import (
    CLASSIFICATION_DIR,
    DATA_DIR,
    DEFAULT_MODEL_PATH,
    DISH_CLASSES,
    EXTRAS_DIR,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    REVIEW_INBOX_DIR,
    REVIEWED_DIR,
)
from canteen_checkout.data_quality import assess_image, hamming_distance_hex
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device


ACTION_FIELDS = [
    "timestamp",
    "item_id",
    "source_path",
    "action",
    "from_class",
    "to_class",
    "output_path",
    "note",
]

DONE_FIELDS = ["timestamp", "item_id", "path", "root", "folder", "class_name", "note"]

MODEL_CACHE: dict[str, object] = {}
METRIC_CACHE: dict[str, tuple[float, int, object]] = {}
PATH_CACHE: dict[tuple[str, str], list[tuple[Path, str, str]]] = {}
COUNT_CACHE: dict[str, dict[str, object]] = {}


@dataclass(frozen=True)
class DataItem:
    item_id: str
    path: Path
    rel_path: str
    split: str
    class_name: str
    source: str
    filename: str
    sha256: str
    phash: str
    blur_score: float
    brightness: float


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def source_from_name(path: Path) -> str:
    if path.name.startswith("old_"):
        return "old"
    if path.name.startswith("reviewed_"):
        return "reviewed"
    return "unmanaged"


def stable_item_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def cached_assess_metrics(path: Path):
    stat = path.stat()
    key = stable_item_id(path)
    cached = METRIC_CACHE.get(key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    _, metrics, _ = assess_image(path)
    METRIC_CACHE[key] = (stat.st_mtime, stat.st_size, metrics)
    return metrics


def safe_label(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in value.strip().lower())
    value = "_".join(value.split())
    return value or "future_use"


def invalidate_data_cache() -> None:
    PATH_CACHE.clear()
    COUNT_CACHE.clear()


def unique_destination(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if not target.exists():
        return target
    idx = 1
    while True:
        candidate = folder / f"{target.stem}_{idx:03d}{target.suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def roots() -> list[dict[str, str]]:
    candidates = [
        ("classification", CLASSIFICATION_DIR, "classification"),
        ("inbox_review", REVIEW_INBOX_DIR, "folder"),
        ("reviewed", REVIEWED_DIR, "folder"),
        ("extras", EXTRAS_DIR, "folder"),
        ("quarantine", DATA_DIR / "quarantine", "folder"),
        ("data_workspace", DATA_DIR, "folder"),
    ]
    return [{"name": name, "path": relative_or_absolute(path), "mode": mode} for name, path, mode in candidates if path.exists()]


def target_roots() -> list[dict[str, str]]:
    candidates = [
        ("reviewed", REVIEWED_DIR),
        ("inbox_review", REVIEW_INBOX_DIR),
        ("extras", EXTRAS_DIR),
        ("quarantine", DATA_DIR / "quarantine"),
    ]
    return [{"name": name, "path": relative_or_absolute(path)} for name, path in candidates if path.exists()]


def root_mode(root: Path) -> str:
    try:
        if root.resolve() == CLASSIFICATION_DIR.resolve():
            return "classification"
    except FileNotFoundError:
        pass
    return "folder"


def image_folders(root: Path) -> list[str]:
    folders: set[str] = set()
    if not root.exists():
        return []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_parent = path.parent.resolve().relative_to(root.resolve())
            folders.add(rel_parent.as_posix() or ".")
    return sorted(folders, key=lambda value: (value.count("/"), value))


def infer_split_class(root: Path, path: Path) -> tuple[str, str] | None:
    rel = path.resolve().relative_to(root.resolve())
    parts = rel.parts
    if root_mode(root) == "classification":
        if len(parts) >= 3 and parts[0] in {"train", "val", "test"} and parts[1] in DISH_CLASSES:
            return parts[0], parts[1]
        return None
    if len(parts) >= 2:
        folder = Path(*parts[:-1]).as_posix()
        return folder, parts[-2]
    return None


def list_labeled_paths(root: Path, split: str = "", class_name: str = "", folder: str = "") -> list[tuple[Path, str, str]]:
    paths: list[tuple[Path, str, str]] = []
    if not root.exists():
        return paths
    mode = root_mode(root)
    cache_key = (str(root.resolve()), folder if mode == "folder" else "")
    cached = PATH_CACHE.get(cache_key)
    if cached is not None:
        if mode == "classification":
            return [
                (path, item_split, item_class)
                for path, item_split, item_class in cached
                if (not split or item_split == split) and (not class_name or item_class == class_name)
            ]
        return list(cached)
    if mode == "folder" and folder:
        search_root = (root / folder).resolve()
        if not is_inside(search_root, root) or not search_root.exists():
            return paths
        candidates = search_root.rglob("*")
    else:
        candidates = root.rglob("*")
    for path in sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        inferred = infer_split_class(root, path)
        if inferred is None:
            continue
        item_split, item_class = inferred
        if mode == "classification" and split and item_split != split:
            continue
        if mode == "classification" and class_name and item_class != class_name:
            continue
        paths.append((path, item_split, item_class))
    if mode == "classification":
        if not split and not class_name:
            PATH_CACHE[cache_key] = list(paths)
    else:
        PATH_CACHE[cache_key] = list(paths)
    return paths


def make_item(path: Path, item_split: str, item_class: str, include_metrics: bool = False) -> DataItem:
    sha256 = ""
    phash = ""
    blur_score = 0.0
    brightness = 0.0
    if include_metrics:
        metrics = cached_assess_metrics(path)
        if metrics is not None:
            sha256 = metrics.sha256
            phash = metrics.phash
            blur_score = metrics.blur_score
            brightness = metrics.brightness
    return DataItem(
        item_id=stable_item_id(path),
        path=path,
        rel_path=relative_or_absolute(path),
        split=item_split,
        class_name=item_class,
        source=source_from_name(path),
        filename=path.name,
        sha256=sha256,
        phash=phash,
        blur_score=blur_score,
        brightness=brightness,
    )


def list_items(root: Path, split: str = "", class_name: str = "", folder: str = "", include_metrics: bool = False) -> list[DataItem]:
    items: list[DataItem] = []
    for path, item_split, item_class in list_labeled_paths(root, split, class_name, folder):
        items.append(make_item(path, item_split, item_class, include_metrics))
    return items


def count_tree(root: Path) -> dict[str, object]:
    cache_key = str(root.resolve())
    cached = COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result: dict[str, object] = {"total": 0, "splits": {}, "classes": {}, "folders": {}, "sources": {}}
    for path, item_split, item_class in list_labeled_paths(root):
        result["total"] = int(result["total"]) + 1
        split_key = item_split or "root"
        result["splits"].setdefault(split_key, 0)
        result["splits"][split_key] += 1
        result["classes"].setdefault(item_class, 0)
        result["classes"][item_class] += 1
        folder = path.parent.resolve().relative_to(root.resolve()).as_posix() or "."
        result["folders"].setdefault(folder, 0)
        result["folders"][folder] += 1
        source = source_from_name(path)
        result["sources"].setdefault(source, 0)
        result["sources"][source] += 1
    COUNT_CACHE[cache_key] = result
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_action(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in ACTION_FIELDS})


def action_log() -> Path:
    return PROJECT_ROOT / "outputs" / "reports" / "data_ide_actions.csv"


def done_log() -> Path:
    return PROJECT_ROOT / "outputs" / "reports" / "data_ide_done.csv"


def done_item_ids() -> set[str]:
    path = done_log()
    if not path.exists():
        return set()
    rows = read_rows(path)
    done: set[str] = set()
    for row in rows:
        item_id = row.get("item_id", "")
        if not item_id:
            continue
        if row.get("note") == "undo_done":
            done.discard(item_id)
        else:
            done.add(item_id)
    return done


def append_done(row: dict[str, str]) -> None:
    path = done_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DONE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in DONE_FIELDS})


def load_model_once(model_path: Path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    device = resolve_device()
    model, class_names, image_size, checkpoint = load_checkpoint(model_path, device)
    cached = (model, class_names, image_size, checkpoint, device)
    MODEL_CACHE[key] = cached
    return cached


@torch.no_grad()
def predict_item(item: DataItem, threshold: float) -> dict[str, object]:
    model, class_names, image_size, checkpoint, device = load_model_once(DEFAULT_MODEL_PATH)
    transform = eval_transforms(image_size)
    image = Image.open(item.path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    values, indices = probs.topk(k=2)
    top1 = class_names[int(indices[0])]
    top2 = class_names[int(indices[1])]
    top1_conf = float(values[0])
    top2_conf = float(values[1])
    margin = top1_conf - top2_conf
    if item.class_name not in DISH_CLASSES:
        decision = "outside_target_classes"
    elif top1 != item.class_name and top1_conf >= threshold:
        decision = "high_confidence_disagreement"
    elif top1_conf < threshold:
        decision = "low_confidence"
    elif margin < 0.15:
        decision = "small_margin"
    else:
        decision = "ok"
    return {
        "top1": top1,
        "top1_confidence": round(top1_conf, 4),
        "top2": top2,
        "top2_confidence": round(top2_conf, 4),
        "margin": round(margin, 4),
        "decision": decision,
        "model_arch": checkpoint.get("arch", ""),
    }


class DataIDE:
    def __init__(self):
        self.last_items: dict[str, DataItem] = {}

    def state(self) -> dict[str, object]:
        root_list = roots()
        return {
            "roots": root_list,
            "target_roots": target_roots(),
            "classes": DISH_CLASSES,
            "log": relative_or_absolute(action_log()),
            "model_path": relative_or_absolute(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else "",
        }

    def folders(self, root_value: str) -> dict[str, object]:
        root = resolve_project_path(root_value)
        return {
            "ok": True,
            "root": relative_or_absolute(root),
            "mode": root_mode(root),
            "folders": image_folders(root),
        }

    def browse(self, root_value: str, split: str = "", class_name: str = "", folder: str = "", page: int = 0, page_size: int = 80) -> dict[str, object]:
        root = resolve_project_path(root_value)
        done_ids = done_item_ids()
        raw_paths = list_labeled_paths(root, split, class_name, folder)
        paths = [entry for entry in raw_paths if stable_item_id(entry[0]) not in done_ids]
        start = max(0, page * page_size)
        end = min(len(paths), start + page_size)
        items = [make_item(path, item_split, item_class) for path, item_split, item_class in paths[start:end]]
        self.last_items.update({item.item_id: item for item in items})
        return {
            "root": relative_or_absolute(root),
            "mode": root_mode(root),
            "folder": folder,
            "counts": count_tree(root),
            "page": page,
            "page_size": page_size,
            "total": len(paths),
            "hidden_done": len(raw_paths) - len(paths),
            "items": [self.serialize_item(item) for item in items],
        }

    def serialize_item(self, item: DataItem) -> dict[str, object]:
        return {
            "id": item.item_id,
            "path": item.rel_path,
            "image_url": f"/file?path={item.rel_path}",
            "split": item.split,
            "class_name": item.class_name,
            "source": item.source,
            "filename": item.filename,
            "sha256": item.sha256,
            "phash": item.phash,
            "blur_score": round(item.blur_score, 2),
            "brightness": round(item.brightness, 2),
        }

    def item_from_id(self, item_id: str) -> DataItem:
        item = self.last_items.get(item_id)
        if item and item.path.exists():
            return item
        path = resolve_project_path(item_id)
        for root_info in roots():
            root = resolve_project_path(root_info["path"])
            if is_inside(path, root):
                inferred = infer_split_class(root, path)
                if inferred is None:
                    break
                split, class_name = inferred
                _, metrics, _ = assess_image(path)
                return DataItem(
                    item_id=stable_item_id(path),
                    path=path,
                    rel_path=relative_or_absolute(path),
                    split=split,
                    class_name=class_name,
                    source=source_from_name(path),
                    filename=path.name,
                    sha256=metrics.sha256 if metrics else "",
                    phash=metrics.phash if metrics else "",
                    blur_score=metrics.blur_score if metrics else 0.0,
                    brightness=metrics.brightness if metrics else 0.0,
                )
        raise ValueError("Unknown item")

    def move_to_class(self, item_id: str, class_name: str) -> dict[str, object]:
        return self.move_to_target(item_id, REVIEWED_DIR, class_name)

    def move_to_target(self, item_id: str, target_root_value: str | Path, target_label: str) -> dict[str, object]:
        item = self.item_from_id(item_id)
        target_root = resolve_project_path(target_root_value)
        if not is_inside(target_root, DATA_DIR):
            raise ValueError("Target root must be inside data/")
        target_label = safe_label(target_label)
        if target_root.resolve() == REVIEWED_DIR.resolve() and target_label not in DISH_CLASSES:
            raise ValueError("Reviewed target must be one of the 11 official classes")
        target_dir = target_root / target_label
        if item.path.resolve() == (target_dir / item.filename).resolve():
            return {"ok": True, "output_path": item.rel_path, "noop": True}
        target = unique_destination(target_dir, item.filename)
        shutil.move(str(item.path), str(target))
        invalidate_data_cache()
        append_action(
            action_log(),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "source_path": item.rel_path,
                "action": "move_target",
                "from_class": item.class_name,
                "to_class": target_label,
                "output_path": relative_or_absolute(target),
                "note": relative_or_absolute(target_root),
            },
        )
        return {"ok": True, "output_path": relative_or_absolute(target)}

    def quarantine(self, item_id: str, label: str = "manual_rejected") -> dict[str, object]:
        return self.move_to_target(item_id, DATA_DIR / "quarantine", label)

    def mark_done(self, item_id: str, note: str = "kept_as_is") -> dict[str, object]:
        item = self.item_from_id(item_id)
        append_done(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "path": item.rel_path,
                "root": "",
                "folder": item.split,
                "class_name": item.class_name,
                "note": note,
            }
        )
        append_action(
            action_log(),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "item_id": item.item_id,
                "source_path": item.rel_path,
                "action": "mark_done",
                "from_class": item.class_name,
                "to_class": item.class_name,
                "output_path": item.rel_path,
                "note": note,
            },
        )
        return {"ok": True, "done": True, "path": item.rel_path}

    def undo(self) -> dict[str, object]:
        rows = read_rows(action_log())
        for row in reversed(rows):
            if row.get("action") == "undo":
                continue
            if row.get("action") == "mark_done":
                if row.get("item_id", "") not in done_item_ids():
                    continue
                append_done(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "item_id": row.get("item_id", ""),
                        "path": row.get("source_path", ""),
                        "root": "",
                        "folder": "",
                        "class_name": row.get("from_class", ""),
                        "note": "undo_done",
                    }
                )
                append_action(
                    action_log(),
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "item_id": row.get("item_id", ""),
                        "source_path": row.get("source_path", ""),
                        "action": "undo",
                        "from_class": row.get("to_class", ""),
                        "to_class": row.get("from_class", ""),
                        "output_path": row.get("source_path", ""),
                        "note": "undo_mark_done",
                    },
                )
                return {"ok": True, "undone": True, "restored": row.get("source_path", ""), "type": "mark_done"}
            source = resolve_project_path(row.get("source_path", ""))
            output = resolve_project_path(row.get("output_path", ""))
            if output.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(source))
                invalidate_data_cache()
                append_action(
                    action_log(),
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "item_id": row.get("item_id", ""),
                        "source_path": row.get("output_path", ""),
                        "action": "undo",
                        "from_class": row.get("to_class", ""),
                        "to_class": row.get("from_class", ""),
                        "output_path": row.get("source_path", ""),
                        "note": "undo_last_move",
                    },
                )
                return {"ok": True, "undone": True, "restored": relative_or_absolute(source)}
        return {"ok": True, "undone": False}

    def model_predict(self, item_ids: list[str], threshold: float) -> dict[str, object]:
        predictions = []
        for item_id in item_ids:
            item = self.item_from_id(item_id)
            predictions.append({"id": item_id, **predict_item(item, threshold)})
        feedback_path = PROJECT_ROOT / "outputs" / "reports" / "model_review_feedback.csv"
        return {"ok": True, "predictions": predictions, "feedback_path": relative_or_absolute(feedback_path)}

    def save_feedback(self, payload: dict[str, object]) -> dict[str, object]:
        path = PROJECT_ROOT / "outputs" / "reports" / "model_review_feedback.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = ["timestamp", "item_id", "current_class", "model_top1", "is_correct", "correct_class", "note"]
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: str(payload.get(field, "")) for field in fields})
        return {"ok": True, "feedback_path": relative_or_absolute(path)}


STORE = DataIDE()


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canteen Data IDE</title>
  <style>
    :root{--bg:#f5f6f8;--panel:#fff;--line:#d9dee5;--ink:#1b222a;--muted:#687382;--accent:#126a5a;--danger:#b42318;--warn:#9a5b00}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;background:#fff;border-bottom:1px solid var(--line)}
    h1{font-size:18px;margin:0}.app{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 56px)}
    aside{background:#fff;border-right:1px solid var(--line);padding:14px;overflow:auto}.main{padding:14px;overflow:auto}
    .group{border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-bottom:12px}
    .group h2{margin:0 0 10px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
    label{display:block;color:var(--muted);font-size:12px;margin:8px 0 4px} select,input,button{width:100%;height:34px;border:1px solid var(--line);border-radius:6px;background:#fff;padding:0 8px;font:inherit}
    button{cursor:pointer;font-weight:650}.primary{background:var(--accent);border-color:var(--accent);color:#fff}.danger{color:var(--danger)}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.row.three{grid-template-columns:repeat(3,1fr)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat{background:#f3f5f7;border-radius:6px;padding:8px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}.card{position:relative;border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}
    .card.selected{outline:3px solid var(--accent)}.card.active-row{box-shadow:inset 0 0 0 3px var(--accent),0 0 12px rgba(18,106,90,.3);border-color:var(--accent);background:rgba(18,106,90,.08)}.card img{width:100%;aspect-ratio:1/1;object-fit:cover;background:#eef1f4}.card .body{padding:8px}.small{font-size:12px;color:var(--muted);word-break:break-word}
    .keycap{position:absolute;top:6px;left:6px;min-width:24px;height:24px;border-radius:999px;background:#1b222a;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;box-shadow:0 2px 8px #0003}.keycap.muted{background:#ffffffde;color:#687382;border:1px solid var(--line)}
    .queuebar{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center}.queuebar button{width:auto}.shortcut{margin-top:8px;padding:8px;border:1px dashed var(--line);border-radius:6px;background:#fafbfc;color:var(--muted)}.hidden{display:none!important}
    .pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eef1f4;font-size:12px;margin:2px}.bad{background:#fee4e2;color:#912018}.warn{background:#fff2cc;color:#7a4a00}.ok{background:#dcfae6;color:#05603a}
    table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left}
    .loading-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:999;color:#fff;font-size:16px;font-weight:700;gap:14px;backdrop-filter:blur(2px)}
    .spinner{width:40px;height:40px;border:4px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
<header><h1>Canteen Data IDE</h1><div id="status" class="small">Ready</div></header>
<div class="app">
  <aside>
    <div class="group"><h2>Nguồn dữ liệu</h2>
      <label>Root</label><select id="rootSelect"></select>
      <label id="splitLabel">Split</label><select id="splitSelect"><option value="">all/root</option><option>train</option><option>val</option><option>test</option></select>
      <label id="classFilterLabel">Class</label><select id="classFilter"></select>
      <label id="folderFilterLabel" class="hidden">Folder / pool</label><select id="folderFilter" class="hidden"></select>
      <div class="row"><button id="loadBtn" class="primary">Load</button><button id="undoBtn">Undo</button></div>
    </div>
    <div class="group"><h2>Action</h2>
      <label>Target root</label><select id="targetRoot"></select>
      <label>Target class / label</label><select id="targetLabel"></select>
      <div class="row"><button id="moveBtn">Move selected</button><button id="bulkMoveBtn">Move visible</button></div>
      <label>Quarantine label</label><input id="quarantineLabel" value="manual_rejected">
      <div class="row"><button id="quarantineBtn" class="danger">Quarantine</button><button id="futureUseBtn">Future use</button></div>
      <label>Keep as-is</label>
      <div class="row"><button id="doneBtn">Mark done</button><button id="doneVisibleBtn">Mark visible done</button></div>
    </div>
    <div class="group"><h2>Model assistant</h2>
      <label>Threshold</label><input id="threshold" type="number" step="0.01" min="0" max="1" value="0.70">
      <button id="predictBtn" class="primary">Predict visible</button>
      <div id="modelStats" class="small"></div>
    </div>
    <div class="group"><h2>Counts</h2><div id="counts" class="small"></div></div>
  </aside>
  <main class="main">
    <div class="group"><h2>Ảnh</h2>
      <div class="queuebar"><div id="queueStatus" class="small">Row 0/0 · selected 0</div><button id="prevRowBtn">Prev row</button><button id="nextRowBtn">Next row</button></div>
      <div class="row three"><button id="selectRowBtn">Select row</button><button id="selectAllBtn">Select visible</button><button id="clearBtn">Clear</button></div>
      <div class="shortcut small">1-0 chọn ảnh trong hàng hiện tại · A chọn cả hàng · D mark done · Space xuống hàng · Shift+Space lên hàng · Enter move · Q quarantine · F future use · P predict · Esc clear</div>
    </div>
    <div id="loadingOverlay" class="loading-overlay hidden"><div class="spinner"></div>Đang xử lý...</div>
    <div id="grid" class="grid"></div>
  </main>
</div>
<script>
const $=id=>document.getElementById(id);
const state={items:[],selected:new Set(),preds:{},classes:[],roots:[],targetRoots:[],cursorRow:0,lastTotal:0,hiddenDone:0,rootMode:'classification',loadSeq:0,activeRoot:'',filterKey:'',initialized:false,watchBusy:false};
function status(t){$('status').textContent=t}
function esc(v){return String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
async function api(url,body=null){const opt=body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{};const r=await fetch(url,opt);const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||r.statusText);return d}
function clsOptions(){return [''].concat(state.classes||[]).map(c=>`<option value="${esc(c)}">${esc(c||'all')}</option>`).join('')}
function isTypingTarget(target){return ['INPUT','SELECT','TEXTAREA'].includes(target?.tagName)||target?.isContentEditable}
function rootInfo(){return state.roots.find(r=>r.path===$('rootSelect').value)||state.roots[0]||{mode:'classification'}}
function targetRootByName(name){return (state.targetRoots.find(r=>r.name===name)||state.targetRoots[0]||{}).path||''}
function targetPayload(){return {target_root:$('targetRoot').value,target_label:$('targetLabel').value}}
function currentFilterKey(){return [$('rootSelect').value,state.rootMode,$('splitSelect').value,$('classFilter').value,$('folderFilter').value].join('|')}
function setHidden(id,hidden){$(id).classList.toggle('hidden',hidden)}
async function syncRootFilters(){const info=rootInfo();state.rootMode=info.mode||'folder';const isClass=state.rootMode==='classification';setHidden('splitLabel',!isClass);setHidden('splitSelect',!isClass);setHidden('classFilterLabel',!isClass);setHidden('classFilter',!isClass);setHidden('folderFilterLabel',isClass);setHidden('folderFilter',isClass);if(isClass){$('folderFilter').innerHTML='<option value="">all folders</option>';return}const d=await api('/api/folders?'+new URLSearchParams({root:$('rootSelect').value}));const opts=[''].concat(d.folders||[]);$('folderFilter').innerHTML=opts.map(f=>`<option value="${esc(f)}">${esc(f||'all folders')}</option>`).join('')}
function columnsPerRow(){const grid=$('grid');const width=grid?.clientWidth||window.innerWidth;return Math.max(1,Math.min(10,Math.floor((width+10)/170)))}
function totalRows(){return state.items.length?Math.ceil(state.items.length/columnsPerRow()):0}
function clampRow(){const rows=totalRows();state.cursorRow=rows?Math.max(0,Math.min(state.cursorRow,rows-1)):0}
function rowBounds(row=state.cursorRow){const cols=columnsPerRow();const start=row*cols;return [start,Math.min(start+cols,state.items.length)]}
function updateQueueStatus(){const rows=totalRows();const hidden=state.hiddenDone?` · done hidden ${state.hiddenDone}`:'';$('queueStatus').textContent=`Row ${rows?state.cursorRow+1:0}/${rows} · selected ${state.selected.size} · visible ${state.items.length}/${state.lastTotal}${hidden}`}
function scrollToRow(behavior='smooth'){const [start]=rowBounds();const card=document.querySelector(`.card[data-index="${start}"]`);if(card)card.scrollIntoView({block:'center',behavior})}
function updateHighlights(){const cols=columnsPerRow();document.querySelectorAll('.card').forEach(card=>{const idx=Number(card.dataset.index);const row=Math.floor(idx/cols);const id=card.dataset.id;card.classList.toggle('active-row',row===state.cursorRow);card.classList.toggle('selected',state.selected.has(id));const kc=card.querySelector('.keycap');if(kc)kc.className=row===state.cursorRow?'keycap':'keycap muted'});updateQueueStatus()}
function setLoading(busy){$('loadingOverlay').classList.toggle('hidden',!busy)}
function setRow(row,behavior='smooth'){state.cursorRow=row;clampRow();updateHighlights();requestAnimationFrame(()=>scrollToRow(behavior))}
function toggleRowKey(position){const [start,end]=rowBounds();const index=start+position;if(index>=end)return;const it=state.items[index];if(!it)return;state.selected.has(it.id)?state.selected.delete(it.id):state.selected.add(it.id);updateHighlights()}
function selectCurrentRow(){const [start,end]=rowBounds();for(let i=start;i<end;i++)state.selected.add(state.items[i].id);updateHighlights()}
async function init(){const s=await api('/api/state');state.classes=s.classes;state.roots=s.roots;state.targetRoots=s.target_roots||[];$('rootSelect').innerHTML=s.roots.map(r=>`<option value="${esc(r.path)}">${esc(r.name)} · ${esc(r.mode)}</option>`).join('');$('targetRoot').innerHTML=state.targetRoots.map(r=>`<option value="${esc(r.path)}">${esc(r.name)}</option>`).join('');$('classFilter').innerHTML=clsOptions();$('targetLabel').innerHTML=(state.classes||[]).concat(['canh_chua_hai_san','khay_background','mon_khac','mon_ngoai_de','future_use']).map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');$('targetLabel').value='com_trang';state.initialized=true;await rootChanged()}
async function load(opts={}){const seq=++state.loadSeq;const nextRow=opts.keepRow?state.cursorRow:0;state.selected.clear();state.preds={};const rootValue=$('rootSelect').value;status('Loading files...');const q=new URLSearchParams({root:rootValue,page_size:'120'});if(state.rootMode==='classification'){q.set('split',$('splitSelect').value);q.set('class_name',$('classFilter').value)}else{q.set('folder',$('folderFilter').value)}const d=await api('/api/browse?'+q);if(seq!==state.loadSeq)return;state.rootMode=d.mode||state.rootMode;state.activeRoot=rootValue;state.filterKey=currentFilterKey();state.items=d.items;state.lastTotal=d.total;state.hiddenDone=d.hidden_done||0;state.cursorRow=nextRow;clampRow();renderCounts(d.counts);renderGrid();status(`${d.total} files · queue refreshed`);scrollToRow('auto')}
function renderCounts(c){const detail=state.rootMode==='classification'?c.classes:c.folders;const detailTitle=state.rootMode==='classification'?'Classes':'Folders';$('counts').innerHTML=`<div class=stats><div class=stat>Total<br><b>${c.total}</b></div><div class=stat>${detailTitle}<br><b>${Object.keys(detail||{}).length}</b></div><div class=stat>Sources<br><b>${esc(Object.keys(c.sources).join(', ')||'-')}</b></div></div><pre>${esc(JSON.stringify(detail||{},null,2))}</pre>`}
function renderGrid(){
  const cols=columnsPerRow();
  $('grid').innerHTML=state.items.map((it,idx)=>{
    const p=state.preds[it.id];
    const dec=p?`<span class="pill ${p.decision==='ok'?'ok':p.decision.includes('disagreement')?'bad':'warn'}">${esc(p.decision)}</span>`:'';
    const row=Math.floor(idx/cols);
    const keyIndex=idx%cols;
    const key=keyIndex===9?'0':String(keyIndex+1);
    const active=row===state.cursorRow;
    return `<div class="card ${state.selected.has(it.id)?'selected':''} ${active?'active-row':''}" data-id="${esc(it.id)}" data-index="${idx}"><div class="${active?'keycap':'keycap muted'}">${key}</div><img src="${esc(it.image_url)}" loading="lazy"><div class=body><b>${esc(it.class_name)}</b> <span class=pill>${esc(it.split||'root')}</span> ${dec}<div class=small>${esc(it.filename)}</div><div class=small>${esc(it.source)}</div>${p?`<div class=small>top1 ${esc(p.top1)} ${esc(p.top1_confidence)}<br>top2 ${esc(p.top2)} ${esc(p.top2_confidence)} · margin ${esc(p.margin)}</div><div class=row><button data-fb="yes">Model đúng</button><button data-fb="no">Model sai</button></div>`:''}</div></div>`;
  }).join('');
  document.querySelectorAll('.card').forEach(card=>{
    card.onclick=e=>{
      if(e.target.dataset.fb){feedback(card.dataset.id,e.target.dataset.fb);return}
      state.cursorRow=Math.floor(Number(card.dataset.index)/columnsPerRow());
      state.selected.has(card.dataset.id)?state.selected.delete(card.dataset.id):state.selected.add(card.dataset.id);
      updateHighlights();
    }
  });
  updateQueueStatus();
}
async function act(action,extra={}){const ids=[...state.selected];if(!ids.length){alert('Chưa chọn ảnh');return}setLoading(true);status(`Đang xử lý ${ids.length} ảnh...`);try{const d=await api('/api/batch-action',{action,item_ids:ids,...extra});await load({keepRow:true});status(`Đã xử lý ${d.processed} ảnh`)}catch(e){status('Lỗi: '+e.message)}finally{setLoading(false)}}
async function actVisible(action,extra={}){const ids=state.items.map(x=>x.id);if(!ids.length){alert('Không có ảnh visible');return}setLoading(true);status(`Đang xử lý ${ids.length} ảnh visible...`);try{const d=await api('/api/batch-action',{action,item_ids:ids,...extra});await load({keepRow:true});status(`Đã xử lý ${d.processed} ảnh visible`)}catch(e){status('Lỗi: '+e.message)}finally{setLoading(false)}}
async function predict(){const ids=state.items.map(x=>x.id);if(!ids.length)return;setLoading(true);status(`Predict ${ids.length} ảnh visible...`);const d=await api('/api/predict',{item_ids:ids,threshold:Number($('threshold').value)});state.preds={};for(const p of d.predictions)state.preds[p.id]=p;const counts={};for(const p of d.predictions)counts[p.decision]=(counts[p.decision]||0)+1;$('modelStats').textContent=JSON.stringify(counts);renderGrid();status('Predict xong');setLoading(false)}
async function feedback(id,val){const it=state.items.find(x=>x.id===id),p=state.preds[id]||{};await api('/api/feedback',{timestamp:new Date().toISOString(),item_id:id,current_class:it.class_name,model_top1:p.top1||'',is_correct:val,correct_class:val==='yes'?p.top1:'',note:''});status('Saved feedback')}
const rootChanged=async()=>{await syncRootFilters();await load({keepRow:false})};
const filterChanged=()=>load({keepRow:false});
$('rootSelect').onchange=rootChanged;$('rootSelect').oninput=rootChanged;
$('folderFilter').onchange=filterChanged;$('folderFilter').oninput=filterChanged;
$('splitSelect').onchange=filterChanged;$('splitSelect').oninput=filterChanged;
$('classFilter').onchange=filterChanged;$('classFilter').oninput=filterChanged;
$('loadBtn').onclick=()=>load({keepRow:false});
$('undoBtn').onclick=async()=>{await api('/api/undo',{});await load({keepRow:true})};
$('moveBtn').onclick=()=>act('move_target',targetPayload());
$('bulkMoveBtn').onclick=()=>actVisible('move_target',targetPayload());
$('quarantineBtn').onclick=()=>act('move_target',{target_root:targetRootByName('quarantine'),target_label:$('quarantineLabel').value});
$('futureUseBtn').onclick=()=>act('move_target',{target_root:targetRootByName('extras'),target_label:'future_use'});
$('doneBtn').onclick=()=>act('mark_done',{note:'kept_as_is'});
$('doneVisibleBtn').onclick=()=>actVisible('mark_done',{note:'visible_kept_as_is'});
$('predictBtn').onclick=predict;
$('selectRowBtn').onclick=selectCurrentRow;
$('selectAllBtn').onclick=()=>{state.items.forEach(x=>state.selected.add(x.id));updateHighlights()};
$('clearBtn').onclick=()=>{state.selected.clear();updateHighlights()};
$('prevRowBtn').onclick=()=>setRow(state.cursorRow-1);
$('nextRowBtn').onclick=()=>setRow(state.cursorRow+1);
document.addEventListener('keydown',e=>{if(isTypingTarget(e.target)||e.ctrlKey||e.metaKey||e.altKey)return;const key=e.key.toLowerCase();if(/^[1-9]$/.test(e.key)){e.preventDefault();toggleRowKey(Number(e.key)-1);return}if(e.key==='0'){e.preventDefault();toggleRowKey(9);return}if(e.code==='Space'){e.preventDefault();setRow(state.cursorRow+(e.shiftKey?-1:1));return}if(e.key==='ArrowDown'){e.preventDefault();setRow(state.cursorRow+1);return}if(e.key==='ArrowUp'){e.preventDefault();setRow(state.cursorRow-1);return}if(e.key==='Enter'){e.preventDefault();$('moveBtn').click();return}if(key==='q'){e.preventDefault();$('quarantineBtn').click();return}if(key==='f'){e.preventDefault();$('futureUseBtn').click();return}if(key==='d'){e.preventDefault();$('doneBtn').click();return}if(key==='p'){e.preventDefault();predict();return}if(key==='a'){e.preventDefault();selectCurrentRow();return}if(e.key==='Escape'){state.selected.clear();updateHighlights();return}});
let resizeTimer=null;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{clampRow();renderGrid();scrollToRow('auto')},120)});
setInterval(async()=>{if(!state.initialized||state.watchBusy)return;const key=currentFilterKey();if(key===state.filterKey)return;state.watchBusy=true;try{if($('rootSelect').value!==state.activeRoot)await syncRootFilters();await load({keepRow:false})}finally{state.watchBusy=false}},500);
init().catch(e=>{status(e.message);alert(e.message)});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                self.send_json(STORE.state())
                return
            if parsed.path == "/api/folders":
                q = parse_qs(parsed.query)
                self.send_json(STORE.folders(q.get("root", [""])[0]))
                return
            if parsed.path == "/api/browse":
                q = parse_qs(parsed.query)
                self.send_json(
                    STORE.browse(
                        q.get("root", [""])[0],
                        q.get("split", [""])[0],
                        q.get("class_name", [""])[0],
                        q.get("folder", [""])[0],
                        int(q.get("page", ["0"])[0]),
                        int(q.get("page_size", ["80"])[0]),
                    )
                )
                return
            if parsed.path == "/file":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                if not path.exists() or not is_inside(path, PROJECT_ROOT):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/action":
                action = payload.get("action")
                if action == "move_class":
                    self.send_json(STORE.move_to_class(str(payload["item_id"]), str(payload["class_name"])))
                    return
                if action == "move_target":
                    self.send_json(
                        STORE.move_to_target(
                            str(payload["item_id"]),
                            str(payload["target_root"]),
                            str(payload["target_label"]),
                        )
                    )
                    return
                if action == "quarantine":
                    self.send_json(STORE.quarantine(str(payload["item_id"]), str(payload.get("label") or "manual_rejected")))
                    return
                if action == "mark_done":
                    self.send_json(STORE.mark_done(str(payload["item_id"]), str(payload.get("note") or "kept_as_is")))
                    return
                raise ValueError("Unsupported action")
            if parsed.path == "/api/batch-action":
                action = payload.get("action")
                item_ids = payload.get("item_ids", [])
                processed = 0
                errors: list[dict[str, str]] = []
                for item_id in item_ids:
                    try:
                        if action == "move_class":
                            STORE.move_to_class(str(item_id), str(payload["class_name"]))
                        elif action == "move_target":
                            STORE.move_to_target(
                                str(item_id),
                                str(payload["target_root"]),
                                str(payload["target_label"]),
                            )
                        elif action == "quarantine":
                            STORE.quarantine(str(item_id), str(payload.get("label") or "manual_rejected"))
                        elif action == "mark_done":
                            STORE.mark_done(str(item_id), str(payload.get("note") or "kept_as_is"))
                        else:
                            raise ValueError("Unsupported action")
                        processed += 1
                    except Exception as exc:
                        errors.append({"item_id": str(item_id), "error": str(exc)})
                self.send_json({"ok": True, "processed": processed, "errors": errors})
                return
            if parsed.path == "/api/undo":
                self.send_json(STORE.undo())
                return
            if parsed.path == "/api/predict":
                self.send_json(STORE.model_predict([str(x) for x in payload["item_ids"]], float(payload.get("threshold", 0.7))))
                return
            if parsed.path == "/api/feedback":
                self.send_json(STORE.save_feedback(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Data IDE for classification/review/quarantine folders.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Data IDE: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
