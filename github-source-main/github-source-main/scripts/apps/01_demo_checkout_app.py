from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shutil
import sys
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch
from PIL import Image

from canteen_checkout.config import (
    BILLS_DIR,
    CROPPED_DISHES_DIR,
    DEFAULT_MODEL_PATH,
    DEMO_TRAYS_DIR,
    DISH_CLASSES,
    ENGAGEMENT_DB_PATH,
    IMAGE_EXTENSIONS,
    PHONE_HMAC_KEY_PATH,
    PROJECT_ROOT,
)
from canteen_checkout.cropping import CropRegion, crop_regions, five_compartment_template, load_regions
from canteen_checkout.engagement import EngagementError, EngagementStore
from canteen_checkout.io_utils import load_prices
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device
from canteen_checkout.pricing import THIT_KHO_TRUNG_CLASS, dish_price


UPLOAD_DIR = DEMO_TRAYS_DIR / "uploads"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
DEMO_TEMPLATE_PATH = TEMPLATE_DIR / "demo_checkout.html"
IGNORE_LABELS = {"ignore", "ignored", "unknown", "other", "extra"}

MODEL_CACHE: dict[str, object] = {}
ENGAGEMENT_STORE: EngagementStore | None = None


def get_engagement_store() -> EngagementStore:
    global ENGAGEMENT_STORE
    if ENGAGEMENT_STORE is None:
        ENGAGEMENT_STORE = EngagementStore(ENGAGEMENT_DB_PATH, phone_key_path=PHONE_HMAC_KEY_PATH)
    return ENGAGEMENT_STORE


def reward_catalog() -> list[dict[str, object]]:
    return [
        {
            "class_name": row.class_name,
            "display_name": row.display_name,
            "points_cost": row.reward_points,
            "discount_vnd": row.price_vnd,
        }
        for row in load_prices().values()
    ]


def write_bill_snapshot(bill: dict) -> None:
    path = resolve_project_path(str(bill["bill_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bill, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def predict_crop(model, class_names: list[str], image_size: int, crop_path: Path, device: torch.device) -> tuple[str, float]:
    transform = eval_transforms(image_size)
    image = Image.open(crop_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, idx = torch.max(probs, dim=0)
    return class_names[int(idx)], float(confidence.cpu().item())


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_safe_project_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def list_demo_images() -> list[dict[str, str]]:
    roots = [DEMO_TRAYS_DIR, PROJECT_ROOT / "Khay_com", PROJECT_ROOT / "data" / "raw_teacher_trays"]
    seen: set[Path] = set()
    images: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append({"path": relative_or_absolute(path), "name": path.name})
    return images


def list_region_templates() -> list[dict[str, str]]:
    templates = [{"path": "", "name": "5-compartment grid"}]
    config_dir = PROJECT_ROOT / "configs"
    if config_dir.exists():
        for path in sorted(config_dir.glob("*regions*.json")):
            templates.append({"path": relative_or_absolute(path), "name": path.name})
    return templates


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def serialize_region(region: CropRegion) -> dict[str, object]:
    return {
        "name": region.name,
        "x": region.x,
        "y": region.y,
        "w": region.w,
        "h": region.h,
        "label": region.label or "",
        "source": region.source,
        "confidence": round(region.confidence, 4) if region.confidence is not None else None,
    }


def region_source(regions: list[CropRegion]) -> str:
    sources = {region.source for region in regions if region.source}
    if not sources:
        return "manual"
    return next(iter(sources)) if len(sources) == 1 else "mixed"


def template_regions(payload: dict, image_path: Path) -> list[CropRegion]:
    template = str(payload.get("template") or "")
    if template:
        return load_regions(resolve_project_path(template))
    width, height = image_size(image_path)
    return five_compartment_template(width, height)


def regions_from_payload(payload: dict, image_path: Path) -> tuple[list[CropRegion], dict[str, object]]:
    raw_regions = payload.get("regions")
    if raw_regions:
        regions = [
            CropRegion(
                name=str(item.get("name") or f"crop_{idx:02d}"),
                x=int(item["x"]),
                y=int(item["y"]),
                w=int(item["w"]),
                h=int(item["h"]),
                label=str(item.get("label") or "").strip() or None,
                source=str(item.get("source") or "manual"),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
            for idx, item in enumerate(raw_regions)
        ]
        metadata = dict(payload.get("region_metadata") or {})
        metadata.update({"requested_mode": str(payload.get("region_mode") or "manual"), "region_source": region_source(regions)})
        return regions, metadata

    mode = str(payload.get("mode") or payload.get("region_mode") or "template")
    metadata: dict[str, object] = {
        "requested_mode": mode,
        "region_source": "template",
    }
    regions = template_regions(payload, image_path)
    return regions, metadata


def load_model_once(model_path: Path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    device = resolve_device()
    model, class_names, model_image_size, checkpoint = load_checkpoint(model_path, device)
    cached = (model, class_names, model_image_size, checkpoint, device)
    MODEL_CACHE[key] = cached
    return cached


def run_checkout(payload: dict) -> dict:
    image_path = resolve_project_path(str(payload["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_safe_project_path(image_path):
        raise ValueError("Only project files can be used in the demo app")

    model_path = resolve_project_path(str(payload.get("model_path") or DEFAULT_MODEL_PATH))
    threshold = float(payload.get("threshold", 0.55))
    regions, region_metadata = regions_from_payload(payload, image_path)

    bill_id = uuid.uuid4().hex
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + bill_id[:6]
    out_dir = CROPPED_DISHES_DIR / f"{image_path.stem}_{run_id}"
    crop_paths = crop_regions(image_path, regions, out_dir)
    prices = load_prices()

    model = None
    class_names: list[str] = []
    model_image_size = 224
    device = torch.device("cpu")
    model_loaded = False
    if model_path.exists():
        model, class_names, model_image_size, _, device = load_model_once(model_path)
        model_loaded = True
    items = []
    total = 0
    for crop_path, region in zip(crop_paths, regions):
        forced_label = region.label or ""
        ignored = forced_label in IGNORE_LABELS
        if ignored:
            class_name = forced_label
            confidence = 1.0
            uncertain = True
        elif forced_label:
            class_name = forced_label
            confidence = 1.0
            uncertain = False
        elif model is not None:
            class_name, confidence = predict_crop(model, class_names, model_image_size, crop_path, device)
            uncertain = confidence < threshold
        else:
            class_name = "unknown"
            confidence = 0.0
            uncertain = True

        final_egg_count = 1 if class_name == THIT_KHO_TRUNG_CLASS else None

        price_row = prices.get(class_name)
        price_info = dish_price(
            class_name,
            prices,
            uncertain=uncertain,
            egg_count=final_egg_count if class_name == THIT_KHO_TRUNG_CLASS else None,
        )
        total += price_info.total_price_vnd
        display_name = class_name if price_row is None else price_row.display_name
        items.append(
            {
                "crop_path": relative_or_absolute(crop_path),
                "crop_url": f"/file?path={relative_or_absolute(crop_path)}",
                "region_name": region.name,
                "region_source": region.source,
                "region_confidence": round(region.confidence, 4) if region.confidence is not None else None,
                "class_name": class_name,
                "display_name": display_name,
                "confidence": round(confidence, 4),
                "uncertain": uncertain,
                "ignored": ignored,
                "egg_count": price_info.egg_count,
                "price_vnd": price_info.total_price_vnd,
                "base_price_vnd": price_info.base_price_vnd,
                "extra_price_vnd": price_info.extra_price_vnd,
            }
        )

    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    bill_path = BILLS_DIR / f"{image_path.stem}_{run_id}_bill.json"
    bill = {
        "image_path": relative_or_absolute(image_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": relative_or_absolute(model_path) if model_path.exists() else None,
        "threshold": threshold,
        "region_source": region_metadata.get("region_source", region_source(regions)),
        "items": items,
        "total_vnd": total,
        "bill_path": relative_or_absolute(bill_path),
        "model_loaded": model_loaded,
    }
    bill = get_engagement_store().create_draft(
        bill_id=bill_id,
        bill_path=relative_or_absolute(bill_path),
        payload=bill,
        customer_id=str(payload.get("customer_id") or "") or None,
        voucher_id=str(payload.get("voucher_id") or "") or None,
    )
    write_bill_snapshot(bill)
    return bill


def confirm_checkout(payload: dict) -> dict:
    bill = get_engagement_store().confirm_bill(str(payload.get("bill_id") or ""))
    write_bill_snapshot(bill)
    return bill


def issue_voucher(payload: dict) -> dict:
    prices = load_prices()
    class_name = str(payload.get("class_name") or "")
    reward = prices.get(class_name)
    if reward is None:
        raise EngagementError("Món đổi thưởng không hợp lệ.", code="reward_not_found", status=404)
    return get_engagement_store().issue_voucher(
        customer_id=str(payload.get("customer_id") or ""),
        source_bill_id=str(payload.get("source_bill_id") or ""),
        class_name=reward.class_name,
        display_name=reward.display_name,
        points_cost=reward.reward_points,
        discount_vnd=reward.price_vnd,
    )


def save_upload(payload: dict) -> dict:
    name = Path(str(payload.get("name") or "upload.jpg")).name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    data_url = str(payload["data_url"])
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(base64.b64decode(encoded))
    return {"path": relative_or_absolute(path), "name": path.name}


def app_state() -> dict:
    prices = load_prices()
    store = get_engagement_store()
    return {
        "project_root": str(PROJECT_ROOT),
        "images": list_demo_images(),
        "templates": list_region_templates(),
        "classes": DISH_CLASSES,
        "labels": ["", "ignore", *DISH_CLASSES],
        "prices": {
            key: {
                "display_name": value.display_name,
                "price_vnd": value.price_vnd,
                "reward_points": value.reward_points,
            }
            for key, value in prices.items()
        },
        "reward_catalog": reward_catalog(),
        "rating_summaries": store.rating_summaries(),
        "default_model_path": relative_or_absolute(DEFAULT_MODEL_PATH),
    }



def load_demo_template() -> str:
    if not DEMO_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Demo template not found: {DEMO_TEMPLATE_PATH}")
    return DEMO_TEMPLATE_PATH.read_text(encoding="utf-8")


def is_safe_static_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(STATIC_DIR.resolve())
        return True
    except ValueError:
        return False


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CanteenDemo/1.0"

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
                body = load_demo_template().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/static/"):
                static_name = parsed.path.removeprefix("/static/")
                path = (STATIC_DIR / static_name).resolve()
                if not path.exists() or not path.is_file() or not is_safe_static_path(path):
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
            if parsed.path == "/api/state":
                self.send_json(app_state())
                return
            if parsed.path == "/api/ratings/summary":
                self.send_json({"ok": True, "summaries": get_engagement_store().rating_summaries()})
                return
            if parsed.path == "/api/image-info":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                width, height = image_size(path)
                self.send_json({"width": width, "height": height})
                return
            if parsed.path == "/file":
                path = resolve_project_path(parse_qs(parsed.query).get("path", [""])[0])
                if not path.exists() or not is_safe_project_path(path):
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
            if parsed.path == "/api/upload":
                self.send_json({"ok": True, **save_upload(payload)})
                return
            if parsed.path == "/api/customers/lookup":
                customer = get_engagement_store().lookup_customer(str(payload.get("phone") or ""))
                self.send_json({"ok": True, "found": customer is not None, "customer": customer})
                return
            if parsed.path == "/api/customers":
                customer, created = get_engagement_store().create_customer(str(payload.get("phone") or ""))
                self.send_json({"ok": True, "created": created, "customer": customer})
                return
            if parsed.path == "/api/regions":
                image_path = resolve_project_path(str(payload["image_path"]))
                regions, metadata = regions_from_payload(payload, image_path)
                self.send_json(
                    {
                        "ok": True,
                        "regions": [serialize_region(region) for region in regions],
                        **metadata,
                    }
                )
                return
            if parsed.path == "/api/run":
                self.send_json({"ok": True, **run_checkout(payload)})
                return
            if parsed.path == "/api/checkout/confirm":
                self.send_json({"ok": True, **confirm_checkout(payload)})
                return
            if parsed.path == "/api/vouchers":
                self.send_json({"ok": True, **issue_voucher(payload)})
                return
            if parsed.path == "/api/ratings":
                result = get_engagement_store().save_rating(
                    bill_item_id=str(payload.get("bill_item_id") or ""),
                    stars=payload.get("stars"),
                    comment=str(payload.get("comment") or ""),
                    customer_id=str(payload.get("customer_id") or "") or None,
                )
                self.send_json({"ok": True, **result})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except EngagementError as exc:
            self.send_json(
                {"ok": False, "error": str(exc), "code": exc.code},
                HTTPStatus(exc.status),
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local web UI for checkout demos.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo checkout app: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
