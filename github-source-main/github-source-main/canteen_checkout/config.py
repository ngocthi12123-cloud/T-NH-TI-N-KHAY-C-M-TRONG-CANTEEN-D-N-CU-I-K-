from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = _env_path("CANTEEN_DATA_DIR", PROJECT_ROOT / "data")
INBOX_DIR = DATA_DIR / "inbox"
REVIEW_INBOX_DIR = INBOX_DIR / "review"
REVIEWED_DIR = DATA_DIR / "reviewed"
EXTRAS_DIR = DATA_DIR / "extras"
RAW_TEACHER_TRAYS_DIR = DATA_DIR / "raw_teacher_trays"
DEMO_TRAYS_DIR = DATA_DIR / "demo"
CLASSIFICATION_DIR = DATA_DIR / "classification"
DOWNLOADS_DIR = DATA_DIR / "downloads"
SCRAPED_CANDIDATES_DIR = DATA_DIR / "scraped_candidates"
PROCESSED_CANDIDATES_DIR = DATA_DIR / "processed_candidates"
REJECTED_CANDIDATES_DIR = DATA_DIR / "rejected_candidates"
TEMP_TEACHER_CROPS_DIR = DATA_DIR / "temp_teacher_crops"
ARCHIVE_DIR = DATA_DIR / "archive"
SCRAPED_MANIFEST_CSV = DATA_DIR / "scraped_manifest.csv"

MODELS_DIR = _env_path("CANTEEN_MODEL_DIR", PROJECT_ROOT / "models")
OUTPUTS_DIR = _env_path("CANTEEN_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
CROPPED_DISHES_DIR = OUTPUTS_DIR / "cropped_dishes"
BILLS_DIR = OUTPUTS_DIR / "bills"
REPORTS_DIR = OUTPUTS_DIR / "reports"
ENGAGEMENT_DB_PATH = _env_path("CANTEEN_ENGAGEMENT_DB_PATH", OUTPUTS_DIR / "canteen_engagement.sqlite3")
PHONE_HMAC_KEY_PATH = _env_path("CANTEEN_PHONE_HMAC_KEY_PATH", OUTPUTS_DIR / "private" / "phone_hmac.key")

PRICES_CSV = PROJECT_ROOT / "prices.csv"
CLASS_NAMES_JSON = MODELS_DIR / "class_names.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "dish_classifier.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DISH_CLASSES = [
    "com_trang",
    "dau_hu_sot_ca",
    "ca_hu_kho",
    "thit_kho_trung",
    "thit_kho",
    "canh_chua_co_ca",
    "canh_chua_khong_ca",
    "suon_nuong",
    "canh_rau",
    "rau_xao",
    "trung_chien",
]

DISPLAY_NAMES = {
    "com_trang": "Com trang",
    "dau_hu_sot_ca": "Dau hu sot ca",
    "ca_hu_kho": "Ca hu kho",
    "thit_kho_trung": "Thit kho trung",
    "thit_kho": "Thit kho",
    "canh_chua_co_ca": "Canh chua co ca",
    "canh_chua_khong_ca": "Canh chua khong ca",
    "suon_nuong": "Suon nuong",
    "canh_rau": "Canh rau",
    "rau_xao": "Rau xao",
    "trung_chien": "Trung chien",
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
