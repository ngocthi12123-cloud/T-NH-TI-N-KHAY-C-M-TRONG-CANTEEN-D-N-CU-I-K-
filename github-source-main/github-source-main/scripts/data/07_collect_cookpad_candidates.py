from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image

from canteen_checkout.config import (
    DATA_DIR,
    DEFAULT_MODEL_PATH,
    DISH_CLASSES,
    IMAGE_EXTENSIONS,
    REPORTS_DIR,
    REVIEW_INBOX_DIR,
)
from canteen_checkout.data_quality import (
    assess_image,
    hamming_distance_hex,
    normalize_image,
    quality_reasons,
)
from canteen_checkout.io_utils import list_images
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device

HASH_CACHE_PATH = REPORTS_DIR / "image_hash_cache.csv"


DEFAULT_QUERIES = [
    "canh chua chay",
    "canh chua chay đơn giản",
    "canh chua rau muống",
    "canh chua rau muống chay",
    "canh chua đậu hũ",
    "canh chua đậu hũ chay",
    "canh chua nấm",
    "canh chua nấm chay",
    "canh chua nấm rơm",
    "canh chua bạc hà",
    "canh chua bạc hà cà chua",
    "canh chua bông súng",
    "canh chua bông súng chay",
    "canh chua giá đậu",
    "canh chua đậu bắp",
    "canh dứa cà chua nấm chay",
    "canh thơm cà chua chay",
    "canh chua kim chi chay",
    "canh chua măng chua chay",
    "canh chua đậu rồng nấm chay",
    "canh chua rau nhút chay",
]

POSITIVE_TERMS = [
    "canh chua",
    "chay",
    "rau muong",
    "rau muống",
    "dau hu",
    "đậu hũ",
    "đậu phụ",
    "nam",
    "nấm",
    "bac ha",
    "bạc hà",
    "ca chua",
    "cà chua",
    "thom",
    "thơm",
    "dua",
    "dứa",
    "bong sung",
    "bông súng",
    "rau nhut",
    "rau nhút",
    "gia dau",
    "giá đậu",
    "dau bap",
    "đậu bắp",
]

NEGATIVE_TERMS = [
    "cá",
    "fish",
    "catfish",
    "tom",
    "tôm",
    "tep",
    "tép",
    "shrimp",
    "muc",
    "mực",
    "squid",
    "hai san",
    "hải sản",
    "seafood",
    "dau ca",
    "đầu cá",
    "ca hu",
    "cá hú",
    "ca loc",
    "cá lóc",
]

ACCENT_SENSITIVE_TERMS = {"cá", "tôm", "tép", "mực", "hải sản", "đầu cá", "cá hú", "cá lóc"}

ALLOWED_MODEL_CLASSES = {"canh_chua_khong_ca", "canh_chua_co_ca", "canh_rau", "dau_hu_sot_ca"}

# Class-specific Cookpad rules. These override the older canh-chua defaults above
# so the crawler can collect other dishes without reusing soup-only filters.
DEFAULT_QUERIES_BY_CLASS = {
    "canh_chua_khong_ca": [
        "canh chua chay",
        "canh chua rau muống",
        "canh chua rau muống chay",
        "canh chua đậu hũ",
        "canh chua đậu hũ chay",
        "canh chua nấm",
        "canh chua nấm chay",
        "canh chua bạc hà cà chua",
        "canh chua bông súng chay",
        "canh chua giá đậu",
        "canh chua đậu bắp",
        "canh thơm cà chua chay",
    ],
    "trung_chien": [
        "trứng chiên",
        "trứng rán",
        "trứng chiên hành",
        "trứng chiên hành lá",
        "trứng chiên cà chua",
        "trứng chiên nấm",
        "trứng chiên rau củ",
        "trứng cuộn",
        "trứng cuộn chiên",
        "omelette Việt Nam",
        "fried omelette",
        "Vietnamese omelette",
    ],
    "dau_hu_sot_ca": [
        "đậu hũ sốt cà",
        "đậu hũ sốt cà chua",
        "đậu phụ sốt cà",
        "đậu phụ sốt cà chua",
        "tàu hủ sốt cà",
        "tàu hủ sốt cà chua",
        "đậu hũ non sốt cà chua",
        "đậu hũ chiên sốt cà",
        "đậu phụ chiên sốt cà chua",
        "tofu tomato sauce",
        "Vietnamese tofu tomato sauce",
    ],
    "suon_nuong": [
        "cơm tấm sườn nướng",
        "cơm sườn nướng",
        "sườn cốt lết nướng",
        "cốt lết nướng",
        "cơm phần sườn nướng",
        "khay cơm sườn nướng",
        "Vietnamese grilled pork chop rice",
    ],
}

TEXT_FILTERS = {
    "canh_chua_khong_ca": {
        "positive": [
            "canh chua",
            "chay",
            "rau muống",
            "đậu hũ",
            "đậu phụ",
            "nấm",
            "bạc hà",
            "cà chua",
            "thơm",
            "dứa",
            "bông súng",
            "rau nhút",
            "giá đậu",
            "đậu bắp",
        ],
        "negative": [
            "cá",
            "fish",
            "catfish",
            "tôm",
            "tép",
            "shrimp",
            "mực",
            "squid",
            "hải sản",
            "seafood",
            "đầu cá",
            "cá hú",
            "cá lóc",
        ],
        "model_allow": {"canh_chua_khong_ca", "canh_chua_co_ca", "canh_rau", "dau_hu_sot_ca"},
    },
    "trung_chien": {
        "positive": [
            "trứng chiên",
            "trứng rán",
            "trứng cuộn",
            "omelette",
            "fried egg",
            "fried omelette",
            "egg omelette",
        ],
        "negative": [
            "thịt kho trứng",
            "trứng kho",
            "canh",
            "súp",
            "cháo",
            "bánh",
            "cơm chiên",
            "mì",
            "bún",
            "luộc",
            "ốp la",
            "salad",
            "tôm",
            "tép",
            "hải sản",
            "mắm tôm",
            "cá cơm",
        ],
        "model_allow": {"trung_chien"},
    },
    "dau_hu_sot_ca": {
        "positive": [
            "đậu hũ sốt cà",
            "đậu hũ sốt cà chua",
            "đậu phụ sốt cà",
            "đậu phụ sốt cà chua",
            "tàu hủ sốt cà",
            "tàu hủ sốt cà chua",
            "tofu tomato",
            "tofu tomato sauce",
            "sốt cà chua",
        ],
        "negative": [
            "nhồi thịt",
            "nhồi",
            "dồn thịt",
            "thịt băm",
            "thịt heo",
            "thịt bò",
            "gà",
            "cá",
            "fish",
            "tôm",
            "shrimp",
            "mực",
            "squid",
            "hải sản",
            "canh",
            "lẩu",
            "phở",
            "miến",
            "bún",
            "mì",
            "bánh",
        ],
        "model_allow": {"dau_hu_sot_ca"},
    },
    "suon_nuong": {
        "positive": [
            "cơm tấm",
            "cơm sườn",
            "sườn nướng",
            "sườn cốt lết",
            "cốt lết nướng",
            "grilled pork chop",
            "pork chop rice",
        ],
        "negative": [
            "sườn non",
            "sườn que",
            "sườn cây",
            "bbq ribs",
            "pork ribs",
            "sườn xào",
            "sườn rim",
            "sườn chua ngọt",
            "canh sườn",
            "cháo sườn",
            "bún",
            "mì",
            "lẩu",
        ],
        "model_allow": {"suon_nuong"},
    },
}

ACCENT_SENSITIVE_TERMS = {
    "cá",
    "tôm",
    "tép",
    "mực",
    "hải sản",
    "đầu cá",
    "cá hú",
    "cá lóc",
}


@dataclass
class RecipeCandidate:
    query: str
    recipe_url: str
    title: str
    text: str
    image_url: str


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def text_key(text: str) -> str:
    return strip_accents(text).lower()


def safe_slug(text: str, max_len: int = 80) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text_key(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text or "item")[:max_len].strip("_")


def term_present(text: str, term: str) -> bool:
    if term in ACCENT_SENSITIVE_TERMS:
        haystack = text.lower()
        needle = term.lower()
        if " " in needle:
            return needle in haystack
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None
    haystack = text_key(text)
    needle = text_key(term)
    if " " in needle:
        return needle in haystack
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def text_filter(text: str, target_class: str) -> tuple[bool, str]:
    config = TEXT_FILTERS.get(
        target_class,
        {
            "positive": [target_class.replace("_", " ")],
            "negative": [],
            "model_allow": {target_class},
        },
    )
    negatives = [term for term in config["negative"] if term_present(text, term)]
    if negatives:
        return False, "negative_text:" + "|".join(sorted(set(negatives)))
    positives = [term for term in config["positive"] if term_present(text, term)]
    if not positives:
        return False, "missing_positive_text"
    return True, "text_ok:" + "|".join(sorted(set(positives))[:8])


def fetch_text(session: requests.Session, url: str, timeout: int = 25) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def search_recipe_links(session: requests.Session, query: str, page: int, delay: float) -> list[str]:
    url = f"https://cookpad.com/vn/tim-kiem/{quote(query)}"
    if page > 1:
        url += f"?page={page}"
    html = fetch_text(session, url)
    time.sleep(delay)
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if re.search(r"/vn/cong-thuc/\d+", href):
            links.append(urljoin(url, href.split("?")[0]))
    return sorted(set(links))


def parse_recipe(session: requests.Session, url: str, query: str, delay: float) -> RecipeCandidate | None:
    html = fetch_text(session, url)
    time.sleep(delay)
    soup = BeautifulSoup(html, "html.parser")
    recipe_data = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text(strip=True))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "Recipe":
            recipe_data = data
            break
    if not recipe_data:
        return None

    title = str(recipe_data.get("name") or "").strip()
    image_url = recipe_data.get("image")
    if isinstance(image_url, list):
        image_url = image_url[0] if image_url else ""
    image_url = str(image_url or "").strip()
    if not title or not image_url:
        return None

    parts = [title, str(recipe_data.get("description") or ""), str(recipe_data.get("keywords") or "")]
    ingredients = recipe_data.get("recipeIngredient") or []
    if isinstance(ingredients, list):
        parts.extend(str(item) for item in ingredients)
    instructions = recipe_data.get("recipeInstructions") or []
    if isinstance(instructions, list):
        for step in instructions:
            if isinstance(step, dict):
                parts.append(str(step.get("text") or ""))
    return RecipeCandidate(query=query, recipe_url=url, title=title, text=" ".join(parts), image_url=image_url)


def download_image(session: requests.Session, url: str, delay: float) -> Image.Image:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(delay)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


@torch.no_grad()
def predict_image(model, class_names: list[str], image_size: int, path: Path, device: torch.device, topk: int = 5) -> list[tuple[str, float]]:
    image = Image.open(path).convert("RGB")
    tensor = eval_transforms(image_size)(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    values, indices = torch.topk(probs, min(topk, len(class_names)))
    return [(class_names[int(idx)], float(value.cpu().item())) for value, idx in zip(values, indices)]


def model_filter(top_predictions: list[tuple[str, float]], target_class: str, min_target_confidence: float) -> tuple[bool, str]:
    top_classes = {name for name, _ in top_predictions[:3]}
    target_confidence = next((conf for name, conf in top_predictions if name == target_class), 0.0)
    allowed_classes = set(TEXT_FILTERS.get(target_class, {}).get("model_allow", {target_class}))
    if top_classes & allowed_classes:
        top_text = "|".join(f"{name}:{conf:.3f}" for name, conf in top_predictions[:3])
        return True, f"allowed_top3:{top_text}"
    if target_confidence >= min_target_confidence:
        top_text = "|".join(f"{name}:{conf:.3f}" for name, conf in top_predictions[:3])
        return True, f"target_confidence:{target_confidence:.4f}|{top_text}"
    top_text = "|".join(f"{name}:{conf:.3f}" for name, conf in top_predictions[:3])
    return False, f"model_reject:target_conf={target_confidence:.4f}|{top_text}"


def collect_existing_hashes(roots: list[Path]) -> tuple[set[str], list[tuple[str, str]]]:
    sha_values: set[str] = set()
    phashes: list[tuple[str, str]] = []
    cache = load_hash_cache()
    changed = False
    scanned = 0
    for root in roots:
        for path in list_images(root):
            scanned += 1
            if scanned % 1000 == 0:
                print(f"dedupe_scanned={scanned}", flush=True)
            resolved = str(path.resolve())
            stat = path.stat()
            cached = cache.get(resolved)
            if (
                cached
                and cached.get("size") == str(stat.st_size)
                and cached.get("mtime_ns") == str(stat.st_mtime_ns)
                and cached.get("sha256")
                and cached.get("phash")
            ):
                sha_values.add(cached["sha256"])
                phashes.append((cached["phash"], resolved))
                continue
            _, metrics, reasons = assess_image(path)
            if metrics is None or reasons:
                continue
            cache[resolved] = {
                "path": resolved,
                "size": str(stat.st_size),
                "mtime_ns": str(stat.st_mtime_ns),
                "sha256": metrics.sha256,
                "phash": metrics.phash,
            }
            changed = True
            sha_values.add(metrics.sha256)
            phashes.append((metrics.phash, resolved))
    if changed:
        save_hash_cache(cache)
    return sha_values, phashes


def duplicate_reason(sha: str, phash: str, sha_values: set[str], phashes: list[tuple[str, str]], threshold: int) -> str | None:
    if sha in sha_values:
        return "duplicate_sha"
    for existing_phash, existing_path in phashes:
        if hamming_distance_hex(phash, existing_phash) <= threshold:
            return f"duplicate_phash:{existing_path}"
    return None


def load_hash_cache(path: Path = HASH_CACHE_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["path"]: row for row in csv.DictReader(f) if row.get("path")}


def save_hash_cache(cache: dict[str, dict[str, str]], path: Path = HASH_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "size", "mtime_ns", "sha256", "phash"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(cache):
            writer.writerow({field: cache[key].get(field, "") for field in fields})


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "reason",
        "query",
        "title",
        "recipe_url",
        "image_url",
        "file_path",
        "sha256",
        "phash",
        "top_predictions",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Cookpad dish candidates into Data IDE review inbox.")
    parser.add_argument("--target-class", default="canh_chua_khong_ca", choices=DISH_CLASSES)
    parser.add_argument("--goal", type=int, default=200)
    parser.add_argument("--max-considered", type=int, default=0, help="Stop after this many recipe pages; 0 means no explicit cap.")
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--min-target-confidence", type=float, default=0.12)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--review-dir", type=Path, default=None, help="Append accepted images to an existing review folder.")
    parser.add_argument("--queries-file", type=Path, default=None, help="UTF-8 text file with one query per line.")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    file_queries: list[str] = []
    if args.queries_file:
        file_queries = [
            line.strip()
            for line in args.queries_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    queries = args.query or file_queries or DEFAULT_QUERIES_BY_CLASS.get(args.target_class, [args.target_class.replace("_", " ")])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_dir = args.review_dir or (REVIEW_INBOX_DIR / f"{args.target_class}_cookpad_{stamp}")
    temp_dir = DATA_DIR / "inbox" / "raw_batches" / f"cookpad_{args.target_class}_{stamp}"
    manifest_path = REPORTS_DIR / f"cookpad_{args.target_class}_{stamp}_manifest.csv"
    summary_path = REPORTS_DIR / f"cookpad_{args.target_class}_{stamp}_summary.json"

    print(f"target_class={args.target_class}")
    print(f"goal={args.goal}")
    print(f"review_dir={review_dir}")
    print(f"manifest={manifest_path}")
    print(f"apply={args.apply}")
    if not args.apply:
        print("Dry run only. Add --apply to download and write images.")
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 CanteenCheckoutStudentProject/1.0",
            "Accept-Language": "vi,en;q=0.8",
        }
    )

    existing_roots = [
        DATA_DIR / "reviewed",
        DATA_DIR / "classification",
        DATA_DIR / "inbox" / "review",
        DATA_DIR / "extras",
    ]
    print("Building dedupe index...")
    existing_sha, existing_phashes = collect_existing_hashes(existing_roots)
    print(f"dedupe_index: sha={len(existing_sha)}, phash={len(existing_phashes)}")

    device = resolve_device()
    model = None
    class_names: list[str] = []
    image_size = 224
    if args.model.exists():
        model, class_names, image_size, _ = load_checkpoint(args.model, device)
        print(f"Loaded classifier: {args.model} on {device}")
    else:
        print(f"Classifier not found: {args.model}. Model gate disabled.")

    review_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    existing_output_count = sum(1 for path in review_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict[str, object]] = []
    seen_recipe_urls: set[str] = set()
    seen_image_urls: set[str] = set()
    accepted = 0
    considered = 0
    downloaded = 0

    for query in queries:
        if accepted >= args.goal:
            break
        if args.max_considered and considered >= args.max_considered:
            break
        print(f"\nQUERY: {query}")
        for page in range(1, args.max_pages + 1):
            if accepted >= args.goal:
                break
            if args.max_considered and considered >= args.max_considered:
                break
            try:
                links = search_recipe_links(session, query, page, args.delay)
            except requests.HTTPError as exc:
                print(f"  page={page} search_error={type(exc).__name__}: {exc}")
                if exc.response is not None and exc.response.status_code == 404:
                    break
                continue
            except Exception as exc:
                print(f"  page={page} search_error={type(exc).__name__}: {exc}")
                continue
            if not links:
                print(f"  page={page} no links")
                break
            print(f"  page={page} links={len(links)} accepted={accepted}")
            for recipe_url in links:
                if accepted >= args.goal:
                    break
                if args.max_considered and considered >= args.max_considered:
                    break
                if recipe_url in seen_recipe_urls:
                    continue
                seen_recipe_urls.add(recipe_url)
                considered += 1
                row: dict[str, object] = {
                    "status": "rejected",
                    "reason": "",
                    "query": query,
                    "recipe_url": recipe_url,
                    "title": "",
                    "image_url": "",
                    "file_path": "",
                    "sha256": "",
                    "phash": "",
                    "top_predictions": "",
                }
                try:
                    candidate = parse_recipe(session, recipe_url, query, args.delay)
                    if candidate is None:
                        row["reason"] = "parse_failed"
                        rows.append(row)
                        continue
                    row["title"] = candidate.title
                    row["image_url"] = candidate.image_url
                    ok, reason = text_filter(candidate.text, args.target_class)
                    if not ok:
                        row["reason"] = reason
                        rows.append(row)
                        continue
                    if candidate.image_url in seen_image_urls:
                        row["reason"] = "duplicate_image_url"
                        rows.append(row)
                        continue
                    seen_image_urls.add(candidate.image_url)

                    image = download_image(session, candidate.image_url, args.delay)
                    downloaded += 1
                    normalized = normalize_image(image, image_size=512, mode="pad")
                    file_name = f"{existing_output_count + accepted + 1:04d}_{safe_slug(candidate.title)}.jpg"
                    temp_path = temp_dir / file_name
                    normalized.save(temp_path, "JPEG", quality=92, optimize=True)

                    _, metrics, assess_reasons = assess_image(temp_path)
                    if metrics is None:
                        row["reason"] = "invalid_after_save:" + "|".join(assess_reasons)
                        temp_path.unlink(missing_ok=True)
                        rows.append(row)
                        continue
                    reasons = quality_reasons(metrics, min_size=220, max_aspect_ratio=2.8, min_blur_score=12.0)
                    if reasons:
                        row["reason"] = "quality:" + "|".join(reasons)
                        row["sha256"] = metrics.sha256
                        row["phash"] = metrics.phash
                        temp_path.unlink(missing_ok=True)
                        rows.append(row)
                        continue

                    dup_reason = duplicate_reason(metrics.sha256, metrics.phash, existing_sha, existing_phashes, args.phash_threshold)
                    if dup_reason:
                        row["reason"] = dup_reason
                        row["sha256"] = metrics.sha256
                        row["phash"] = metrics.phash
                        temp_path.unlink(missing_ok=True)
                        rows.append(row)
                        continue

                    top_predictions: list[tuple[str, float]] = []
                    if model is not None:
                        top_predictions = predict_image(model, class_names, image_size, temp_path, device)
                        row["top_predictions"] = json.dumps(top_predictions, ensure_ascii=False)
                        ok, reason = model_filter(top_predictions, args.target_class, args.min_target_confidence)
                        if not ok:
                            row["reason"] = reason
                            row["sha256"] = metrics.sha256
                            row["phash"] = metrics.phash
                            temp_path.unlink(missing_ok=True)
                            rows.append(row)
                            continue

                    final_path = review_dir / file_name
                    temp_path.replace(final_path)
                    existing_sha.add(metrics.sha256)
                    existing_phashes.append((metrics.phash, str(final_path)))
                    accepted += 1
                    row.update(
                        {
                            "status": "accepted",
                            "reason": "accepted",
                            "file_path": str(final_path),
                            "sha256": metrics.sha256,
                            "phash": metrics.phash,
                        }
                    )
                    rows.append(row)
                    if accepted % 25 == 0 or accepted == args.goal:
                        print(f"    accepted={accepted}/{args.goal} considered={considered} downloaded={downloaded}")
                except Exception as exc:
                    row["reason"] = f"exception:{type(exc).__name__}:{exc}"
                    rows.append(row)

    write_manifest(manifest_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_class": args.target_class,
        "goal": args.goal,
        "max_considered": args.max_considered,
        "accepted": accepted,
        "existing_output_count": existing_output_count,
        "total_output_count": existing_output_count + accepted,
        "considered": considered,
        "downloaded": downloaded,
        "review_dir": str(review_dir),
        "raw_batch_dir": str(temp_dir),
        "manifest": str(manifest_path),
        "queries": queries,
        "status_counts": {},
        "reason_counts": {},
    }
    for row in rows:
        summary["status_counts"][row["status"]] = summary["status_counts"].get(row["status"], 0) + 1
        reason = str(row.get("reason", ""))
        reason_key = reason.split(":", 1)[0]
        summary["reason_counts"][reason_key] = summary["reason_counts"].get(reason_key, 0) + 1
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
