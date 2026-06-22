from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageStat


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    blur_score: float
    phash: str
    sha256: str


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def blur_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(image: Image.Image) -> float:
    stat = ImageStat.Stat(image.convert("L"))
    return float(stat.mean[0])


def assess_image(path: Path) -> tuple[Image.Image | None, ImageMetrics | None, list[str]]:
    try:
        image = open_rgb_image(path)
        width, height = image.size
        aspect = max(width / max(height, 1), height / max(width, 1))
        metrics = ImageMetrics(
            width=width,
            height=height,
            aspect_ratio=aspect,
            brightness=brightness_mean(image),
            blur_score=blur_score(image),
            phash=perceptual_hash(image),
            sha256=file_sha256(path),
        )
        return image, metrics, []
    except Exception as exc:
        return None, None, [f"invalid_image:{type(exc).__name__}"]


def quality_reasons(
    metrics: ImageMetrics,
    *,
    min_size: int = 180,
    max_aspect_ratio: float = 3.0,
    min_blur_score: float = 20.0,
    min_brightness: float = 20.0,
    max_brightness: float = 238.0,
) -> list[str]:
    reasons: list[str] = []
    if min(metrics.width, metrics.height) < min_size:
        reasons.append("too_small")
    if metrics.aspect_ratio > max_aspect_ratio:
        reasons.append("odd_aspect_ratio")
    if metrics.blur_score < min_blur_score:
        reasons.append("too_blurry")
    if metrics.brightness < min_brightness:
        reasons.append("too_dark")
    if metrics.brightness > max_brightness:
        reasons.append("too_bright")
    return reasons


def normalize_image(
    image: Image.Image,
    *,
    image_size: int = 512,
    mode: str = "pad",
    background: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    if mode == "crop":
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if mode != "pad":
        raise ValueError(f"Unknown normalize mode: {mode}")
    resized = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), background)
    x = (image_size - resized.width) // 2
    y = (image_size - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas
