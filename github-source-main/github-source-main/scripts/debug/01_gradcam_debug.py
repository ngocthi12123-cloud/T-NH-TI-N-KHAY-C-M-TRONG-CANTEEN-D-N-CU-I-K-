from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from canteen_checkout.config import CLASSIFICATION_DIR, DEFAULT_MODEL_PATH, DISH_CLASSES, OUTPUTS_DIR, PROJECT_ROOT
from canteen_checkout.io_utils import IMAGE_EXTENSIONS
from canteen_checkout.model import eval_transforms, load_checkpoint, resolve_device


@dataclass
class PredictionRow:
    path: Path
    true_class: str
    pred_class: str
    confidence: float
    true_confidence: float
    margin: float
    correct: bool
    cam_path: Path | None = None


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, target_index].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_tensor = F.relu(cam_tensor)
        cam_tensor = F.interpolate(cam_tensor, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam_tensor[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def list_samples(root: Path, class_names: list[str]) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name))
    return samples


def target_layer_for(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise ValueError("Cannot infer target layer for this architecture.")


def open_resized_rgb(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def overlay_cam(image: Image.Image, cam_map: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = np.asarray(image).astype(np.float32) / 255.0
    heatmap = colormaps.get_cmap("jet")(cam_map)[..., :3].astype(np.float32)
    mixed = np.clip((1 - alpha) * base + alpha * heatmap, 0, 1)
    return Image.fromarray((mixed * 255).astype(np.uint8))


def safe_stem(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def write_csv(path: Path, rows: list[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "true_class",
        "pred_class",
        "confidence",
        "true_confidence",
        "margin",
        "correct",
        "cam_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": relative_or_absolute(row.path),
                    "true_class": row.true_class,
                    "pred_class": row.pred_class,
                    "confidence": f"{row.confidence:.6f}",
                    "true_confidence": f"{row.true_confidence:.6f}",
                    "margin": f"{row.margin:.6f}",
                    "correct": str(row.correct),
                    "cam_path": relative_or_absolute(row.cam_path) if row.cam_path else "",
                }
            )


def annotate_pair(original: Image.Image, cam_image: Image.Image, title: str, width: int = 224) -> Image.Image:
    text_height = 54
    panel = Image.new("RGB", (width * 2, width + text_height), (245, 245, 245))
    panel.paste(original, (0, text_height))
    panel.paste(cam_image, (width, text_height))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(20, 20, 20), font=font)
    draw.text((6, 30), "left: original   right: Grad-CAM", fill=(80, 80, 80), font=font)
    return panel


def make_contact_sheet(rows: list[PredictionRow], output: Path, image_size: int, max_items: int) -> None:
    selected = rows[:max_items]
    if not selected:
        return
    panels: list[Image.Image] = []
    for row in selected:
        if row.cam_path is None or not row.cam_path.exists():
            continue
        original = open_resized_rgb(row.path, image_size)
        cam_image = Image.open(row.cam_path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        title = f"T:{row.true_class}  P:{row.pred_class}  conf={row.confidence:.2f}  true={row.true_confidence:.2f}"
        panels.append(annotate_pair(original, cam_image, title, image_size))
    if not panels:
        return
    columns = 2
    rows_count = int(np.ceil(len(panels) / columns))
    panel_w, panel_h = panels[0].size
    sheet = Image.new("RGB", (columns * panel_w, rows_count * panel_h), (255, 255, 255))
    for idx, panel in enumerate(panels):
        x = (idx % columns) * panel_w
        y = (idx // columns) * panel_h
        sheet.paste(panel, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM debug images for weak food classes.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=CLASSIFICATION_DIR / "test")
    parser.add_argument("--classes", default="dau_hu_sot_ca,thit_kho,canh_chua_co_ca,canh_chua_khong_ca")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "gradcam_debug")
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--include-correct-low-confidence", type=int, default=4)
    args = parser.parse_args()

    device = resolve_device()
    model, class_names, image_size, checkpoint = load_checkpoint(args.model, device)
    transform = eval_transforms(image_size)
    weak_classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    unknown = [name for name in weak_classes if name not in class_names]
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}")

    run_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    target_layer = target_layer_for(model)
    gradcam = GradCAM(model, target_layer)
    rows: list[PredictionRow] = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    try:
        for path, true_class in list_samples(args.data, weak_classes):
            image = open_resized_rgb(path, image_size)
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            pred_idx = int(probs.argmax().item())
            true_idx = class_to_idx[true_class]
            top2 = torch.topk(probs, k=2)
            confidence = float(probs[pred_idx].item())
            true_confidence = float(probs[true_idx].item())
            margin = float(top2.values[0].item() - top2.values[1].item())
            cam_map = gradcam(input_tensor, pred_idx)
            cam_image = overlay_cam(image, cam_map)
            pred_class = class_names[pred_idx]
            correct = pred_class == true_class
            cam_name = (
                f"{safe_stem(true_class)}__pred_{safe_stem(pred_class)}__"
                f"conf_{confidence:.2f}__{safe_stem(path.stem)}.jpg"
            )
            cam_path = run_dir / "overlays" / true_class / cam_name
            cam_path.parent.mkdir(parents=True, exist_ok=True)
            cam_image.save(cam_path, format="JPEG", quality=92)
            rows.append(
                PredictionRow(
                    path=path,
                    true_class=true_class,
                    pred_class=pred_class,
                    confidence=confidence,
                    true_confidence=true_confidence,
                    margin=margin,
                    correct=correct,
                    cam_path=cam_path,
                )
            )
    finally:
        gradcam.close()

    write_csv(run_dir / "gradcam_predictions.csv", rows)
    summary: dict[str, object] = {
        "model": relative_or_absolute(args.model),
        "data": relative_or_absolute(args.data),
        "checkpoint": checkpoint.get("metadata", {}),
        "classes": {},
    }
    for class_name in weak_classes:
        class_rows = [row for row in rows if row.true_class == class_name]
        mistakes = [row for row in class_rows if not row.correct]
        correct_low_conf = sorted(
            [row for row in class_rows if row.correct],
            key=lambda row: (row.confidence, row.margin),
        )[: args.include_correct_low_confidence]
        selected = sorted(mistakes, key=lambda row: (-row.confidence, row.margin)) + correct_low_conf
        make_contact_sheet(selected, run_dir / f"contact_{class_name}.jpg", image_size, args.max_per_class)
        pred_counts: dict[str, int] = {}
        for row in class_rows:
            pred_counts[row.pred_class] = pred_counts.get(row.pred_class, 0) + 1
        summary["classes"][class_name] = {
            "total": len(class_rows),
            "correct": sum(row.correct for row in class_rows),
            "wrong": len(mistakes),
            "accuracy": round(sum(row.correct for row in class_rows) / max(len(class_rows), 1), 4),
            "pred_counts": pred_counts,
            "contact_sheet": relative_or_absolute(run_dir / f"contact_{class_name}.jpg"),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": relative_or_absolute(run_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
