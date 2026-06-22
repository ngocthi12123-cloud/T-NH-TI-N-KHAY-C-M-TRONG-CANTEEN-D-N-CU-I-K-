from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from canteen_checkout.config import CLASSIFICATION_DIR, DEFAULT_MODEL_PATH, DISH_CLASSES, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.io_utils import IMAGE_EXTENSIONS, save_class_names
from canteen_checkout.model import (
    SUPPORTED_AUGMENTATIONS,
    SUPPORTED_ARCHES,
    build_classifier,
    eval_transforms,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_transforms,
)


class FixedClassImageDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], transform=None, class_transforms: dict[str, object] | None = None):
        self.root = root
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.class_transforms = class_transforms or {}
        self.samples: list[tuple[Path, int]] = []
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = default_loader(path)
        class_name = self.class_names[label]
        transform = self.class_transforms.get(class_name, self.transform)
        if transform is not None:
            image = transform(image)
        return image, label

    def counts_by_class(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def class_weight_tensor(counts: dict[str, int], class_names: list[str], device: torch.device) -> torch.Tensor:
    total = sum(counts.values())
    weights = []
    for class_name in class_names:
        count = max(counts.get(class_name, 0), 1)
        weights.append(total / (len(class_names) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        weighted_ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * weighted_ce).mean()


def parse_class_augmentations(items: list[str], class_names: list[str], image_size: int) -> dict[str, object]:
    overrides = {}
    valid_classes = set(class_names)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --class-augmentation '{item}'. Use class_name=none|light|medium|strong.")
        class_name, augmentation = [part.strip() for part in item.split("=", 1)]
        if class_name not in valid_classes:
            raise SystemExit(f"Unknown class in --class-augmentation: {class_name}")
        if augmentation not in SUPPORTED_AUGMENTATIONS:
            raise SystemExit(f"Unknown augmentation '{augmentation}'. Supported: {', '.join(SUPPORTED_AUGMENTATIONS)}")
        overrides[class_name] = train_transforms(image_size, augmentation)
    return overrides


def parse_oversample_items(items: list[str], class_names: list[str]) -> dict[str, float]:
    boosts = {}
    valid_classes = set(class_names)
    for item in items:
        if ":" in item:
            class_name, value = [part.strip() for part in item.split(":", 1)]
        elif "=" in item:
            class_name, value = [part.strip() for part in item.split("=", 1)]
        else:
            class_name, value = item.strip(), "2.0"
        if class_name not in valid_classes:
            raise SystemExit(f"Unknown class in --oversample-class: {class_name}")
        factor = float(value)
        if factor <= 0:
            raise SystemExit(f"Oversample factor must be positive for {class_name}: {factor}")
        boosts[class_name] = factor
    return boosts


def build_train_sampler(dataset: FixedClassImageDataset, sampler_mode: str, class_boosts: dict[str, float]):
    if sampler_mode == "shuffle" and not class_boosts:
        return None

    counts = dataset.counts_by_class()
    weights = []
    for _, label in dataset.samples:
        class_name = dataset.class_names[label]
        if sampler_mode == "balanced":
            base = 1.0 / max(counts.get(class_name, 0), 1)
        else:
            base = 1.0
        weights.append(base * class_boosts.get(class_name, 1.0))

    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def plot_history(history: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dish classifier.")
    parser.add_argument("--data", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize from a compatible classifier checkpoint instead of ImageNet weights.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without better validation accuracy; use 0 to disable.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--arch", choices=SUPPORTED_ARCHES, default="mobilenet_v3_small")
    parser.add_argument("--augmentation", choices=SUPPORTED_AUGMENTATIONS, default="medium")
    parser.add_argument(
        "--class-augmentation",
        action="append",
        default=[],
        help="Per-class augmentation override, e.g. canh_chua_khong_ca=strong. Can be repeated.",
    )
    parser.add_argument("--sampler", choices=("shuffle", "balanced"), default="shuffle")
    parser.add_argument(
        "--oversample-class",
        action="append",
        default=[],
        help="Boost one class in the train sampler, e.g. canh_chua_khong_ca:2.5. Can be repeated.",
    )
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-weighted-loss", action="store_true", help="Disable class-balanced loss weights.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-empty-val", action="store_true")
    args = parser.parse_args()

    class_transform_overrides = parse_class_augmentations(args.class_augmentation, DISH_CLASSES, args.image_size)
    oversample_boosts = parse_oversample_items(args.oversample_class, DISH_CLASSES)

    train_ds = FixedClassImageDataset(
        args.data / "train",
        DISH_CLASSES,
        train_transforms(args.image_size, args.augmentation),
        class_transforms=class_transform_overrides,
    )
    val_ds = FixedClassImageDataset(args.data / "val", DISH_CLASSES, eval_transforms(args.image_size))
    test_ds = FixedClassImageDataset(args.data / "test", DISH_CLASSES, eval_transforms(args.image_size))

    print("train counts:", train_ds.counts_by_class())
    print("val counts:", val_ds.counts_by_class())
    print("test counts:", test_ds.counts_by_class())
    print(f"arch: {args.arch}")
    print(f"augmentation: {args.augmentation}")
    print(f"class_augmentation: {args.class_augmentation}")
    print(f"sampler: {args.sampler}")
    print(f"oversample_class: {args.oversample_class}")
    print(f"loss: {args.loss}")

    if len(train_ds) == 0:
        raise SystemExit("No training images found. Put labeled crops under data/classification/train/<class_name>/")
    if len(val_ds) == 0 and not args.allow_empty_val:
        raise SystemExit("No validation images found. Add val images or pass --allow-empty-val for a smoke run.")

    device = resolve_device()
    print(f"device: {device}")

    train_sampler = build_train_sampler(train_ds, args.sampler, oversample_boosts)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds if len(val_ds) else train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds if len(test_ds) else val_ds if len(val_ds) else train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint if args.init_checkpoint.is_absolute() else PROJECT_ROOT / args.init_checkpoint
        model, init_classes, init_image_size, init_payload = load_checkpoint(init_path, device)
        init_arch = str(init_payload.get("arch") or "")
        if init_classes != DISH_CLASSES:
            raise SystemExit(f"Init checkpoint class order does not match DISH_CLASSES: {init_path}")
        if init_arch != args.arch:
            raise SystemExit(f"Init checkpoint architecture is {init_arch}, requested {args.arch}")
        if init_image_size != args.image_size:
            raise SystemExit(f"Init checkpoint image size is {init_image_size}, requested {args.image_size}")
        print(f"initialized_from: {init_path}")
    else:
        model = build_classifier(len(DISH_CLASSES), pretrained=not args.no_pretrained, arch=args.arch).to(device)
    loss_weights = None if args.no_weighted_loss else class_weight_tensor(train_ds.counts_by_class(), DISH_CLASSES, device)
    if args.loss == "focal":
        criterion = FocalLoss(weight=loss_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.model_out,
                model,
                DISH_CLASSES,
                args.image_size,
                metadata={
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "arch": args.arch,
                    "augmentation": args.augmentation,
                    "class_augmentation": args.class_augmentation,
                    "sampler": args.sampler,
                    "oversample_class": args.oversample_class,
                    "loss": args.loss,
                    "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
                    "weighted_loss": not args.no_weighted_loss,
                    "label_smoothing": args.label_smoothing,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                },
                arch=args.arch,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    save_class_names()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, REPORTS_DIR / "training_history.png")

    if args.model_out.exists():
        model, _, _, checkpoint = load_checkpoint(args.model_out, device)
        print(f"Loaded best checkpoint for test: epoch={checkpoint.get('metadata', {}).get('epoch')}, val_acc={checkpoint.get('metadata', {}).get('best_val_acc')}")
    y_true, y_pred = collect_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, labels=list(range(len(DISH_CLASSES))), target_names=DISH_CLASSES, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DISH_CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISH_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(report)
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
