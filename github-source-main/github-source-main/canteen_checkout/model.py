from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

SUPPORTED_ARCHES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "resnet18",
    "resnet50",
    "convnext_tiny",
)

SUPPORTED_AUGMENTATIONS = ("none", "light", "medium", "strong")


def _replace_classifier_tail(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _replace_resnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_classifier(num_classes: int, pretrained: bool = True, arch: str = "mobilenet_v3_small") -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        return _replace_resnet_head(model, num_classes)
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        return _replace_resnet_head(model, num_classes)
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        return _replace_classifier_tail(model, num_classes)
    raise ValueError(f"Unsupported architecture: {arch}. Supported: {', '.join(SUPPORTED_ARCHES)}")


def train_transforms(image_size: int = 224, augmentation: str = "medium") -> transforms.Compose:
    if augmentation not in SUPPORTED_AUGMENTATIONS:
        raise ValueError(f"Unsupported augmentation: {augmentation}. Supported: {', '.join(SUPPORTED_AUGMENTATIONS)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augmentation == "none":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if augmentation == "light":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if augmentation == "medium":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.78, 1.0),
                    ratio=(0.85, 1.18),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.06),
                    shear=3,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomPerspective(distortion_scale=0.08, p=0.2),
                transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.15, hue=0.02),
                transforms.RandomAutocontrast(p=0.15),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.5, 2.0), value="random"),
            ]
        )

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.68, 1.0),
                ratio=(0.75, 1.33),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=18,
                translate=(0.06, 0.06),
                scale=(0.9, 1.12),
                shear=5,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.22, hue=0.03),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.12),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.18, scale=(0.02, 0.12), ratio=(0.4, 2.5), value="random"),
        ]
    )


def eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model, class_names, image_size: int, metadata: dict | None = None, arch: str = "mobilenet_v3_small") -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": image_size,
        "arch": arch,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    arch = checkpoint.get("arch", "mobilenet_v3_small")
    model = build_classifier(num_classes=len(class_names), pretrained=False, arch=arch)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    return model, class_names, image_size, checkpoint
