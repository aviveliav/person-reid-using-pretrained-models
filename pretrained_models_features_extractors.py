# === imports ===
from __future__ import annotations
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNeXt50_32X4D_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
    resnext50_32x4d,
)

# Utilities
def _apply_batch_transform(img, transform):
    """
    Apply a transform to a batch [B,C,H,W]
    """
    if transform is None:
        return img
    return transform(img)


def _ensure_float_unit(img: torch.Tensor) -> torch.Tensor:
    # Make sure inputs are float in [0,1] before Normalize/Resize (torchvision expects float)
    if img.dtype != torch.float32 and img.dtype != torch.float16 and img.dtype != torch.bfloat16:
        img = img.float()
    # If likely in 0..255, scale down
    if img.max() > 1.5:
        img = img / 255.0
    return img


# Common, batch-friendly transform builders (tensor-only; no PIL)
def make_imagenet_224() -> Callable:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def make_clip_vitb32_224() -> Callable:
    # OpenAI CLIP normalization (tensor pipeline, no PIL, batch-safe)
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

def make_resize_to_640() -> Callable:
    return T.Compose([T.Resize((640, 640), antialias=True)])


# Base class
class BaseExtractor(nn.Module):
    def __init__(self, device: torch.device, preprocess: Optional[Callable] = None):
        super().__init__()
        self.device = device
        self.preprocess = preprocess

    def _pre(self, img: torch.Tensor) -> torch.Tensor:
        x = _ensure_float_unit(img).to(self.device, non_blocking=True)
        if self.preprocess is None:
            return x
        # GPU-capable transforms (Resize/CenterCrop/Normalize) will run on device
        return _apply_batch_transform(x, self.preprocess)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ------------------------------ Concrete extractors ------------------------------ #
class ResNet50Extractor(BaseExtractor):
    def __init__(self, device: torch.device):
        super().__init__(device, preprocess=make_imagenet_224())
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        # trunk to global pool
        self.trunk = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            x = self.trunk(x)            # [B, 2048, 1, 1]
            x = torch.flatten(x, 1)      # [B, 2048]
        return x


class ResNeXt50_32x4DExtractor(BaseExtractor):
    def __init__(self, device: torch.device):
        super().__init__(device, preprocess=make_imagenet_224())
        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2).to(device).eval()
        self.trunk = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            x = self.trunk(x)
            x = torch.flatten(x, 1)      # [B, 2048]
        return x


class DenseNet121Extractor(BaseExtractor):
    def __init__(self, device: torch.device):
        super().__init__(device, preprocess=make_imagenet_224())
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(device).eval()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            x = self.backbone.features(x)     # [B, 1024, H, W]
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)           # [B, 1024]
        return x


class EfficientNetB0Extractor(BaseExtractor):
    def __init__(self, device: torch.device):
        super().__init__(device, preprocess=make_imagenet_224())
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device).eval()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            x = self.backbone.features(x)     # [B, 1280, H, W]
            x = self.backbone.avgpool(x)      # [B, 1280, 1, 1]
            x = torch.flatten(x, 1)           # [B, 1280]
        return x


class DINOv2ViTSExtractor(BaseExtractor):
    """
    timm 'vit_small_patch14_dinov2' → [B, 384] from CLS
    """
    def __init__(self, device: torch.device):
        import timm
        model = timm.create_model("vit_small_patch14_dinov2", pretrained=True).to(device).eval()
        cfg = model.pretrained_cfg
        size = int(cfg["input_size"][-1])
        mean = tuple(cfg["mean"])
        std = tuple(cfg["std"])
        preprocess = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Normalize(mean=mean, std=std),
        ])
        super().__init__(device, preprocess=preprocess)
        self.backbone = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats: x = feats["x_norm_clstoken"]
                elif "cls_token" in feats:     x = feats["cls_token"]
                elif "x" in feats and feats["x"].dim() == 3: x = feats["x"][:, 0]
                else:
                    anyv = next(iter(feats.values()))
                    x = anyv[:, 0] if anyv.dim() == 3 else anyv
            else:
                x = feats[:, 0] if feats.dim() == 3 else feats
        return x


class MaskRCNNExtractor(BaseExtractor):
    """
    Pools FPN pyramid features to a single vector.
    """
    def __init__(self, device: torch.device):
        # Mask R-CNN expects 0..1 tensors; keep aspect with min side ~800 like torchvision default
        def resize_longest(img: torch.Tensor) -> torch.Tensor:
            if img.dim() == 4:   # batch
                return torch.stack([resize_longest(x) for x in img], dim=0)
            elif img.dim() == 3: # single image
                _, h, w = img.shape
                scale = 800.0 / min(h, w)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                return torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
                ).squeeze(0)
            else:
                raise ValueError(f"Unexpected shape {img.shape}")
        
        preprocess = T.Compose([resize_longest])
        super().__init__(device, preprocess=preprocess)
        self.backbone = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True).to(device).eval()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)
        with torch.no_grad():
            feats = self.backbone.backbone(x)        # dict of pyramid maps
            pooled = [F.adaptive_avg_pool2d(t, 1).flatten(1) for t in feats.values()]
            x = torch.cat(pooled, dim=1)
        return x


class YOLOExtractor(BaseExtractor):
    def __init__(self, device, weights="yolov8n.pt", layer_indices=(16,19,22)):
        from ultralytics import YOLO
        super().__init__(device, preprocess=make_resize_to_640())

        # load YOLO wrapper
        self.model = YOLO(weights).model.to(device).eval()
        self.layers = self.model.model
        
        self.layer_indices = layer_indices


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img).to(self.device)

        local, handles = [], []
        for idx in self.layer_indices:
            def _hook():
                return lambda m, i, o: local.append(o)
            handles.append(self.layers[idx].register_forward_hook(_hook()))

        with torch.no_grad():
            _ = self.model(x)

        for h in handles:
            h.remove()

        pooled = []
        for f in local:
            if isinstance(f, torch.Tensor):
                pooled.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
            elif isinstance(f, (list, tuple)):
                for sub in f:
                    if isinstance(sub, torch.Tensor):
                        pooled.append(F.adaptive_avg_pool2d(sub, 1).flatten(1))

        if not pooled:
            raise RuntimeError("No valid features captured from YOLO")

        return torch.cat(pooled, dim=1)



class CLIPViTB32Extractor(BaseExtractor):
    """
    Batch-friendly CLIP extractor (no PIL, no loop).
    Uses tensor-native transforms equivalent to OpenAI CLIP.
    """
    def __init__(self, device: torch.device, model_name: str = "ViT-B/32"):
        import clip as openai_clip
        super().__init__(device, preprocess=make_clip_vitb32_224())
        model, _ = openai_clip.load(model_name, device=device, jit=False)
        self.clip = model.eval()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._pre(img)  # already normalized/resized for CLIP
        with torch.no_grad():
            feats = self.clip.encode_image(x)
        return feats


_EXTRACTOR_REGISTRY: Dict[str, Callable[..., BaseExtractor]] = {
    "resnet50":        ResNet50Extractor,
    "resnext50_32x4d": ResNeXt50_32x4DExtractor,
    "densenet121":     DenseNet121Extractor,
    "efficientnet_b0": EfficientNetB0Extractor,
    "dinov2_vits":     DINOv2ViTSExtractor,
    "maskrcnn":        MaskRCNNExtractor,
    "yolo":            YOLOExtractor,
    "clip":            CLIPViTB32Extractor,
}

def make_extractor(name: str, device: torch.device, **kwargs) -> BaseExtractor:
    key = name.lower()
    if key not in _EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor '{name}'. Available: {list(_EXTRACTOR_REGISTRY.keys())}")
    return _EXTRACTOR_REGISTRY[key](device=device, **kwargs)

