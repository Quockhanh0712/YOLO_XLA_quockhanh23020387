from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F_nn

CLASSES = ["person", "car", "dog", "cat", "chair"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)

def move_targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    moved = []
    for target in targets:
        moved.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()})
    return moved

def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def extract_map50(score_path: str | Path) -> float | None:
    try:
        data = load_json(score_path)
    except Exception:
        return None
    keys = ["mAP@0.5", "map@0.5", "mAP50", "map50", "AP50", "ap50", "score"]
    for key in keys:
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                for key in keys:
                    inner = value.get(key)
                    if isinstance(inner, (int, float)):
                        return float(inner)
    return None

def run_external_eval(gt_path: str, pred_path: str, out_path: str, tool_path: str = "public/tools/evaluate_predictions.py") -> float | None:
    if not Path(tool_path).exists():
        return None
    try:
        subprocess.run(
            ["python", tool_path, "--ground_truth", gt_path, "--predictions", pred_path, "--output", out_path],
            check=True,
        )
    except Exception:
        return None
    return extract_map50(out_path)

def list_images(image_dir: str | Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])

class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        import copy

        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])
