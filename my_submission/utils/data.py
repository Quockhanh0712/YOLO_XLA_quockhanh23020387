from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from .loss import clip_boxes_to_image, remove_small_boxes
from .utils import CLASSES, load_json, IMAGENET_MEAN, IMAGENET_STD

def _filter_boxes(boxes: torch.Tensor, labels: torch.Tensor, height: int, width: int, min_size: float = 2.0):
    boxes = clip_boxes_to_image(boxes, height, width)
    keep = remove_small_boxes(boxes, min_size)
    return boxes[keep], labels[keep]

class DetectionTransform:
    def __init__(self, train: bool, img_size: int = 512, multiscale: Sequence[int] | None = None):
        self.train = train
        self.img_size = img_size
        self.multiscale = list(multiscale or [img_size])

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        boxes = target["boxes"].clone().float()
        labels = target["labels"].clone().long()
        if self.train:
            image, boxes, labels = self._augment(image, boxes, labels)
        size = self.img_size
        image, boxes = self._resize_square(image, boxes, size)
        boxes, labels = _filter_boxes(boxes, labels, size, size)
        tensor = F.to_tensor(image)
        tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = dict(target)
        target["boxes"] = boxes
        target["labels"] = labels
        target["size"] = torch.tensor([size, size], dtype=torch.long)
        return tensor, target

    def _augment(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        if random.random() < 0.5:
            w, _ = image.size
            image = ImageOps.mirror(image)
            if boxes.numel() > 0:
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2

        image = self._color_jitter(image)

        scale = random.uniform(0.7, 1.3)
        if abs(scale - 1.0) > 1e-3:
            w, h = image.size
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((nw, nh), Image.BILINEAR)
            boxes = boxes * torch.tensor([nw / w, nh / h, nw / w, nh / h], dtype=torch.float32)

        image, boxes, labels = self._safe_random_crop(image, boxes, labels)
        image, boxes, labels = self._random_translate(image, boxes, labels)

        if random.random() < 0.05:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < 0.05:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return image, boxes, labels

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        for enhancer, amount in [
            (ImageEnhance.Brightness, 0.2),
            (ImageEnhance.Contrast, 0.2),
            (ImageEnhance.Color, 0.2),
        ]:
            image = enhancer(image).enhance(random.uniform(1 - amount, 1 + amount))
        return image

    def _safe_random_crop(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        w, h = image.size
        if w < 32 or h < 32 or random.random() > 0.7:
            return image, boxes, labels
        for _ in range(10):
            crop_w = random.randint(max(16, int(0.6 * w)), w)
            crop_h = random.randint(max(16, int(0.6 * h)), h)
            if boxes.numel() > 0:
                idx = random.randrange(boxes.shape[0])
                cx = float((boxes[idx, 0] + boxes[idx, 2]) * 0.5)
                cy = float((boxes[idx, 1] + boxes[idx, 3]) * 0.5)
                left = int(min(max(0, cx - random.uniform(0.2, 0.8) * crop_w), w - crop_w))
                top = int(min(max(0, cy - random.uniform(0.2, 0.8) * crop_h), h - crop_h))
            else:
                left = random.randint(0, w - crop_w)
                top = random.randint(0, h - crop_h)
            new_boxes = boxes.clone()
            if new_boxes.numel() > 0:
                new_boxes[:, [0, 2]] -= left
                new_boxes[:, [1, 3]] -= top
                new_boxes, new_labels = _filter_boxes(new_boxes, labels, crop_h, crop_w)
                if labels.numel() > 0 and new_boxes.numel() == 0:
                    continue
            else:
                new_labels = labels
            return image.crop((left, top, left + crop_w, top + crop_h)), new_boxes, new_labels
        return image, boxes, labels

    def _random_translate(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        w, h = image.size
        max_dx, max_dy = int(0.1 * w), int(0.1 * h)
        if max_dx == 0 and max_dy == 0:
            return image, boxes, labels
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(image, (dx, dy))
        if boxes.numel() > 0:
            boxes = boxes + torch.tensor([dx, dy, dx, dy], dtype=torch.float32)
            boxes, labels = _filter_boxes(boxes, labels, h, w)
        return canvas, boxes, labels

    def _resize_square(self, image: Image.Image, boxes: torch.Tensor, size: int):
        w, h = image.size
        image = image.resize((size, size), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([size / w, size / h, size / w, size / h], dtype=torch.float32)
        return image, boxes

class DetectionDataset(Dataset):
    def __init__(self, annotation_path: str, image_dir: str, transforms=None):
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        data = load_json(self.annotation_path)
        self.classes = data.get("classes", CLASSES)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.images = data.get("images", [])
        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[ann.get("image_id")].append(ann)
        self.anns_by_image = anns_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image_id = info["id"]
        file_name = Path(info.get("file_name", image_id)).name
        image_path = self.image_dir / file_name
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []
        for ann in self.anns_by_image.get(image_id, []):
            cls = ann.get("class")
            if cls not in self.class_to_idx:
                continue
            box = ann.get("bbox", [])
            if len(box) != 4:
                continue
            boxes.append([float(v) for v in box])
            labels.append(self.class_to_idx[cls])

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.long)
        if boxes_t.numel() > 0:
            boxes_t = clip_boxes_to_image(boxes_t, height, width)
            keep = remove_small_boxes(boxes_t, 2.0)
            boxes_t = boxes_t[keep]
            labels_t = labels_t[keep]

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
