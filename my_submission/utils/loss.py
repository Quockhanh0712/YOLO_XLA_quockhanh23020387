from __future__ import annotations

import torch
import torch.nn.functional as F

LEVEL_STRIDES = [8, 16, 32, 64]
LEVEL_RANGES = [(0, 96), (96, 192), (192, 384), (384, 10**8)]

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou
    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter_lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return iou - (area - union) / area.clamp(min=1e-6)

def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0, width)
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0, height)
    return clipped

def remove_small_boxes(boxes: torch.Tensor, min_size: float = 1.0) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return torch.where((ws >= min_size) & (hs >= min_size))[0]

def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).sum()

def make_grid_points(features: list[torch.Tensor], strides: list[int] = LEVEL_STRIDES) -> tuple[list[torch.Tensor], list[int]]:
    points = []
    for feat, stride in zip(features, strides):
        h, w = feat.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(h, device=feat.device, dtype=torch.float32),
            torch.arange(w, device=feat.device, dtype=torch.float32),
            indexing="ij",
        )
        pts = torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).reshape(-1, 2)
        points.append(pts)
    return points, strides

def flatten_predictions(outputs: dict[str, list[torch.Tensor]]):
    cls = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]) for x in outputs["cls"]], dim=1)
    reg = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4) for x in outputs["bbox"]], dim=1)
    ctr = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1) for x in outputs["centerness"]], dim=1)
    return cls, reg, ctr

def assign_fcos_targets(points_per_level: list[torch.Tensor], targets: list[dict], num_classes: int):
    device = points_per_level[0].device
    all_points = torch.cat(points_per_level, dim=0)
    labels_batch, reg_batch, ctr_batch, pos_batch = [], [], [], []
    level_ids = []
    for level, pts in enumerate(points_per_level):
        level_ids.append(torch.full((pts.shape[0],), level, device=device, dtype=torch.long))
    level_ids = torch.cat(level_ids)

    for target in targets:
        boxes = target["boxes"].to(device).float()
        labels = target["labels"].to(device).long()
        n = all_points.shape[0]
        cls_targets = torch.zeros((n, num_classes), device=device)
        reg_targets = torch.zeros((n, 4), device=device)
        ctr_targets = torch.zeros((n,), device=device)
        pos_mask = torch.zeros((n,), dtype=torch.bool, device=device)
        if boxes.numel() == 0:
            labels_batch.append(cls_targets)
            reg_batch.append(reg_targets)
            ctr_batch.append(ctr_targets)
            pos_batch.append(pos_mask)
            continue

        xs, ys = all_points[:, 0], all_points[:, 1]
        l = xs[:, None] - boxes[:, 0]
        t = ys[:, None] - boxes[:, 1]
        r = boxes[:, 2] - xs[:, None]
        b = boxes[:, 3] - ys[:, None]
        reg = torch.stack([l, t, r, b], dim=2)
        inside_box = reg.min(dim=2).values > 0

        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        inside_center = torch.zeros_like(inside_box)
        for level, stride in enumerate(LEVEL_STRIDES):
            idx = level_ids == level
            radius = 1.5 * stride
            cb = torch.cat([centers - radius, centers + radius], dim=1)
            cb[:, 0::2] = torch.maximum(cb[:, 0::2], boxes[:, 0::2])
            cb[:, 1::2] = torch.maximum(cb[:, 1::2], boxes[:, 1::2])
            cb[:, 2] = torch.minimum(cb[:, 2], boxes[:, 2])
            cb[:, 3] = torch.minimum(cb[:, 3], boxes[:, 3])
            c_l = xs[idx, None] - cb[:, 0]
            c_t = ys[idx, None] - cb[:, 1]
            c_r = cb[:, 2] - xs[idx, None]
            c_b = cb[:, 3] - ys[idx, None]
            inside_center[idx] = torch.stack([c_l, c_t, c_r, c_b], dim=2).min(dim=2).values > 0

        max_reg = reg.max(dim=2).values
        in_range = torch.zeros_like(inside_box)
        for level, (lo, hi) in enumerate(LEVEL_RANGES):
            idx = level_ids == level
            in_range[idx] = (max_reg[idx] >= lo) & (max_reg[idx] <= hi)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        candidates = inside_box & inside_center & in_range
        candidate_areas = torch.where(candidates, areas[None, :], torch.full_like(candidates.float(), 1e18))
        min_area, matched = candidate_areas.min(dim=1)
        pos_mask = min_area < 1e18
        if pos_mask.any():
            matched_reg = reg[torch.arange(n, device=device), matched]
            reg_targets[pos_mask] = matched_reg[pos_mask]
            matched_labels = labels[matched[pos_mask]]
            cls_targets[pos_mask, matched_labels] = 1.0
            lr_min = torch.minimum(matched_reg[:, 0], matched_reg[:, 2])
            lr_max = torch.maximum(matched_reg[:, 0], matched_reg[:, 2]).clamp(min=1e-6)
            tb_min = torch.minimum(matched_reg[:, 1], matched_reg[:, 3])
            tb_max = torch.maximum(matched_reg[:, 1], matched_reg[:, 3]).clamp(min=1e-6)
            ctr_batch_vals = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max)).clamp(0, 1)
            ctr_targets[pos_mask] = ctr_batch_vals[pos_mask]
        labels_batch.append(cls_targets)
        reg_batch.append(reg_targets)
        ctr_batch.append(ctr_targets)
        pos_batch.append(pos_mask)

    return torch.stack(labels_batch), torch.stack(reg_batch), torch.stack(ctr_batch), torch.stack(pos_batch)

def distances_to_boxes(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        points[:, 0] - distances[:, 0],
        points[:, 1] - distances[:, 1],
        points[:, 0] + distances[:, 2],
        points[:, 1] + distances[:, 3],
    ], dim=1)

def detection_loss(outputs: dict[str, list[torch.Tensor]], targets: list[dict], num_classes: int = 5) -> dict[str, torch.Tensor]:
    cls_logits, reg_pred, ctr_logits = flatten_predictions(outputs)
    points_per_level, _ = make_grid_points(outputs["cls"])
    cls_t, reg_t, ctr_t, pos = assign_fcos_targets(points_per_level, targets, num_classes)
    num_pos = pos.sum().clamp(min=1).float()
    cls_loss = sigmoid_focal_loss(cls_logits, cls_t) / num_pos

    if pos.any():
        points = torch.cat(points_per_level, dim=0)
        pred_boxes = []
        target_boxes = []
        for b in range(cls_logits.shape[0]):
            p = pos[b]
            pred_boxes.append(distances_to_boxes(points[p], F.relu(reg_pred[b, p])))
            target_boxes.append(distances_to_boxes(points[p], reg_t[b, p]))
        pred_boxes = torch.cat(pred_boxes, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)
        giou = generalized_box_iou(pred_boxes, target_boxes).diag()
        box_loss = (1.0 - giou).mean()
        centerness_loss = F.binary_cross_entropy_with_logits(ctr_logits[pos], ctr_t[pos], reduction="mean")
    else:
        box_loss = torch.tensor(0.0, device=cls_logits.device)
        centerness_loss = torch.tensor(0.0, device=cls_logits.device)

    total = cls_loss + 2.0 * box_loss + centerness_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "box_loss": box_loss.detach(),
        "centerness_loss": centerness_loss.detach(),
        "num_pos": pos.sum().detach(),
    }
