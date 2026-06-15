from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

from .loss import box_iou
from .utils import CLASSES, load_json


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def batched_class_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    max_detections: int = 100,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep_all = []
    for cls in labels.unique():
        idx = torch.where(labels == cls)[0]
        keep = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[keep])
    keep = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep[:max_detections]


def soft_nms(
    boxes: torch.Tensor, 
    scores: torch.Tensor, 
    method: str = "gaussian", 
    sigma: float = 0.5, 
    iou_threshold: float = 0.3, 
    score_threshold: float = 0.001
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device), torch.empty((0,), device=scores.device)
    
    boxes = boxes.clone()
    scores = scores.clone()
    N = boxes.shape[0]
    indices = torch.arange(N, device=boxes.device)
    
    for i in range(N):
        max_score, max_pos = torch.max(scores[i:], dim=0)
        max_pos = max_pos + i
        
        b_temp, s_temp, i_temp = boxes[i].clone(), scores[i].clone(), indices[i].clone()
        boxes[i], scores[i], indices[i] = boxes[max_pos], scores[max_pos], indices[max_pos]
        boxes[max_pos], scores[max_pos], indices[max_pos] = b_temp, s_temp, i_temp
        
        if i + 1 < N:
            ious = box_iou(boxes[i].unsqueeze(0), boxes[i+1:]).squeeze(0)
            if method == "gaussian":
                weight = torch.exp(-(ious * ious) / sigma)
            elif method == "linear":
                weight = torch.ones_like(ious)
                weight[ious > iou_threshold] = 1 - ious[ious > iou_threshold]
            else:
                weight = torch.ones_like(ious)
                weight[ious > iou_threshold] = 0.0
            scores[i+1:] *= weight
            
    keep = scores >= score_threshold
    return indices[keep], scores[keep]


def batched_class_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    method: str = "gaussian",
    sigma: float = 0.5,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device), torch.empty((0,), device=scores.device)
        
    keep_indices, keep_scores = [], []
    for cls in labels.unique():
        idx = torch.where(labels == cls)[0]
        cls_idx, cls_scores = soft_nms(boxes[idx], scores[idx], method, sigma, iou_threshold, score_threshold)
        keep_indices.append(idx[cls_idx])
        keep_scores.append(cls_scores)
        
    if not keep_indices:
        return torch.empty((0,), dtype=torch.long, device=boxes.device), torch.empty((0,), device=scores.device)
        
    keep = torch.cat(keep_indices)
    new_scores = torch.cat(keep_scores)
    
    order = new_scores.argsort(descending=True)
    keep = keep[order][:max_detections]
    new_scores = new_scores[order][:max_detections]
    
    return keep, new_scores


def iou_np(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(gt_json: str, predictions: list[dict], iou_thr: float = 0.5) -> tuple[float, dict]:
    gt = load_json(gt_json)
    classes = gt.get("classes", CLASSES)
    gt_by_class = {c: defaultdict(list) for c in classes}
    for ann in gt.get("annotations", []):
        if ann.get("class") in gt_by_class:
            gt_by_class[ann["class"]][ann["image_id"]].append(ann["bbox"])
    pred_by_class = {c: [] for c in classes}
    for item in predictions:
        image_id = item.get("image_id")
        for box in item.get("boxes", []):
            cls = box.get("class")
            if cls in pred_by_class:
                pred_by_class[cls].append((image_id, float(box.get("confidence", 0.0)), box.get("bbox", [0, 0, 0, 0])))
    aps = []
    per_class_stats = {}
    for cls in classes:
        npos = sum(len(v) for v in gt_by_class[cls].values())
        if npos == 0:
            per_class_stats[cls] = {"AP50": 0.0, "precision": 0.0, "recall": 0.0, "num_pred": len(pred_by_class[cls]), "num_gt": 0}
            continue
        preds = sorted(pred_by_class[cls], key=lambda x: x[1], reverse=True)
        matched = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_by_class[cls].items()}
        tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
        for i, (img, _score, box) in enumerate(preds):
            gts = gt_by_class[cls].get(img, [])
            if not gts:
                fp[i] = 1
                continue
            ious = np.array([iou_np(box, gt_box) for gt_box in gts])
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not matched[img][j]:
                tp[i] = 1; matched[img][j] = True
            else:
                fp[i] = 1
        if len(preds) == 0:
            aps.append(0.0)
            per_class_stats[cls] = {"AP50": 0.0, "precision": 0.0, "recall": 0.0, "num_pred": 0, "num_gt": npos}
            continue
        fp_cum = np.cumsum(fp); tp_cum = np.cumsum(tp)
        rec = tp_cum / max(npos, np.finfo(np.float64).eps)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        aps.append(ap)
        
        final_tp = tp.sum()
        final_fp = fp.sum()
        final_prec = final_tp / max(final_tp + final_fp, np.finfo(np.float64).eps)
        final_rec = final_tp / max(npos, np.finfo(np.float64).eps)
        
        per_class_stats[cls] = {
            "AP50": float(ap), 
            "precision": float(final_prec), 
            "recall": float(final_rec), 
            "num_pred": len(preds), 
            "num_gt": int(npos)
        }
        
    map50 = float(np.mean(aps)) if aps else 0.0
    return map50, per_class_stats
