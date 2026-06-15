from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils.model import FCOSDetector
from utils.loss import clip_boxes_to_image, remove_small_boxes, LEVEL_STRIDES, flatten_predictions, make_grid_points, distances_to_boxes
from utils.utils import CLASSES, IMAGENET_MEAN, IMAGENET_STD, list_images, load_json, save_json
from utils.eval import batched_class_nms, nms, batched_class_soft_nms
import torchvision.transforms.functional as TF


def preprocess(image: Image.Image, img_size: int):
    ow, oh = image.size
    resized = image.resize((img_size, img_size), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, (oh, ow)


def decode_outputs(outputs, img_size: int, config: dict, topk: int = 300):
    cls_logits, reg_pred, ctr_logits = flatten_predictions(outputs)
    points = torch.cat(make_grid_points(outputs["cls"])[0], dim=0)
    results = []
    
    nms_type = config.get("nms_type", "hard")
    soft_method = config.get("soft_nms_method", "gaussian")
    soft_sigma = float(config.get("soft_nms_sigma", 0.5))
    max_det = int(config.get("max_detections", 100))
    per_class_cfg = config.get("per_class", None)
    global_conf = float(config.get("conf_threshold", 0.01))
    global_nms = float(config.get("nms_iou", 0.5))

    for b in range(cls_logits.shape[0]):
        cls_prob = cls_logits[b].sigmoid()
        ctr_prob = ctr_logits[b].sigmoid().unsqueeze(1)
        scores_all = cls_prob * ctr_prob
        scores, labels = scores_all.max(dim=1)
        
        keep = torch.zeros_like(scores, dtype=torch.bool)
        if per_class_cfg:
            for cls_idx, cls_name in enumerate(CLASSES):
                cls_mask = labels == cls_idx
                cls_conf = per_class_cfg.get(cls_name, {}).get("conf", global_conf)
                keep[cls_mask] = scores[cls_mask] >= cls_conf
        else:
            keep = scores >= global_conf
            
        if keep.sum() == 0:
            results.append({"boxes": scores.new_zeros((0, 4)), "scores": scores.new_zeros((0,)), "labels": labels.new_zeros((0,), dtype=torch.long)})
            continue
            
        scores, labels = scores[keep], labels[keep]
        regs = F.relu(reg_pred[b, keep])
        pts = points[keep]
        
        if scores.numel() > topk:
            top_scores, idx = scores.topk(topk)
            scores, labels, regs, pts = top_scores, labels[idx], regs[idx], pts[idx]
            
        boxes = distances_to_boxes(pts, regs)
        boxes = clip_boxes_to_image(boxes, img_size, img_size)
        valid = remove_small_boxes(boxes, 1.0)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        
        if nms_type == "soft":
            keep_idx, scores_out = batched_class_soft_nms(
                boxes, scores, labels, method=soft_method, sigma=soft_sigma, 
                iou_threshold=global_nms, score_threshold=0.001, max_detections=max_det
            )
        else:
            keep_all = []
            for cls_idx in labels.unique():
                idx = torch.where(labels == cls_idx)[0]
                c_nms = global_nms
                if per_class_cfg and CLASSES[int(cls_idx)] in per_class_cfg:
                    c_nms = per_class_cfg[CLASSES[int(cls_idx)]].get("nms_iou", global_nms)
                k = nms(boxes[idx], scores[idx], c_nms)
                keep_all.append(idx[k])
            keep_idx = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.long, device=boxes.device)
            keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)][:max_det]
            scores_out = scores[keep_idx]

        results.append({"boxes": boxes[keep_idx], "scores": scores_out, "labels": labels[keep_idx]})
    return results


def predict_one(model, image: Image.Image, device, img_size, config, topk, tta=False):
    tensor, (oh, ow) = preprocess(image, img_size)
    with torch.no_grad():
        det = decode_outputs(model(tensor.unsqueeze(0).to(device)), img_size, config, topk)[0]
    boxes, scores, labels = det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()
    if tta:
        flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        ftensor, _ = preprocess(flip, img_size)
        with torch.no_grad():
            fdet = decode_outputs(model(ftensor.unsqueeze(0).to(device)), img_size, config, topk)[0]
        fboxes = fdet["boxes"].cpu()
        if fboxes.numel() > 0:
            x1 = img_size - fboxes[:, 2]
            x2 = img_size - fboxes[:, 0]
            fboxes[:, 0], fboxes[:, 2] = x1, x2
            boxes = torch.cat([boxes, fboxes], dim=0)
            scores = torch.cat([scores, fdet["scores"].cpu()], dim=0)
            labels = torch.cat([labels, fdet["labels"].cpu()], dim=0)
            
            # Apply NMS again after merging
            nms_type = config.get("nms_type", "hard")
            max_det = int(config.get("max_detections", 100))
            if nms_type == "soft":
                keep, scores = batched_class_soft_nms(
                    boxes, scores, labels, 
                    method=config.get("soft_nms_method", "gaussian"),
                    sigma=float(config.get("soft_nms_sigma", 0.5)),
                    iou_threshold=float(config.get("nms_iou", 0.5)),
                    max_detections=max_det
                )
                boxes, labels = boxes[keep], labels[keep]
            else:
                per_class_cfg = config.get("per_class", None)
                global_nms = float(config.get("nms_iou", 0.5))
                keep_all = []
                for cls_idx in labels.unique():
                    idx = torch.where(labels == cls_idx)[0]
                    c_nms = global_nms
                    if per_class_cfg and CLASSES[int(cls_idx)] in per_class_cfg:
                        c_nms = per_class_cfg[CLASSES[int(cls_idx)]].get("nms_iou", global_nms)
                    k = nms(boxes[idx], scores[idx], c_nms)
                    keep_all.append(idx[k])
                keep = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.long, device=boxes.device)
                keep = keep[scores[keep].argsort(descending=True)][:max_det]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                
    if boxes.numel() > 0:
        boxes[:, [0, 2]] *= ow / img_size
        boxes[:, [1, 3]] *= oh / img_size
        boxes = clip_boxes_to_image(boxes, oh, ow)
    return boxes, scores, labels, (oh, ow)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--checkpoint", default="./models/best.pth")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--nms_iou", type=float, default=None)
    p.add_argument("--nms_type", choices=["hard", "soft"], default=None)
    p.add_argument("--max_detections", type=int, default=None)
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--tta", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = Path(args.checkpoint)
    # Auto download weights from Hugging Face if not present locally
    if not ckpt_path.exists() and args.checkpoint == "./models/best.pth":
        print(f"Checkpoint not found at {args.checkpoint}.")
        print("Downloading 'best.pth' from Hugging Face (Quockhanh05/YOLO_quockhanh)...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/Quockhanh05/YOLO_quockhanh/resolve/main/best.pth"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(ckpt_path))
            print("Successfully downloaded weights!")
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print(f"Please download manually from: {url}")
            return

    config_path = ckpt_path.with_name("best_config.json")
    if not config_path.exists() and args.checkpoint == "./models/best.pth":
        print("Downloading 'best_config.json' from Hugging Face...")
        url_cfg = "https://huggingface.co/Quockhanh05/YOLO_quockhanh/resolve/main/best_config.json"
        try:
            import urllib.request
            urllib.request.urlretrieve(url_cfg, str(config_path))
            print("Successfully downloaded config file!")
        except Exception as e:
            print(f"Error downloading config: {e}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    img_size = int(args.img_size or ckpt.get("img_size", 512))
    
    config = {
        "conf_threshold": 0.01,
        "nms_iou": 0.5,
        "nms_type": "hard",
        "max_detections": 100
    }
    
    if config_path.exists():
        cfg = load_json(config_path)
        config.update(cfg)
        
    if args.conf is not None:
        config["conf_threshold"] = args.conf
    if args.nms_iou is not None:
        config["nms_iou"] = args.nms_iou
    if args.nms_type is not None:
        config["nms_type"] = args.nms_type
    if args.max_detections is not None:
        config["max_detections"] = args.max_detections
        
    model = FCOSDetector(num_classes=len(CLASSES), backbone_name=ckpt.get("backbone", "convnext_tiny")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    image_paths = list_images(args.image_dir)
    outputs = []

    try:
        from tqdm import tqdm
        pbar = tqdm(image_paths, desc="Inference")
    except ImportError:
        pbar = image_paths

    for image_path in pbar:
        image = Image.open(image_path).convert("RGB")
        boxes, scores, labels, (oh, ow) = predict_one(model, image, device, img_size, config, args.topk, args.tta)
        entry = {"image_id": image_path.name, "boxes": []}
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(ow, int(round(x1))))
            y1 = max(0, min(oh, int(round(y1))))
            x2 = max(0, min(ow, int(round(x2))))
            y2 = max(0, min(oh, int(round(y2))))
            if x2 > x1 and y2 > y1:
                entry["boxes"].append({"class": CLASSES[int(label)], "confidence": float(max(0.0, min(1.0, float(score)))), "bbox": [x1, y1, x2, y2]})
        outputs.append(entry)

    output_path = Path(args.output)
    if output_path.suffix.lower() == ".csv":
        rows = []
        for item in outputs:
            image_id = item['image_id']
            output_boxes = []
            for box in item.get('boxes', []):
                bbox = box['bbox']
                output_boxes.append({
                    "x_min": float(bbox[0]),
                    "y_min": float(bbox[1]),
                    "x_max": float(bbox[2]),
                    "y_max": float(bbox[3]),
                    "class": box['class'],
                    "confidence": float(box['confidence'])
                })
            rows.append({"image_id": image_id, "bounding_boxes": json.dumps(output_boxes)})
        
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(rows)} predictions to CSV at {output_path}")
    else:
        save_json(outputs, args.output)
        print(f"Wrote {args.output} with {len(outputs)} images")


if __name__ == "__main__":
    main()
