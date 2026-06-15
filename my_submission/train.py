from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model import FCOSDetector
from utils.data import DetectionDataset, DetectionTransform
from utils.loss import detection_loss
from utils.eval import evaluate_map50
from utils.utils import CLASSES, ModelEma, collate_fn, ensure_dir, load_json, move_targets_to_device, save_json, seed_everything, run_external_eval


FULL_RESUME_KEYS = {"epoch", "optimizer", "scheduler", "scaler"}


def set_backbone_trainable(model, trainable: bool):
    target_model = model.module if hasattr(model, "module") else model
    for p in target_model.backbone.parameters():
        p.requires_grad_(trainable)


def build_optimizer(model, backbone_lr, lr, weight_decay):
    target_model = model.module if hasattr(model, "module") else model
    return torch.optim.AdamW([
        {"params": [p for p in target_model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": [p for n, p in target_model.named_parameters() if not n.startswith("backbone") and p.requires_grad], "lr": lr},
    ], weight_decay=weight_decay)


def checkpoint_has_full_state(ckpt: dict) -> bool:
    return FULL_RESUME_KEYS.issubset(set(ckpt.keys()))


def load_map_from_config(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    value = data.get("map50", data.get("mAP@0.5")) if isinstance(data, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")


def load_per_class_stats(path: Path) -> dict | None:
    try:
        data = load_json(path)
    except Exception:
        return None
    stats = data.get("per_class") if isinstance(data, dict) else None
    return stats if isinstance(stats, dict) else None


def class_ap_value(stats: dict, cls: str) -> float | None:
    item = stats.get(cls)
    if not isinstance(item, dict):
        return None
    value = item.get("AP50", item.get("ap"))
    return float(value) if isinstance(value, (int, float)) else None


def resolve_resume_path(path: str | None, is_rank0: bool) -> tuple[Path | None, bool]:
    if not path:
        return None, False
    resume_path = Path(path)
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume checkpoint does not exist: {resume_path}")

    ckpt = torch.load(resume_path, map_location="cpu")
    if checkpoint_has_full_state(ckpt):
        return resume_path, False

    sibling_last = resume_path.with_name("last.pth")
    if sibling_last.exists():
        sibling = torch.load(sibling_last, map_location="cpu")
        if checkpoint_has_full_state(sibling):
            if is_rank0:
                print(f"WARNING: {resume_path} is weights-only; using sibling full resume checkpoint {sibling_last}.")
            return sibling_last, False

    if is_rank0:
        print(f"WARNING: {resume_path} is weights-only; loading it as --weights and starting at epoch 1.")
    return resume_path, True


def save_best_checkpoint(path: Path, model, args, epoch: int, best_map: float, bad_epochs: int, source: str):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "classes": CLASSES,
        "img_size": args.img_size,
        "backbone": getattr(model, "backbone_name", "convnext_tiny"),
        "best_map": best_map,
        "bad_epochs": bad_epochs,
        "config_source": source,
    }, path)


def predict_dataset(model, loader, device, out_path, config=None, topk=300):
    from predict import decode_outputs
    model.eval()
    if config is None:
        config = {"conf_threshold": 0.15, "nms_iou": 0.5, "nms_type": "hard", "max_detections": 100}
    results = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            decoded = decode_outputs(outputs, images.shape[-1], config, topk)
            for det, tgt in zip(decoded, targets):
                oh, ow = int(tgt["orig_size"][0]), int(tgt["orig_size"][1])
                scale_x, scale_y = ow / images.shape[-1], oh / images.shape[-2]
                boxes = det["boxes"].cpu()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                entry = {"image_id": tgt["image_id"], "boxes": []}
                for box, score, label in zip(boxes, det["scores"].cpu(), det["labels"].cpu()):
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)
                    if x2 > x1 and y2 > y1:
                        entry["boxes"].append({"class": CLASSES[int(label)], "confidence": float(score), "bbox": [x1, y1, x2, y2]})
                results.append(entry)
    save_json(results, out_path)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--val_image_dir", required=True)
    p.add_argument("--checkpoint_dir", default="./models/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--resume", default=None, help="Path to checkpoint (.pth) to resume training from")
    p.add_argument("--weights", default=None, help="Path to weights-only checkpoint (.pth) to fine-tune from epoch 1")
    p.add_argument("--version", default="", help="Version directory name to save checkpoints and outputs")
    p.add_argument("--max_bad_batches", type=int, default=5, help="Stop an epoch if this many non-finite losses/gradients are seen")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    
    # DDP Initialization
    is_dist = "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    if is_dist:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    is_rank0 = not is_dist or (dist.get_rank() == 0)
    
    ckpt_dir = ensure_dir(Path(args.checkpoint_dir) / args.version)
    train_tf = DetectionTransform(True, args.img_size, [448, 512, 576])
    val_tf = DetectionTransform(False, args.img_size)
    train_ds = DetectionDataset(args.train_data, args.image_dir, train_tf)
    val_ds = DetectionDataset(args.val_data, args.val_image_dir, val_tf)
    
    if is_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        
    # Non-distributed validation loader for rank-0 validation evaluations
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    
    model = FCOSDetector(num_classes=len(CLASSES)).to(device)
    
    start_epoch = 1
    best_map, bad_epochs = -1.0, 0
    ckpt = None
    weight_source = None

    best_cfg_map = load_map_from_config(ckpt_dir / "best_config.json") if (ckpt_dir / "best.pth").exists() else None
    if best_cfg_map is not None:
        best_map = best_cfg_map
        if is_rank0:
            print(f"Loaded protected best mAP from {ckpt_dir / 'best_config.json'}: {best_map:.6f}")

    if args.resume:
        resolved, as_weights = resolve_resume_path(args.resume, is_rank0)
        if as_weights:
            args.weights = str(resolved)
        else:
            args.resume = str(resolved)

    if args.weights:
        if is_rank0:
            print(f"Loading weights from {args.weights}; fine-tuning from epoch 1.")
        weights_ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_ckpt["model"])
        weight_source = str(args.weights)
        ckpt_best = weights_ckpt.get("best_map")
        if isinstance(ckpt_best, (int, float)):
            best_map = max(best_map, float(ckpt_best))

    if args.resume and not args.weights:
        if is_rank0:
            print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        if not checkpoint_has_full_state(ckpt):
            raise ValueError(f"--resume requires a full training checkpoint with {sorted(FULL_RESUME_KEYS)}: {args.resume}")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_map = max(best_map, float(ckpt.get("best_map", -1.0)))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        if is_rank0:
            print(f"Resuming from epoch {start_epoch}")
            
    set_backbone_trainable(model, start_epoch > 4)
    optimizer = build_optimizer(model, args.backbone_lr, args.lr, args.weight_decay)
    
    # Wrap with DDP
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    if hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_steps = max(1, args.epochs * len(train_loader))
    if start_epoch > 4:
        total_steps = max(1, (args.epochs - 4 + 1) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Instantiate ModelEma using the base model
    base_model = model.module if is_dist else model
    ema = None if args.no_ema else ModelEma(base_model)
    
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            ema.module.load_state_dict(ckpt["ema"])

    for epoch in range(start_epoch, args.epochs + 1):
        if is_dist:
            train_sampler.set_epoch(epoch)
            
        if epoch == 4:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, args.backbone_lr, args.lr, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (args.epochs - epoch + 1) * len(train_loader)))
            
        model.train()
        sums = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "centerness_loss": 0.0, "num_pos": 0.0}
        
        bad_batches = 0
        good_batches = 0
        epoch_failed = False
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss_dict = detection_loss(model(images), targets, len(CLASSES))
                loss = loss_dict["loss"]
            bad_loss = torch.tensor([0 if torch.isfinite(loss) else 1], device=device, dtype=torch.int)
            if is_dist:
                dist.all_reduce(bad_loss, op=dist.ReduceOp.MAX)
            if bad_loss.item():
                bad_batches += 1
                if is_rank0:
                    print(f"WARNING: non-finite loss at epoch {epoch}; skipping batch ({bad_batches}/{args.max_bad_batches}).")
                optimizer.zero_grad(set_to_none=True)
                if bad_batches >= args.max_bad_batches:
                    epoch_failed = True
                    break
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if ema is not None:
                base_model = model.module if is_dist else model
                ema.update(base_model)
                
            for k in sums:
                sums[k] += float(loss_dict[k].detach().cpu())
            good_batches += 1
                
            lr = optimizer.param_groups[-1]["lr"]

        stop_training = epoch_failed
        
        # Only run validation and checkpoint saving on Rank 0
        if is_rank0:
            base_model = model.module if is_dist else model
            eval_model = ema.module.to(device) if ema is not None else base_model
            val_loss, val_batches = 0.0, 0
            eval_model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets_dev = move_targets_to_device(targets, device)
                    loss_dict = detection_loss(eval_model(images), targets_dev, len(CLASSES))
                    val_loss += float(loss_dict["loss"])
                    val_batches += 1

            pred_path = ckpt_dir / "val_predictions.json"
            score_path = ckpt_dir / "val_score.json"
            cfg = {"conf_threshold": 0.15, "nms_iou": 0.5, "nms_type": "hard", "max_detections": 100}
            preds = predict_dataset(eval_model, val_loader, device, pred_path, config=cfg)
            
            map50 = run_external_eval(args.val_data, str(pred_path), str(score_path))
            eval_source = "external_eval"
            stats = None
            if map50 is None:
                eval_source = "internal_eval"
                map50, stats = evaluate_map50(args.val_data, preds)
                save_json({"mAP@0.5": map50, "source": "internal", "per_class": stats}, score_path)
            else:
                stats = load_per_class_stats(score_path)
            
            
            improved = map50 > best_map
            if improved:
                best_map = map50
                bad_epochs = 0
                save_best_checkpoint(ckpt_dir / "best.pth", eval_model, args, epoch, best_map, bad_epochs, eval_source)
            else:
                bad_epochs += 1
            
            # Save last checkpoint for resuming
            torch.save({
                "epoch": epoch,
                "model": base_model.state_dict(),
                "ema": ema.module.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_map": best_map,
                "bad_epochs": bad_epochs,
                "weights_source": weight_source,
            }, ckpt_dir / "last.pth")

            n = max(1, good_batches)
            train_loss_avg = sums["loss"] / n
            val_loss_avg = val_loss / max(1, val_batches)
            print(f"Epoch {epoch}: train_loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f} mAP@0.5={map50:.4f} best={best_map:.4f}")
            append_jsonl(ckpt_dir / "train_history.jsonl", {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "mAP@0.5": map50,
                "best_map": best_map,
                "bad_epochs": bad_epochs,
                "good_batches": good_batches,
                "bad_batches": bad_batches,
                "eval_source": eval_source,
                "lr": optimizer.param_groups[-1]["lr"],
            })
            if is_rank0 and stats is not None:
                for c, s in stats.items():
                    num_gt = s.get("num_gt", s.get("num_ground_truth", 0))
                    num_pred = s.get("num_pred", s.get("num_predictions", 0))
                    ap = s.get("AP50", s.get("ap", 0.0))
                    if num_gt > 0:
                        print(f"  {c}: AP50={ap:.4f} P={s['precision']:.4f} R={s['recall']:.4f} (GT:{num_gt} PRED:{num_pred})")
                        
            if epoch_failed:
                print(f"Stopping after epoch {epoch}: too many non-finite batches.")
            if bad_epochs >= args.patience:
                stop_training = True

        if is_dist:
            # Broadcast the stop signal to all other ranks
            stop_tensor = torch.tensor([1.0 if stop_training else 0.0], device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() > 0.5:
                break
            # Sync ranks before starting next epoch
            dist.barrier()
        elif stop_training:
            break

    # Tune thresholds on the best model (Only on Rank 0)
    if is_rank0:
        ckpt = torch.load(ckpt_dir / "best.pth", map_location=device)
        tune_model = FCOSDetector(num_classes=len(CLASSES), backbone_name=ckpt.get("backbone", "convnext_tiny")).to(device)
        tune_model.load_state_dict(ckpt["model"])
        
        confs = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
        ious = [0.45, 0.5, 0.55, 0.6, 0.65]
        max_detections = [100, 200, 300]
        nms_types = ["hard", "soft"]
        
        best_global_map = -1
        best_global_cfg = {}
        class_best = {c: {"AP50": -1, "conf": 0.15, "nms_iou": 0.5} for c in CLASSES}
        
        for nms_type in nms_types:
            for max_det in max_detections:
                for conf in confs:
                    for nms_iou in ious:
                        pred_path = ckpt_dir / "val_predictions.json"
                        score_path = ckpt_dir / "val_score.json"
                        cfg = {
                            "conf_threshold": conf,
                            "nms_iou": nms_iou,
                            "nms_type": nms_type,
                            "max_detections": max_det,
                        }
                        if nms_type == "soft":
                            cfg.update({"soft_nms_method": "gaussian", "soft_nms_sigma": 0.5})
                        preds = predict_dataset(tune_model, val_loader, device, pred_path, config=cfg)
                        
                        map50 = run_external_eval(args.val_data, str(pred_path), str(score_path))
                        stats = None
                        if map50 is None:
                            map50, stats = evaluate_map50(args.val_data, preds)
                            save_json({"mAP@0.5": map50, "source": "internal", "per_class": stats}, score_path)
                        else:
                            stats = load_per_class_stats(score_path)
                        
                        if map50 > best_global_map:
                            best_global_map = map50
                            best_global_cfg = dict(cfg)
                            best_global_cfg["map50"] = map50
                            
                        if stats is not None:
                            for c in CLASSES:
                                ap = class_ap_value(stats, c)
                                if ap is not None and ap > class_best[c]["AP50"]:
                                    class_best[c]["AP50"] = ap
                                    class_best[c]["conf"] = conf
                                    class_best[c]["nms_iou"] = nms_iou

        best_cfg = dict(best_global_cfg)
        if any(v["AP50"] >= 0 for v in class_best.values()):
            candidate_cfg = dict(best_global_cfg)
            candidate_cfg["per_class"] = {c: {"conf": class_best[c]["conf"], "nms_iou": class_best[c]["nms_iou"], "AP50": class_best[c]["AP50"]} for c in CLASSES}
            pred_path = ckpt_dir / "val_predictions.json"
            score_path = ckpt_dir / "val_score.json"
            preds = predict_dataset(tune_model, val_loader, device, pred_path, config=candidate_cfg)
            candidate_map = run_external_eval(args.val_data, str(pred_path), str(score_path))
            if candidate_map is None:
                candidate_map, _stats = evaluate_map50(args.val_data, preds)
                save_json({"mAP@0.5": candidate_map, "source": "internal"}, score_path)
            if candidate_map >= best_global_map:
                candidate_cfg["map50"] = candidate_map
                best_cfg = candidate_cfg
        
        save_json(best_cfg, ckpt_dir / "best_config.json")
        try:
            (ckpt_dir / "val_predictions.json").unlink()
        except FileNotFoundError:
            pass
        print(f"Best config: {best_cfg}")
        
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
