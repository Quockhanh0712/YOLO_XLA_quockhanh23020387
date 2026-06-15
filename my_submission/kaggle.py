# Copy this whole file into one Kaggle Notebook cell. It does not generate source code.
from __future__ import annotations

import shutil
import subprocess
import zipfile
import json
from pathlib import Path

# Edit these to download or use specific paths.
# They can be:
# - Kaggle dataset slug (e.g. "username/dataset-slug")
# - Kaggle dataset URL (e.g. "https://www.kaggle.com/datasets/username/dataset-slug")
# - Direct HTTP/HTTPS link (e.g. "https://example.com/dataset.zip")
# - Local path (e.g. "/kaggle/input/dataset-slug")
# If left as None, the script will auto-detect them in /kaggle/input
CODE_LINK_OR_PATH = "/kaggle/input/datasets/tqkhanh05/code-convex/my_submission"
DATA_LINK_OR_PATH = "/kaggle/input/datasets/tqkhanh05/data-yolo/public"
# Paste your Kaggle Model or Checkpoint link/path here to resume training or load a specific pre-trained checkpoint.
# Supports Kaggle Model/Dataset URL, Kaggle slug, or local file/directory path.
MODEL_LINK_OR_PATH = ""
VERSION = "v1"

KAGGLE_INPUT = Path("/kaggle/input")
WORK_REPO = Path("/kaggle/working/my_submission")
PUBLIC_DST = WORK_REPO / "public"
OUTPUT_ROOT = Path("/kaggle/working/outputs") / VERSION


def short_tree(root: Path, max_files: int = 100) -> None:
    print(f"\nTree: {root}")
    count = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        if depth > 3:
            continue
        if count >= max_files:
            print("  ...")
            break
        print("  " + "  " * depth + rel.name + ("/" if path.is_dir() else ""))
        count += 1


def list_input() -> str:
    if not KAGGLE_INPUT.exists():
        return "Missing /kaggle/input"
    paths = [str(p) for p in sorted(KAGGLE_INPUT.rglob("*")) if len(p.relative_to(KAGGLE_INPUT).parts) <= 4]
    return "\n".join(paths) if paths else "No files under /kaggle/input"


def unzip_if_needed(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".zip":
        out = Path("/kaggle/working/_unzipped") / path.stem
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
        return out
    return path


def resolve_dataset(link_or_path: str | None, kind: str) -> Path | None:
    if not link_or_path:
        return None
    
    # If it's a local path on disk that already exists
    path = Path(link_or_path)
    if path.exists():
        return unzip_if_needed(path)
    
    link_or_path = link_or_path.strip()
    
    # Parse Kaggle URLs or slugs
    import re
    kaggle_url_match = re.search(r"kaggle\.com/datasets/([^/]+)/([^/]+)", link_or_path)
    slug_match = re.match(r"^([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_]+)$", link_or_path)
    
    slug = None
    if kaggle_url_match:
        slug = f"{kaggle_url_match.group(1)}/{kaggle_url_match.group(2)}"
    elif slug_match:
        slug = link_or_path
        
    if slug:
        print(f"Downloading Kaggle dataset '{slug}' using kagglehub...")
        try:
            import kagglehub
            downloaded = kagglehub.dataset_download(slug)
            return unzip_if_needed(Path(downloaded))
        except Exception as e:
            print(f"Error downloading via kagglehub: {e}")
            print("Trying fallback options...")
            
    # Check if it starts with http:// or https://
    if link_or_path.startswith("http://") or link_or_path.startswith("https://"):
        print(f"Downloading from URL: {link_or_path}")
        import urllib.request
        temp_dir = Path("/kaggle/working/_temp_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        filename = link_or_path.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = f"dataset_{kind}.zip"
        dest_file = temp_dir / filename
        try:
            urllib.request.urlretrieve(link_or_path, dest_file)
            return unzip_if_needed(dest_file)
        except Exception as e:
            raise RuntimeError(f"Failed to download from URL '{link_or_path}': {e}")
            
    # If it's a string path that doesn't exist yet, we raise error
    raise RuntimeError(f"Could not resolve link or path for {kind}: {link_or_path}")


def is_code_repo(path: Path) -> bool:
    return path.is_dir() and (path / "train.py").exists() and (path / "predict.py").exists()


def is_public_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "annotations/train.json").exists()
        and (path / "annotations/val.json").exists()
        and (path / "train/images").exists()
        and (path / "val/images").exists()
    )


def find_code_dataset() -> Path:
    resolved = resolve_dataset(CODE_LINK_OR_PATH, "code")
    if resolved:
        if is_code_repo(resolved):
            return resolved
        nested = [p for p in resolved.rglob("*") if is_code_repo(p)]
        if nested:
            return sorted(nested, key=lambda x: len(x.parts))[0]
        raise RuntimeError(
            f"Resolved code path exists but does not contain train.py and predict.py: {resolved}"
        )

    candidates = []
    for path in KAGGLE_INPUT.rglob("*"):
        path = unzip_if_needed(path)
        if is_code_repo(path):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(
            "Could not find code dataset containing train.py and predict.py.\n\n"
            "Kaggle input listing:\n" + list_input()
        )
    return sorted(candidates, key=lambda x: len(x.parts))[0]


def find_public_dataset() -> Path:
    resolved = resolve_dataset(DATA_LINK_OR_PATH, "data")
    if resolved:
        if is_public_dataset(resolved):
            return resolved
        nested = [p for p in resolved.rglob("*") if is_public_dataset(p)]
        if nested:
            return sorted(nested, key=lambda x: len(x.parts))[0]
        raise RuntimeError(
            f"Resolved data path exists but does not have annotations/train.json, annotations/val.json, train/images, val/images: {resolved}"
        )

    candidates = []
    for path in KAGGLE_INPUT.rglob("*"):
        path = unzip_if_needed(path)
        if is_public_dataset(path):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(
            "Could not find public dataset with annotations/train.json, annotations/val.json, train/images, val/images.\n\n"
            "Kaggle input listing:\n" + list_input()
        )
    return sorted(candidates, key=lambda x: len(x.parts))[0]


def py_files_for_compile() -> list[str]:
    files = ["train.py", "predict.py"]
    files += [str(p.relative_to(WORK_REPO)) for p in sorted((WORK_REPO / "core").glob("*.py"))]
    return files


def copy_seed_model_artifacts(src: Path, dst: Path) -> None:
    src_models = src / "models"
    if not src_models.exists():
        return
    for name in ["best.pth", "last.pth", "best_config.json"]:
        for src_file in src_models.rglob(name):
            rel = src_file.relative_to(src_models)
            if len(rel.parts) == 1:
                rel = Path(VERSION) / rel
            dst_file = dst / "models" / rel
            if not dst_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"Seeded model artifact: {dst_file}")


def checkpoint_has_full_state(path: Path) -> bool:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Could not inspect checkpoint {path}: {e}")
        return False
    return all(key in ckpt for key in ["epoch", "optimizer", "scheduler", "scaler"])


def newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return paths[0]


def find_training_checkpoint(model_root: Path) -> tuple[Path | None, str | None]:
    if model_root.is_file() and model_root.suffix.lower() == ".pth":
        return (model_root, "resume") if checkpoint_has_full_state(model_root) else (model_root, "weights")
    if not model_root.exists():
        return None, None
    version_dir = model_root / "models" / VERSION if (model_root / "models" / VERSION).exists() else model_root / VERSION
    search_roots = [version_dir, model_root]
    for root in search_roots:
        if not root.exists():
            continue
        last = newest(list(root.rglob("last.pth")))
        if last is not None and checkpoint_has_full_state(last):
            return last, "resume"
    for root in search_roots:
        if not root.exists():
            continue
        best = newest(list(root.rglob("best.pth")))
        if best is not None:
            return best, "weights"
    any_ckpt = newest(list(model_root.rglob("*.pth"))) if model_root.is_dir() else None
    if any_ckpt is not None:
        return (any_ckpt, "resume") if checkpoint_has_full_state(any_ckpt) else (any_ckpt, "weights")
    return None, None


def add_file_to_zip(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.exists() and path.is_file():
        zf.write(path, arcname)


def make_clean_submission_zip(repo: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    artifact_dir = repo / "models" / VERSION
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["train.py", "predict.py", "kaggle_onecell.py", "README.md", "requirements.txt"]:
            add_file_to_zip(zf, repo / name, name)
        for path in sorted((repo / "core").glob("*.py")):
            add_file_to_zip(zf, path, str(path.relative_to(repo)))
        for name in ["best.pth", "last.pth", "best_config.json", "train_history.jsonl", "train.log", "val_score.json"]:
            add_file_to_zip(zf, artifact_dir / name, f"models/{VERSION}/{name}")


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def make_training_outputs(repo: Path) -> Path:
    artifact_dir = repo / "models" / VERSION
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_files = [
        "best.pth",
        "last.pth",
        "best_config.json",
        "train_history.jsonl",
        "train.log",
        "val_score.json",
    ]
    copied = []
    for name in output_files:
        src = artifact_dir / name
        if src.exists():
            dst = OUTPUT_ROOT / name
            shutil.copy2(src, dst)
            copied.append(name)

    summary = {
        "version": VERSION,
        "artifact_dir": str(artifact_dir),
        "output_dir": str(OUTPUT_ROOT),
        "copied_files": copied,
        "val_score": read_json_if_exists(artifact_dir / "val_score.json"),
        "best_config": read_json_if_exists(artifact_dir / "best_config.json"),
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_lines = [
        f"version: {VERSION}",
        f"output_dir: {OUTPUT_ROOT}",
        "",
        "files:",
    ]
    for name in sorted(copied + ["run_summary.json"]):
        manifest_lines.append(f"- {name}")
    (OUTPUT_ROOT / "README_OUTPUT.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    output_zip = Path("/kaggle/working") / f"training_outputs_{VERSION}.zip"
    if output_zip.exists():
        output_zip.unlink()
    shutil.make_archive(str(output_zip.with_suffix("")), "zip", OUTPUT_ROOT)
    return output_zip


code_src = find_code_dataset()
public_src = find_public_dataset()
print("Code dataset:", code_src)
print("Public dataset:", public_src)

if WORK_REPO.exists():
    # Preserve models/ directory across cell restarts
    for child in WORK_REPO.iterdir():
        if child.name == "models":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
shutil.copytree(code_src, WORK_REPO, ignore=shutil.ignore_patterns("public", "*.zip", "__pycache__", "models"), dirs_exist_ok=True)
copy_seed_model_artifacts(code_src, WORK_REPO)

if PUBLIC_DST.exists():
    shutil.rmtree(PUBLIC_DST)
shutil.copytree(public_src, PUBLIC_DST)

short_tree(WORK_REPO)

subprocess.run(["python", "-m", "py_compile", *py_files_for_compile()], check=True, cwd=WORK_REPO)

train_cmd = [
    "python", "train.py",
    "--train_data", "./public/annotations/train.json",
    "--val_data", "./public/annotations/val.json",
    "--image_dir", "./public/train/images",
    "--val_image_dir", "./public/val/images",
    "--checkpoint_dir", "./models/",
    "--epochs", "50",
    "--batch_size", "2",
    "--img_size", "512",
    "--amp",
    "--version", VERSION,
]

existing_ckpt, checkpoint_mode = find_training_checkpoint(WORK_REPO / "models" / VERSION)
if existing_ckpt is None:
    existing_ckpt, checkpoint_mode = find_training_checkpoint(WORK_REPO / "models")
if MODEL_LINK_OR_PATH:
    try:
        resolved_model = resolve_dataset(MODEL_LINK_OR_PATH, "model")
        if resolved_model and resolved_model.exists():
            model_ckpt, model_mode = find_training_checkpoint(resolved_model)
            if model_ckpt is not None:
                existing_ckpt, checkpoint_mode = model_ckpt, model_mode
                print(f"Resolved MODEL_LINK_OR_PATH to checkpoint: {existing_ckpt} ({checkpoint_mode})")
    except Exception as e:
        print(f"Error resolving MODEL_LINK_OR_PATH ({MODEL_LINK_OR_PATH}): {e}. Will fallback to auto-detection.")

if existing_ckpt is None:
    code_ckpt, code_mode = find_training_checkpoint(code_src / "models" / VERSION)
    existing_ckpt, checkpoint_mode = code_ckpt, code_mode

if existing_ckpt is None:
    input_last = newest(list(KAGGLE_INPUT.rglob(f"**/{VERSION}/last.pth")) or list(KAGGLE_INPUT.rglob("**/last.pth")))
    if input_last is not None and checkpoint_has_full_state(input_last):
        existing_ckpt, checkpoint_mode = input_last, "resume"
    else:
        input_best = newest(list(KAGGLE_INPUT.rglob(f"**/{VERSION}/best.pth")) or list(KAGGLE_INPUT.rglob("**/best.pth")))
        if input_best is not None:
            existing_ckpt, checkpoint_mode = input_best, "weights"

if existing_ckpt is not None and existing_ckpt.exists():
    if checkpoint_mode == "resume":
        print(f"\nFound full checkpoint at {existing_ckpt}. Resuming training...")
        train_cmd.extend(["--resume", str(existing_ckpt)])
    else:
        print(f"\nFound weights-only checkpoint at {existing_ckpt}. Fine-tuning from epoch 1...")
        train_cmd.extend(["--weights", str(existing_ckpt)])
else:
    print("\nNo existing checkpoint found. Training from scratch.")

# Multi-GPU auto-detection for torchrun (DDP)
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"\nDetected {num_gpus} GPUs. Running with torchrun (DDP)...")
    train_cmd = ["torchrun", f"--nproc_per_node={num_gpus}"] + train_cmd[1:]
else:
    print(f"\nDetected {num_gpus} GPU(s). Running with standard single-process python...")

import sys

log_file = WORK_REPO / "models" / VERSION / "train.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\nStarting training... Logs are being saved to {log_file}")

def run_training(cmd: list[str]) -> tuple[int, str]:
    recent_lines: list[str] = []
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        process = subprocess.Popen(
            cmd, 
            cwd=WORK_REPO, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
            recent_lines.append(line)
            recent_lines[:] = recent_lines[-200:]
        process.wait()
        return process.returncode, "".join(recent_lines)


returncode, tail = run_training(train_cmd)
if returncode != 0 and "--batch_size" in train_cmd and "out of memory" in tail.lower():
    print("\nTraining hit OOM with batch_size=2. Retrying with batch_size=1...")
    idx = train_cmd.index("--batch_size")
    train_cmd[idx + 1] = "1"
    returncode, tail = run_training(train_cmd)
if returncode != 0:
    print(f"\nTraining failed with return code {returncode}.")
    raise subprocess.CalledProcessError(returncode, train_cmd)

subprocess.run([
    "python", "predict.py",
    "--image_dir", "./public/val/images",
    "--output", f"./models/{VERSION}/val_predictions.json",
    "--checkpoint", f"./models/{VERSION}/best.pth",
    "--tta",
], check=True, cwd=WORK_REPO)

eval_tool = WORK_REPO / "public/tools/evaluate_predictions.py"
score_path = WORK_REPO / "models" / VERSION / "val_score.json"
if eval_tool.exists():
    subprocess.run([
        "python", "public/tools/evaluate_predictions.py",
        "--ground_truth", "./public/annotations/val.json",
        "--predictions", f"./models/{VERSION}/val_predictions.json",
        "--output", f"./models/{VERSION}/val_score.json",
    ], check=True, cwd=WORK_REPO)
    print("\nval_score.json:")
    print(score_path.read_text())
else:
    print("No evaluate_predictions.py found; skipping external evaluation.")

zip_path = Path("/kaggle/working/my_submission.zip")
try:
    (WORK_REPO / "models" / VERSION / "val_predictions.json").unlink()
except FileNotFoundError:
    pass
output_zip = make_training_outputs(WORK_REPO)
make_clean_submission_zip(WORK_REPO, zip_path)

print("\nCreated repo:", WORK_REPO)
print("Best checkpoint:", WORK_REPO / "models" / VERSION / "best.pth")
print("Last checkpoint:", WORK_REPO / "models" / VERSION / "last.pth")
print("Train history:", WORK_REPO / "models" / VERSION / "train_history.jsonl")
print("Train log:", WORK_REPO / "models" / VERSION / "train.log")
print("Validation score:", score_path if score_path.exists() else "not created")
print("Kaggle output folder:", OUTPUT_ROOT)
print("Kaggle output zip:", output_zip)
print("Submission zip:", zip_path)
