"""
OOD robustness eval (inference only) for a trained backbone: top-1 on ImageNet-R,
ImageNet-Sketch, ImageNet-A, ImageNet-V2, etc.

For ImageNet-R / ImageNet-A, which cover a 200-class SUBSET of ImageNet-1K, pass
--subset: the logits are restricted to the classes present in the val directory
(identified by their wnid folder names), matching standard practice. The 1000-class
ordering is derived from the training directory's sorted wnid folders (torchvision
ImageFolder convention), so no hard-coded class lists are needed.

Run per (model, dataset) on ONE GPU (FlashSwin needs flash-attn / Ampere+):
    python robustness_eval.py \
        --cfg configs/flash_swin/flash_swin_tiny_patch4_window8_256.yaml \
        --checkpoint /.../ckpt_epoch_299.pth --window 8 \
        --train-dir /path/to/data \
        --val-dir /path/to/imagenet-r --subset --tag flash_w8_inr
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build(cfg_path, window, img_size, checkpoint):
    import copy
    from config import _C
    from models import build_model
    cfg = copy.deepcopy(_C); cfg.defrost()
    cfg.merge_from_file(cfg_path); cfg.defrost()
    if window:
        # Iterate every MODEL sub-section that has a WINDOW_SIZE rather than a hardcoded
        # list. The old tuple ("SWIN","SWINV2","FlashSWIN") silently IGNORED --window for
        # number rather than an error -- exactly the way a window-transfer result would
        # be quietly invalid.
        touched = []
        for sec in cfg.MODEL:
            sub = getattr(cfg.MODEL, sec, None)
            if hasattr(sub, "WINDOW_SIZE"):
                sub.WINDOW_SIZE = window
                touched.append(sec)
        if not touched:
            raise SystemExit(f"--window {window} given but no MODEL.* section has "
                             f"WINDOW_SIZE; refusing to run with an unapplied window.")
    if img_size:
        cfg.DATA.IMG_SIZE = img_size
    cfg.TRAIN.USE_CHECKPOINT = False; cfg.FUSED_WINDOW_PROCESS = False
    cfg.freeze()
    model = build_model(cfg)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model.cuda().eval(), cfg.DATA.IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--train-dir", required=True, help="IN-1k train dir (for wnid->index ordering)")
    ap.add_argument("--val-dir", required=True, help="OOD val dir (ImageFolder, wnid subfolders)")
    ap.add_argument("--subset", action="store_true", help="restrict logits to classes present (IN-R/A)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    model, img_size = build(args.cfg, args.window, args.img_size, args.checkpoint)

    # 1000-class wnid -> index (torchvision sorts folder names)
    wnids_1k = sorted(d.name for d in os.scandir(args.train_dir) if d.is_dir())
    wnid2idx = {w: i for i, w in enumerate(wnids_1k)}

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val = datasets.ImageFolder(args.val_dir, tf)                 # val.classes = present folder names (sorted)
    # map each val ImageFolder label -> true 1000-class index.
    # Folders may be wnids (n01234567, e.g. IN-R/A) or integer class ids (0..999, e.g. IN-V2).
    if all(c.startswith("n") and c[1:].isdigit() for c in val.classes):
        val_to_1k = torch.tensor([wnid2idx[w] for w in val.classes], dtype=torch.long)
    else:
        val_to_1k = torch.tensor([int(c) for c in val.classes], dtype=torch.long)  # V2: folder == class idx
    present = val_to_1k.tolist()
    if args.subset:
        keep = torch.tensor(sorted(set(present)), dtype=torch.long).cuda()
        print(f"[subset] restricting to {len(keep)} classes present in {os.path.basename(args.val_dir)}")

    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    val_to_1k = val_to_1k.cuda()
    correct = total = 0
    with torch.no_grad():
        for imgs, y_local in loader:
            imgs = imgs.cuda(non_blocking=True)
            y_true = val_to_1k[y_local.cuda()]                  # -> 1000-class label
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(imgs).float()
            if args.subset:
                masked = torch.full_like(logits, float("-inf"))
                masked[:, keep] = logits[:, keep]
                logits = masked
            pred = logits.argmax(1)
            correct += (pred == y_true).sum().item()
            total += y_true.numel()
    acc = 100.0 * correct / total
    print(f"[{args.tag}] top-1 = {acc:.2f}%   ({correct}/{total}, {len(val)} images)")


if __name__ == "__main__":
    main()
