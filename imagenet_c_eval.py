"""
ImageNet-C robustness eval: per-corruption top-1, mean corruption error, and mCE.

Layout expected (standard ImageNet-C):  <root>/<corruption>/<severity 1..5>/<wnid>/*.JPEG
Auto-detects which of the 15 standard corruptions are present, so it can be run on
a partial download and re-run when complete. Full 1000-class eval (wnid->index from
the training dir's sorted folders).

Metrics:
  * per-corruption mean top-1 error (over available severities)
  * mean Corruption Error (mCE_raw): unnormalized mean error over all corruptions
    -- protocol-independent, the clean number for a relative FlashSwin-vs-Swin comparison
  * mCE: error normalized by AlexNet's per-corruption error (Hendrycks & Dietterich,
    2019), averaged over the 15 corruptions -- the standard headline metric. The
    AlexNet constants assume the standard 224 eval protocol, so treat as approximate
    when models are evaluated at other resolutions.

Run per model on ONE GPU (FlashSwin needs flash-attn / Ampere+):
    python imagenet_c_eval.py \
        --cfg configs/flash_swin/flash_swin_tiny_patch4_window8_256.yaml \
        --checkpoint /.../ckpt_epoch_299.pth --window 8 \
        --train-dir /path/to/data \
        --c-root /path/to/scratch --tag flash_w8
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 15 standard corruptions, grouped
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise",
               "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
               "snow", "frost", "fog", "brightness",
               "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
# AlexNet mean top-1 error per corruption (over 5 severities), Hendrycks 2019 -> mCE norm
ALEXNET_ERR = {"gaussian_noise": 0.886, "shot_noise": 0.894, "impulse_noise": 0.923,
               "defocus_blur": 0.820, "glass_blur": 0.826, "motion_blur": 0.786,
               "zoom_blur": 0.798, "snow": 0.867, "frost": 0.827, "fog": 0.819,
               "brightness": 0.565, "contrast": 0.853, "elastic_transform": 0.646,
               "pixelate": 0.718, "jpeg_compression": 0.607}


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


@torch.no_grad()
def eval_dir(model, wnid2idx, path, img_size, batch):
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(path, tf)
    to_1k = torch.tensor([wnid2idx[w] for w in ds.classes]).cuda()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    correct = total = 0
    for imgs, y in loader:
        imgs = imgs.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(imgs).float().argmax(1)
        correct += (pred == to_1k[y.cuda()]).sum().item()
        total += y.numel()
    return correct / total                              # accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--c-root", required=True, help="dir containing <corruption>/<severity>/ subdirs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    model, img_size = build(args.cfg, args.window, args.img_size, args.checkpoint)
    wnids = sorted(d.name for d in os.scandir(args.train_dir) if d.is_dir())
    wnid2idx = {w: i for i, w in enumerate(wnids)}

    per_corr_err = {}
    for c in CORRUPTIONS:
        cdir = os.path.join(args.c_root, c)
        sevs = [s for s in ("1", "2", "3", "4", "5") if os.path.isdir(os.path.join(cdir, s))]
        if not sevs:
            continue
        errs = []
        for s in sevs:
            acc = eval_dir(model, wnid2idx, os.path.join(cdir, s), img_size, args.batch_size)
            errs.append(1.0 - acc)
        per_corr_err[c] = float(np.mean(errs))
        print(f"[{args.tag}] {c:18s} sev={len(sevs)}  mean top-1 err = {100*per_corr_err[c]:.2f}%")

    if not per_corr_err:
        print("No corruption dirs found under", args.c_root); return
    raw = 100 * np.mean(list(per_corr_err.values()))
    print(f"\n[{args.tag}] corruptions evaluated: {len(per_corr_err)}/15")
    print(f"[{args.tag}] mean Corruption Error (unnormalized) = {raw:.2f}%   "
          f"(mean corruption top-1 = {100 - raw:.2f}%)")
    # mCE (normalized by AlexNet), over the corruptions present
    ces = [per_corr_err[c] / ALEXNET_ERR[c] for c in per_corr_err]
    print(f"[{args.tag}] mCE (AlexNet-normalized, {len(ces)} corruptions) = {100*np.mean(ces):.2f}"
          f"   (lower is better; 100 = AlexNet)")


if __name__ == "__main__":
    main()
