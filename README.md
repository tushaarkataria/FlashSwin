# FlashSwin

**Unlocking Large Windows and Dense Tokens in Swin Vision Transformers with Memory Efficient Attention**

Tushar Kataria, Gerald Sabin, Ponnuswamy Sadayappan, Shireen Y. Elhabian
Scientific Computing and Imaging Institute & Kahlert School of Computing, University of Utah · RNET Technologies

Standard windowed attention materialises an `M² × M²` score matrix per window, costing `O(M⁴)`
memory, and Swin's learned relative-position bias is added elementwise to that matrix — so the
bias is exactly the term that keeps the fused kernel from being usable. FlashSwin removes the
score matrix with a FlashAttention kernel, restores position as a **window-local learnable 2D
RoPE** applied to queries and keys, and replaces the disjoint patch stem with an overlapping one.
Peak training memory becomes nearly flat in window size: **12.4 GB at a 32×32 window against
70.4 GB for SwinV2-T**, which is what makes the dense-token, wide-window regime trainable at all.

At matched compute (5.9 GFLOPs) FlashSwin-T reaches **82.3%** ImageNet-1K against 81.9% for both
Swin-T and SwinV2-T. Scaling the same ≈28M backbone to `p=2, M=32` reaches **84.1%**, and the
gains are larger under distribution shift (**42.8%** ImageNet-A) and on boundary-sensitive dense
prediction than on clean accuracy.

> **Scope.** This repository is the FlashSwin model and the code to train, evaluate and deploy it.
> It is not a reproduction harness, so the paper's baselines, controls and ablation arms are not
> here: Swin and SwinV2 live upstream at
> [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

## One model, any input size

`models/flash_swin_transformer.py` is the **size-adaptive** implementation. It pads arbitrary
`H, W` up to the patch and window multiples and exposes `forward_stages()` for dense-task
backbones, so the same file serves 256×256 classification crops and 512×1024 Cityscapes crops.
Per-block `window_size` and `shift_size` are fixed at `__init__`, so checkpoints load with no
parameter-shape changes.

If you compare against the research code, that one has a second, fixed-resolution file. The two
are weight-identical — same 28.27M parameters, no shape mismatch — and differ only in five
`attn_mask` **buffers** that the fixed version stores and this one derives per forward. The fixed
version silently squares non-square inputs, which is why it is not the one shipped.

## Install

```bash
conda create -n flashswin python=3.10 -y && conda activate flashswin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm yacs termcolor einops flash-attn --no-build-isolation
```

**FlashSwin requires Ampere or newer** (sm_80+). The fused kernel has no fallback path, and on
older cards model construction itself fails rather than degrading.

## Train

ImageNet is a standard `train/` + `val/` ImageFolder tree.

```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py \
  --cfg configs/flash_swin/flash_swin_tiny_patch4_window8_256.yaml \
  --data-path /path/to/imagenet --output /path/to/output --batch-size 256
```

**Effective batch is 1024** (4 GPUs × 256), and the learning rate is scaled from it as
`BASE_LR × batch × world / 512 × accumulation`. Changing the batch without compensating with
`--accumulation-steps` changes the learning rate, and the numbers will not reproduce.

| Config | Window | Patch | ImageNet-1K |
|---|---|---|---|
| `flash_swin_tiny_patch4_window8_256.yaml` | 8 | 4 | 82.3 |
| `flash_swin_tiny_patch4_window16_256.yaml` | 16 | 4 | 82.9 |
| `flash_swin_tiny_patch4_window32_256.yaml` | 32 | 4 | 82.9 |
| **`flash_swin_tiny_patch2_window32_256_emb96.yaml`** | 32 | 2 | **84.1** |

The dense config is the **`_emb96`** one, and it is the only patch-2 config here on purpose: the
research tree also had a `patch2_window32_256.yaml` at `EMBED_DIM 48`, a 7.3M model that is not
the 84.1% row.

### Large (≈197M)

```bash
# 1. IN-22K pretrain at 192², patch 2, window 32, 90 epochs
python -m torch.distributed.launch --nproc_per_node 8 main.py \
  --cfg configs/flash_swin/flash_swin_large_patch2_window32_192_22k.yaml --data-path /path/to/in22k
# 2. 22K -> 1K finetune (runs at 256²)
python -m torch.distributed.launch --nproc_per_node 8 main.py \
  --cfg configs/flash_swin/flash_swin_large_patch2_window32_256_22kto1k_finetune_fixedmap.yaml \
  --pretrained /path/to/in22k_checkpoint.pth --data-path /path/to/imagenet
```

**The class map matters.** `data/` ships two 22K→1K maps and they disagree in **281 of 1000
slots**: `map22kto1k.txt` follows Microsoft's class convention and is right for *their*
checkpoints, while `map22kto1k_local.txt` matches a sorted-21841 IN-22K copy and is the one our
pretrain used. The `_fixedmap` config selects the local map. Picking the wrong one costs accuracy
silently, with no error.

At 192² with patch 2, stage 1 is a 96² token grid — 4× the tokens of a patch-4 backbone at the
same input resolution.

## Evaluate

```bash
python image_net_v2_evaluation.py --cfg <config> --resume <ckpt> --data-path /path/to/imagenet-v2
python robustness_eval.py         --cfg <config> --resume <ckpt>   # ImageNet-A / -R / Sketch
python imagenet_c_eval.py         --cfg <config> --resume <ckpt>   # mCE
```

ImageNet-A and -R are scored over their own 200-class subsets. ImageNet-C uses the released 224²
images, so the resulting 1.30× upsampling makes the absolute mCE incomparable to the 224 protocol
— applied identically to every model.

The memory and throughput benchmarks and the effective-receptive-field analysis are not part of
this release: they exist to compare backbones against the paper's baselines, which are not here.

## Downstream

`downstream/` holds the backbone registration and configs for
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (`ade20k/`, `cityscapes/`) and
[mmdetection](https://github.com/open-mmlab/mmdetection) (`coco/`). Copy each `local_backbones/`
into that repo's `projects/` and point a config's `checkpoint_path` at a classification
checkpoint — **no conversion step**. `LocalSwinBackbone.init_weights()` reads the training
checkpoint as-is: it unwraps `state_dict`/`model`/`module`, strips the window-dependent buffers
(`attn_mask`, `rope_module.pos`, `relative_position_index`) so a backbone at a different window
size still loads, and calls `load_state_dict`.

Configs select the backbone with `variant='flash_swin'`, which resolves to the one model file.
mmdetection and mmsegmentation both expose a top-level `projects` package, so when both are
installed editable, run detection with `PYTHONPATH` set to the mmdetection root or the registry
lookup fails.

Boundary-IoU uses a band of width `0.02d`; the reported BF-score uses a matching tolerance
`τ/d = 0.008`. Both accumulate per-class across the dataset and reduce once at the end, matching
how mIoU is reported.

## Checkpoints

Released separately — see the repository's release page. Weights are not stored in this repo.

## Citation

```bibtex
```

## License and attribution

MIT. This repository is derived from Microsoft's
[Swin-Transformer](https://github.com/microsoft/Swin-Transformer); `LICENSE` retains their
copyright, and the training scaffolding (`main.py`, `optimizer.py`, `lr_scheduler.py`, `data/`)
is theirs, kept close to upstream. FlashAttention is by
[Dao et al.](https://github.com/Dao-AILab/flash-attention)
