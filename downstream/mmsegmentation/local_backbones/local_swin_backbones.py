import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mmseg.registry import MODELS


def _repo_root_from_this_file() -> Path:
    # .../SEGMENTATION_PROJECT/mmsegmentation/projects/local_backbones/local_swin_backbones.py
    return Path(__file__).resolve().parents[3]


def _load_class_from_file(file_path: Path, class_name: str):
    module_name = f"_local_swin_{file_path.stem}_{class_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _resolve_swin_root(swin_root: str = None) -> Path:
    if swin_root is not None:
        root = Path(swin_root)
    elif os.environ.get("LOCAL_SWIN_ROOT"):
        root = Path(os.environ["LOCAL_SWIN_ROOT"])
    else:
        root = _repo_root_from_this_file() / "Swin-Transformer"
    if not root.exists():
        raise FileNotFoundError(f"Swin root not found: {root}")
    return root


def _load_checkpoint_state_dict(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")


@MODELS.register_module()
class LocalSwinBackbone(nn.Module):
    """Wrapper that ports local Swin/Flash-Swin variants into OpenMMLab backbones.

    Supported `variant` values:
      - swin
      - swinv2
      - flash_swin
    """

    _VARIANT_TO_FILE_CLASS: Dict[str, Tuple[str, str]] = {
        # One backbone. The shipped flash_swin_transformer.py is the SIZE-ADAPTIVE
        # implementation: it pads arbitrary H,W to the patch and window multiples, so it
        # serves square classification crops and non-square dense-task crops alike.
        "flash_swin": ("flash_swin_transformer.py", "FlashSwinTransformer"),
    }

    def __init__(
        self,
        variant: str = "swin",
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
        pretrained_window_sizes: Tuple[int, int, int, int] = (0, 0, 0, 0),
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        out_norm: bool = True,
        checkpoint_path: str = None,
        swin_root: str = None,
        strict_load: bool = False,
        **kwargs,
    ):
        super().__init__()
        if variant not in self._VARIANT_TO_FILE_CLASS:
            raise ValueError(f"Unsupported variant: {variant}")

        self.variant = variant
        self.out_indices = tuple(out_indices)
        self.out_norm_enabled = out_norm
        self.strict_load = strict_load

        swin_root_path = _resolve_swin_root(swin_root)
        models_dir = swin_root_path / "models"
        file_name, class_name = self._VARIANT_TO_FILE_CLASS[variant]
        model_class = _load_class_from_file(models_dir / file_name, class_name)

        model_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )
        if "swinv2" in variant:
            model_kwargs["pretrained_window_sizes"] = pretrained_window_sizes
        else:
            model_kwargs["fused_window_process"] = fused_window_process

        model_kwargs.update(kwargs)
        self.backbone = model_class(**model_kwargs)

        # Classification tail is not used by dense tasks; freeze it to avoid DDP unused-param errors.
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = False

        self.stage_dims = [int(embed_dim * (2 ** i)) for i in range(len(depths))]
        self.out_norms = nn.ModuleList()
        for i, dim in enumerate(self.stage_dims):
            self.out_norms.append(nn.LayerNorm(dim) if out_norm else nn.Identity())

        if checkpoint_path:
            self.init_weights(checkpoint_path)

    def init_weights(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            return
        state_dict = _load_checkpoint_state_dict(checkpoint_path)

        # Buffers that are DETERMINISTIC functions of window size / input resolution
        # (not learned weights) and must be dropped + re-derived when window or
        # resolution changes across pretrain -> finetune -> seg/det, mirroring
        # Swin-Transformer/utils.py load_pretrained. strict=False only tolerates
        # missing/unexpected KEYS, NOT shape mismatches on keys present in both --
        # a mismatched Swin-V1/SwinV2 relative_position_index or a FlashSwin
        # rope_module.pos otherwise crashes load_state_dict with a RuntimeError.
        _DROP_SUFFIXES = (
            "attn_mask",
            "relative_position_index",       # Swin V1 / SwinV2
            "relative_coords_table",         # SwinV2
            "rope_module.pos",               # FlashSwin
        )
        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ["module.", "backbone.", "encoder."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            if nk.startswith("head."):
                continue
            if nk.endswith(_DROP_SUFFIXES):
                continue
            cleaned[nk] = v

        # FlashBias's q_bias_pos / k_bias_pos are LEARNED parameters shaped
        # (1, M^2, heads, R), indexed by position WITHIN a window. They are the one
        # window-dependent tensor here that cannot be dropped and re-derived -- doing so
        # discards the trained bias. Their shape changes whenever the EFFECTIVE window
        # changes, which a resolution change triggers because the window clamps to the
        # stage grid (256px stage 2 -> grid 16, N=256; 512px stage 2 -> grid 32, N=1024).
        # Resize as Swin resizes relative_position_bias_table: bicubic over the M x M
        # layout, so the learned structure carries across instead of being lost.
        model_sd = self.backbone.state_dict()
        resized = []
        for k in list(cleaned.keys()):
            if not k.endswith(("q_bias_pos", "k_bias_pos")) or k not in model_sd:
                continue
            v, tgt = cleaned[k], model_sd[k]
            if v.shape == tgt.shape:
                continue
            n_src, n_dst = v.shape[1], tgt.shape[1]
            m_src, m_dst = int(round(n_src ** 0.5)), int(round(n_dst ** 0.5))
            if m_src * m_src != n_src or m_dst * m_dst != n_dst or v.shape[2:] != tgt.shape[2:]:
                raise ValueError(f"cannot resize {k}: {tuple(v.shape)} -> {tuple(tgt.shape)}")
            import torch.nn.functional as _F
            h, r = v.shape[2], v.shape[3]
            x = v.reshape(m_src, m_src, h * r).permute(2, 0, 1).unsqueeze(0)
            x = _F.interpolate(x.float(), size=(m_dst, m_dst), mode="bicubic",
                               align_corners=False)
            cleaned[k] = x.squeeze(0).permute(1, 2, 0).reshape(1, n_dst, h, r).to(v.dtype)
            resized.append((m_src, m_dst))
        if resized:
            byshape = {}
            for a, b in resized:
                byshape[(a, b)] = byshape.get((a, b), 0) + 1
            for (a, b), n in sorted(byshape.items()):
                print(f"[LocalSwinBackbone] bicubic-resized {n} FlashBias factor tensors "
                      f"from window {a} to {b}")

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=self.strict_load)
        if not self.strict_load:
            print(f"[LocalSwinBackbone] loaded {checkpoint_path}")
            print(f"[LocalSwinBackbone] missing keys: {len(missing)}")
            print(f"[LocalSwinBackbone] unexpected keys: {len(unexpected)}")

    def forward(self, x):
        # Size-adaptive variants ingest arbitrary input resolution natively.
        if hasattr(self.backbone, "forward_stages"):
            feats = self.backbone.forward_stages(x)
            outs = []
            for i, (tokens, H, W) in enumerate(feats):
                if i in self.out_indices:
                    t = self.out_norms[i](tokens)
                    B, L, C = t.shape
                    out = t.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)
            return tuple(outs)

        _, _, in_h, in_w = x.shape
        target_h, target_w = tuple(self.backbone.patch_embed.img_size)

        # Keep local Swin block resolutions/masks consistent by enforcing fixed backbone input size.
        if (in_h, in_w) != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        self.backbone.patch_embed.patches_resolution = [
            target_h // self.backbone.patch_embed.patch_size[0],
            target_w // self.backbone.patch_embed.patch_size[1],
        ]
        self.backbone.patch_embed.num_patches = (
            self.backbone.patch_embed.patches_resolution[0] * self.backbone.patch_embed.patches_resolution[1]
        )

        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        H, W = self.backbone.patch_embed.patches_resolution
        outs = []

        for i, layer in enumerate(self.backbone.layers):
            for blk in layer.blocks:
                if layer.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            x_out = self.out_norms[i](x)
            if i in self.out_indices:
                B, L, C = x_out.shape
                assert L == H * W, f"Stage {i}: token length {L} != H*W {H*W}"
                out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

            if layer.downsample is not None:
                x = layer.downsample(x)
                H, W = H // 2, W // 2

        return tuple(outs)
