# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# FlashSwin release: this file is deliberately much shorter than the research version it comes
# from. That one dispatches a dozen backbones -- baselines, controls and exploratory variants --
# because it had to reproduce an ablation table. A code release has one job, so there is one
# model here.
import torch.nn as nn

from .flash_swin_transformer import FlashSwinTransformer


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type != 'flash_swin':
        raise ValueError(
            f"this release builds FlashSwin only, got MODEL.TYPE={model_type!r}. "
            f"Baselines (Swin, SwinV2) live upstream at "
            f"https://github.com/microsoft/Swin-Transformer.")

    # LayerNorm in fp32 under AMP. Swin's recipe depends on this: in fp16 the normaliser
    # underflows at depth and training diverges late, long after the run looks healthy.
    layernorm = nn.LayerNorm
    if config.FUSED_LAYERNORM:
        try:
            from apex.normalization import FusedLayerNorm
            layernorm = FusedLayerNorm
        except ImportError:
            pass

    C = config.MODEL.FlashSWIN
    return FlashSwinTransformer(
        img_size=config.DATA.IMG_SIZE,
        patch_size=C.PATCH_SIZE,
        in_chans=C.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=C.EMBED_DIM,
        depths=C.DEPTHS,
        num_heads=C.NUM_HEADS,
        window_size=C.WINDOW_SIZE,
        mlp_ratio=C.MLP_RATIO,
        qkv_bias=C.QKV_BIAS,
        qk_scale=C.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=C.APE,
        norm_layer=layernorm,
        patch_norm=C.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
