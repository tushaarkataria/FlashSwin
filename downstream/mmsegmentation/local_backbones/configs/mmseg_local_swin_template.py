_base_ = ['../../../configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(
    imports=['projects.local_backbones'],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='LocalSwinBackbone',
        variant='flash_swin',
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        out_indices=(0, 1, 2, 3),
        out_norm=True,
        checkpoint_path=None,
        strict_load=False,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(in_channels=384),
)

# Your frequently used presets (change model.backbone values above):
# 1) Swin V1-T:  window=7, patch=4, embed=96, img=224
# 2) Swin V1-T:  window=8, patch=4, embed=96, img=256
# 3) Swin V2-T:  variant='swinv2', window=8,  patch=4, embed=96, img=256
# 4) Swin V2-T:  variant='swinv2', window=16, patch=4, embed=96, img=256
# 5) FlashSwin-T: variant='flash_swin', window=8/16/32, patch=4, embed=96, img=256
# 6) FlashSwin-T: variant='flash_swin', window=32, patch=8, embed=96, img=512
# 7) FlashSwin-T: variant='flash_swin', window=32, patch=4, embed=96, img=512
# 8) FlashSwin-T: variant='flash_swin', window=32, patch=2, embed=96, img=256
