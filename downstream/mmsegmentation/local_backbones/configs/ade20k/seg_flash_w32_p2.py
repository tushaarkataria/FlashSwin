# AUTO-GENERATED -- ADE20K UPerNet, 160k, adaptive(keep_ratio)+TTA eval.
# Backbone: flash_swin, window 32, patch 2; init from IN-1K classification checkpoint.
_base_ = ['../../../../configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(imports=['projects.local_backbones'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='LocalSwinBackbone',
        variant='flash_swin',
        img_size=512,                 # seg crop; 256-trained ckpt loads (same window; attn_mask/rope rebuilt)
        patch_size=2,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=32,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=True,          # seg @512 memory; numerically identical
        fused_window_process=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        out_indices=(0, 1, 2, 3),
        out_norm=True,
        checkpoint_path='/path/to/scratch',
        strict_load=False,
        swin_root='/path/to',
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150),
)
