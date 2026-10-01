_base_ = ['../../../configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py']

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
    neck=dict(in_channels=[96, 192, 384, 768]),
)

# Pipeline and backbone img_size must match so FPN features align with image
# space. All images are resized (keep_ratio=True) then padded to 512×512.
# Override img_size in backbone cfg-options to change resolution consistently.
_img_size = 512

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(_img_size, _img_size), keep_ratio=True),
    dict(type='Pad', size=(_img_size, _img_size),
         pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(_img_size, _img_size), keep_ratio=True),
    dict(type='Pad', size=(_img_size, _img_size),
         pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# Same preset matrix as mmseg template; switch `variant/img_size/patch_size/window_size`.
