# AUTO-GENERATED -- COCO Mask R-CNN 1x, flash_swin_adaptive, window 32, patch 4.
# Init from IN-1K classification checkpoint. Inherits the 1x template (512x512 pad
# pipeline + FPN); only the backbone and the pyramid strides are swapped.
_base_ = ['../mmdet_local_swin_template.py']

model = dict(
    backbone=dict(
        variant='flash_swin',
        img_size=512,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=32,
        use_checkpoint=True,
        checkpoint_path='/path/to/scratch',
        strict_load=False,
        swin_root='/path/to',
    ),
    # STRIDES MUST FOLLOW THE STEM. The backbone's four outputs are at strides
    # patch * 2^i, so patch 4 gives (4,8,16,32) and patch 2 gives (2,4,8,16).
    # The template hardcodes the patch-4 values; leaving them in place for a patch-2
    # backbone is silent -- channel dims are identical either way, so nothing raises --
    # but every stride-dependent component then runs at 2x the wrong scale: RPN anchor
    # placement, RoIAlign for the box AND mask heads, and box decoding. The first
    # det_flash_w32_p2 run (bbox 37.8 / mask 32.4) was invalidated by exactly this;
    # verified with a forward pass, which printed (2,4,8,16) against a config
    # declaring [4,8,16,32].
    # NOTE: mmdet sizes anchors as octave_base_scale * stride, so a patch-2 model also
    # gets proportionally smaller anchors. That is the right pairing for a finer
    # pyramid, but it means a patch-2 row differs from the patch-4 rows in anchor
    # distribution as well as backbone -- state this when reporting it.
    rpn_head=dict(anchor_generator=dict(strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[4, 8, 16, 32]),
        mask_roi_extractor=dict(featmap_strides=[4, 8, 16, 32])),
)
