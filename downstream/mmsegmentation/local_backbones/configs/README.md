# Local Swin/Flash-Swin Backbones

This project package registers `LocalSwinBackbone` so you can use your local
`/Swin-Transformer/models` implementations directly in OpenMMLab pipelines.

## Import Hook

Add to config:

```python
custom_imports = dict(
    imports=['projects.local_backbones'],
    allow_failed_imports=False,
)
```

## Backbone Type

Use:

```python
backbone=dict(
    type='LocalSwinBackbone',
    variant='flash_swin',
    # ...
)
```

Supported `variant` values:
- `swin`
- `swinv2`
- `flash_swin`
- `flash_swin_v2`
- `flash_swin_v2_qknorm`
- `flash_swin_v3`
- `flash_swin_wo_overlaptoken_with_rope`
- `flash_swin_wo_overlaptoken_wo_rope`

## Your Configuration Matrix

```text
Swin V1-T:
  window=7, patch=4, embed_dim=96, img=224
  window=8, patch=4, embed_dim=96, img=256

Swin V2-T:
  window=8, patch=4, embed_dim=96, img=256
  window=16, patch=4, embed_dim=96, img=256

FlashSwin-T:
  window=8, patch=4, embed_dim=96, img=256
  window=16, patch=4, embed_dim=96, img=256
  window=32, patch=4, embed_dim=96, img=256
  window=32, patch=8, embed_dim=96, img=512
  window=32, patch=4, embed_dim=96, img=512
  window=32, patch=2, embed_dim=96, img=256
```

## Notes

- Set `checkpoint_path='...'` in backbone config to load your local pretrained checkpoint.
- Wrapper strips common prefixes (`module.`, `backbone.`, `encoder.`) and drops `head.*`.
- For Flash variants, make sure `flash-attn` is installed in the active env.
