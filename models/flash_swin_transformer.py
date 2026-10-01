# --------------------------------------------------------
# Swin Transformer (size-adaptive variant)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
#
# Size-adaptive fork of flash_swin_transformer.py.
# Differences from the fixed-size original:
#   * PatchEmbed accepts arbitrary input H,W (pads to a multiple of patch size)
#     and returns the actual feature-map resolution.
#   * SwinTransformerBlock / PatchMerging take (H, W) at forward time and pad
#     feature maps to a multiple of the window size, so any input resolution
#     works while feature-map strides stay exactly 4/8/16/32.
#   * Per-block window_size / shift_size are still fixed at __init__ (clamped
#     from the configured img_size), so they match the values used during
#     training -> the checkpoint loads with NO parameter-shape changes.
#   * RoPE is unchanged: it is window-local and its only learned parameter
#     (log_inv_freq) is resolution-independent.
#   * The shifted-window attn_mask is intentionally a no-op here, exactly as in
#     the original (the FlashAttention path ignores it), so behaviour matches
#     the trained checkpoints.
#   * forward_stages() exposes per-stage (tokens, H, W) for dense-task backbones.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image (padded)
        W (int): Width of image (padded)

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def rotate_every_two(x):
    # x: (..., dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x, sin, cos):
    # x: (..., dim)
    # sin, cos: broadcastable to x[..., :dim]
    x_rot = (x * cos) + (rotate_every_two(x) * sin)
    return x_rot


class LearnableFreqRoPE(nn.Module):
    def __init__(self, rotary_dim, max_pos, device=None):
        """
        rotary_dim: even number of rotary dims (per head)
        max_pos: maximum coordinate (window size)
        """
        super().__init__()
        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim
        self.max_pos = max_pos
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dim_idx = torch.arange(0, rotary_dim // 2, device=device).float()  # pairs
        init_inv_freq = 1.0 / (10000 ** (dim_idx / (rotary_dim // 2)))
        self.log_inv_freq = nn.Parameter(init_inv_freq.log())  # (rotary_dim//2,)

        pos = torch.arange(max_pos, device=device).float()  # (max_pos,)
        self.register_buffer('pos', pos)

    def build_sin_cos_for_grid(self, H, W):
        inv_freq = self.log_inv_freq.exp()  # (rotary_dim//2,)
        px = torch.einsum('p,d->pd', self.pos[:W], inv_freq)   # (W, dim/2)
        py = torch.einsum('p,d->pd', self.pos[:H], inv_freq)   # (H, dim/2)
        sin_x = torch.sin(px)
        cos_x = torch.cos(px)
        sin_y = torch.sin(py)
        cos_y = torch.cos(py)
        sin = torch.zeros(H, W, self.rotary_dim, device=sin_x.device, dtype=sin_x.dtype)
        cos = torch.zeros_like(sin)
        dim_half = self.rotary_dim // 2
        sin[:, :, :dim_half] = sin_y.unsqueeze(1).expand(-1, W, -1)
        sin[:, :, dim_half:] = sin_x.unsqueeze(0).expand(H, -1, -1)
        cos[:, :, :dim_half] = cos_y.unsqueeze(1).expand(-1, W, -1)
        cos[:, :, dim_half:] = cos_x.unsqueeze(0).expand(H, -1, -1)
        return sin, cos  # (H, W, rotary_dim)

    def forward_build(self, H, W):
        return self.build_sin_cos_for_grid(H, W)


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) with RoPE, FlashAttention backend. """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_out_probability = proj_drop

        self.softmax = nn.Softmax(dim=-1)
        self.rotary_dim = self.head_dim  # apply to whole head by default
        self.rope_module = LearnableFreqRoPE(self.rotary_dim, self.window_size[0])

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)  with N == window_size*window_size
            mask: ignored (kept for signature compatibility with the original)
        """
        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, head_dim)

        sin, cos = self.rope_module.forward_build(self.window_size[0], self.window_size[1])
        sin = sin.reshape(1, 1, N, self.rotary_dim).to(q.device)
        cos = cos.reshape(1, 1, N, self.rotary_dim).to(q.device)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)
        q = q.to(torch.bfloat16).permute(0, 2, 1, 3)
        k = k.to(torch.bfloat16).permute(0, 2, 1, 3)
        v = v.to(torch.bfloat16).permute(0, 2, 1, 3)

        out = flash_attn_func(
            q, k, v,
            dropout_p=self.drop_out_probability if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        x = out.reshape(B_, N, C).float()
        x = self.proj(x)
        x = self.proj_drop(x).to(torch.float32)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block (size-adaptive forward). """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # Clamp window/shift at construction from the configured resolution so
        # they match the values used during training (-> checkpoint compatible).
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # attn_mask is a no-op in the FlashAttention path; kept None for parity.
        self.fused_window_process = fused_window_process

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        ws = self.window_size
        shift = self.shift_size

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature map so H, W are multiples of the window size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # cyclic shift
        if shift > 0:
            shifted_x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, ws)               # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, ws * ws, C)                # nW*B, ws*ws, C

        # W-MSA / SW-MSA (mask is a no-op, matching the trained checkpoints)
        attn_windows = self.attn(x_windows, mask=None)
        attn_windows = attn_windows.view(-1, ws, ws, C)

        # merge windows
        shifted_x = window_reverse(attn_windows, ws, Hp, Wp)      # B, Hp, Wp, C

        # reverse cyclic shift
        if shift > 0:
            x = torch.roll(shifted_x, shifts=(shift, shift), dims=(1, 2))
        else:
            x = shifted_x

        # remove padding
        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer (size-adaptive forward). """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        returns: x_merged, H_out, W_out
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # pad to even spatial dims
        pad_b = H % 2
        pad_r = W % 2
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            H, W = H + pad_b, W + pad_r

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage (size-adaptive forward). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, use_reentrant=False)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding (size-adaptive forward). """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        kernel_size = 2 * patch_size[0] - 1
        padding = kernel_size // 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=patch_size[0],
            padding=padding,
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        # pad input so H, W are multiples of the patch size
        pad_b = (ph - H % ph) % ph
        pad_r = (pw - W % pw) % pw
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.proj(x)                      # B, embed_dim, Ho, Wo
        Ho, Wo = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)      # B, Ho*Wo, C
        if self.norm is not None:
            x = self.norm(x)
        return x, Ho, Wo


class FlashSwinTransformer(nn.Module):
    r""" Size-adaptive FlashSwin Transformer.

    Construct with the SAME img_size used during training so per-stage window
    sizes (and RoPE buffers) match the checkpoint; inputs of any resolution are
    then accepted at forward time.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_stages(self, x):
        """Return a list of (tokens B,L,C, H, W) at each stage's resolution,
        taken BEFORE that stage's downsample. Used by dense-task backbones."""
        x, H, W = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        for layer in self.layers:
            for blk in layer.blocks:
                if layer.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, H, W, use_reentrant=False)
                else:
                    x = blk(x, H, W)
            outs.append((x, H, W))
            if layer.downsample is not None:
                x, H, W = layer.downsample(x, H, W)
        return outs

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
