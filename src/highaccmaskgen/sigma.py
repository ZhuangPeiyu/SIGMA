from __future__ import annotations

import contextlib
import importlib.util
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESOURCE_ROOT = _REPO_ROOT / "resource"
_DEFAULT_GDINO_SNAPSHOT = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
_DEFAULT_SAM_CKPT = str(_RESOURCE_ROOT / "sam2.1_hiera_small.pt")
_DEFAULT_GDINO_DIR = str(
    _RESOURCE_ROOT / "models--IDEA-Research--grounding-dino-base" / "snapshots" / _DEFAULT_GDINO_SNAPSHOT
)

# ===== Inlined from causal_model.py =====

class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, k=3, s=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class DriftAlignStem(nn.Module):
    """
    Predicts a small dense flow field and aligns the before image to after image.
    """

    def __init__(self, hidden: int = 32, max_flow: float = 0.06):
        super().__init__()
        self.max_flow = max_flow
        self.flow_head = nn.Sequential(
            ConvNormAct(6, hidden, 3, 1),
            ConvNormAct(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 2, 1),
            nn.Tanh(),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow_head(torch.cat([before, after], dim=1)) * self.max_flow
        aligned_before = self._warp(before, flow)
        return aligned_before, flow

    @staticmethod
    def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device),
            torch.linspace(-1.0, 1.0, w, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        sample_grid = base_grid + flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def resolve_hf_hub_cache_dir(explicit: str | None = None) -> str | None:
    """
    Pick a Hugging Face Hub cache directory so timm/HF can reuse downloaded weights.

    Tries, in order: ``explicit`` (and ``explicit/hub``), env ``HF_HUB_CACHE``,
    ``$HF_HOME/hub``, then local ``resource`` fallbacks.
    """
    candidates: list[str] = []
    if explicit:
        e = os.path.expanduser(explicit.strip())
        candidates.append(e)
        candidates.append(os.path.join(e, "hub"))
    v = os.environ.get("HF_HUB_CACHE", "").strip()
    if v:
        candidates.append(os.path.expanduser(v))
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        candidates.append(os.path.join(os.path.expanduser(hf_home), "hub"))
    candidates.append(str(_RESOURCE_ROOT))
    candidates.append(str(_RESOURCE_ROOT / "models--timm--vit_base_patch14_dinov2.lvd142m"))
    candidates.append(str(_RESOURCE_ROOT / "models--Qwen--Qwen2.5-0.5B-Instruct"))

    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.isdir(c):
            return c
    return None


@contextlib.contextmanager
def _hf_hub_cache_env(hub_dir: str | None) -> Iterator[None]:
    if not hub_dir:
        yield
        return
    prev = os.environ.get("HF_HUB_CACHE")
    os.environ["HF_HUB_CACHE"] = hub_dir
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = prev


def _vit_depth(backbone: torch.nn.Module) -> int:
    blocks = getattr(backbone, "blocks", None)
    if blocks is not None:
        return len(blocks)
    d = getattr(backbone, "depth", None)
    if d is not None:
        return int(d)
    return 12


def _scaled_vit_intermediate_indices(depth: int) -> tuple[int, int, int]:
    """
    Match legacy ViT-B/12 indices (2, 5, 11); scale for ViT-S/L and DINOv3 depths.
    """
    if depth < 3:
        raise ValueError(f"Expected ViT depth >= 3, got {depth}")
    # Map block indices proportionally: k * (depth-1) / 11 for k in {2,5,11}
    raw = [int(round(k * (depth - 1) / 11.0)) for k in (2, 5, 11)]
    raw[0] = max(0, min(raw[0], depth - 3))
    raw[1] = max(raw[0] + 1, min(raw[1], depth - 2))
    raw[2] = max(raw[1] + 1, min(raw[2], depth - 1))
    return (raw[0], raw[1], raw[2])


class DINOv2PyramidEncoder(nn.Module):
    """
    timm ViT backbone (DINOv2 / DINOv3 / …) with intermediate block features projected into a pyramid.
    """

    def __init__(
        self,
        model_name: str = "timm/vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        trainable: bool = False,
        out_dims: tuple[int, int, int] = (64, 128, 256),
        hf_hub_cache: str | None = None,
        intermediate_indices: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for DINOv2 encoder. Install with `pip install timm`.") from e

        hub_dir = resolve_hf_hub_cache_dir(hf_hub_cache)

        try:
            with _hf_hub_cache_env(hub_dir):
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    dynamic_img_size=True,
                )
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained DINOv2 weights ({e}). Falling back to random initialization.",
                stacklevel=2,
            )
            with _hf_hub_cache_env(hub_dir):
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False,
                    dynamic_img_size=True,
                )

        if not hasattr(self.backbone, "forward_intermediates"):
            raise RuntimeError(f"Backbone `{model_name}` does not support forward_intermediates.")

        self.patch_size = int(getattr(self.backbone.patch_embed, "patch_size", (14, 14))[0])
        in_dim = int(self.backbone.num_features)
        depth = _vit_depth(self.backbone)
        self.indices = (
            intermediate_indices
            if intermediate_indices is not None
            else _scaled_vit_intermediate_indices(depth)
        )
        self.backbone_trainable = bool(trainable)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dims[0], 1, bias=False),
                    nn.BatchNorm2d(out_dims[0]),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dims[1], 1, bias=False),
                    nn.BatchNorm2d(out_dims[1]),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dims[2], 1, bias=False),
                    nn.BatchNorm2d(out_dims[2]),
                    nn.SiLU(inplace=True),
                ),
            ]
        )
        self._set_trainable(trainable)

    def _set_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = bool(trainable)
        if not trainable:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()
        return self

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # print("x.shape：{}".format(x.shape))
        x_pad, _ = self._pad_to_patch(x)
        # print("x_pad.shape：{}".format(x_pad.shape))
        _, intermediates = self.backbone.forward_intermediates(
            x_pad,
            indices=self.indices,
            return_prefix_tokens=False,
            norm=True,
            output_fmt="NCHW",
        )
        # print("intermediates[0].shape：{}".format(intermediates[0].shape))
        # print("intermediates[1].shape：{}".format(intermediates[1].shape))
        # print("intermediates[2].shape：{}".format(intermediates[2].shape))
        f1 = self.proj[0](intermediates[0])
        f2 = self.proj[1](intermediates[1])
        f3 = self.proj[2](intermediates[2])
        # print("f1.shape：{}".format(f1.shape))
        # print("f2.shape：{}".format(f2.shape))
        # print("f3.shape：{}".format(f3.shape))
        return [f1, f2, f3]


class StructureBranch(nn.Module):
    """
    Learns structure-sensitive differences from high-frequency residuals.
    """

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            ConvNormAct(3, out_channels, 3, 1),
            ResidualBlock(out_channels),
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _edge(self, x: torch.Tensor) -> torch.Tensor:
        xg = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(xg, self.sobel_x, padding=1)
        gy = F.conv2d(xg, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag.repeat(1, 3, 1, 1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        e_before = self._edge(before)
        e_after = self._edge(after)
        d = torch.abs(e_after - e_before)
        return self.proj(d)


class TransportBranch(nn.Module):
    """
    Local transport discrepancy by feature-wise cosine distance.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.refine = nn.Sequential(
            ConvNormAct(1, channels // 2, 3, 1),
            ConvNormAct(channels // 2, channels // 2, 3, 1),
        )

    def forward(self, before_feat: torch.Tensor, after_feat: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        q = F.normalize(self.q_proj(after_feat), dim=1)
        k = F.normalize(self.k_proj(before_feat), dim=1)
        cost = 1.0 - (q * k).sum(dim=1, keepdim=True)
        cost = self.refine(cost)
        return F.interpolate(cost, size=out_size, mode="bilinear", align_corners=False)


class CausalEvidenceRouter(nn.Module):
    def __init__(self, sem_ch: int, str_ch: int, trp_ch: int, out_ch: int = 64):
        super().__init__()
        in_ch = sem_ch + str_ch + trp_ch + 1
        self.gate = nn.Sequential(
            ConvNormAct(in_ch, out_ch, 3, 1),
            nn.Conv2d(out_ch, 3, 1),
        )
        self.sem_proj = nn.Conv2d(sem_ch, out_ch, 1)
        self.str_proj = nn.Conv2d(str_ch, out_ch, 1)
        self.trp_proj = nn.Conv2d(trp_ch, out_ch, 1)
        self.out = nn.Sequential(ResidualBlock(out_ch), ResidualBlock(out_ch))

    def forward(self, sem: torch.Tensor, strc: torch.Tensor, trp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = torch.var(torch.stack([sem.mean(dim=1), strc.mean(dim=1), trp.mean(dim=1)], dim=1), dim=1, keepdim=True)
        gate_logits = self.gate(torch.cat([sem, strc, trp, uncertainty], dim=1))
        gates = torch.softmax(gate_logits, dim=1)
        fused = (
            gates[:, 0:1] * self.sem_proj(sem)
            + gates[:, 1:2] * self.str_proj(strc)
            + gates[:, 2:3] * self.trp_proj(trp)
        )
        return self.out(fused), uncertainty


class FPNBoundaryDecoder(nn.Module):
    """
    Top-down FPN on multi-scale (before||after) features, fused with router logits at full resolution.
    Pyramid spatial layout: coarse (H/4) <- mid (H/2) <- fine (H) on the encoder patch grid.

    If ``mask_aux_ch > 0``, the fused map ``g`` (``head_hidden`` channels) is concatenated with
    ``mask_aux`` along channel before ``mask_head`` only; ``boundary_head`` still uses ``g``.
    """

    def __init__(
        self,
        in_chs: tuple[int, int, int] = (128, 256, 512),
        fpn_dim: int = 128,
        router_ch: int = 64,
        head_hidden: int = 128,
        mask_aux_ch: int = 0,
    ):
        super().__init__()
        c1, c2, c3 = in_chs
        self.mask_aux_ch = int(mask_aux_ch)
        self.lat1 = nn.Conv2d(c1, fpn_dim, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1, bias=False)
        self.smooth3 = ConvNormAct(fpn_dim, fpn_dim, 3, 1)
        self.smooth2 = ConvNormAct(fpn_dim, fpn_dim, 3, 1)
        self.smooth1 = ConvNormAct(fpn_dim, fpn_dim, 3, 1)
        self.fuse_router = nn.Sequential(
            ConvNormAct(fpn_dim + router_ch, head_hidden, 3, 1),
            ResidualBlock(head_hidden),
        )
        mask_in = head_hidden + self.mask_aux_ch
        self.mask_head = nn.Sequential(
            ConvNormAct(mask_in, head_hidden, 3, 1),
            nn.Conv2d(head_hidden, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            ConvNormAct(head_hidden, head_hidden // 2, 3, 1),
            nn.Conv2d(head_hidden // 2, 1, 1),
        )

    @staticmethod
    def _up(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        pyr: list[torch.Tensor],
        router_feat: torch.Tensor,
        out_hw: tuple[int, int],
        mask_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d1, d2, d3 = pyr[0], pyr[1], pyr[2]
        h, w = d1.shape[2], d1.shape[3]
        h4, w4 = max(1, h // 4), max(1, w // 4)
        h2, w2 = max(1, h // 2), max(1, w // 2)

        p3 = self.lat3(F.adaptive_avg_pool2d(d3, (h4, w4)))
        p3 = self.smooth3(p3)

        p2 = self.lat2(F.adaptive_avg_pool2d(d2, (h2, w2)))
        p2 = p2 + self._up(p3, (h2, w2))
        p2 = self.smooth2(p2)

        p1 = self.lat1(d1)
        p1 = p1 + self._up(p2, (h, w))
        p1 = self.smooth1(p1)

        p1_up = self._up(p1, out_hw)
        g = self.fuse_router(torch.cat([p1_up, router_feat], dim=1))
        if self.mask_aux_ch > 0:
            if mask_aux is None:
                raise ValueError("mask_aux is required when mask_aux_ch > 0")
            if mask_aux.shape[1] != self.mask_aux_ch:
                raise ValueError(
                    f"mask_aux has {mask_aux.shape[1]} channels, expected {self.mask_aux_ch}"
                )
            x_mask = torch.cat([g, mask_aux], dim=1)
        else:
            if mask_aux is not None:
                raise ValueError("mask_aux was passed but decoder was built with mask_aux_ch=0")
            x_mask = g
        return self.mask_head(x_mask), self.boundary_head(g)


class SingleScaleBoundaryDecoder(nn.Module):
    """
    Fuse finest-scale MSDM difference features with ``router_feat`` for mask/boundary.

    ``d_feat`` is kept at **encoder / patch-grid resolution** (same spatial size as the finest
    pyramid level). Fusion happens there, then features are upsampled to ``out_hw`` for the heads.
    This matches the FPN idea (avoid fusing 256-channel tensors at full image resolution) and
    reduces activation memory vs. upsampling ``d_feat`` before ``fuse_router``.
    """

    def __init__(
        self,
        in_ch: int = 256,
        router_ch: int = 64,
        head_hidden: int = 128,
        mask_aux_ch: int = 0,
    ):
        super().__init__()
        self.mask_aux_ch = int(mask_aux_ch)
        self.fuse_router = nn.Sequential(
            ConvNormAct(in_ch + router_ch, head_hidden, 3, 1),
            ResidualBlock(head_hidden),
        )
        mask_in = head_hidden + self.mask_aux_ch
        self.mask_head = nn.Sequential(
            ConvNormAct(mask_in, head_hidden, 3, 1),
            nn.Conv2d(head_hidden, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            ConvNormAct(head_hidden, head_hidden // 2, 3, 1),
            nn.Conv2d(head_hidden // 2, 1, 1),
        )

    def forward(
        self,
        d_feat: torch.Tensor,
        router_feat: torch.Tensor,
        out_hw: tuple[int, int],
        mask_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_d, w_d = d_feat.shape[2], d_feat.shape[3]
        r = F.interpolate(
            router_feat, size=(h_d, w_d), mode="bilinear", align_corners=False
        )
        g = self.fuse_router(torch.cat([d_feat, r], dim=1))
        g_up = F.interpolate(g, size=out_hw, mode="bilinear", align_corners=False)
        if self.mask_aux_ch > 0:
            if mask_aux is None:
                raise ValueError("mask_aux is required when mask_aux_ch > 0")
            if mask_aux.shape[1] != self.mask_aux_ch:
                raise ValueError(
                    f"mask_aux has {mask_aux.shape[1]} channels, expected {self.mask_aux_ch}"
                )
            x_mask = torch.cat([g_up, mask_aux], dim=1)
        else:
            if mask_aux is not None:
                raise ValueError("mask_aux was passed but decoder was built with mask_aux_ch=0")
            x_mask = g_up
        return self.mask_head(x_mask), self.boundary_head(g_up)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    d = (2.0 * inter + 1e-6) / (den + 1e-6)
    return 1.0 - d.mean()


def mask_to_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable boundary approximation from binary mask.
    """

    laplace_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    edge = torch.abs(F.conv2d(mask, laplace_kernel, padding=1))
    return (edge > 0.01).float()


def causal_diff_loss(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    w_bce: float = 1.0,
    w_dice: float = 1.0,
    w_boundary: float = 0.4,
    w_smooth: float = 0.02,
) -> torch.Tensor:
    logits = outputs["mask_logits"]
    bnd_logits = outputs["boundary_logits"]
    flow = outputs["flow"]
    bce = F.binary_cross_entropy_with_logits(logits, target_mask)
    d = dice_loss(logits, target_mask)

    target_bnd = mask_to_boundary(target_mask)
    bnd = F.binary_cross_entropy_with_logits(bnd_logits, target_bnd)

    # Flow smoothness keeps the alignment module from introducing noisy warps.
    smooth_x = torch.mean(torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]))
    smooth_y = torch.mean(torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]))
    smooth = smooth_x + smooth_y
    return w_bce * bce + w_dice * d + w_boundary * bnd + w_smooth * smooth

# ===== Inlined from causal_model_new.py =====


class MultiScaleDifferencingModule(nn.Module):
    """
    For each encoder level l, compute d_abs, d_mul, concat with f_o, f_e, and 1x1 project to C.
    f_o = features of original (aligned before), f_e = edited (after).
    """

    def __init__(
        self,
        level_dims: tuple[int, int, int] = (64, 128, 256),
        out_ch: int = 256,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(4 * d, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
                for d in level_dims
            ]
        )

    def forward(self, fb: list[torch.Tensor], fa: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(fb) != len(fa) or len(fb) != len(self.projs):
            raise ValueError("fb, fa, and MSDM levels must align.")
        out: list[torch.Tensor] = []
        for b_l, a_l, proj in zip(fb, fa, self.projs):
            d_abs = torch.abs(b_l - a_l)
            d_mul = b_l * a_l
            x = torch.cat([d_abs, d_mul, b_l, a_l], dim=1)
            out.append(proj(x))
        return out


class PixelDifferenceAuxHead(nn.Module):
    """
    Auxiliary head: D_pixel = |I_o - I_e| (RGB), three 3×3 convs → single-channel map for mask path.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_ch, hidden, 3, 1),
            ConvNormAct(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=False),
        )

    def forward(self, d_pixel: torch.Tensor) -> torch.Tensor:
        return self.net(d_pixel)


class DifferenceRefinementBlock(nn.Module):
    """
    One DRB: (1) spatial self-attention on scale l, (2) cross-scale attention where
    queries are from l and keys/values are other scales interpolated to (H_l, W_l),
    (3) FFN with GELU.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(dim)
        self.norm_ca_q = nn.LayerNorm(dim)
        self.norm_ca_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(xs) != 3:
            raise ValueError("DRB expects exactly three scale tensors.")
        outs: list[torch.Tensor] = []
        for l, x in enumerate(xs):
            b, c, h, w = x.shape
            q_seq = self._to_seq(x)

            t = self.norm_sa(q_seq)
            sa_out, _ = self.self_attn(t, t, t, need_weights=False)
            q_seq = q_seq + self.dropout(sa_out)

            kv_parts: list[torch.Tensor] = []
            for l2, x2 in enumerate(xs):
                if l2 == l:
                    continue
                x2r = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
                kv_parts.append(self._to_seq(x2r))
            kv = torch.cat(kv_parts, dim=1)

            qc = self.norm_ca_q(q_seq)
            kvc = self.norm_ca_kv(kv)
            ca_out, _ = self.cross_attn(qc, kvc, kvc, need_weights=False)
            q_seq = q_seq + self.dropout(ca_out)

            q_seq = q_seq + self.dropout(self.ffn(self.norm_ffn(q_seq)))

            y = q_seq.transpose(1, 2).reshape(b, c, h, w)
            outs.append(y)
        return outs


def _sem_feat_cosine(
    fb: list[torch.Tensor],
    fa: list[torch.Tensor],
    hw: tuple[int, int],
) -> torch.Tensor:
    """Build (B,3,H,W) cosine-dissimilarity stack at full resolution (for cosine sem path)."""
    sem_levels: list[torch.Tensor] = []
    for b_l, a_l in zip(fb, fa):
        cos = F.cosine_similarity(
            F.normalize(a_l, dim=1), F.normalize(b_l, dim=1), dim=1
        ).unsqueeze(1)
        sem = 1.0 - cos
        sem_levels.append(
            F.interpolate(sem, size=hw, mode="bilinear", align_corners=False)
        )
    return torch.cat(sem_levels, dim=1)


class CausalDiffNetNew(nn.Module):
    """
    CausalDiffNet + MSDM (+ optional DRB) + FPN or single-scale decoder.

    Ablation switches (消融表): multi-scale MSDM, DRB 数量, semantic / structural / transport
    evidence branches, Evidence Router, Pixel Aux, and ``SingleScaleBoundaryDecoder`` vs
    ``FPNBoundaryDecoder``. When semantic/structural/transport are all off, fusion uses a
    zero ``router_ch`` map (no trainable evidence fusion).
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_dinov2.lvd142m",
        encoder_pretrained: bool = True,
        encoder_trainable: bool = False,
        msdm_out_ch: int = 256,
        drb_num_heads: int = 8,
        drb_ffn_mult: int = 4,
        drb_dropout: float = 0.0,
        num_drb_blocks: int = 2,
        multiscale_msdm: bool = True,
        sem_source: str = "d_levels",
        use_pixel_aux: bool = True,
        pixel_aux_hidden: int = 32,
        use_sem_branch: bool = True,
        use_structure_branch: bool = True,
        use_transport_branch: bool = True,
        use_evidence_router: bool = True,
        hf_hub_cache: str | None = None,
    ):
        super().__init__()
        self.encoder_level_dims = (64, 128, 256)
        self.msdm_out_ch = int(msdm_out_ch)
        self.multiscale_msdm = bool(multiscale_msdm)
        self.sem_source = str(sem_source).lower().strip()
        if self.sem_source not in ("d_levels", "cosine"):
            raise ValueError("sem_source must be 'd_levels' or 'cosine'")
        self.use_sem_branch = bool(use_sem_branch)
        self.use_structure_branch = bool(use_structure_branch)
        self.use_transport_branch = bool(use_transport_branch)
        self.use_evidence_router = bool(use_evidence_router)
        self._no_evidence_branches = not (
            self.use_sem_branch or self.use_structure_branch or self.use_transport_branch
        )
        if self.use_evidence_router and self._no_evidence_branches:
            raise ValueError("Evidence router requires at least one of semantic/structure/transport branches.")
        n_drb = int(num_drb_blocks)
        if n_drb < 0:
            raise ValueError("num_drb_blocks must be >= 0")
        if not self.multiscale_msdm and n_drb > 0:
            raise ValueError("Single-scale SSD baseline requires num_drb_blocks=0 (no DRB).")

        self.align = DriftAlignStem(hidden=32, max_flow=0.06)
        self.encoder = DINOv2PyramidEncoder(
            model_name=encoder_name,
            pretrained=encoder_pretrained,
            trainable=encoder_trainable,
            out_dims=self.encoder_level_dims,
            hf_hub_cache=hf_hub_cache,
        )
        self.msdm = MultiScaleDifferencingModule(
            level_dims=self.encoder_level_dims,
            out_ch=self.msdm_out_ch,
        )
        self.drb_blocks = nn.ModuleList(
            [
                DifferenceRefinementBlock(
                    dim=self.msdm_out_ch,
                    num_heads=drb_num_heads,
                    ffn_mult=drb_ffn_mult,
                    dropout=drb_dropout,
                )
                for _ in range(n_drb)
            ]
        )
        self.structure = StructureBranch(out_channels=32) if self.use_structure_branch else None
        self.transport = TransportBranch(channels=256) if self.use_transport_branch else None

        if self.sem_source == "cosine" and self.use_sem_branch:
            self.sem_cosine_refine = nn.Sequential(
                ConvNormAct(3, 96, 3, 1),
                ResidualBlock(96),
            )
        else:
            self.sem_cosine_refine = None  # type: ignore[assignment]

        if self.multiscale_msdm and self.sem_source == "d_levels" and self.use_sem_branch:
            self.sem_level_proj = nn.ModuleList(
                [ConvNormAct(self.msdm_out_ch, 32, 1) for _ in range(3)]
            )
            self.sem_refine = nn.Sequential(
                ConvNormAct(96, 96, 3, 1),
                ResidualBlock(96),
            )
        else:
            self.sem_level_proj = None  # type: ignore[assignment]
            self.sem_refine = None  # type: ignore[assignment]

        if not self.multiscale_msdm and self.sem_source == "d_levels" and self.use_sem_branch:
            self.sem_ssd = nn.Sequential(
                ConvNormAct(self.msdm_out_ch, 96, 3, 1),
                ResidualBlock(96),
            )
        else:
            self.sem_ssd = None  # type: ignore[assignment]

        self.router = (
            CausalEvidenceRouter(sem_ch=96, str_ch=32, trp_ch=128, out_ch=64)
            if (self.use_evidence_router and not self._no_evidence_branches)
            else None
        )
        # Router-disabled ablation: directly fuse active branches by concat+conv.
        no_router_in_ch = 0
        if self.use_sem_branch:
            no_router_in_ch += 96
        if self.use_structure_branch:
            no_router_in_ch += 32
        if self.use_transport_branch:
            no_router_in_ch += 128
        self.no_router_fuse = (
            nn.Sequential(
                ConvNormAct(no_router_in_ch, 64, 3, 1),
                ResidualBlock(64),
                ResidualBlock(64),
            )
            if (not self.use_evidence_router and not self._no_evidence_branches)
            else None
        )
        c = self.msdm_out_ch
        self.use_pixel_aux = bool(use_pixel_aux)
        if self.use_pixel_aux:
            self.pixel_diff_head = PixelDifferenceAuxHead(3, int(pixel_aux_hidden))
            mask_aux_ch = 1
        else:
            self.pixel_diff_head = None
            mask_aux_ch = 0

        if self.multiscale_msdm:
            self.decoder = FPNBoundaryDecoder(
                in_chs=(c, c, c),
                fpn_dim=128,
                router_ch=64,
                head_hidden=128,
                mask_aux_ch=mask_aux_ch,
            )
            self.single_scale_decoder = None  # type: ignore[assignment]
        else:
            self.single_scale_decoder = SingleScaleBoundaryDecoder(
                in_ch=c,
                router_ch=64,
                head_hidden=128,
                mask_aux_ch=mask_aux_ch,
            )
            self.decoder = None  # type: ignore[assignment]

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> dict[str, torch.Tensor]:
        aligned_before, flow = self.align(before, after)
        fb = self.encoder(aligned_before)
        fa = self.encoder(after)
        # for b_l, a_l in zip(fb, fa):
        #     print("b_l.shape：{}, a_l.shape：{}".format(b_l.shape, a_l.shape))
            
        hw = before.shape[-2:]
        out_hw = (before.shape[-2], before.shape[-1])

        if self._no_evidence_branches:
            if self.multiscale_msdm:
                d_levels = self.msdm(fb, fa)
                for drb in self.drb_blocks:
                    d_levels = drb(d_levels)
                pyr = d_levels
                fused = pyr[0].new_zeros(pyr[0].shape[0], 64, hw[0], hw[1])
                uncertainty = pyr[0].new_zeros(pyr[0].shape[0], 1, hw[0], hw[1])
                mask_aux = None
                if self.use_pixel_aux and self.pixel_diff_head is not None:
                    d_pixel = torch.abs(aligned_before - after)
                    mask_aux = self.pixel_diff_head(d_pixel)
                assert self.decoder is not None
                if mask_aux is not None:
                    mask_logits, boundary_logits = self.decoder(
                        pyr, fused, out_hw, mask_aux=mask_aux
                    )
                else:
                    mask_logits, boundary_logits = self.decoder(pyr, fused, out_hw)
            else:
                b0, a0 = fb[0], fa[0]
                x0 = torch.cat(
                    [torch.abs(b0 - a0), b0 * a0, b0, a0],
                    dim=1,
                )
                d0 = self.msdm.projs[0](x0)
                fused = d0.new_zeros(d0.shape[0], 64, hw[0], hw[1])
                uncertainty = d0.new_zeros(d0.shape[0], 1, hw[0], hw[1])
                mask_aux = None
                if self.use_pixel_aux and self.pixel_diff_head is not None:
                    d_pixel = torch.abs(aligned_before - after)
                    mask_aux = self.pixel_diff_head(d_pixel)
                assert self.single_scale_decoder is not None
                if mask_aux is not None:
                    mask_logits, boundary_logits = self.single_scale_decoder(
                        d0, fused, out_hw, mask_aux=mask_aux
                    )
                else:
                    mask_logits, boundary_logits = self.single_scale_decoder(d0, fused, out_hw)
            return {
                "mask_logits": mask_logits,
                "boundary_logits": boundary_logits,
                "uncertainty": uncertainty,
                "flow": flow,
            }

        if self.multiscale_msdm:
            d_levels = self.msdm(fb, fa)
            for drb in self.drb_blocks:
                d_levels = drb(d_levels)
            pyr = d_levels

            if self.use_sem_branch:
                if self.sem_source == "d_levels":
                    assert self.sem_level_proj is not None and self.sem_refine is not None
                    sem_parts = [
                        proj(F.interpolate(d_l, size=hw, mode="bilinear", align_corners=False))
                        for d_l, proj in zip(d_levels, self.sem_level_proj)
                    ]
                    sem_feat = self.sem_refine(torch.cat(sem_parts, dim=1))
                else:
                    assert self.sem_cosine_refine is not None
                    sem_feat = self.sem_cosine_refine(_sem_feat_cosine(fb, fa, hw))
            else:
                sem_feat = pyr[0].new_zeros(pyr[0].shape[0], 96, hw[0], hw[1])
        else:
            b0, a0 = fb[0], fa[0]
            x0 = torch.cat(
                [torch.abs(b0 - a0), b0 * a0, b0, a0],
                dim=1,
            )
            d0 = self.msdm.projs[0](x0)

            if self.use_sem_branch:
                if self.sem_source == "d_levels":
                    assert self.sem_ssd is not None
                    d_up_sem = F.interpolate(d0, size=hw, mode="bilinear", align_corners=False)
                    sem_feat = self.sem_ssd(d_up_sem)
                else:
                    assert self.sem_cosine_refine is not None
                    sem_feat = self.sem_cosine_refine(_sem_feat_cosine(fb, fa, hw))
            else:
                sem_feat = d0.new_zeros(d0.shape[0], 96, hw[0], hw[1])

        sem_feat_96 = sem_feat
        if self.use_structure_branch:
            assert self.structure is not None
            str_feat_32 = self.structure(aligned_before, after)
        else:
            str_feat_32 = torch.zeros(
                sem_feat.shape[0], 32, sem_feat.shape[2], sem_feat.shape[3], device=sem_feat.device, dtype=sem_feat.dtype
            )
        if self.use_transport_branch:
            assert self.transport is not None
            trp_feat_128 = self.transport(fb[-1], fa[-1], out_size=before.shape[-2:])
        else:
            trp_feat_128 = torch.zeros(
                sem_feat.shape[0], 128, sem_feat.shape[2], sem_feat.shape[3], device=sem_feat.device, dtype=sem_feat.dtype
            )

        if self.use_evidence_router:
            assert self.router is not None
            fused, uncertainty = self.router(sem_feat_96, str_feat_32, trp_feat_128)
        else:
            assert self.no_router_fuse is not None
            fuse_parts = []
            if self.use_sem_branch:
                fuse_parts.append(sem_feat_96)
            if self.use_structure_branch:
                fuse_parts.append(str_feat_32)
            if self.use_transport_branch:
                fuse_parts.append(trp_feat_128)
            fused = self.no_router_fuse(torch.cat(fuse_parts, dim=1))
            uncertainty = torch.zeros(
                sem_feat.shape[0], 1, sem_feat.shape[2], sem_feat.shape[3], device=sem_feat.device, dtype=sem_feat.dtype
            )

        mask_aux = None
        if self.use_pixel_aux and self.pixel_diff_head is not None:
            d_pixel = torch.abs(aligned_before - after)
            mask_aux = self.pixel_diff_head(d_pixel)

        if self.multiscale_msdm:
            assert self.decoder is not None
            if mask_aux is not None:
                mask_logits, boundary_logits = self.decoder(
                    pyr, fused, out_hw, mask_aux=mask_aux
                )
            else:
                mask_logits, boundary_logits = self.decoder(pyr, fused, out_hw)
        else:
            assert self.single_scale_decoder is not None
            if mask_aux is not None:
                mask_logits, boundary_logits = self.single_scale_decoder(
                    d0, fused, out_hw, mask_aux=mask_aux
                )
            else:
                mask_logits, boundary_logits = self.single_scale_decoder(d0, fused, out_hw)

        return {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "uncertainty": uncertainty,
            "flow": flow,
        }

# Presets for ablation comparisons. Values override YAML/CLI when set.
# Columns: Multi-scale MSDM, DRB blocks, Sem / Struct / Trp branches, Evidence Router,
# Pixel Aux, Decoder (SingleScaleBoundaryDecoder vs FPNBoundaryDecoder — implied by MSDM/DRB).
TABLE1_ABLATIONS: dict[str, dict[str, object]] = {
    # baseline: no multi-scale MSDM, no evidence branches, SingleScaleBoundaryDecoder
    "baseline": {
        "multiscale_msdm": False,
        "num_drb_blocks": 0,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "ssd": {  # alias for baseline (historical name)
        "multiscale_msdm": False,
        "num_drb_blocks": 0,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    # Multi-scale MSDM, no DRB, FPN; no evidence fusion modules
    "msdm_nodrb": {
        "multiscale_msdm": True,
        "num_drb_blocks": 0,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    # MSDM + DRB depth sweep (FPN); no sem/struct/trp/router/pixel
    "msdm_drb2": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "msdm_drb4": {
        "multiscale_msdm": True,
        "num_drb_blocks": 4,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "msdm_drb6": {
        "multiscale_msdm": True,
        "num_drb_blocks": 6,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "msdm_drb8": {
        "multiscale_msdm": True,
        "num_drb_blocks": 8,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": False,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    # Branch ablations (DRB=2); router off until sem_struct_trp_router / all_strategies
    "sem_only": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": False,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "sem_struct": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": False,
        "use_evidence_router": False,
    },
    "sem_struct_trp": {  # alias: same as sem_struct_trp_router
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    "sem_struct_trp_no_router": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": False,
    },
    "sem_struct_trp_router": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    # Legacy / full-model names
    "ours_no_pixel": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": False,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    "full": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    "full_drb8": {
        "multiscale_msdm": True,
        "num_drb_blocks": 8,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    "all_strategies": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
    },
    # all_strategies + alternate timm backbones (encoder_name; HF weights via hf_hub_cache in yaml/CLI)
    "all_strategies_vit_small_patch14_dinov2": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
        "encoder_name": "timm/vit_small_patch14_dinov2.lvd142m",
    },
    "all_strategies_vit_large_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
        "encoder_name": "timm/vit_large_patch16_dinov3.lvd1689m",
    },
    "all_strategies_vit_base_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
        "encoder_name": "timm/vit_base_patch16_dinov3.lvd1689m",
    },
    "all_strategies_vit_small_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "sem_source": "d_levels",
        "use_pixel_aux": True,
        "use_sem_branch": True,
        "use_structure_branch": True,
        "use_transport_branch": True,
        "use_evidence_router": True,
        "encoder_name": "timm/vit_small_patch16_dinov3.lvd1689m",
    },
}


def build_causal_diff_net_new(
    mcfg: dict,
    encoder_pretrained: bool,
) -> CausalDiffNetNew:
    """Build ``CausalDiffNetNew`` from a ``model`` config dict (see ``TABLE1_ABLATIONS``)."""
    preset = (mcfg.get("ablation_preset") or mcfg.get("table1_preset") or "").strip()
    kwargs: dict = dict(
        encoder_name=mcfg.get("encoder_name", "vit_base_patch14_dinov2.lvd142m"),
        encoder_pretrained=encoder_pretrained,
        encoder_trainable=bool(mcfg.get("encoder_trainable", False)),
        msdm_out_ch=int(mcfg.get("msdm_out_ch", 256)),
        drb_num_heads=int(mcfg.get("drb_num_heads", 8)),
        drb_ffn_mult=int(mcfg.get("drb_ffn_mult", 4)),
        drb_dropout=float(mcfg.get("drb_dropout", 0.0)),
        num_drb_blocks=int(mcfg.get("num_drb_blocks", 2)),
        multiscale_msdm=bool(mcfg.get("multiscale_msdm", True)),
        sem_source=str(mcfg.get("sem_source", "d_levels")),
        use_pixel_aux=bool(mcfg.get("use_pixel_aux", True)),
        pixel_aux_hidden=int(mcfg.get("pixel_aux_hidden", 32)),
        use_sem_branch=bool(mcfg.get("use_sem_branch", True)),
        use_structure_branch=bool(mcfg.get("use_structure_branch", True)),
        use_transport_branch=bool(mcfg.get("use_transport_branch", True)),
        use_evidence_router=bool(mcfg.get("use_evidence_router", True)),
        hf_hub_cache=(
            str(mcfg.get("hf_hub_cache", "")).strip() or None
        ),
    )
    if preset:
        if preset not in TABLE1_ABLATIONS:
            raise ValueError(
                f"Unknown ablation_preset={preset!r}; choose one of {list(TABLE1_ABLATIONS)}"
            )
        kwargs.update(TABLE1_ABLATIONS[preset])
    return CausalDiffNetNew(**kwargs)

# ===== Inlined from FFD.py =====

SemanticTransformationParser = None  # type: ignore[assignment]
LangSAMGuidedSegmenter = None  # type: ignore[assignment]
_stp_spec = importlib.util.find_spec("semantic_transformation_parser_module")
if _stp_spec is not None:
    _stp_mod = importlib.util.module_from_spec(_stp_spec)
    assert _stp_spec.loader is not None
    _stp_spec.loader.exec_module(_stp_mod)
    SemanticTransformationParser = getattr(_stp_mod, "SemanticTransformationParser", None)
_ls_spec = importlib.util.find_spec("langsam_guided_module")
if _ls_spec is not None:
    _ls_mod = importlib.util.module_from_spec(_ls_spec)
    assert _ls_spec.loader is not None
    _ls_spec.loader.exec_module(_ls_mod)
    LangSAMGuidedSegmenter = getattr(_ls_mod, "LangSAMGuidedSegmenter", None)

# Fallback: load modules from scripts/SCPModule when not installed on PYTHONPATH.
if SemanticTransformationParser is None or LangSAMGuidedSegmenter is None:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _scp_dir = os.path.join(_repo_root, "scripts", "SCPModule")
    if SemanticTransformationParser is None:
        _stp_file = os.path.join(_scp_dir, "semantic_transformation_parser_module.py")
        if os.path.isfile(_stp_file):
            _stp_file_spec = importlib.util.spec_from_file_location(
                "semantic_transformation_parser_module",
                _stp_file,
            )
            if _stp_file_spec is not None and _stp_file_spec.loader is not None:
                _stp_mod = importlib.util.module_from_spec(_stp_file_spec)
                _stp_file_spec.loader.exec_module(_stp_mod)
                SemanticTransformationParser = getattr(_stp_mod, "SemanticTransformationParser", None)
    if LangSAMGuidedSegmenter is None:
        _ls_file = os.path.join(_scp_dir, "langsam_guided_module.py")
        if os.path.isfile(_ls_file):
            _ls_file_spec = importlib.util.spec_from_file_location(
                "langsam_guided_module",
                _ls_file,
            )
            if _ls_file_spec is not None and _ls_file_spec.loader is not None:
                _ls_mod = importlib.util.module_from_spec(_ls_file_spec)
                _ls_file_spec.loader.exec_module(_ls_mod)
                LangSAMGuidedSegmenter = getattr(_ls_mod, "LangSAMGuidedSegmenter", None)

# Single-scale (``multiscale_msdm=False``) MSDM uses one encoder level: ``0..2`` for ``[f1,f2,f3]``
# in ``DINOv2PyramidEncoder``; use index **2** (``d2``), not the first scale.
_FFD_SSD_MSDM_LEVEL: int = 2


class ActionConditionedSpatialFusion(nn.Module):
    """ACSF: fuse A^- and A^+ by action-conditioned rule (no spatial softmax)."""

    ACTION2ID = {
        "add": 0,
        "remove": 1,
        "replace": 2,
        "attribute": 3,
        "attribute change": 3,
        "global": 4,
    }

    def __init__(self, out_ch: int = 64, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)
        self.proj = nn.Conv2d(1, out_ch, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    @staticmethod
    def _normalize_action(action: str) -> str:
        a = (action or "global").strip().lower()
        if a == "attribute change":
            return "attribute"
        if a not in {"add", "remove", "replace", "attribute", "global"}:
            return "global"
        return a

    def _fuse_single(self, a_minus: torch.Tensor, a_plus: torch.Tensor, action: str) -> torch.Tensor:
        # a_minus/a_plus: [1,H,W]
        tau = self._normalize_action(action)
        if tau == "add":
            return a_plus
        if tau == "remove":
            return a_minus
        if tau in {"replace", "attribute"}:
            return a_minus * a_plus
        return torch.ones_like(a_minus) * self.epsilon

    def forward(
        self,
        a_minus: torch.Tensor,
        a_plus: torch.Tensor,
        action_types: list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a_minus/a_plus: [B,1,H,W] in [0,1]
            action_types: len B
        Returns:
            m_intent: [B,1,H,W]
            e_t0: [B,C',H,W]
        """
        b, _, h, w = a_minus.shape
        if len(action_types) != b:
            raise ValueError(f"len(action_types)={len(action_types)} but batch={b}")
        fused = []
        for i in range(b):
            fused.append(self._fuse_single(a_minus[i], a_plus[i], action_types[i]))
        m_intent = torch.stack(fused, dim=0).view(b, 1, h, w)
        e_t0 = self.relu(self.proj(m_intent))
        return m_intent, e_t0


class ICMVRouter(nn.Module):
    """
    Iterative Cross-Modal Verification router:
    E_v^k = E_v^(k-1) + CrossAttn(Q=E_v^(k-1), K,V=E_t^(k-1))
    E_t^k = E_t^(k-1) + CrossAttn(Q=E_t^(k-1), K,V=E_v^k)
    Z = Proj([E_v^K || E_t^K])
    """

    def __init__(self, c: int = 64, num_heads: int = 4, k_iter: int = 2):
        super().__init__()
        self.c = int(c)
        self.k_iter = int(k_iter)
        self.v_to_t = nn.MultiheadAttention(embed_dim=self.c, num_heads=num_heads, batch_first=True)
        self.t_to_v = nn.MultiheadAttention(embed_dim=self.c, num_heads=num_heads, batch_first=True)
        self.v_norm = nn.LayerNorm(self.c)
        self.t_norm = nn.LayerNorm(self.c)
        self.consensus_proj = nn.Conv2d(self.c * 2, self.c, kernel_size=1, bias=True)

    def forward(
        self,
        e_v0: torch.Tensor,
        e_t0: torch.Tensor,
        process_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        # e_v0/e_t0: [B,C,H,W]
        b, c, h_full, w_full = e_v0.shape
        h_proc, w_proc = (process_hw if process_hw is not None else (h_full, w_full))
        if (h_proc, w_proc) != (h_full, w_full):
            e_v0 = F.interpolate(e_v0, size=(h_proc, w_proc), mode="bilinear", align_corners=False)
            e_t0 = F.interpolate(e_t0, size=(h_proc, w_proc), mode="bilinear", align_corners=False)
        elif e_t0.shape[-2:] != (h_proc, w_proc):
            e_t0 = F.interpolate(e_t0, size=(h_proc, w_proc), mode="bilinear", align_corners=False)
        if e_t0.shape[1] != c:
            raise ValueError(f"ICMV channel mismatch: Ev={c}, Et={e_t0.shape[1]}")

        ev = e_v0.flatten(2).transpose(1, 2)  # [B,N,C]
        et = e_t0.flatten(2).transpose(1, 2)  # [B,N,C]
        for _ in range(self.k_iter):
            dv, _ = self.v_to_t(query=ev, key=et, value=et, need_weights=False)
            ev = self.v_norm(ev + dv)
            dt, _ = self.t_to_v(query=et, key=ev, value=ev, need_weights=False)
            et = self.t_norm(et + dt)

        ev_hw = ev.transpose(1, 2).reshape(b, c, h_proc, w_proc)
        et_hw = et.transpose(1, 2).reshape(b, c, h_proc, w_proc)
        z = self.consensus_proj(torch.cat([ev_hw, et_hw], dim=1))  # [B,C,H,W]
        z_flat = z.flatten(2).transpose(1, 2)  # [B,N,C]
        return {
            "e_v_k": ev_hw,
            "e_t_k": et_hw,
            "z": z,
            "z_flat": z_flat,
            "process_hw": torch.tensor([h_proc, w_proc], device=z.device),
        }


class TextBranch(nn.Module):
    """
    Text branch = (optional parser/langsam wrappers) + ACSF feature extractor.
    Runtime training path expects precomputed A^- / A^+ / action_type in text_inputs.
    """

    def __init__(self, out_ch: int = 64, epsilon: float = 1e-3):
        super().__init__()
        self.acsf = ActionConditionedSpatialFusion(out_ch=out_ch, epsilon=epsilon)
        self.semantic_parser = None
        self.langsam_guided = None
        self._parser_kwargs: dict = {}
        self._langsam_kwargs: dict = {}

    def set_external_module_kwargs(self, parser_kwargs: dict | None = None, langsam_kwargs: dict | None = None) -> None:
        self._parser_kwargs = dict(parser_kwargs or {})
        self._langsam_kwargs = dict(langsam_kwargs or {})

    def build_external_modules(self, parser_kwargs: dict | None = None, langsam_kwargs: dict | None = None) -> None:
        """Optional online modules for raw instruction+image use-cases."""
        if parser_kwargs is not None:
            self._parser_kwargs = dict(parser_kwargs)
        if langsam_kwargs is not None:
            self._langsam_kwargs = dict(langsam_kwargs)
        if SemanticTransformationParser is not None:
            pk = dict(self._parser_kwargs)
            mp = str(pk.get("model_path", "") or "").strip()
            if mp:
                self.semantic_parser = SemanticTransformationParser(**pk)
            else:
                self.semantic_parser = None
        if LangSAMGuidedSegmenter is not None:
            self.langsam_guided = LangSAMGuidedSegmenter(**self._langsam_kwargs)

    @staticmethod
    def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
        """
        Convert [3,H,W] tensor to PIL RGB.
        Works with either [0,1] or normalized ranges by min-max remapping per image.
        """
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W], got {tuple(img.shape)}")
        x = img.detach().cpu().float()
        mn, mx = float(x.min()), float(x.max())
        if mx > 1.0 or mn < 0.0:
            # robust fallback for normalized tensors
            x = (x - mn) / (mx - mn + 1e-6)
        x = x.clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def infer_online(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        instructions: Sequence[str] | str,
        target_hw: tuple[int, int],
        out_device: torch.device,
        out_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Online path: instruction -> parser -> LangSAM -> A-/A+ -> ACSF."""
        if self.semantic_parser is None or self.langsam_guided is None:
            self.build_external_modules()
        if self.langsam_guided is None:
            raise RuntimeError(
                "Online LangSAM path unavailable (langsam_guided_module import/checkpoints). "
                "Or supply offline text_inputs covering the batch."
            )
        if self.semantic_parser is None:
            raise RuntimeError(
                "instruction-based online text branch needs model.parser_model_path "
                "(directory or HF id for SemanticTransformationParser; same as scripts/test_FFD_text_branch*.py "
                "`--parser-model-path` / YAML `parser_model_path`). Without it, rely on ACS offline text_inputs "
                "(`text_inputs=` from build_batch_text_inputs) for every batch item."
            )
        if isinstance(instructions, str):
            instructions = [instructions] * before.shape[0]
        if len(instructions) != before.shape[0]:
            raise ValueError(f"len(instructions)={len(instructions)} but batch={before.shape[0]}")

        a_minus_list: list[torch.Tensor] = []
        a_plus_list: list[torch.Tensor] = []
        actions: list[str] = []
        for i, inst in enumerate(instructions):
            parsed = self.semantic_parser.parse(inst)
            actions.append(parsed.get("action type", "global"))
            b_pil = self._tensor_to_pil(before[i])
            a_pil = self._tensor_to_pil(after[i])
            a_minus_np, a_plus_np = self.langsam_guided.predict_before_after(
                before_image=b_pil,
                after_image=a_pil,
                before_concept=parsed.get("original concept"),
                after_concept=parsed.get("edited concept"),
            )
            a_minus_list.append(torch.from_numpy(a_minus_np).unsqueeze(0))
            a_plus_list.append(torch.from_numpy(a_plus_np).unsqueeze(0))

        a_minus = torch.stack(a_minus_list, dim=0).to(device=out_device, dtype=out_dtype)
        a_plus = torch.stack(a_plus_list, dim=0).to(device=out_device, dtype=out_dtype)
        return self.forward(
            text_inputs={"a_minus": a_minus, "a_plus": a_plus, "action_type": actions},
            target_hw=target_hw,
        )

    def infer_fallback(
        self,
        batch_size: int,
        target_hw: tuple[int, int],
        out_device: torch.device,
        out_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """
        Stable fallback when instruction/text_inputs are missing.
        Use weak global prior via ACSF(global): near-zero uniform intent.
        """
        h, w = target_hw
        a_minus = torch.zeros((batch_size, 1, h, w), device=out_device, dtype=out_dtype)
        a_plus = torch.zeros((batch_size, 1, h, w), device=out_device, dtype=out_dtype)
        actions = ["global"] * batch_size
        return self.forward(
            text_inputs={"a_minus": a_minus, "a_plus": a_plus, "action_type": actions},
            target_hw=target_hw,
        )

    def forward(self, text_inputs: dict, target_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        if text_inputs is None:
            raise ValueError("TextBranch requires text_inputs with a_minus/a_plus/action_type.")
        a_minus = text_inputs.get("a_minus")
        a_plus = text_inputs.get("a_plus")
        action_type = text_inputs.get("action_type")
        if a_minus is None or a_plus is None or action_type is None:
            raise ValueError("text_inputs must contain keys: a_minus, a_plus, action_type.")

        if not isinstance(a_minus, torch.Tensor) or not isinstance(a_plus, torch.Tensor):
            raise TypeError("a_minus and a_plus must be torch.Tensor.")
        if a_minus.ndim == 3:
            a_minus = a_minus.unsqueeze(1)
        if a_plus.ndim == 3:
            a_plus = a_plus.unsqueeze(1)
        if a_minus.ndim != 4 or a_plus.ndim != 4:
            raise ValueError("a_minus/a_plus must have shape [B,1,H,W] or [B,H,W].")
        if a_minus.shape != a_plus.shape:
            raise ValueError(f"a_minus/a_plus shape mismatch: {a_minus.shape} vs {a_plus.shape}")

        if isinstance(action_type, str):
            action_types = [action_type] * a_minus.shape[0]
        else:
            action_types = list(action_type)
        m_intent, e_t0 = self.acsf(a_minus, a_plus, action_types)
        if e_t0.shape[-2:] != target_hw:
            e_t0 = F.interpolate(e_t0, size=target_hw, mode="bilinear", align_corners=False)
            m_intent = F.interpolate(m_intent, size=target_hw, mode="bilinear", align_corners=False)
        return {"m_intent": m_intent, "e_t0": e_t0}


class FFDDiffNet(nn.Module):
    """
    MSDM + optional DRB + FPN (or single-scale) decoder, **semantic difference branch only**.

    No ``StructureBranch`` or ``TransportBranch``. When ``use_sem_branch`` is true, semantic
    features are fused to 64 channels with ``ConvNormAct(96,64)`` + two ``ResidualBlock``s
    (same as the old ``no_router`` + ``sem_only`` path in ``causal_model_new``).
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_dinov2.lvd142m",
        encoder_pretrained: bool = True,
        encoder_trainable: bool = False,
        msdm_out_ch: int = 256,
        drb_num_heads: int = 8,
        drb_ffn_mult: int = 4,
        drb_dropout: float = 0.0,
        num_drb_blocks: int = 2,
        multiscale_msdm: bool = True,
        use_pixel_aux: bool = True,
        pixel_aux_hidden: int = 32,
        use_sem_branch: bool = True,
        use_text_branch: bool = False,
        enable_online_text_modules: bool = False,
        text_fallback_mode: str = "weak_global",  # weak_global | bypass
        acsf_epsilon: float = 1e-3,
        icmv_num_heads: int = 4,
        icmv_k_iter: int = 2,
        parser_model_path: str | None = None,
        parser_device_map: str = "auto",
        parser_torch_dtype: str = "bfloat16",
        parser_max_memory: dict | None = None,
        langsam_sam_type: str = "sam2.1_hiera_small",
        langsam_sam_ckpt_path: str = _DEFAULT_SAM_CKPT,
        langsam_gdino_model_ckpt_path: str = _DEFAULT_GDINO_DIR,
        langsam_gdino_processor_ckpt_path: str = _DEFAULT_GDINO_DIR,
        hf_hub_cache: str | None = None,
    ):
        super().__init__()
        self.encoder_level_dims = (64, 128, 256)
        self.msdm_out_ch = int(msdm_out_ch)
        self.multiscale_msdm = bool(multiscale_msdm)
        self.use_sem_branch = bool(use_sem_branch)
        self.use_text_branch = bool(use_text_branch)
        self.text_fallback_mode = str(text_fallback_mode).strip().lower()
        if self.text_fallback_mode not in {"weak_global", "bypass"}:
            raise ValueError("text_fallback_mode must be 'weak_global' or 'bypass'")
        self._no_evidence_branches = not self.use_sem_branch
        n_drb = int(num_drb_blocks)
        if n_drb < 0:
            raise ValueError("num_drb_blocks must be >= 0")
        if not self.multiscale_msdm and n_drb > 0:
            raise ValueError("Single-scale SSD baseline requires num_drb_blocks=0 (no DRB).")

        self.align = DriftAlignStem(hidden=32, max_flow=0.06)
        self.encoder = DINOv2PyramidEncoder(
            model_name=encoder_name,
            pretrained=encoder_pretrained,
            trainable=encoder_trainable,
            out_dims=self.encoder_level_dims,
            hf_hub_cache=hf_hub_cache,
        )
        self.msdm = MultiScaleDifferencingModule(
            level_dims=self.encoder_level_dims,
            out_ch=self.msdm_out_ch,
        )
        self.drb_blocks = nn.ModuleList(
            [
                DifferenceRefinementBlock(
                    dim=self.msdm_out_ch,
                    num_heads=drb_num_heads,
                    ffn_mult=drb_ffn_mult,
                    dropout=drb_dropout,
                )
                for _ in range(n_drb)
            ]
        )

        if self.multiscale_msdm and self.use_sem_branch:
            self.sem_level_proj = nn.ModuleList(
                [ConvNormAct(self.msdm_out_ch, 32, 1) for _ in range(3)]
            )
            self.sem_refine = nn.Sequential(
                ConvNormAct(96, 96, 3, 1),
                ResidualBlock(96),
            )
        else:
            self.sem_level_proj = None  # type: ignore[assignment]
            self.sem_refine = None  # type: ignore[assignment]

        if not self.multiscale_msdm and self.use_sem_branch:
            self.sem_ssd = nn.Sequential(
                ConvNormAct(self.msdm_out_ch, 96, 3, 1),
                ResidualBlock(96),
            )
        else:
            self.sem_ssd = None  # type: ignore[assignment]

        self.sem_fuse = (
            nn.Sequential(
                ConvNormAct(96, 64, 3, 1),
                ResidualBlock(64),
                ResidualBlock(64),
            )
            if not self._no_evidence_branches
            else None
        )
        if self.use_text_branch:
            # Keep text branch output in router_ch=64 to align with decoder and sem branch.
            self.text_branch = TextBranch(out_ch=64, epsilon=acsf_epsilon)
            parser_kwargs = {}
            if parser_model_path:
                parser_kwargs = {
                    "model_path": str(parser_model_path),
                    "device_map": str(parser_device_map),
                    "torch_dtype": str(parser_torch_dtype),
                }
                if parser_max_memory is not None:
                    parser_kwargs["max_memory"] = parser_max_memory
            langsam_kwargs = {
                "sam_type": str(langsam_sam_type),
                "sam_ckpt_path": str(langsam_sam_ckpt_path),
                "gdino_model_ckpt_path": str(langsam_gdino_model_ckpt_path),
                "gdino_processor_ckpt_path": str(langsam_gdino_processor_ckpt_path),
            }
            self.text_branch.set_external_module_kwargs(
                parser_kwargs=parser_kwargs,
                langsam_kwargs=langsam_kwargs,
            )
            if enable_online_text_modules:
                self.text_branch.build_external_modules()
            self.icmv_router = ICMVRouter(c=64, num_heads=icmv_num_heads, k_iter=icmv_k_iter)
        else:
            self.text_branch = None  # type: ignore[assignment]
            self.icmv_router = None  # type: ignore[assignment]
        c = self.msdm_out_ch
        self.use_pixel_aux = bool(use_pixel_aux)
        if self.use_pixel_aux:
            self.pixel_diff_head = PixelDifferenceAuxHead(3, int(pixel_aux_hidden))
            mask_aux_ch = 1
        else:
            self.pixel_diff_head = None
            mask_aux_ch = 0

        if self.multiscale_msdm:
            self.decoder = FPNBoundaryDecoder(
                in_chs=(c, c, c),
                fpn_dim=128,
                router_ch=64,
                head_hidden=128,
                mask_aux_ch=mask_aux_ch,
            )
            self.single_scale_decoder = None  # type: ignore[assignment]
        else:
            self.single_scale_decoder = SingleScaleBoundaryDecoder(
                in_ch=c,
                router_ch=64,
                head_hidden=128,
                mask_aux_ch=mask_aux_ch,
            )
            self.decoder = None  # type: ignore[assignment]

    def _msdm_single_level(
        self, fb: list[torch.Tensor], fa: list[torch.Tensor], level: int
    ) -> torch.Tensor:
        b_l, a_l = fb[level], fa[level]
        x = torch.cat(
            [torch.abs(b_l - a_l), b_l * a_l, b_l, a_l],
            dim=1,
        )
        return self.msdm.projs[level](x)

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        text_inputs: dict | None = None,
        instructions: Sequence[str] | str | None = None,
    ) -> dict[str, torch.Tensor]:
        aligned_before, flow = self.align(before, after)
        fb = self.encoder(aligned_before)
        fa = self.encoder(after)
        hw = before.shape[-2:]
        out_hw = (before.shape[-2], before.shape[-1])
        ssd_level = _FFD_SSD_MSDM_LEVEL

        if self._no_evidence_branches:
            if self.multiscale_msdm:
                d_levels = self.msdm(fb, fa)
                for drb in self.drb_blocks:
                    d_levels = drb(d_levels)
                pyr = d_levels
                fused = pyr[0].new_zeros(pyr[0].shape[0], 64, hw[0], hw[1])
                mask_aux = None
                if self.use_pixel_aux and self.pixel_diff_head is not None:
                    d_pixel = torch.abs(aligned_before - after)
                    mask_aux = self.pixel_diff_head(d_pixel)
                assert self.decoder is not None
                if mask_aux is not None:
                    mask_logits, boundary_logits = self.decoder(
                        pyr, fused, out_hw, mask_aux=mask_aux
                    )
                else:
                    mask_logits, boundary_logits = self.decoder(pyr, fused, out_hw)
            else:
                d_ssd = self._msdm_single_level(fb, fa, ssd_level)
                fused = d_ssd.new_zeros(d_ssd.shape[0], 64, hw[0], hw[1])
                mask_aux = None
                if self.use_pixel_aux and self.pixel_diff_head is not None:
                    d_pixel = torch.abs(aligned_before - after)
                    mask_aux = self.pixel_diff_head(d_pixel)
                assert self.single_scale_decoder is not None
                if mask_aux is not None:
                    mask_logits, boundary_logits = self.single_scale_decoder(
                        d_ssd, fused, out_hw, mask_aux=mask_aux
                    )
                else:
                    mask_logits, boundary_logits = self.single_scale_decoder(d_ssd, fused, out_hw)
            return {
                "mask_logits": mask_logits,
                "boundary_logits": boundary_logits,
                "flow": flow,
            }

        if self.multiscale_msdm:
            d_levels = self.msdm(fb, fa)
            for drb in self.drb_blocks:
                d_levels = drb(d_levels)
            pyr = d_levels
            # Use DINO/MSDM feature resolution for ICMV to reduce O(N^2) attention cost.
            icmv_hw = tuple(pyr[0].shape[-2:])

            if self.use_sem_branch:
                assert self.sem_level_proj is not None and self.sem_refine is not None
                sem_parts = [
                    proj(F.interpolate(d_l, size=hw, mode="bilinear", align_corners=False))
                    for d_l, proj in zip(d_levels, self.sem_level_proj)
                ]
                sem_feat = self.sem_refine(torch.cat(sem_parts, dim=1))
            else:
                sem_feat = pyr[0].new_zeros(pyr[0].shape[0], 96, hw[0], hw[1])
        else:
            d_ssd = self._msdm_single_level(fb, fa, ssd_level)
            icmv_hw = tuple(d_ssd.shape[-2:])

            if self.use_sem_branch:
                assert self.sem_ssd is not None
                d_up_sem = F.interpolate(d_ssd, size=hw, mode="bilinear", align_corners=False)
                sem_feat = self.sem_ssd(d_up_sem)
            else:
                sem_feat = d_ssd.new_zeros(d_ssd.shape[0], 96, hw[0], hw[1])

        assert self.sem_fuse is not None
        sem_fused = self.sem_fuse(sem_feat)
        fused = sem_fused
        router_debug: dict[str, torch.Tensor] | None = None
        text_debug: dict[str, torch.Tensor] | None = None
        if self.use_text_branch and (text_inputs is not None):
            assert self.text_branch is not None and self.icmv_router is not None
            text_out = self.text_branch(text_inputs=text_inputs, target_hw=hw)
            text_debug = text_out
            icmv_out = self.icmv_router(e_v0=fused, e_t0=text_out["e_t0"], process_hw=icmv_hw)
            router_debug = icmv_out
            # consensus Z as fused evidence for decoder
            fused = F.interpolate(icmv_out["z"], size=hw, mode="bilinear", align_corners=False)
        elif self.use_text_branch and (instructions is not None):
            assert self.text_branch is not None and self.icmv_router is not None
            text_out = self.text_branch.infer_online(
                before=before,
                after=after,
                instructions=instructions,
                target_hw=hw,
                out_device=fused.device,
                out_dtype=fused.dtype,
            )
            text_debug = text_out
            icmv_out = self.icmv_router(e_v0=fused, e_t0=text_out["e_t0"], process_hw=icmv_hw)
            router_debug = icmv_out
            fused = F.interpolate(icmv_out["z"], size=hw, mode="bilinear", align_corners=False)
        elif self.use_text_branch and (text_inputs is None) and (instructions is None):
            # automatic stable fallback for missing instruction/text_inputs
            if self.text_fallback_mode == "weak_global":
                assert self.text_branch is not None and self.icmv_router is not None
                text_out = self.text_branch.infer_fallback(
                    batch_size=before.shape[0],
                    target_hw=hw,
                    out_device=fused.device,
                    out_dtype=fused.dtype,
                )
                text_debug = text_out
                icmv_out = self.icmv_router(e_v0=fused, e_t0=text_out["e_t0"], process_hw=icmv_hw)
                router_debug = icmv_out
                fused = F.interpolate(icmv_out["z"], size=hw, mode="bilinear", align_corners=False)
            # mode == bypass: keep original sem-only fused as-is

        mask_aux = None
        if self.use_pixel_aux and self.pixel_diff_head is not None:
            d_pixel = torch.abs(aligned_before - after)
            mask_aux = self.pixel_diff_head(d_pixel)

        if self.multiscale_msdm:
            assert self.decoder is not None
            if mask_aux is not None:
                mask_logits, boundary_logits = self.decoder(
                    pyr, fused, out_hw, mask_aux=mask_aux
                )
            else:
                mask_logits, boundary_logits = self.decoder(pyr, fused, out_hw)
        else:
            assert self.single_scale_decoder is not None
            if mask_aux is not None:
                mask_logits, boundary_logits = self.single_scale_decoder(
                    d_ssd, fused, out_hw, mask_aux=mask_aux
                )
            else:
                mask_logits, boundary_logits = self.single_scale_decoder(d_ssd, fused, out_hw)

        out = {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "flow": flow,
            # semantic branch debug outputs for visualization.
            "sem_feat": sem_feat,
            "sem_fused": sem_fused,
            "decoder_fused": fused,
        }
        if text_debug is not None:
            out["m_intent"] = text_debug["m_intent"]
            out["e_t0"] = text_debug["e_t0"]
        if router_debug is not None:
            out["e_v_k"] = F.interpolate(router_debug["e_v_k"], size=hw, mode="bilinear", align_corners=False)
            out["e_t_k"] = F.interpolate(router_debug["e_t_k"], size=hw, mode="bilinear", align_corners=False)
            out["z_consensus"] = F.interpolate(router_debug["z"], size=hw, mode="bilinear", align_corners=False)
            out["z_consensus_flat"] = router_debug["z_flat"]
        return out

# Ablation presets for ``FFDDiffNet`` (semantic path is fixed to MSDM/DRB ``d_levels``).
# ``use_pixel_aux`` is not listed: it defaults to True in :func:`build_ffd_net` for all runs.
# Keys per preset: multiscale_msdm, num_drb_blocks, use_sem_branch, and optional ``encoder_name``.
#
# Aliases: ``ffd_sem_only`` and ``ffd_all_strategies`` match the old naming; they are the same
#          configuration as ``ffd_msdm_drb2`` (MSDM + 2xDRB + semantic branch).
FFD_PRESETS: dict[str, dict[str, object]] = {
    # --- Scale / capacity (default ViT backbone from YAML/CLI) ---
    # Baseline: single-scale SSD (``multiscale_msdm=False``), no DRB, with semantic branch on ``d2``
    # (encoder level index 2, see ``_FFD_SSD_MSDM_LEVEL`` in ``FFD.py``).
    "ffd_baseline": {
        "multiscale_msdm": False,
        "num_drb_blocks": 0,
        "use_sem_branch": True,
    },
    "ffd_msdm_nodrb": {
        "multiscale_msdm": True,
        "num_drb_blocks": 0,
        "use_sem_branch": False,
    },
    "ffd_msdm_nodrb_sem": {
        "multiscale_msdm": True,
        "num_drb_blocks": 0,
        "use_sem_branch": True,
    },
    "ffd_msdm_drb2": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
    },
    "ffd_msdm_drb2_textbranch": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
        "use_text_branch": True,
        # In training/eval without instruction, auto-fallback keeps branch stable.
        "text_fallback_mode": "weak_global",
    },
    "ffd_msdm_drb4": {
        "multiscale_msdm": True,
        "num_drb_blocks": 4,
        "use_sem_branch": True,
    },
    "ffd_msdm_drb6": {
        "multiscale_msdm": True,
        "num_drb_blocks": 6,
        "use_sem_branch": True,
    },
    "ffd_msdm_drb8": {
        "multiscale_msdm": True,
        "num_drb_blocks": 8,
        "use_sem_branch": True,
    },
    "ffd_sem_only": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
    },
    "ffd_all_strategies": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
    },
    # --- Backbones (MSDM + 2xDRB + sem; only ``encoder_name`` changes) ---
    "ffd_all_strategies_vit_small_patch14_dinov2": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
        "encoder_name": "timm/vit_small_patch14_dinov2.lvd142m",
    },
    "ffd_all_strategies_vit_small_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
        "encoder_name": "timm/vit_small_patch16_dinov3.lvd1689m",
    },
    "ffd_all_strategies_vit_base_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
        "encoder_name": "timm/vit_base_patch16_dinov3.lvd1689m",
    },
    "ffd_all_strategies_vit_large_patch16_dinov3": {
        "multiscale_msdm": True,
        "num_drb_blocks": 2,
        "use_sem_branch": True,
        "encoder_name": "timm/vit_large_patch16_dinov3.lvd1689m",
    },
}


def build_ffd_net(
    mcfg: dict,
    encoder_pretrained: bool,
) -> FFDDiffNet:
    """Build :class:`FFDDiffNet` from a ``model`` config dict (see ``FFD_PRESETS``)."""
    preset = (mcfg.get("ablation_preset") or mcfg.get("ffd_preset") or mcfg.get("table1_preset") or "").strip()
    kwargs: dict = dict(
        encoder_name=mcfg.get("encoder_name", "vit_base_patch14_dinov2.lvd142m"),
        encoder_pretrained=encoder_pretrained,
        encoder_trainable=bool(mcfg.get("encoder_trainable", False)),
        msdm_out_ch=int(mcfg.get("msdm_out_ch", 256)),
        drb_num_heads=int(mcfg.get("drb_num_heads", 8)),
        drb_ffn_mult=int(mcfg.get("drb_ffn_mult", 4)),
        drb_dropout=float(mcfg.get("drb_dropout", 0.0)),
        num_drb_blocks=int(mcfg.get("num_drb_blocks", 2)),
        multiscale_msdm=bool(mcfg.get("multiscale_msdm", True)),
        use_pixel_aux=bool(mcfg.get("use_pixel_aux", True)),
        pixel_aux_hidden=int(mcfg.get("pixel_aux_hidden", 32)),
        use_sem_branch=bool(mcfg.get("use_sem_branch", True)),
        use_text_branch=bool(mcfg.get("use_text_branch", False)),
        enable_online_text_modules=bool(mcfg.get("enable_online_text_modules", False)),
        text_fallback_mode=str(mcfg.get("text_fallback_mode", "weak_global")),
        acsf_epsilon=float(mcfg.get("acsf_epsilon", 1e-3)),
        icmv_num_heads=int(mcfg.get("icmv_num_heads", 4)),
        icmv_k_iter=int(mcfg.get("icmv_k_iter", 2)),
        parser_model_path=(str(mcfg.get("parser_model_path", "")).strip() or None),
        parser_device_map=str(mcfg.get("parser_device_map", "auto")),
        parser_torch_dtype=str(mcfg.get("parser_torch_dtype", "bfloat16")),
        parser_max_memory=mcfg.get("parser_max_memory", None),
        langsam_sam_type=str(mcfg.get("langsam_sam_type", "sam2.1_hiera_small")),
        langsam_sam_ckpt_path=str(
            mcfg.get(
                "langsam_sam_ckpt_path",
                _DEFAULT_SAM_CKPT,
            )
        ),
        langsam_gdino_model_ckpt_path=str(
            mcfg.get(
                "langsam_gdino_model_ckpt_path",
                _DEFAULT_GDINO_DIR,
            )
        ),
        langsam_gdino_processor_ckpt_path=str(
            mcfg.get(
                "langsam_gdino_processor_ckpt_path",
                _DEFAULT_GDINO_DIR,
            )
        ),
        hf_hub_cache=(str(mcfg.get("hf_hub_cache", "")).strip() or None),
    )
    if preset:
        if preset not in FFD_PRESETS:
            raise ValueError(
                f"Unknown FFD preset={preset!r}; choose one of {list(FFD_PRESETS)}"
            )
        kwargs.update(FFD_PRESETS[preset])
    return FFDDiffNet(**kwargs)

# ===== Local sigma helpers =====

def extract_state_dict(ckpt: object) -> dict:
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(ckpt)}")
    for key in ("state_dict", "model", "net"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and inner:
            first = next(iter(inner.values()))
            if isinstance(first, torch.Tensor):
                return inner
    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Could not find a tensor state_dict in checkpoint.")


def load_checkpoint_compatible(model: torch.nn.Module, state: dict, *, strict: bool, log) -> dict:
    model_state = model.state_dict()
    model_has_module = any(k.startswith("module.") for k in model_state.keys())
    ckpt_has_module = any(k.startswith("module.") for k in state.keys())
    adapted = "none"
    if model_has_module != ckpt_has_module:
        if model_has_module:
            state = {f"module.{k}" if not k.startswith("module.") else k: v for k, v in state.items()}
            log("resume: adapting checkpoint keys single-gpu -> DataParallel")
            adapted = "single-gpu -> DataParallel"
        else:
            state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            log("resume: adapting checkpoint keys DataParallel -> single-gpu")
            adapted = "DataParallel -> single-gpu"

    if strict:
        model.load_state_dict(state, strict=True)
        log(f"resume: loaded weights (strict), num_keys={len(state)}")
        return {
            "mode": "strict",
            "loaded": int(len(state)),
            "from_ckpt": int(len(state)),
            "missing": 0,
            "unexpected": 0,
            "adapted": adapted,
        }

    filtered: dict = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    msg = model.load_state_dict(filtered, strict=False)
    log(f"resume: compatible load loaded={len(filtered)} from_ckpt={len(state)}")
    if msg.missing_keys:
        log(f"resume: missing_keys after load: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        log(f"resume: unexpected_keys after load: {len(msg.unexpected_keys)}")
    return {
        "mode": "non_strict",
        "loaded": int(len(filtered)),
        "from_ckpt": int(len(state)),
        "missing": int(len(msg.missing_keys)),
        "unexpected": int(len(msg.unexpected_keys)),
        "adapted": adapted,
    }


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    *,
    device: str,
    strict: bool = False,
    log=print,
) -> dict:
    try:
        ckpt_obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(str(ckpt_path), map_location=device)
    state = extract_state_dict(ckpt_obj)
    return load_checkpoint_compatible(model, state, strict=strict, log=log)


def state_dict_for_save(model: torch.nn.Module) -> dict:
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return real_model.state_dict()


def extract_instructions_from_batch(batch: dict, bs: int) -> list[str] | None:
    meta = batch.get("meta")
    if not isinstance(meta, dict):
        return None
    inst = meta.get("instruction")
    if inst is None:
        return None
    if isinstance(inst, (list, tuple)):
        out = [str(x) for x in inst]
    else:
        out = [str(inst)] * bs
    if len(out) != bs:
        if len(out) == 1:
            out = out * bs
        else:
            out = out[:bs]
            if len(out) < bs:
                out.extend([""] * (bs - len(out)))
    return out


def _normalize_path_str(p: str | Path) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


class OfflineTextInputStore:
    def __init__(self, root: Path):
        self.root = root
        self.samples_root = root / "samples"
        if not self.samples_root.is_dir():
            raise FileNotFoundError(f"acsf samples dir not found: {self.samples_root}")
        self.index: dict[str, dict] = {}
        self.path_index: dict[tuple[str, str], dict] = {}
        self._tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
        self._path_tensor_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        summary_path = self.root / "summary.json"
        built_from_summary = False
        if summary_path.is_file():
            try:
                import json

                with open(summary_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        if row.get("status", "ok") != "ok":
                            continue
                        before = str(row.get("before_image", "")).strip()
                        after = str(row.get("after_image", "")).strip()
                        fig = str(row.get("comparison_figure", "")).strip()
                        if not before or not after or not fig:
                            continue
                        d = Path(fig).parent
                        a_minus = d / "A_minus.npy"
                        a_plus = d / "A_plus.npy"
                        meta = d / "meta.json"
                        if not (a_minus.is_file() and a_plus.is_file() and meta.is_file()):
                            continue
                        try:
                            with open(meta, "r", encoding="utf-8") as fm:
                                meta_obj = json.load(fm)
                            action_type = str(meta_obj.get("action_type", "global"))
                        except Exception:
                            action_type = "global"
                        key = ""
                        if "_" in d.name:
                            key = d.name.split("_", 1)[1]
                        if key:
                            self.index[key] = {
                                "a_minus": a_minus,
                                "a_plus": a_plus,
                                "action_type": action_type,
                            }
                        pkey = (_normalize_path_str(before), _normalize_path_str(after))
                        self.path_index[pkey] = {
                            "a_minus": a_minus,
                            "a_plus": a_plus,
                            "action_type": action_type,
                        }
                    built_from_summary = (len(self.path_index) > 0 and len(self.index) > 0)
            except Exception:
                pass

        if built_from_summary:
            return
        for d in sorted(self.samples_root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if "_" not in name:
                continue
            key = name.split("_", 1)[1]
            a_minus = d / "A_minus.npy"
            a_plus = d / "A_plus.npy"
            meta = d / "meta.json"
            if not (a_minus.is_file() and a_plus.is_file() and meta.is_file()):
                continue
            try:
                import json

                with open(meta, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
                action_type = str(meta_obj.get("action_type", "global"))
            except Exception:
                action_type = "global"
            self.index[key] = {
                "a_minus": a_minus,
                "a_plus": a_plus,
                "action_type": action_type,
            }

    def get(self, key: str) -> Optional[tuple[torch.Tensor, torch.Tensor, str]]:
        if key in self._tensor_cache:
            return self._tensor_cache[key]
        rec = self.index.get(key)
        if rec is None:
            return None
        a_minus_np = np.load(rec["a_minus"]).astype(np.float32)
        a_plus_np = np.load(rec["a_plus"]).astype(np.float32)
        a_minus_t = torch.from_numpy(a_minus_np).unsqueeze(0)
        a_plus_t = torch.from_numpy(a_plus_np).unsqueeze(0)
        out = (a_minus_t, a_plus_t, rec["action_type"])
        self._tensor_cache[key] = out
        return out

    def get_by_paths(self, real_path: str, forged_path: str) -> Optional[tuple[torch.Tensor, torch.Tensor, str]]:
        pkey = (_normalize_path_str(real_path), _normalize_path_str(forged_path))
        if pkey in self._path_tensor_cache:
            return self._path_tensor_cache[pkey]
        rec = self.path_index.get(pkey)
        if rec is None:
            return None
        a_minus_np = np.load(rec["a_minus"]).astype(np.float32)
        a_plus_np = np.load(rec["a_plus"]).astype(np.float32)
        a_minus_t = torch.from_numpy(a_minus_np).unsqueeze(0)
        a_plus_t = torch.from_numpy(a_plus_np).unsqueeze(0)
        out = (a_minus_t, a_plus_t, rec["action_type"])
        self._path_tensor_cache[pkey] = out
        return out

    def has_sample(self, sample) -> bool:
        pkey = (_normalize_path_str(sample.real_path), _normalize_path_str(sample.forged_path))
        if pkey in self.path_index:
            return True
        return str(sample.key) in self.index

    def build_batch_text_inputs(self, batch: dict, device: str) -> Optional[dict]:
        meta = batch.get("meta")
        if not isinstance(meta, dict):
            return None
        keys = meta.get("key")
        real_paths = meta.get("real_path")
        forged_paths = meta.get("forged_path")
        if keys is None:
            return None
        if isinstance(keys, (list, tuple)):
            key_list = [str(k) for k in keys]
        else:
            key_list = [str(keys)] * int(batch["real"].shape[0])
        if isinstance(real_paths, (list, tuple)) and isinstance(forged_paths, (list, tuple)):
            rp_list = [str(x) for x in real_paths]
            fp_list = [str(x) for x in forged_paths]
        else:
            rp_list = []
            fp_list = []
        src_raw = meta.get("data_source")
        if isinstance(src_raw, (list, tuple)):
            src_list = [str(x) for x in src_raw]
        elif src_raw is not None and str(src_raw).strip():
            src_list = [str(src_raw)] * len(key_list)
        else:
            src_list = ["deal300k"] * len(key_list)
        a_minus_list: list[torch.Tensor] = []
        a_plus_list: list[torch.Tensor] = []
        actions: list[str] = []
        target_hw = tuple(int(x) for x in batch["real"].shape[-2:])
        h, w = target_hw
        for i, k in enumerate(key_list):
            is_brgen = i < len(src_list) and src_list[i] == "brgen"
            if is_brgen:
                a_minus_list.append(torch.zeros((1, h, w), dtype=torch.float32))
                a_plus_list.append(torch.zeros((1, h, w), dtype=torch.float32))
                actions.append("global")
                continue
            got = None
            if i < len(rp_list) and i < len(fp_list):
                got = self.get_by_paths(rp_list[i], fp_list[i])
            if got is None:
                got = self.get(k)
            if got is None:
                return None
            am, ap, act = got
            if tuple(int(x) for x in am.shape[-2:]) != target_hw:
                am = F.interpolate(am.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
            if tuple(int(x) for x in ap.shape[-2:]) != target_hw:
                ap = F.interpolate(ap.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
            a_minus_list.append(am)
            a_plus_list.append(ap)
            actions.append(act)
        a_minus = torch.stack(a_minus_list, dim=0).to(device, non_blocking=True)
        a_plus = torch.stack(a_plus_list, dim=0).to(device, non_blocking=True)
        return {"a_minus": a_minus, "a_plus": a_plus, "action_type": actions}


__all__ = [
    "FFDDiffNet",
    "FFD_PRESETS",
    "CausalDiffNetNew",
    "TABLE1_ABLATIONS",
    "DifferenceRefinementBlock",
    "MultiScaleDifferencingModule",
    "PixelDifferenceAuxHead",
    "build_ffd_net",
    "build_causal_diff_net_new",
    "causal_diff_loss",
    "dice_loss",
    "mask_to_boundary",
    "resolve_hf_hub_cache_dir",
    "extract_state_dict",
    "load_checkpoint_compatible",
    "load_checkpoint",
    "state_dict_for_save",
    "extract_instructions_from_batch",
    "OfflineTextInputStore",
]
