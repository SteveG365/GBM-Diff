from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

# Code here was adapted from the CONFLUX implementation by 
# Max Van Puyvelde (Stanford University) from 3D into 2D.
# ----------------------------------------------------------------------------------------------------------------
# Citation:
# M. Van Puyvelde, H. I. Gulluk, W. Van Criekinge, and O. Gevaert, 
# "CONFLUX: A latent diffusion model for 3D chest-CT synthesis with RL post-training," 2026, arXiv:2607.02998.
# ----------------------------------------------------------------------------------------------------------------

class VAELossMRI(nn.Module):

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-6,
        perceptual_weight: float = 0.0
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        target,
        reconstruction,
        posterior,
        perceptual_loss=None
    ):
        reconstruction_loss = F.l1_loss(reconstruction, target)

        kl = posterior.kl()

        if perceptual_loss is None:
            perceptual = torch.zeros((), device=target.device)
        else:
            perceptual = perceptual_loss(reconstruction, target)

        total = self.reconstruction_weight * reconstruction_loss + self.kl_weight * kl + self.perceptual_weight * perceptual

        return {
            "loss": total,
            "reconstruction": reconstruction_loss,
            "kl": kl,
            "perceptual": perceptual
        }

class VAELossSegmentation(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        kl_weight: float = 1e-4
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        logits,
        target,
        posterior,
        kl_weight=None
    ):
        logits = logits.float()

        # target: [B, H, W], integer indices 0/1/2/3
        # logits: [B, 4, H, W], unrestricted decoder outputs
        ce_loss = F.cross_entropy(logits, target)

        probabilities = logits.softmax(dim=1)

        target_one_hot = F.one_hot(
            target,
            num_classes=4
        ).permute(0, 3, 1, 2).float()

        # Exclude background from the Dice term.
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

        intersection = (probabilities * target_one_hot).sum(dim=(2, 3))

        predicted_area = probabilities.sum(dim=(2, 3))
        target_area = target_one_hot.sum(dim=(2, 3))

        dice_scores = (2.0 * intersection + 1e-6) / (predicted_area + target_area + 1e-6)

        # Only average Dice over classes present in each target slice.
        present = target_area > 0

        dice_loss_per_slice = ((1.0 - dice_scores) * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
        dice_loss = dice_loss_per_slice.mean()

        mean = posterior.mean.float()
        logvar = posterior.logvar.float()

        kl_loss = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()

        # Optional override allows KL warmup in the training loop.
        beta = self.kl_weight if kl_weight is None else kl_weight

        reconstruction_loss = (self.ce_weight * ce_loss + self.dice_weight * dice_loss)

        weighted_kl_loss = beta * kl_loss
        total_loss = reconstruction_loss + weighted_kl_loss

        return {
            "loss": total_loss,
            "reconstruction": reconstruction_loss,
            "ce": ce_loss,
            "dice_loss": dice_loss,
            "kl": kl_loss,
            "weighted_kl": weighted_kl_loss
        }

# Kept this function the same, dimension agnostic
def _gn(channels: int, groups: int=32) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels, eps=1e-06, affine=True)


# Changed Conv3d -> Conv2d, Dropout3d -> Dropout2d
class ResBlock2D(nn.Module):

    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        groups: int=32, 
        dropout: float=0.0
    ):
        super().__init__()

        self.norm1 = _gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = _gn(out_ch, groups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# Changed:
# 1) Removed D dimension
# 2) Conv3d -> Conv2d
class AttnBlock2D(nn.Module):

    def __init__(
        self, 
        channels: int, 
        num_heads: int=1, 
        groups: int=32
    ):
        super().__init__()

        if channels % num_heads != 0:
            for h in (8, 4, 2, 1):
                if channels % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _gn(channels, groups)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:

        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)

# Conv3d -> Conv2d
class Downsample2D(nn.Module):

    def __init__(
        self, 
        channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        return self.conv(x)

# Changed: All the 3D logic in the if else: deemed unnecesary for 2D 
class Upsample2D(nn.Module):

    def __init__(
        self, 
        channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# Changed:
# Conv3d -> Conv2d
# Renamed all functions 3d ->2d
class Encoder2D(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        base_channels: int, 
        channel_mults: Sequence[int], 
        latent_channels: int, 
        num_res_blocks: int, 
        groups: int, 
        dropout: float, 
        use_checkpoint: bool
    ):

        super().__init__()
        self.use_checkpoint = use_checkpoint
        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        in_ch = chs[0]

        for level, out_ch in enumerate(chs):
            blocks = nn.ModuleList(
                [
                    ResBlock2D(in_ch if i == 0 else out_ch, out_ch, groups, dropout) 
                    for i in range(num_res_blocks)
                ]
            )
            downsample = Downsample2D(out_ch) if level < len(chs) - 1 else nn.Identity()
            self.down.append(nn.ModuleDict({'blocks': blocks, 'downsample': downsample}))
            in_ch = out_ch
            
        self.mid_block1 = ResBlock2D(in_ch, in_ch, groups, dropout)
        self.mid_attn = AttnBlock2D(in_ch, groups=groups)
        self.mid_block2 = ResBlock2D(in_ch, in_ch, groups, dropout)
        self.out_norm = _gn(in_ch, groups)
        self.out_conv = nn.Conv2d(in_ch, 2 * latent_channels, 3, padding=1)

    def _maybe_ckpt(self, fn, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return ckpt.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.in_conv(x)

        for level in self.down:
            for block in level['blocks']:
                x = self._maybe_ckpt(block, x)
            x = level['downsample'](x)

        x = self._maybe_ckpt(self.mid_block1, x)
        x = self.mid_attn(x)
        x = self._maybe_ckpt(self.mid_block2, x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x

# Changed: same as Encoder
class Decoder2D(nn.Module):

    def __init__(
        self, 
        out_channels: int, 
        base_channels: int, 
        channel_mults: Sequence[int], 
        latent_channels: int, 
        num_res_blocks: int, 
        groups: int, 
        dropout: float, 
        use_checkpoint: bool
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv2d(latent_channels, chs[-1], kernel_size=3, padding=1)
        self.mid_block1 = ResBlock2D(chs[-1], chs[-1], groups, dropout)
        self.mid_attn = AttnBlock2D(chs[-1], groups=groups)
        self.mid_block2 = ResBlock2D(chs[-1], chs[-1], groups, dropout)
        self.up = nn.ModuleList()
        in_ch = chs[-1]

        for level, out_ch in enumerate(reversed(chs)):
            blocks = nn.ModuleList([ResBlock2D(in_ch if i == 0 else out_ch, out_ch, groups, dropout) for i in range(num_res_blocks)])
            upsample = Upsample2D(out_ch) if level < len(chs) - 1 else nn.Identity()
            self.up.append(nn.ModuleDict({'blocks': blocks, 'upsample': upsample}))
            in_ch = out_ch
        
        self.out_norm = _gn(in_ch, groups)
        self.out_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def _maybe_ckpt(self, fn, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return ckpt.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:

        x = self.in_conv(z)
        x = self._maybe_ckpt(self.mid_block1, x)
        x = self.mid_attn(x)
        x = self._maybe_ckpt(self.mid_block2, x)
        for level in self.up:
            for block in level['blocks']:
                x = self._maybe_ckpt(block, x)
            x = level['upsample'](x)
        return self.out_conv(F.silu(self.out_norm(x)))

    def get_last_layer(self) -> torch.Tensor:
        return self.out_conv.weight

@dataclass
class DiagonalGaussianDistribution:
    mean: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self):
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar).mean()

class VAE2D(nn.Module):

    def __init__(
        self, 
        in_channels: int=1, 
        out_channels: int=1, 
        base_channels: int=64, # halved this for 2D
        channel_mults: Sequence[int]=(1, 2, 4, 4), 
        latent_channels: int=16, # /4 this for 2d
        num_res_blocks: int=2, 
        num_res_blocks_decoder: Optional[int]=None, 
        norm_groups: int=32, 
        dropout: float=0.0, 
        use_checkpoint: bool=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_downs = len(channel_mults) - 1
        self.spatial_compression = 2 ** self.num_downs

        if num_res_blocks_decoder is None:
            num_res_blocks_decoder = num_res_blocks + 1

        self.encoder = Encoder2D(in_channels=in_channels, base_channels=base_channels, channel_mults=channel_mults, latent_channels=latent_channels, num_res_blocks=num_res_blocks, groups=norm_groups, dropout=dropout, use_checkpoint=use_checkpoint)
        self.decoder = Decoder2D(out_channels=out_channels, base_channels=base_channels, channel_mults=channel_mults, latent_channels=latent_channels, num_res_blocks=num_res_blocks_decoder, groups=norm_groups, dropout=dropout, use_checkpoint=use_checkpoint)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return DiagonalGaussianDistribution(mean, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample: bool=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample else posterior.mode()
        recon = self.decode(z)
        return (recon, posterior)

    def get_last_layer(self) -> torch.Tensor:
        return self.decoder.get_last_layer()

