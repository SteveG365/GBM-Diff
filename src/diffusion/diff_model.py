import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Some of the code here inspired by the CONFLUX implementation by
# Max Van Puyvelde (Stanford University) from 3D into 2D.
# ----------------------------------------------------------------------------------------------------------------
# Citation:
# M. Van Puyvelde, H. I. Gulluk, W. Van Criekinge, and O. Gevaert, 
# "CONFLUX: A latent diffusion model for 3D chest-CT synthesis with RL post-training," 2026, arXiv:2607.02998.
# ----------------------------------------------------------------------------------------------------------------

def _gn(channels: int, groups: int=32) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels, eps=1e-06, affine=True)


def sinusoid_time_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000
) -> torch.Tensor:

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        time_dim: int
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(sinusoid_time_embedding(t, self.emb_dim))


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        self.dropout = (
            nn.Dropout2d(p=dropout)
            if dropout != 0.0
            else nn.Identity()
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = _gn(in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = _gn(out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(
        self, 
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:

        h = self.conv1(F.silu(self.norm1(x)))
        t_proj = self.time_proj(t_emb)

        h = h + t_proj[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        return h + self.skip(x)

        

class AttentionBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        head_dim: int = 64  
    ):
        super().__init__()
        
        self.channels = channels
        self.groups = groups
        if channels % head_dim != 0:
            raise ValueError(f"{channels=} must be divisible by {head_dim=}")
        self.head_dim = head_dim
        self.num_heads = channels // head_dim   # inspired from Dhariwal et al. (2021)
        self.norm = _gn(channels, groups)

        self.qkv = nn.Conv2d(channels, 3*channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)

        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H*W)
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)

class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)   # div 2

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# ---------------------- UNet Architecture ----------------------

class LatentDiffusionUNet_v1(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 128,
        time_dim: int = 512,
        num_modalities: int = 4,
        norm_groups: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4

        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3

        self.time_embedding = TimeEmbedding(emb_dim=base_channels, time_dim=time_dim)
        self.num_modalities = num_modalities
        self.null_class = num_modalities

        self.modality_embedding = nn.Embedding(
            num_modalities + 1, # extra token for null class
            time_dim
        )

        self.in_conv = nn.Conv2d(latent_channels, ch1, kernel_size=3, padding=1)

        # -------------------------- Encoder Blocks --------------------------
        self.enc1_block1 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.enc1_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.down1 = Downsample2D(ch1)

        self.enc2_block1 = ResBlock2D(ch1, ch2, time_dim, norm_groups, dropout)
        self.enc2_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.enc2att = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.down2 = Downsample2D(ch2)

        # -------------------------- Mid Block --------------------------
        self.mid_block1 = ResBlock2D(ch2, ch3, time_dim, norm_groups, dropout)
        self.mid_att1 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.mid_att2 = AttentionBlock2D(ch3, norm_groups, head_dim=64) # 2 attention blocks here, none for block 1 as 32 x 32 too beeg
        self.mid_block2 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)

        # -------------------------- Decoder Blocks --------------------------
        self.up1 = Upsample2D(ch3)
        self.dec1_block1 = ResBlock2D(ch3 + ch2, ch2, time_dim, norm_groups, dropout)
        self.dec1_att = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.dec1_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        
        self.up2 = Upsample2D(ch2)
        self.dec2_block1 = ResBlock2D(ch2 + ch1, ch1, time_dim, norm_groups, dropout)
        self.dec2_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.out_norm = _gn(ch1, norm_groups)
        self.out_conv = nn.Conv2d(ch1, latent_channels, kernel_size=3, padding=1)


    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        modality: torch.Tensor | None = None
    ):

        t_emb = self.time_embedding(timesteps)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.null_class,
                device=z_t.device,
                dtype=torch.long
            )

        m_emb = self.modality_embedding(modality)

        cond_emb = t_emb + m_emb

        x = self.in_conv(z_t)

        x = self.enc1_block1(x, cond_emb)
        x = self.enc1_block2(x, cond_emb)
        skip1 = x
        x = self.down1(x)

        x = self.enc2_block1(x, cond_emb)
        x = self.enc2att(x)
        x = self.enc2_block2(x, cond_emb)
        skip2 = x
        x = self.down2(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_att1(x)
        x = self.mid_block2(x, cond_emb)
        x = self.mid_att2(x)

        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1_block1(x, cond_emb)
        x = self.dec1_att(x)
        x = self.dec1_block2(x, cond_emb)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2_block1(x, cond_emb)
        x = self.dec2_block2(x, cond_emb)

        x = self.out_conv(F.silu(self.out_norm(x)))

        return x

class LatentDiffusionUNet_v2(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 128,
        time_dim: int = 512,
        num_modalities: int = 4,
        norm_groups: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.ch4 = ch4

        self.time_embedding = TimeEmbedding(emb_dim=base_channels, time_dim=time_dim)
        self.num_modalities = num_modalities
        self.null_class = num_modalities

        self.modality_embedding = nn.Embedding(
            num_modalities + 1, # extra token for null class
            time_dim
        )

        self.in_conv = nn.Conv2d(latent_channels, ch1, kernel_size=3, padding=1)

        # -------------------------- Encoder Blocks --------------------------
        self.enc1_block1 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.enc1_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.enc1_block3 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.down1 = Downsample2D(ch1)

        self.enc2_block1 = ResBlock2D(ch1, ch2, time_dim, norm_groups, dropout)
        self.enc2_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.enc2_block3 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.enc2att1 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.enc2att2 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.enc2att3 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.down2 = Downsample2D(ch2)

        self.enc3_block1 = ResBlock2D(ch2, ch3, time_dim, norm_groups, dropout)
        self.enc3_block2 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.enc3_block3 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.enc3att1 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.enc3att2 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.enc3att3 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.down3 = Downsample2D(ch3)

        # -------------------------- Mid Block --------------------------
        self.mid_block1 = ResBlock2D(ch3, ch4, time_dim, norm_groups, dropout)
        self.mid_att1 = AttentionBlock2D(ch4, norm_groups, head_dim=64)
        self.mid_block2 = ResBlock2D(ch4, ch4, time_dim, norm_groups, dropout)

        # -------------------------- Decoder Blocks --------------------------
        self.up1 = Upsample2D(ch4)
        self.dec1_block1 = ResBlock2D(ch4 + ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_block2 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_block3 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_att1 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.dec1_att2 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.dec1_att3 = AttentionBlock2D(ch3, norm_groups, head_dim=64)

        self.up2 = Upsample2D(ch3)
        self.dec2_block1 = ResBlock2D(ch3 + ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_block3 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_att1 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.dec2_att2 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.dec2_att3 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        
        self.up3 = Upsample2D(ch2)
        self.dec3_block1 = ResBlock2D(ch2 + ch1, ch1, time_dim, norm_groups, dropout)
        self.dec3_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.dec3_block3 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.out_norm = _gn(ch1, norm_groups)
        self.out_conv = nn.Conv2d(ch1, latent_channels, kernel_size=3, padding=1)


    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        modality: torch.Tensor | None = None
    ):

        t_emb = self.time_embedding(timesteps)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.null_class,
                device=z_t.device,
                dtype=torch.long
            )

        m_emb = self.modality_embedding(modality)

        cond_emb = t_emb + m_emb

        x = self.in_conv(z_t)

        x = self.enc1_block1(x, cond_emb)
        x = self.enc1_block2(x, cond_emb)
        x = self.enc1_block3(x, cond_emb)
        skip1 = x
        x = self.down1(x)

        x = self.enc2_block1(x, cond_emb)
        x = self.enc2att1(x)
        x = self.enc2_block2(x, cond_emb)
        x = self.enc2att2(x)
        x = self.enc2_block3(x, cond_emb)
        x = self.enc2att3(x)
        skip2 = x
        x = self.down2(x)

        x = self.enc3_block1(x, cond_emb)
        x = self.enc3att1(x)
        x = self.enc3_block2(x, cond_emb)
        x = self.enc3att2(x)
        x = self.enc3_block3(x, cond_emb)
        x = self.enc3att3(x)
        skip3 = x
        x = self.down3(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_att1(x)
        x = self.mid_block2(x, cond_emb)

        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1_block1(x, cond_emb)
        x = self.dec1_att1(x)
        x = self.dec1_block2(x, cond_emb)
        x = self.dec1_att2(x)
        x = self.dec1_block3(x, cond_emb)
        x = self.dec1_att3(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2_block1(x, cond_emb)
        x = self.dec2_att1(x)
        x = self.dec2_block2(x, cond_emb)
        x = self.dec2_att2(x)
        x = self.dec2_block3(x, cond_emb)
        x = self.dec2_att3(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3_block1(x, cond_emb)
        x = self.dec3_block2(x, cond_emb)
        x = self.dec3_block3(x, cond_emb)

        x = self.out_conv(F.silu(self.out_norm(x)))

        return x


# --------------------- Segmentation Mask Diffusion Model ---------------------

class MaskLatentDiffusionUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        time_dim: int = 512,
        #num_modalities: int = 4,
        norm_groups: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.ch4 = ch4

        self.time_embedding = TimeEmbedding(emb_dim=base_channels, time_dim=time_dim)

        self.condition_embedding = nn.Sequential(
            nn.Linear(4, time_dim), # input: [slice idx, p1, p2, p3]
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.null_condition_embedding = nn.Parameter(
            torch.zeros(1, time_dim)
        )

        self.in_conv = nn.Conv2d(latent_channels + 1, ch1, kernel_size=3, padding=1) # will add circle embedding here

        # -------------------------- Encoder Blocks --------------------------
        self.enc1_block1 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.enc1_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.enc1_block3 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.down1 = Downsample2D(ch1)

        self.enc2_block1 = ResBlock2D(ch1, ch2, time_dim, norm_groups, dropout)
        self.enc2_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.enc2_block3 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.enc2att1 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.enc2att2 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.enc2att3 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.down2 = Downsample2D(ch2)

        self.enc3_block1 = ResBlock2D(ch2, ch3, time_dim, norm_groups, dropout)
        self.enc3_block2 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.enc3_block3 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.enc3att1 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.enc3att2 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.enc3att3 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.down3 = Downsample2D(ch3)

        # -------------------------- Mid Block --------------------------
        self.mid_block1 = ResBlock2D(ch3, ch4, time_dim, norm_groups, dropout)
        self.mid_att1 = AttentionBlock2D(ch4, norm_groups, head_dim=64)
        self.mid_block2 = ResBlock2D(ch4, ch4, time_dim, norm_groups, dropout)

        # -------------------------- Decoder Blocks --------------------------
        self.up1 = Upsample2D(ch4)
        self.dec1_block1 = ResBlock2D(ch4 + ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_block2 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_block3 = ResBlock2D(ch3, ch3, time_dim, norm_groups, dropout)
        self.dec1_att1 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.dec1_att2 = AttentionBlock2D(ch3, norm_groups, head_dim=64)
        self.dec1_att3 = AttentionBlock2D(ch3, norm_groups, head_dim=64)

        self.up2 = Upsample2D(ch3)
        self.dec2_block1 = ResBlock2D(ch3 + ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_block2 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_block3 = ResBlock2D(ch2, ch2, time_dim, norm_groups, dropout)
        self.dec2_att1 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.dec2_att2 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        self.dec2_att3 = AttentionBlock2D(ch2, norm_groups, head_dim=64)
        
        self.up3 = Upsample2D(ch2)
        self.dec3_block1 = ResBlock2D(ch2 + ch1, ch1, time_dim, norm_groups, dropout)
        self.dec3_block2 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.dec3_block3 = ResBlock2D(ch1, ch1, time_dim, norm_groups, dropout)
        self.out_norm = _gn(ch1, norm_groups)
        self.out_conv = nn.Conv2d(ch1, latent_channels, kernel_size=3, padding=1)


    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        circle: torch.Tensor,
        conditions: torch.Tensor,
        conditions_present=True
    ):
        present = torch.as_tensor(conditions_present, device=z_t.device, dtype=torch.bool)
        if present.ndim == 0:
            present = present.expand(z_t.shape[0])

        conditions = conditions.to(device=z_t.device, dtype=z_t.dtype)
        global_embedding = self.condition_embedding(conditions)
        null_embedding = self.null_condition_embedding.expand(
            z_t.shape[0], -1
        ).to(dtype=global_embedding.dtype)
        global_embedding = torch.where(present[:, None], global_embedding, null_embedding)
        
        t_emb = self.time_embedding(timesteps)
        cond_emb = t_emb + global_embedding

        circle_latent = F.interpolate(circle.float(), z_t.shape[-2:], mode="area").to(dtype=z_t.dtype)
        circle_latent = torch.where(
            present[:, None, None, None],
            circle_latent,
            torch.zeros_like(circle_latent)
        )
        x = torch.concat([z_t, circle_latent], dim=1)

        x = self.in_conv(x)

        x = self.enc1_block1(x, cond_emb)
        x = self.enc1_block2(x, cond_emb)
        x = self.enc1_block3(x, cond_emb)
        skip1 = x
        x = self.down1(x)

        x = self.enc2_block1(x, cond_emb)
        x = self.enc2att1(x)
        x = self.enc2_block2(x, cond_emb)
        x = self.enc2att2(x)
        x = self.enc2_block3(x, cond_emb)
        x = self.enc2att3(x)
        skip2 = x
        x = self.down2(x)

        x = self.enc3_block1(x, cond_emb)
        x = self.enc3att1(x)
        x = self.enc3_block2(x, cond_emb)
        x = self.enc3att2(x)
        x = self.enc3_block3(x, cond_emb)
        x = self.enc3att3(x)
        skip3 = x
        x = self.down3(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_att1(x)
        x = self.mid_block2(x, cond_emb)

        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1_block1(x, cond_emb)
        x = self.dec1_att1(x)
        x = self.dec1_block2(x, cond_emb)
        x = self.dec1_att2(x)
        x = self.dec1_block3(x, cond_emb)
        x = self.dec1_att3(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2_block1(x, cond_emb)
        x = self.dec2_att1(x)
        x = self.dec2_block2(x, cond_emb)
        x = self.dec2_att2(x)
        x = self.dec2_block3(x, cond_emb)
        x = self.dec2_att3(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3_block1(x, cond_emb)
        x = self.dec3_block2(x, cond_emb)
        x = self.dec3_block3(x, cond_emb)

        x = self.out_conv(F.silu(self.out_norm(x)))

        return x