from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

@dataclass
class BrainMRIControlState_v1:
    """
    Residual corrections intended for the BrainMRI UNet.
    """
    skip1_residual: torch.Tensor
    skip2_residual: torch.Tensor
    mid_residual: torch.Tensor
@dataclass
class BrainMRIControlState_v2:
    """
    Residual corrections intended for the BrainMRI UNet.
    """
    skip1_residual: torch.Tensor
    skip2_residual: torch.Tensor
    skip3_residual: torch.Tensor
    mid_residual: torch.Tensor


class BrainMRIControlAdapter_v1:
    """
    Defines how the ControlNet residuals plug
    into the Brain MRI UNet.
    """
    def __init__(
        self,
        backbone
    ):
        self.backbone = backbone
        self.unet = backbone.model

    def validate_control_state(
        self,
        skip1,
        skip2,
        mid,
        control_state: BrainMRIControlState_v1
    ) -> None:
        if skip1.shape != control_state.skip1_residual.shape:
            raise ValueError(
                "skip1 shape mismatch: "
                f"{skip1.shape} vs "
                f"{control_state.skip1_residual.shape}"
            )
        if skip2.shape != control_state.skip2_residual.shape:
            raise ValueError(
                "skip2 shape mismatch: "
                f"{skip2.shape} vs "
                f"{control_state.skip2_residual.shape}"
            )
        if mid.shape != control_state.mid_residual.shape:
            raise ValueError(
                "mid shape mismatch: "
                f"{mid.shape} vs "
                f"{control_state.mid_residual.shape}"
            )

    def predict_noise(
        self,
        z_t,
        timestep,
        control_state,
        class_labels: Optional[torch.Tensor] = None,
        modality: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        unet = self.unet
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
            [timestep],
            device=z_t.device,
            dtype=torch.long
        )
        else:
            timestep = timestep.to(z_t.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and z_t.shape[0] > 1:
            timestep = timestep.expand(z_t.shape[0])
        t_emb = unet.time_embedding(timestep)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.unet.null_class,
                device=z_t.device,
                dtype=torch.long
            )
                
        m_emb = unet.modality_embedding(modality)
        t_m_emb = t_emb + m_emb
        x = unet.in_conv(z_t)

        x = unet.enc1_block1(x, t_m_emb)
        x = unet.enc1_block2(x, t_m_emb)
        skip1 = x
        x = unet.down1(x)

        x = unet.enc2_block1(x, t_m_emb)
        x = unet.enc2att(x)
        x = unet.enc2_block2(x, t_m_emb)
        skip2 = x
        x = unet.down2(x)

        x = unet.mid_block1(x, t_m_emb)
        x = unet.mid_att1(x)
        x = unet.mid_block2(x, t_m_emb)
        x = unet.mid_att2(x)

        # CNet Injection
        self.validate_control_state(skip1, skip2, x, control_state)
        skip1 += control_state.skip1_residual
        skip2 += control_state.skip2_residual
        x += control_state.mid_residual

        x = unet.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = unet.dec1_block1(x, t_m_emb)
        x = unet.dec1_att(x)
        x = unet.dec1_block2(x, t_m_emb)

        x = unet.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = unet.dec2_block1(x, t_m_emb)
        x = unet.dec2_block2(x, t_m_emb)

        x = unet.out_conv(F.silu(unet.out_norm(x)))

        return x


class BrainMRIControlAdapter_v2:
    """
    Defines how the ControlNet residuals plug
    into the Brain MRI UNet.
    """
    def __init__(
        self,
        backbone
    ):
        self.backbone = backbone
        self.unet = backbone.model

    def validate_control_state(
        self,
        skip1,
        skip2,
        skip3,
        mid,
        control_state: BrainMRIControlState_v2
    ) -> None:
        if skip1.shape != control_state.skip1_residual.shape:
            raise ValueError(
                "skip1 shape mismatch: "
                f"{skip1.shape} vs "
                f"{control_state.skip1_residual.shape}"
            )
        if skip2.shape != control_state.skip2_residual.shape:
            raise ValueError(
                "skip2 shape mismatch: "
                f"{skip2.shape} vs "
                f"{control_state.skip2_residual.shape}"
            )
        if skip3.shape != control_state.skip3_residual.shape:
                raise ValueError(
                    "skip2 shape mismatch: "
                    f"{skip3.shape} vs "
                    f"{control_state.skip3_residual.shape}"
                )
        if mid.shape != control_state.mid_residual.shape:
            raise ValueError(
                "mid shape mismatch: "
                f"{mid.shape} vs "
                f"{control_state.mid_residual.shape}"
            )

    def predict_noise(
        self,
        z_t,
        timestep,
        control_state,
        class_labels: Optional[torch.Tensor] = None,
        modality: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        unet = self.unet
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
            [timestep],
            device=z_t.device,
            dtype=torch.long
        )
        else:
            timestep = timestep.to(z_t.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and z_t.shape[0] > 1:
            timestep = timestep.expand(z_t.shape[0])
        t_emb = unet.time_embedding(timestep)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.unet.null_class,
                device=z_t.device,
                dtype=torch.long
            )
                
        m_emb = unet.modality_embedding(modality)
        t_m_emb = t_emb + m_emb
        x = unet.in_conv(z_t)

        x = unet.enc1_block1(x, t_m_emb)
        x = unet.enc1_block2(x, t_m_emb)
        x = unet.enc1_block3(x, t_m_emb)
        skip1 = x
        x = unet.down1(x)

        x = unet.enc2_block1(x, t_m_emb)
        x = unet.enc2att1(x)
        x = unet.enc2_block2(x, t_m_emb)
        x = unet.enc2att2(x)
        x = unet.enc2_block3(x, t_m_emb)
        x = unet.enc2att3(x)
        skip2 = x
        x = unet.down2(x)

        x = unet.enc3_block1(x, t_m_emb)
        x = unet.enc3att1(x)
        x = unet.enc3_block2(x, t_m_emb)
        x = unet.enc3att2(x)
        x = unet.enc3_block3(x, t_m_emb)
        x = unet.enc3att3(x)
        skip3 = x
        x = unet.down3(x)

        x = unet.mid_block1(x, t_m_emb)
        x = unet.mid_att1(x)
        x = unet.mid_block2(x, t_m_emb)

        # CNet Injection
        self.validate_control_state(skip1, skip2, skip3, x, control_state)
        skip1 += control_state.skip1_residual
        skip2 += control_state.skip2_residual
        skip3 += control_state.skip3_residual
        x += control_state.mid_residual

        x = unet.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = unet.dec1_block1(x, t_m_emb)
        x = unet.dec1_att1(x)
        x = unet.dec1_block2(x, t_m_emb)
        x = unet.dec1_att2(x)
        x = unet.dec1_block3(x, t_m_emb)
        x = unet.dec1_att3(x)

        x = unet.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = unet.dec2_block1(x, t_m_emb)
        x = unet.dec2_att1(x)
        x = unet.dec2_block2(x, t_m_emb)
        x = unet.dec2_att2(x)
        x = unet.dec2_block3(x, t_m_emb)
        x = unet.dec2_att3(x)

        x = unet.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = unet.dec3_block1(x, t_m_emb)
        x = unet.dec3_block2(x, t_m_emb)
        x = unet.dec3_block3(x, t_m_emb)

        x = unet.out_conv(F.silu(unet.out_norm(x)))

        return x