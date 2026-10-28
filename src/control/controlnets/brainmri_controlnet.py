import torch
from src.control.base import ControlModule
import torch.nn as nn
import copy

from src.control.adapters.brainmri_adapter import BrainMRIControlState_v1, BrainMRIControlState_v2

def make_zero_conv(channels: int) -> nn.Conv2d:
    """
    1x1 convolution initialized to output exactly zero.
    """

    layer = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=1
    )

    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)

    return layer

class BrainMRIControlNet_v1(
    nn.Module,
    ControlModule
):
    def __init__(
        self,
        backbone,
        device,
        condition_channels = 1
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.condition_channels = condition_channels
        
        unet = backbone.model

        # -------- Copy Pretrained Encoder --------
        self.time_embedding = copy.deepcopy(unet.time_embedding)

        self.in_conv = copy.deepcopy(unet.in_conv)

        self.enc1_block1 = copy.deepcopy(unet.enc1_block1)
        self.enc1_block2 = copy.deepcopy(unet.enc1_block2)
        self.down1 = copy.deepcopy(unet.down1)

        self.enc2_block1 = copy.deepcopy(unet.enc2_block1)
        self.enc2att = copy.deepcopy(unet.enc2att)
        self.enc2_block2 = copy.deepcopy(unet.enc2_block2)
        self.down2 = copy.deepcopy(unet.down2)

        self.mid_block1 = copy.deepcopy(unet.mid_block1)
        self.mid_att1 = copy.deepcopy(unet.mid_att1)
        self.mid_block2 = copy.deepcopy(unet.mid_block2)
        self.mid_att2 = copy.deepcopy(unet.mid_att2)

        # -------- Encode Segmentation Map --------

        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, unet.ch1, kernel_size=3, stride=2, padding=1) # [B, 128, 32, 32]
        )

        # Make zero-convolutions
        self.skip1_zero = make_zero_conv(unet.ch1)
        self.skip2_zero = make_zero_conv(unet.ch2)
        self.mid_zero = make_zero_conv(unet.ch3)

        self.modality_embedding = copy.deepcopy(unet.modality_embedding)
    

    def get_control(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        modality: torch.Tensor | None = None
    ) -> BrainMRIControlState_v1:
        """
        Function that returns a ControlState object -
        containing the residuals to be inserted
        """
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
        
        t_emb = self.time_embedding(timestep)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.backbone.model.null_class,
                device=z_t.device,
                dtype=torch.long
            )
        
        m_emb = self.modality_embedding(modality)

        t_m_emb = t_emb + m_emb

        # 1) Add Condition encoding w/ image
        x = self.in_conv(z_t)
        condition = self.condition_encoder(condition.to(device=x.device, dtype=x.dtype))
        x = x + condition

        # 2) Encoder Block 1
        x = self.enc1_block1(x, t_m_emb)
        x = self.enc1_block2(x, t_m_emb)
        skip1 = x
        x = self.down1(x)

        # 3) Encoder Block 2
        x = self.enc2_block1(x, t_m_emb)
        x = self.enc2att(x)
        x = self.enc2_block2(x, t_m_emb)
        skip2 = x
        x = self.down2(x)

        # 4) Mid Block
        x = self.mid_block1(x, t_m_emb)
        x = self.mid_att1(x)
        x = self.mid_block2(x, t_m_emb)
        x = self.mid_att2(x)

        return BrainMRIControlState_v1(
            skip1_residual=self.skip1_zero(skip1),
            skip2_residual=self.skip2_zero(skip2),
            mid_residual=self.mid_zero(x)
        )
        

class BrainMRIControlNet_v2(
    nn.Module,
    ControlModule
):
    def __init__(
        self,
        backbone,
        device,
        condition_channels = 1
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.condition_channels = condition_channels
        
        unet = backbone.model

        # -------- Copy Pretrained Encoder --------
        self.time_embedding = copy.deepcopy(unet.time_embedding)

        self.in_conv = copy.deepcopy(unet.in_conv)

        self.enc1_block1 = copy.deepcopy(unet.enc1_block1)
        self.enc1_block2 = copy.deepcopy(unet.enc1_block2)
        self.enc1_block3 = copy.deepcopy(unet.enc1_block3)
        self.down1 = copy.deepcopy(unet.down1)

        self.enc2_block1 = copy.deepcopy(unet.enc2_block1)
        self.enc2_block2 = copy.deepcopy(unet.enc2_block2)
        self.enc2_block3 = copy.deepcopy(unet.enc2_block3)
        self.enc2att1 = copy.deepcopy(unet.enc2att1)
        self.enc2att2 = copy.deepcopy(unet.enc2att2)
        self.enc2att3 = copy.deepcopy(unet.enc2att3)
        self.down2 = copy.deepcopy(unet.down2)

        self.enc3_block1 = copy.deepcopy(unet.enc3_block1)
        self.enc3_block2 = copy.deepcopy(unet.enc3_block2)
        self.enc3_block3 = copy.deepcopy(unet.enc3_block3)
        self.enc3att1 = copy.deepcopy(unet.enc3att1)
        self.enc3att2 = copy.deepcopy(unet.enc3att2)
        self.enc3att3 = copy.deepcopy(unet.enc3att3)
        self.down3 = copy.deepcopy(unet.down3)

        self.mid_block1 = copy.deepcopy(unet.mid_block1)
        self.mid_att1 = copy.deepcopy(unet.mid_att1)
        self.mid_block2 = copy.deepcopy(unet.mid_block2)
        

        # -------- Encode Segmentation Map --------

        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, unet.ch1, kernel_size=3, stride=2, padding=1) # [B, 128, 32, 32]
        )

        # Make zero-convolutions
        self.c_zero = make_zero_conv(unet.ch1)
        self.skip1_zero = make_zero_conv(unet.ch1)
        self.skip2_zero = make_zero_conv(unet.ch2)
        self.skip3_zero = make_zero_conv(unet.ch3)
        self.mid_zero = make_zero_conv(unet.ch4)

        self.modality_embedding = copy.deepcopy(unet.modality_embedding)
    

    def get_control(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        modality: torch.Tensor | None = None,
        zero_cond: bool = False
    ) -> BrainMRIControlState_v2:
        """
        Function that returns a ControlState object -
        containing the residuals to be inserted
        """
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
        
        t_emb = self.time_embedding(timestep)

        if modality is None:
            modality = torch.full(
                (z_t.shape[0],), # batch size
                self.backbone.model.null_class,
                device=z_t.device,
                dtype=torch.long
            )
        
        m_emb = self.modality_embedding(modality)

        t_m_emb = t_emb + m_emb

        # 1) Add Condition encoding w/ image
        x = self.in_conv(z_t)
        condition = self.condition_encoder(condition.to(device=x.device, dtype=x.dtype))
        if zero_cond:
            condition = self.c_zero(condition)
        x = x + condition

        # 2) Encoder Block 1
        x = self.enc1_block1(x, t_m_emb)
        x = self.enc1_block2(x, t_m_emb)
        x = self.enc1_block3(x, t_m_emb)
        skip1 = x
        x = self.down1(x)

        # 3) Encoder Block 2
        x = self.enc2_block1(x, t_m_emb)
        x = self.enc2att1(x)
        x = self.enc2_block2(x, t_m_emb)
        x = self.enc2att2(x)
        x = self.enc2_block3(x, t_m_emb)
        x = self.enc2att3(x)
        skip2 = x
        x = self.down2(x)

        # 4) Encoder Block 3
        x = self.enc3_block1(x, t_m_emb)
        x = self.enc3att1(x)
        x = self.enc3_block2(x, t_m_emb)
        x = self.enc3att2(x)
        x = self.enc3_block3(x, t_m_emb)
        x = self.enc3att3(x)
        skip3 = x
        x = self.down3(x)

        # 5) Mid Block
        x = self.mid_block1(x, t_m_emb)
        x = self.mid_att1(x)
        x = self.mid_block2(x, t_m_emb)

        return BrainMRIControlState_v2(
            skip1_residual=self.skip1_zero(skip1),
            skip2_residual=self.skip2_zero(skip2),
            skip3_residual=self.skip3_zero(skip3),
            mid_residual=self.mid_zero(x)
        )
        
