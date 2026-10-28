from src.backbones.base import DiffusionBackbone
import torch
from src.utils.diffusion import load_diffusion_from_checkpoint
from src.control.adapters.brainmri_adapter import BrainMRIControlAdapter_v1, BrainMRIControlAdapter_v2

class BrainMRIBackbone(DiffusionBackbone):

    def __init__(
        self,
        checkpoint_path: str,
        diff_model,
        architecture: str,
        device
    ):
        super().__init__()
        self.model = load_diffusion_from_checkpoint(
            path=checkpoint_path,
            model = diff_model,
            device=device
        ).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        if architecture == "latent_v1":
            self.control_adapter = BrainMRIControlAdapter_v1(self)
        elif architecture == "latent_v2":
            self.control_adapter = BrainMRIControlAdapter_v2(self)
        else:
            raise ValueError("Unknown Architecture")

    def predict_noise(
        self, 
        z_t, 
        timestep,
        control_state = None,
        modality = None
    ):
        if control_state == None:
            return self.model(z_t, timestep, modality=modality)

        # Controlled Path
        return self.control_adapter.predict_noise(
            z_t=z_t,
            timestep=timestep,
            control_state=control_state,
            modality=modality
        )