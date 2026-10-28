from src.backbones.segguideddiff_backbone import SegGuidedDiffBackbone
from src.backbones.brainmri_backbone import BrainMRIBackbone
from src.utils.diffusion import build_diffusion_model

def build_backbone(config, device):
    #print(config)
    if config['backbone']['type'] == "segguideddiff":
        return SegGuidedDiffBackbone(
            checkpoint_path=config['backbone']["checkpoint"],
            device=device
        )

    if config["backbone"]["type"] == "brainmri":
        diff_model = build_diffusion_model(config, device)

        return BrainMRIBackbone(
            checkpoint_path=config["backbone"]["checkpoint"],
            diff_model=diff_model,
            device=device
        )

    raise ValueError(f"Unknown backbone type:{config['backbone']['type']}")