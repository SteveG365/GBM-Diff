from src.control.controlnets.segguideddiff_controlnet import SegGuidedDiffControlNet
from src.control.controlnets.brainmri_controlnet import BrainMRIControlNet_v1, BrainMRIControlNet_v2

def build_control(
    config,
    backbone,
    device
):
    """
    Converts the Config.yml file into 
    the approapriate ControlNet object
    """

    if config["control"]["enabled"] == False:
        return None

    control_type = config["control"]["type"]

    if control_type == "segguideddiff_controlnet":

        controlnet = SegGuidedDiffControlNet(
            backbone=backbone,
            device=device,
            condition_channels=config["control"]["condition_channels"]
        )
    elif control_type == "brainmri_controlnet_v1":
        controlnet = BrainMRIControlNet_v1(
            backbone=backbone,
            device=device,
            condition_channels=config["control"]["condition_channels"]
        )
    elif control_type == "brainmri_controlnet_v2":
        controlnet = BrainMRIControlNet_v2(
            backbone=backbone,
            device=device,
            condition_channels=config["control"]["condition_channels"]
        )
    else:
        raise ValueError(
            f"Unknown control type: {control_type}"
        )
    return controlnet.to(device)