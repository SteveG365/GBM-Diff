from src.diffusion.diff_model import LatentDiffusionUNet_v1, LatentDiffusionUNet_v2

def build_diffusion(config):
    cfg = config["diffusion"]
    architecture = cfg.get("architecture", "latent_v1")

    if architecture == "latent_v1":
        print("Using v1 Architecture")
        return LatentDiffusionUNet_v1(
            latent_channels=config["vae"]["latent_channels"],
            base_channels=cfg.get("base_channels", 128),
            time_dim=cfg.get("time_dim", 512),
            num_modalities=cfg.get("num_modalities", 4),
            norm_groups=cfg.get("norm_groups", 32),
            dropout=cfg.get("dropout", 0.0)
        )

    if architecture == "latent_v2":
        print("Using v2 Architecture")
        return LatentDiffusionUNet_v2(
            latent_channels=config["vae"]["latent_channels"],
            base_channels=cfg.get("base_channels", 128),
            time_dim=cfg.get("time_dim", 512),
            num_modalities=cfg.get("num_modalities", 4),
            norm_groups=cfg.get("norm_groups", 32),
            dropout=cfg.get("dropout", 0.0)
        )

    raise ValueError(f"Unknown diffusion architecture: {architecture}")