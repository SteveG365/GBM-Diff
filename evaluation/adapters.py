from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


# --------------------- Input Checking / Provenance ---------------------

def check_conditioning_mask(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("Expected a nonempty 2D conditioning grid")

    if not np.isin(mask, [0, 1, 2, 4]).all():
        raise ValueError(
            f"Expected raw mask labels 0/1/2/4, got {np.unique(mask)}"
        )

    return mask.astype(np.uint8)


def checkpoint_digest(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

# --------------------- Controlled / Backbone-Only Prediction ---------------------

def predict_evaluation_noise(
    helper,
    backbone,
    controlnet,
    z_t,
    timesteps,
    condition,
    modality,
    guidance_scale,
    control_enabled
):
    if control_enabled:
        return helper.predict_noise(
            backbone=backbone,
            controlnet=controlnet,
            z_t=z_t,
            timesteps=timesteps,
            condition=condition,
            modality=modality,
            guidance_scale=guidance_scale
        )

    # Preserve modality conditioning while removing mask control
    epsilon_cond = backbone.predict_noise(
        z_t=z_t,
        timestep=timesteps,
        control_state=None,
        modality=modality
    )

    if guidance_scale == 1.0:
        return epsilon_cond

    null_modality = torch.full_like(
        modality,
        backbone.model.null_class
    )

    epsilon_null = backbone.predict_noise(
        z_t=z_t,
        timestep=timesteps,
        control_state=None,
        modality=null_modality
    )

    return epsilon_null + guidance_scale * (epsilon_cond - epsilon_null)

# --------------------- ControlNet Adapter Construction ---------------------

def make_controlnet(config):
    helper = importlib.import_module(
        config.get(
            "helper_module",
            "scripts.visualise_mri_controlnet",
        )
    )

    device = torch.device(
        config.get("device", "cuda")
    )

    config_path = Path(
        config["config_path"]
    ).expanduser()

    with config_path.open() as handle:
        model_config = yaml.safe_load(handle)

    guidance_scale = float(
        config.get("guidance_scale", 1.0)
    )

    if not np.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError(
            "guidance_scale must be finite and nonnegative"
        )

    latent_shape = tuple(config["latent_shape"])

    if (
        len(latent_shape) != 3
        or any(type(value) is not int or value < 1 for value in latent_shape)
    ):
        raise ValueError(
            "latent_shape must contain three positive integers: C, H, W"
        )

    control_path = Path(config["control_checkpoint"]).expanduser()

    vae_path = Path(config["vae_checkpoint"]).expanduser()

    use_ema = config.get("use_ema", True)

    checkpoint = torch.load(
        control_path,
        map_location="cpu",
        weights_only=False
    )

    if use_ema and checkpoint.get("ema_controlnet") is None:
        raise ValueError("EMA weights are unavailable; set use_ema: false")

    backbone_path = (
        config.get("backbone_checkpoint")
        or checkpoint.get("backbone_checkpoint")
    )

    if not backbone_path:
        raise ValueError("No backbone checkpoint path supplied")

    backbone_path = Path(backbone_path).expanduser()

    del checkpoint

    backbone, controlnet, vae, latent_scale = helper.load_models(
        model_config,
        str(control_path),
        str(backbone_path),
        str(vae_path),
        device,
        use_ema,
    )

    latent_scale = float(latent_scale)

    if not np.isfinite(latent_scale) or latent_scale <= 0:
        raise ValueError("Loaded latent scale must be finite and positive")

    for model in (backbone.model, controlnet, vae):
        model.eval()
        model.requires_grad_(False)

    diffusion = model_config["diffusion"]

    schedule = helper.DDPMSchedule(
        int(diffusion.get("num_timesteps", 1000)),
        float(diffusion.get("beta_start", 0.0001)),
        float(diffusion.get("beta_end", 0.02)),
        device,
    )

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        config.get("deterministic", True)
    )

    # --------------------- MRI Generation ---------------------

    @torch.inference_mode()
    def generate(
        mask,
        seed,
        modalities,
        control_enabled=True
    ):
        masks = np.asarray(mask)

        # Preserve support for existing single-mask calls
        single_sample = masks.ndim == 2

        if single_sample:
            masks = masks[None]

        if masks.ndim != 3:
            raise ValueError("Expected conditioning mask with shape H×W or B×H×W")

        # Validate every mask using the existing validation function
        masks = np.stack([
            check_conditioning_mask(sample)
            for sample in masks
        ])

        batch_size = masks.shape[0]

        # Single-image generation still accepts an integer seed.
        if type(seed) is int:
            if batch_size != 1:
                raise ValueError(
                    "Batched generation requires one seed per mask"
                )

            seeds = [seed]

        else:
            seeds = list(seed)

            if len(seeds) != batch_size:
                raise ValueError(
                    "Number of seeds must equal number of masks"
                )

            if any(
                type(value) is not int or value < 0
                for value in seeds
            ):
                raise ValueError(
                    "Seeds must be nonnegative integers"
                )

        expected_size = model_config.get("data", {}).get("image_size")

        if expected_size is not None:
            expected_grid = (int(expected_size), int(expected_size))

            if tuple(masks.shape[1:]) != expected_grid:
                raise ValueError(
                    f"Expected mask grid {expected_grid}, "
                    f"got {tuple(masks.shape[1:])}"
                )

        condition = torch.from_numpy(masks.astype(np.float32))[:, None].to(device)

        images = {}

        for name in modalities:
            if name not in helper.MODALITY_TO_ID:
                raise ValueError(f"Unknown MRI modality: {name}")

            rngs = [
                torch.Generator(device=device).manual_seed(value)
                for value in seeds
            ]

            z = torch.stack(
                [
                    torch.randn(latent_shape, generator=rng, device=device)
                    for rng in rngs
                ],
                dim=0
            )

            modality = torch.full(
                (batch_size,),
                helper.MODALITY_TO_ID[name],
                dtype=torch.long,
                device=device
            )

            sampling_steps = tqdm(
                reversed(range(schedule.num_timesteps)),
                total=schedule.num_timesteps,
                desc=f"{name}, seed {seed}",
                leave=False,
                disable=not config.get("progress", True)
            )

            for step in sampling_steps:
                timesteps = torch.full(
                    (batch_size,),
                    step,
                    dtype=torch.long,
                    device=device
                )

                epsilon = predict_evaluation_noise(
                    helper=helper,
                    backbone=backbone,
                    controlnet=controlnet,
                    z_t=z,
                    timesteps=timesteps,
                    condition=condition,
                    modality=modality,
                    guidance_scale=guidance_scale,
                    control_enabled=control_enabled
                )

                beta = schedule.betas[step]
                alpha = schedule.alphas[step]
                alpha_bar = schedule.alpha_bars[step]

                z = torch.rsqrt(alpha) * (z - beta / torch.sqrt(1 - alpha_bar) * epsilon)

                if step > 0:
                    previous_alpha_bar = schedule.alpha_bars[step - 1]

                    variance = (beta * (1 - previous_alpha_bar) / (1 - alpha_bar))

                    noise = torch.stack(
                        [
                            torch.randn(latent_shape, generator=rng, device=device)
                            for rng in rngs
                        ],
                        dim=0
                    )

                    z = z + variance.sqrt() * noise

            decoded = vae.decode(z / latent_scale).float()

            expected_shape = (batch_size, 1, *masks.shape[1:])

            if tuple(decoded.shape) != expected_shape:
                raise ValueError(
                    "MRI decoder output must have shape "
                    f"{expected_shape}, got {tuple(decoded.shape)}"
                )

            if not torch.isfinite(decoded).all():
                raise ValueError("MRI decoder produced non-finite values")

            image = decoded.clamp(-1, 1).add(1).div(2)

            batch_images = image[:, 0].cpu().numpy()

            if single_sample:
                images[name] = batch_images[0]
            else:
                images[name] = batch_images

        return images

    # --------------------- Adapter Metadata ---------------------

    generate.metadata = {
        "torch_version": str(torch.__version__),
        "latent_scale": latent_scale,
        "latent_shape": list(latent_shape),
        "guidance_scale": guidance_scale,
        "use_ema": use_ema,
        "model_config": model_config,
        "checkpoint_sha256": {
            "controlnet": checkpoint_digest(control_path),
            "backbone": checkpoint_digest(backbone_path),
            "vae": checkpoint_digest(vae_path)
        }
    }

    return generate