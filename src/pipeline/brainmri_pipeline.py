from pathlib import Path

import torch
import yaml
from src.control.controlnets.brainmri_controlnet import BrainMRIControlNet_v1, BrainMRIControlNet_v2
from src.diffusion.diff_model import MaskLatentDiffusionUNet
from src.diffusion.sampler import sample_ddpm_with_predictor, sample_mask_ddpm
from src.backbones.brainmri_backbone import BrainMRIBackbone
from src.diffusion.factory import build_diffusion
from src.diffusion.scheduler import DDPMSchedule
from src.vae.factory import build_vae, load_vae_state_dict
from src.data.load_masks import restore_mask_labels

MODALITY_TO_ID = {
    "t1": 0,
    "t1gd": 1,
    "t2": 2,
    "flair": 3
}


class BrainMRIGenerativePipeline:
    """
    High-level interface for the Brain MRI generative pipeline.

    This class is responsible for orchestrating the trained components,
    rather than implementing any neural-network architecture itself.
    """

    def __init__(
        self,
        backbone,
        controlnet,
        vae,
        schedule,
        latent_shape,
        latent_scale,
        device,
        default_guidance_scale=1.0,
        default_seed=42,
        mask_model=None,
        mask_vae=None,
        mask_schedule=None,
        mask_latent_shape=None,
        mask_latent_scale=None,
        default_mask_guidance_scale=1.0
    ):
        self.backbone = backbone
        self.controlnet = controlnet
        self.vae = vae
        self.schedule = schedule

        self.latent_shape = tuple(latent_shape)
        self.latent_scale = float(latent_scale)

        self.device = torch.device(device)
        self.default_guidance_scale = default_guidance_scale
        self.default_seed = default_seed

        # Mask generator
        self.mask_model = mask_model
        self.mask_vae = mask_vae
        self.mask_schedule = mask_schedule

        self.mask_latent_shape = (
            tuple(mask_latent_shape)
            if mask_latent_shape is not None
            else None
        )

        self.mask_latent_scale = (
            float(mask_latent_scale)
            if mask_latent_scale is not None
            else None
        )

        self.default_mask_guidance_scale = float(default_mask_guidance_scale)

        # Ensure inference behaviour
        self.backbone.model.eval()
        self.controlnet.eval()
        self.vae.eval()
        if self.mask_model is not None:
            self.mask_model.eval()
        if self.mask_vae is not None:
            self.mask_vae.eval()

    

    def _prepare_seg_map(
        self,
        seg_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper such that [H, W], [1, H, W] and [B, H, W]
        all work for our pipeline.
        """

        if not torch.is_tensor(seg_map):
            seg_map = torch.as_tensor(seg_map)

        # [H, W] -> [1, 1, H, W]
        if seg_map.ndim == 2:
            seg_map = seg_map.unsqueeze(0).unsqueeze(0)

        # [B, H, W] -> [B, 1, H, W]
        elif seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)

        elif seg_map.ndim != 4:
            raise ValueError(
                "seg_map must have shape [H, W], [B, H, W], "
                f"or [B, 1, H, W], got {tuple(seg_map.shape)}"
            )

        if seg_map.shape[1] != 1:
            raise ValueError(
                "ControlNet expects a single-channel segmentation map, "
                f"got shape {tuple(seg_map.shape)}"
            )

        return seg_map.to(
            device=self.device,
            dtype=torch.float32
        )

    def _prepare_modality(
        self,
        modality: str,
        batch_size: int
    ) -> torch.Tensor:
        """
        Converts 't1', 't1gd', 't2' and 'flair'
        to our internal representation
        """
        modality = modality.lower()

        if modality not in MODALITY_TO_ID:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Expected one of {list(MODALITY_TO_ID)}."
            )

        return torch.full(
            (batch_size,),
            MODALITY_TO_ID[modality],
            device=self.device,
            dtype=torch.long
        )

    def _prepare_mask_conditions(
        self,
        slice_position,
        centre_rc,
        radius,
        proportions,
        image_size=256,
    ):
        """
        Construct the spatial circle condition and global tumour conditions.
        Parameters
        ----------
        slice_position:
            Normalised axial slice position in [0, 1]
        centre_rc:
            Circle centre as (row, column), in image pixel coordinates
        radius:
            Circle radius in pixels
        proportions:
            Tumour-class proportions ordered as (p1, p2, p4)
        image_size:
            Spatial mask size.
        """

        # Validate global conditions
        slice_position = float(slice_position)

        if not 0.0 <= slice_position <= 1.0:
            raise ValueError("slice_position must lie between 0 and 1.")
        if len(centre_rc) != 2:
            raise ValueError("centre_rc must contain (row, column).")

        centre_row = float(centre_rc[0])
        centre_col = float(centre_rc[1])

        radius = float(radius)

        if radius <= 0:
            raise ValueError("radius must be greater than zero.")
        if len(proportions) != 3:
            raise ValueError("proportions must contain (p1, p2, p4).")

        proportions = torch.tensor(
            proportions,
            device=self.device,
            dtype=torch.float32
        )

        if (proportions < 0).any():
            raise ValueError("Tumour proportions cannot be negative.")

        if not torch.isclose(
            proportions.sum(),
            torch.tensor(
                1.0,
                device=self.device,
                dtype=torch.float32
            ),
            atol=1e-4
        ):
            raise ValueError("Tumour proportions must sum to 1.")

        # Construct filled disk

        rows, cols = torch.meshgrid(
            torch.arange(
                image_size,
                device=self.device,
                dtype=torch.float32
            ),
            torch.arange(
                image_size,
                device=self.device,
                dtype=torch.float32
            ),
            indexing="ij"
        )

        distance_squared = ((rows - centre_row).square() + (cols - centre_col).square())

        circle = (distance_squared <= radius ** 2).float()

        # [H, W] -> [1, 1, H, W]
        circle = circle.unsqueeze(0).unsqueeze(0)

        # Global conditions
        conditions = torch.cat(
            [
                torch.tensor(
                    [slice_position],
                    device=self.device,
                    dtype=torch.float32
                ),
                proportions
            ]
        )

        # [4] -> [1, 4]
        conditions = conditions.unsqueeze(0)
        return circle, conditions

    def _predict_controlled_noise(
        self,
        z_t,
        timesteps,
        condition,
        modality,
        guidance_scale
    ):
        """
        Predict epsilon using the ControlNet-conditioned backbone w/ CFG.
        """

        control_state = self.controlnet.get_control(
            z_t=z_t,
            timestep=timesteps,
            condition=condition,
            modality=modality
        )

        epsilon_cond = self.backbone.predict_noise(
            z_t=z_t,
            timestep=timesteps,
            control_state=control_state,
            modality=modality,
        )

        if guidance_scale == 1.0:
            return epsilon_cond

        null_modality = torch.full_like(
            modality,
            self.backbone.model.null_class
        )

        epsilon_null = self.backbone.predict_noise(
            z_t=z_t,
            timestep=timesteps,
            control_state=control_state,
            modality=null_modality
        )

        return epsilon_null + guidance_scale * (epsilon_cond - epsilon_null)

    @staticmethod
    def _load_mask_generator(
        mask_config,
        device,
    ):
        """
        Load the trained mask diffusion model, segmentation VAE,
        diffusion schedule, latent scale, and latent shape.
        """

        # Paths

        diffusion_checkpoint_path = Path(mask_config["diffusion"]["checkpoint"])

        vae_checkpoint_path = Path(mask_config["vae"]["checkpoint"])

        if not diffusion_checkpoint_path.is_file():
            raise FileNotFoundError(diffusion_checkpoint_path)

        if not vae_checkpoint_path.is_file():
            raise FileNotFoundError(vae_checkpoint_path)

        # Mask VAE

        vae_checkpoint = torch.load(
            vae_checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        vae_architecture = vae_checkpoint["config"]["vae"]

        mask_vae = build_vae({"vae": vae_architecture})

        mask_vae.load_state_dict(
            vae_checkpoint["vae"],
            strict=True
        )

        mask_vae = mask_vae.to(device=device, dtype=torch.float32)

        mask_vae.requires_grad_(False)
        mask_vae.eval()

        # Mask diffusion model
        diffusion_config = mask_config["diffusion"]

        mask_model = MaskLatentDiffusionUNet(
            latent_channels=int(diffusion_config.get("latent_channels", 4)),
            base_channels=int(diffusion_config.get("base_channels", 128)),
            time_dim=int(diffusion_config.get("time_dim", 512)),
            norm_groups=int(diffusion_config.get("norm_groups", 32)),
            dropout=float(diffusion_config.get("dropout", 0.0)),
        ).to(device)

        diffusion_checkpoint = torch.load(
            diffusion_checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        mask_model.load_state_dict(
            diffusion_checkpoint["model"],
            strict=True
        )

        mask_model.requires_grad_(False)
        mask_model.eval()

        # Latent scale
        if "latent_scale" not in diffusion_checkpoint:
            raise RuntimeError(
                "Mask diffusion checkpoint does not contain 'latent_scale'."
            )

        mask_latent_scale = float(diffusion_checkpoint["latent_scale"])

        # DDPM schedule

        mask_schedule = DDPMSchedule(
            num_timesteps=int(diffusion_config.get("num_timesteps", 1000)),
            beta_start=float(diffusion_config.get("beta_start", 1e-4)),
            beta_end=float(diffusion_config.get("beta_end", 0.02)),
            device=device
        )

        # Latent shape

        image_size = int(mask_config["data"].get("image_size", 256))

        if image_size % mask_vae.spatial_compression != 0:
            raise ValueError(
                "Mask image size must be divisible by the VAE spatial compression."
            )

        latent_size = (image_size // mask_vae.spatial_compression)

        mask_latent_shape = (
            mask_vae.latent_channels,
            latent_size,
            latent_size
        )

        # Check diffusion/VAE compatibility.
        expected_channels = int(
            diffusion_config.get("latent_channels", 4))

        if mask_vae.latent_channels != expected_channels:
            raise ValueError(
                f"Mask VAE has {mask_vae.latent_channels} "
                f"latent channels but diffusion expects "
                f"{expected_channels}."
            )

        return {
            "model": mask_model,
            "vae": mask_vae,
            "schedule": mask_schedule,
            "latent_scale": mask_latent_scale,
            "latent_shape": mask_latent_shape,
        }

    @torch.inference_mode()
    def generate_mri(
        self,
        seg_map,
        modality="t1",
        guidance_scale=None,
        seed=None
    ):
        """
        Generate MRI slices conditioned on segmentation maps.

        Parameters
        ----------
        seg_map:
            Segmentation map(s) with shape:
                [H, W] or [B, H, W], or [B, 1, H, W]
        modality:
            One of: "t1", "t1gd", "t2", "flair".
        guidance_scale:
            Global CFG modality guidance scale
        seed:
            Random seed used for reverse diffusion
        Returns
        -------
        torch.Tensor
            Generated MRI images with shape [B, 1, H, W],
            normalised to [0, 1].
        """
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        if seed is None:
            seed = self.default_seed

        condition = self._prepare_seg_map(seg_map)

        batch_size = condition.shape[0]

        modality_tensor = self._prepare_modality(modality=modality, batch_size=batch_size)

        def predict_noise(z_t, timesteps):
            return self._predict_controlled_noise(
                z_t=z_t,
                timesteps=timesteps,
                condition=condition,
                modality=modality_tensor,
                guidance_scale=guidance_scale
            )

        latents = sample_ddpm_with_predictor(
            predict_noise=predict_noise,
            schedule=self.schedule,
            shape=(batch_size, *self.latent_shape),
            device=self.device,
            seed=seed,
            desc=f"Generating {modality.upper()} MRI"
        )

        images = self.vae.decode(latents / self.latent_scale)
        images = images.float().clamp(-1, 1).add(1.0).div(2.0)

        return images

    @torch.inference_mode()
    def generate_seg_map(
        self,
        slice_position,
        centre_rc,
        radius,
        proportions,
        guidance_scale=None,
        seed=None,
        image_size=256
    ):
        """
        Generate a tumour segmentation map.
        Parameters
        ----------
        slice_position:
            Normalised axial slice position in [0, 1]
        centre_rc:
            Approximate tumour centre as (row, column)
        radius:
            Approximate tumour radius in pixels
        proportions:
            Relative proportions of BraTS tumour labels (1, 2, 4)
        guidance_scale:
            Classifier-free guidance scale. If None, use the pipeline default.
        seed:
            Random sampling seed.

        Returns
        -------
        torch.Tensor
            Generated BraTS segmentation with shape [1, H, W] and labels {0, 1, 2, 4}.
        """

        # Ensure mask generator is available
        if self.mask_model is None:
            raise RuntimeError("Mask diffusion model has not been loaded.")
        if self.mask_vae is None:
            raise RuntimeError("Mask VAE has not been loaded.")
        if self.mask_schedule is None:
            raise RuntimeError("Mask diffusion schedule has not been loaded.")
        if self.mask_latent_shape is None:
            raise RuntimeError("Mask latent shape is not defined.")
        if self.mask_latent_scale is None:
            raise RuntimeError("Mask latent scale is not defined.")

        # Defaults
        if guidance_scale is None:
            guidance_scale = self.default_mask_guidance_scale
        if seed is None:
            seed = self.default_seed

        # Prepare conditioning

        circle, conditions = self._prepare_mask_conditions(
            slice_position=slice_position,
            centre_rc=centre_rc,
            radius=radius,
            proportions=proportions,
            image_size=image_size
        )

        # Diffusion sampling
        latents = sample_mask_ddpm(
            model=self.mask_model,
            schedule=self.mask_schedule,
            shape=(1, *self.mask_latent_shape),
            device=self.device,
            circle=circle,
            conditions=conditions,
            guidance_scale=guidance_scale,
            seed=seed
        )

        # Decode segmentation latent

        decoded = self.mask_vae.decode(latents / self.mask_latent_scale)

        if decoded.shape[1] != 4:
            raise RuntimeError(
                "Expected mask VAE to decode to four class channels, "
                f"but got shape {tuple(decoded.shape)}."
            )

        # VAE output channels correspond to internal classes: 0, 1, 2, 3
        class_indices = decoded.argmax(dim=1)

        # Convert internal class 3 back to BraTS label 4.
        seg_map = restore_mask_labels(class_indices)

        return seg_map

    @classmethod
    def from_config(
        cls,
        mri_config_path,
        mask_config_path=None,
        device=None,
        use_ema=True
    ):
        """
        Build the complete MRI generative pipeline from a YAML config.

        Parameters
        ----------
        mri_config_path:
            Path to the MRI pipeline/training YAML configuration
        mask_config_path:
            Path to the Mask pipelein/training YAML configuration
        device:
            Torch device. If None, choose CUDA, then MPS, then CPU
        use_ema:
            If True, use EMA ControlNet weights when available

        Returns
        -------
        BrainMRIGenerativePipeline
        """

        mri_config_path = Path(mri_config_path)
        with open(mri_config_path, "r") as file:
            config = yaml.safe_load(file)

        mask_config = None

        if mask_config_path is not None:
            mask_config_path = Path(mask_config_path)
            with open(mask_config_path, "r") as file:
                mask_config = yaml.safe_load(file)

        # Device

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        # Checkpoint paths

        backbone_path = Path(config["diffusion"]["checkpoint"])
        vae_path = Path(config["vae"]["checkpoint"])
        control_path = config.get("control", {}).get("checkpoint")

        if control_path is None:
            raise ValueError(
                "No ControlNet checkpoint was specified. "
                "Add 'control.checkpoint' to the config."
            )

        control_path = Path(control_path)

        # Backbone

        diffusion_model = build_diffusion(config).to(device)

        backbone = BrainMRIBackbone(
            checkpoint_path=backbone_path,
            diff_model=diffusion_model,
            architecture=config["diffusion"]["architecture"],
            device=device
        )

        backbone.model.requires_grad_(False)
        backbone.model.eval()

        # ControlNet

        architecture = config["diffusion"]["architecture"]

        condition_channels = int(
            config.get("control", {}).get("condition_channels", 1)
        )

        if architecture == "latent_v1":
            controlnet = BrainMRIControlNet_v1(
                backbone=backbone,
                device=device,
                condition_channels=condition_channels,
            ).to(device)

        elif architecture == "latent_v2":
            controlnet = BrainMRIControlNet_v2(
                backbone=backbone,
                device=device,
                condition_channels=condition_channels,
            ).to(device)
        else:
            raise ValueError(f"Unknown diffusion architecture: {architecture}")

        control_checkpoint = torch.load(
            control_path,
            map_location=device,
            weights_only=False
        )

        control_state_dict = control_checkpoint.get("controlnet")

        if (
            use_ema
            and control_checkpoint.get("ema_controlnet") is not None
        ):
            control_state_dict = control_checkpoint["ema_controlnet"]

        if control_state_dict is None:
            raise RuntimeError("ControlNet checkpoint does not contain 'controlnet' weights.")

        missing_keys, unexpected_keys = controlnet.load_state_dict(control_state_dict, strict=False)

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        controlnet.requires_grad_(False)
        controlnet.eval()

        # MRI VAE

        vae = build_vae(config).to(device)

        vae_checkpoint = torch.load(
            vae_path,
            map_location=device,
            weights_only=False
        )

        load_vae_state_dict(vae, vae_checkpoint)

        vae.requires_grad_(False)
        vae.eval()

        # Latent scale

        latent_scale = control_checkpoint.get("latent_scale")

        if latent_scale is None:

            backbone_checkpoint = torch.load(
                backbone_path,
                map_location="cpu",
                weights_only=False
            )

            latent_scale = backbone_checkpoint.get("latent_scale")

        if latent_scale is None:
            raise RuntimeError(
                "Could not determine latent_scale from either "
                "the ControlNet or backbone checkpoint."
            )

        latent_scale = float(latent_scale)

        # DDPM schedule

        diffusion_config = config["diffusion"]

        schedule = DDPMSchedule(
            num_timesteps=int(diffusion_config.get("num_timesteps", 1000)),
            beta_start=float(diffusion_config.get("beta_start", 1e-4)),
            beta_end=float(diffusion_config.get("beta_end", 2e-2)),
            device=device
        )

        # Latent dimensions

        image_size = int(config["data"]["image_size"])
        latent_channels = int(config["vae"]["latent_channels"])
        channel_mults = config["vae"]["channel_mults"]
        downsample_factor = 2 ** (len(channel_mults) - 1)
        latent_size = image_size // downsample_factor

        latent_shape = (latent_channels, latent_size, latent_size)

        # Mask Paths
        mask_model = None
        mask_vae = None
        mask_schedule = None
        mask_latent_scale = None
        mask_latent_shape = None
        default_mask_guidance_scale = 1.0

        if mask_config is not None:
            mask_components = cls._load_mask_generator(
                mask_config=mask_config,
                device=device
            )
            mask_model = mask_components["model"]
            mask_vae = mask_components["vae"]
            mask_schedule = mask_components["schedule"]
            mask_latent_scale = mask_components["latent_scale"]
            mask_latent_shape = mask_components["latent_shape"]

            default_mask_guidance_scale = float(mask_config.get("evaluation", {}).get("guidance_scale", 1.0))
        # Construct high-level pipeline

        return cls(
            backbone=backbone,
            controlnet=controlnet,
            vae=vae,
            schedule=schedule,
            latent_shape=latent_shape,
            latent_scale=latent_scale,
            device=device,
            default_guidance_scale=float(diffusion_config.get("guidance_scale", 1.0)),
            default_seed=int(config.get("seed", 42)),
            mask_model=mask_model,
            mask_vae=mask_vae,
            mask_schedule=mask_schedule,
            mask_latent_shape=mask_latent_shape,
            mask_latent_scale=mask_latent_scale,
            default_mask_guidance_scale=default_mask_guidance_scale
        )