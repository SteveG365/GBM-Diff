import torch
from tqdm import tqdm

def get_raw_model(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model

@torch.no_grad()
def sample_ddpm(
    model,
    schedule,
    shape,
    device,
    modality_id,
    guidance_scale=1.0,
    seed=42
):
    """Ancestral DDPM sampling with classifier-free guidance."""
    sampling_model = get_raw_model(model)
    sampling_model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Initial noise sample
    z = torch.randn(shape, device=device, generator=generator)
    
    modality = torch.full((shape[0],), modality_id, device=device, dtype=torch.long)
    null_modality = torch.full((shape[0],), sampling_model.null_class, device=device, dtype=torch.long)

    for t in tqdm(
        reversed(range(schedule.num_timesteps)),
        total=schedule.num_timesteps,
        desc="Sampling",
        leave=False
    ):
        timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)

        epsilon_cond = sampling_model(z, timesteps, modality).clone()
        if guidance_scale == 1.0:
            epsilon_pred = epsilon_cond
        else:
            epsilon_null = sampling_model(z, timesteps, null_modality).clone()
            epsilon_pred = epsilon_null + guidance_scale * (epsilon_cond - epsilon_null)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]

        # See Eq. 2.15
        mean = torch.rsqrt(alpha_t) * (z - beta_t / torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred)

        if t > 0:
            alpha_bar_prev = schedule.alpha_bars[t - 1]
            # See Eq. 2.9
            posterior_variance = (beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
            z = mean + torch.sqrt(posterior_variance) * noise
        else:
            z = mean

    return z


@torch.no_grad()
def sample_mask_ddpm(
    model,
    schedule,
    shape,
    device,
    circle,
    conditions,
    guidance_scale=1.0,
    seed=42
):
    """Ancestral Mask DDPM sampling with classifier-free guidance"""
    sampling_model = get_raw_model(model)
    sampling_model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Initial noise sample
    z = torch.randn(shape, device=device, generator=generator)

    circle = circle.to(device=device, dtype=torch.float32)
    conditions = conditions.to(device=device, dtype=torch.float32)

    for t in tqdm(
        reversed(range(schedule.num_timesteps)),
        total=schedule.num_timesteps,
        desc="Sampling",
        leave=False
    ):
        timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)

        epsilon_cond = sampling_model(z, timesteps, circle, conditions, conditions_present=True).clone()
        if guidance_scale == 1.0:
            epsilon_pred = epsilon_cond
        else:
            epsilon_null = sampling_model(z, timesteps, circle, conditions, conditions_present=False).clone()
            epsilon_pred = epsilon_null + guidance_scale * (epsilon_cond - epsilon_null)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]

        # See Eq. 2.15
        mean = torch.rsqrt(alpha_t) * (z - beta_t / torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred)

        if t > 0:
            alpha_bar_prev = schedule.alpha_bars[t - 1]
            # See Eq. 2.9
            posterior_variance = (beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=generator)
            z = mean + torch.sqrt(posterior_variance) * noise
        else:
            z = mean

    return z

@torch.no_grad()
def sample_ddpm_with_predictor(
    predict_noise,
    schedule,
    shape,
    device,
    seed=42,
    desc="Sampling"
):
    """
    Generic ancestral DDPM sampler.

    `predict_noise` must be a callable with signature:
        epsilon_pred = predict_noise(z_t, timesteps)
    where:
        z_t:        [B, C, H, W]
        timesteps:  [B]
    and epsilon_pred must have the same shape as z_t.

    This keeps the DDPM reverse process independent of how the
    noise prediction is produced
    """

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # z_T ~ N(0, I)
    z = torch.randn(shape, device=device, generator=generator)

    for t in tqdm(
        reversed(range(schedule.num_timesteps)),
        total=schedule.num_timesteps,
        desc=desc,
        leave=False
    ):
        timesteps = torch.full(
            (shape[0],),
            t,
            device=device,
            dtype=torch.long
        )

        epsilon_pred = predict_noise(z, timesteps)

        if epsilon_pred.shape != z.shape:
            raise ValueError(
                "Noise predictor returned incorrect shape: "
                f"expected {z.shape}, got {epsilon_pred.shape}"
            )

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]

        # DDPM posterior mean: Eq 2.15
        mean = torch.rsqrt(alpha_t) * (z - beta_t / torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred)

        if t > 0:
            alpha_bar_prev = schedule.alpha_bars[t - 1]

            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

            noise = torch.randn(
                z.shape,
                device=device,
                dtype=z.dtype,
                generator=generator
            )

            z =  mean + torch.sqrt(posterior_variance) * noise

        else:
            z = mean

    return z