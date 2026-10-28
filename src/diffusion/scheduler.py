import torch

class DDPMSchedule:

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None
    ):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            beta_start, # beta_0
            beta_end,   # beta_T
            num_timesteps,
            dtype=torch.float32,
            device=device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Samples z_t from the forward noising process"""

        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)

        return sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise

