import torch
import numpy as np  

class DDPMSampler:

    def __init__(self,
        generator: torch.Generator,
        num_training_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_bounds: tuple[float, float] = (1e-4, 2e-2), # (beta_1, beta_T)
    ):  
        self.dtype = torch.float32 # TODO: check if this is correct
        self.generator = generator
        self.num_training_timesteps = num_training_timesteps
        
        self.timesteps = torch.arange(
            start=0,
            end=self.num_training_timesteps,
            step=1,
            dtype=torch.int64
        ).flip(dims=[0]) # Writing the timesteps in reverse order, [::-1] slicing does not work on cuda tensors

        self.beta_bounds = beta_bounds
        self.beta_schedule = beta_schedule
        self.betas = self.scheduler(self.beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = self.alphas.cumprod(dim=0)

    def scheduler(self, schedule_type: str = 'linear') -> torch.FloatTensor:
        if schedule_type == 'linear':
            return torch.linspace( # we are first taking sqrt and then squaring to keep sqrt(betas) linear
                self.beta_bounds[0] ** 0.5,
                self.beta_bounds[1] ** 0.5,
                self.num_training_timesteps,
                dtype=self.dtype
            ) ** 2 
        elif schedule_type == 'cosine':
            raise NotImplementedError("Cosine schedule is not implemented")
        else:
            raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    def set_inference_timesteps(self, num_inference_timesteps: int = 50):
        self.num_inference_timesteps = num_inference_timesteps
        step_ratio = self.num_training_timesteps // self.num_inference_timesteps # 20 for 1000 & 50
        timesteps = torch.arange(
            start=0, 
            end=self.num_training_timesteps,
            step=step_ratio,
            dtype=torch.int64
        ).flip(dims=[0]) # Writing the timesteps in reverse order

        self.timesteps = timesteps # Updating the timesteps for Inference
        print(f"self.timesteps are set to Inference Timesteps: {self.num_inference_timesteps}")

    def _get_prev_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_training_timesteps // self.num_inference_timesteps
        return prev_t

    def denoise_step(self, timestep: int, model_outs: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: The timestep to denoise where 1 < t <= T
            model_outs: The model outputs
            x_t: The noisy samples

        Returns:
            The denoised samples
        """
        t = timestep
        prev_t = self._get_prev_timestep(t)

        beta_t = self.betas[timestep].to(x_t.device)
        alpha_cumprod_t = self.alpha_cumprod[timestep].to(x_t.device)
        alpha_cumprod_t_prev = self.alpha_cumprod[prev_t].to(x_t.device) if prev_t >= 0 else torch.tensor(1.0).to(x_t.device) # Very Important!
        sqrt_alpha_t = (1 - beta_t) ** 0.5
        one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t)
        one_minus_alpha_cumprod_t_prev = (1 - alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t = one_minus_alpha_cumprod_t ** 0.5

        while len(beta_t.shape) < len(x_t.shape):
            beta_t = beta_t.unsqueeze(-1)
        while len(sqrt_alpha_t.shape) < len(x_t.shape):
            sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)

        while len(one_minus_alpha_cumprod_t.shape) < len(x_t.shape):
            one_minus_alpha_cumprod_t = one_minus_alpha_cumprod_t.unsqueeze(-1)
        while len(one_minus_alpha_cumprod_t_prev.shape) < len(x_t.shape):
            one_minus_alpha_cumprod_t_prev = one_minus_alpha_cumprod_t_prev.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_cumprod_t.shape) < len(x_t.shape):
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        mean = (1/sqrt_alpha_t) * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * model_outs)
        variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
        variance = variance.clamp(min=1e-20) # Clamping the variance to avoid division by zero

        while len(variance.shape) < len(x_t.shape):
            variance = variance.unsqueeze(-1)

        z = torch.randn(x_t.shape, generator=self.generator, device=x_t.device, dtype=x_t.dtype)
        x_denoised = mean + (variance ** 0.5) * z
        return x_denoised
    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # Setting the device and dtype of the tensors
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_cumprod_t = alpha_cumprod[timesteps] ** 0.5 # Shape: (batch_size,)
        while len(sqrt_alpha_cumprod_t.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1) # Shape: (batch_size, 1,  ...)

        sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod[timesteps]) ** 0.5 # Shape: (batch_size,) For some reason, the shape came back to (batch_size,)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.flatten()
        while len(sqrt_one_minus_alpha_cumprod_t.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1) # Shape: (batch_size, 1,  ...)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        )
        noisy_samples = sqrt_alpha_cumprod_t * original_samples + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_samples, noise

if __name__ == "__main__":
    sampler = DDPMSampler(
        generator=torch.Generator(device='cuda'),
        num_training_timesteps=1000,
        beta_schedule='linear',
        beta_bounds=(1e-4, 2e-2),
    )
    sampler.set_inference_timesteps(num_inference_timesteps=50)

    x_0 = torch.randn(
        (5, 3, 256, 256),
        generator=sampler.generator,
        device='cuda',
        dtype=sampler.dtype
    )
    timesteps = torch.randint(
        low=0,
        high=sampler.num_training_timesteps,
        size=(5,),
        device='cuda',
        dtype=torch.int64
    )
    print(f"x_0.shape: {x_0.shape}")
    print(f"timesteps.shape: {timesteps.shape}")
    x_t, noise = sampler.add_noise(x_0, timesteps)
    print(f"x_t.shape: {x_t.shape}")
    print(f"noise.shape: {noise.shape}")