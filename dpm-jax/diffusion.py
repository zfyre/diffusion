import jax
import jax.numpy as jnp
from model import ReverseDiffusion
import jax.random as random
import flax.linen as nn

class GaussianScheduler:
    def __init__(self,
                type:str='linear',
                beta_bounds:tuple[float]=(0.0001,0.02), # Taking the schedule limits from DDPM -> (beta_1, beta_T)
                diffusion_steps:int=40,# 40 steps taken in DPM,
                batch_size:int=16,
                is_trainable:bool=False,
                verbose:bool=False
            ):
        self.type = type
        self.beta_bounds = beta_bounds
        self.steps = diffusion_steps
        self.batch_size = batch_size
        self.is_trainable = is_trainable
        self.verbose = verbose
        if is_trainable:
            # TODO: initialize the model whih will produce the schedule
            pass
        else:
            self.betas = self.get_betas()
            self.alphas = 1 - self.betas
            self.alpha_cumprod = jnp.cumprod(self.alphas)
            assert not jnp.any(jnp.isnan(self.alpha_cumprod)), "NaN values detected in alpha_cumprod"
    
    def get_betas(self):
        betas=None
        if self.type == 'linear':
            betas = jnp.linspace(self.beta_bounds[0], self.beta_bounds[1], self.steps)
        elif self.type == 'cosine':
            # TODO: Understand the need and do later
            pass
        
        assert betas is not None, "Beta schedule type not recognized. Use 'linear' or 'cosine'."
        assert not jnp.any(jnp.isnan(betas)), "NaN values detected in betas"
        return betas
    
    def forward(self):
        if self.is_trainable:
        # TODO: call the forward of the model which produces schedule
            pass
        else:
            return jnp.broadcast_to(self.alpha_cumprod, (self.batch_size, self.steps))

class GaussianKernel:

    def __init__(self, diffusion_steps, batch_size, scheduler=None, verbose=False):
        self.verbose = verbose
        if scheduler is None:
            self.scheduler = GaussianScheduler(diffusion_steps=diffusion_steps, batch_size=batch_size, verbose=self.verbose)
            if self.verbose:
                print(f"[INFO] Using default GaussianScheduler: {self.scheduler}")
        else:
            if self.verbose:
                print(f"[INFO] Using provided Scheduler: {scheduler}")
            self.scheduler = scheduler
    
    # TODO: currently the dtype of the timesteps in int but it'll be float when used with models, and eventually it'll be between 0 and 1
    def forward(self, x_0: jnp.ndarray, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        B, C, H, W = x_0.shape
        # print(f"t: {t.shape}, B: {B}")
        assert t.shape[0] == B, "t should have the same batch size as x_0"
        # assert jnp.all(t <= self.scheduler.steps) and jnp.all(t >= 1), "t should be less than or equal to the number of diffusion steps"
        
        t = t-1  # Adjusting t to be zero-indexed
        
        # Since the N(0,I) has identiy covariance hence each noise can be sampled independently, hence multivariate normal is not needed
        key, subkey = jax.random.split(key) # Important to split key to ensure diverse random numbers
        epsilon = jax.random.normal(subkey, shape=x_0.shape)
        
        alpha_cumprod_t = jnp.take(self.scheduler.alpha_cumprod, t)  # Shape: (batch_size,)
        alpha_cumprod_t = jnp.expand_dims(alpha_cumprod_t, axis=(-1, -2, -3))  # Shape: (batch_size, 1, 1, 1)
        assert alpha_cumprod_t.shape == (B, 1, 1, 1), "alpha_cumprod_t should have shape (batch_size, 1, 1, 1)"
        # assert not jnp.any(jnp.isnan(alpha_cumprod_t)), "NaN values detected in alpha_cumprod_t"

        alpha_cumprod_t = jnp.broadcast_to(alpha_cumprod_t, (B, C, H, W))
        x_t = jnp.sqrt(alpha_cumprod_t) * x_0 + jnp.sqrt(1 - alpha_cumprod_t) * epsilon
        # assert not jnp.any(jnp.isnan(x_t)), "NaN values detected in x_t"
        
        return x_t, key  # Returning the noisy sample and the key for further operations

    def reverse(self, x_t: jnp.ndarray, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        assert self.scheduler is not None
        pass

    def sample(self, shape: tuple, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        assert self.scheduler is not None
        pass
     
class Diffusion:
    def __init__(self, model: nn.Module, input_shape, key, diffusion_steps=1000, diffusion_kernel=None, device=None, verbose=False):
        self.verbose = verbose
        self.device = jax.default_backend() if device is None else device
        self.steps = diffusion_steps
        self.kernel = GaussianKernel(diffusion_steps=diffusion_steps, batch_size=16, verbose=self.verbose) if diffusion_kernel is None else diffusion_kernel
        self.model = model
        self.B, self.C, self.H, self.W = input_shape
        self.key = key # This key is necessary to initialize the model
        self.variables = self.model.init(
            self.key,
            jnp.ones((self.B, self.C, self.H, self.W)),
            t=jnp.ones((self.B,), dtype=jnp.int32),
            beta_t=jnp.ones((self.B,)),
        )

    def forward_trajectory(self, x_0: jnp.ndarray, key=None) -> tuple:
        if key is None:
            key = self.key
        forward_trajectory = []
        B, C, H, W = x_0.shape
        for step in range(self.steps):
            timestep = jnp.broadcast_to(jnp.array([step+1]), shape=(B,))
            x_t, key = self.kernel.forward(x_0, timestep, key)
            if self.verbose:
                print(f"[INFO] Step {step+1}/{self.steps}, x_t shape: {x_t.shape}, key: {key}, device: {self.device}, dtype: {x_t.dtype}")
            forward_trajectory.append(x_t)
        return jnp.stack(forward_trajectory, axis=1), key  # [B, T, C, H, W], key

    def reverse_trajectory(self, x_T: jnp.ndarray, key=None) -> tuple:
        if key is None:
            key = self.key
        x_t = x_T
        reverse_trajectory = [x_T]
        B, C, H, W = x_t.shape
        for step in range(self.steps):
            beta_t = self.kernel.scheduler.betas[step]
            beta_t = jnp.broadcast_to(beta_t, shape=(B,))
            timestep = jnp.broadcast_to(jnp.array([step+1]), shape=(B,))
            x_t_minus_1, _, _, key = self.reverse(self.variables, x_t, timestep, key, sigma=self.kernel.scheduler.betas[step])
            if self.verbose:
                print(f"[INFO] Step {step+1}/{self.steps}, x_t shape: {x_t.shape}, key: {key}, device: {self.device}, dtype: {x_t.dtype}")
            reverse_trajectory.append(x_t_minus_1)
            x_t = x_t_minus_1
        return jnp.stack(reverse_trajectory[::-1], axis=1), key  # [B, T, C, H, W], key

    def reverse(self, params, x_t: jnp.ndarray, t: jnp.ndarray, key=None, sigma=None) -> tuple:
        if key is None:
            key = self.key
        B, C, H, W = x_t.shape
        t = t-1 # Adjusting t to be zero-indexed
        beta_t = jnp.take(self.kernel.scheduler.betas, t)
        # jax.debug.print("beta_t: {}", beta_t)
        beta_t = jnp.broadcast_to(beta_t, shape=(B,))
        timesteps = t
        if sigma is None:
            mu, sigma = self.model.apply(params, x_t, timesteps, beta_t=beta_t)
        else:
            mu, _ = self.model.apply(params, x_t, timesteps, beta_t=beta_t)
        
        # Sampling from the mean and sigma
        key, subkey = jax.random.split(key)
        x_t_minus_1 = mu + jnp.sqrt(sigma) * jax.random.normal(subkey, mu.shape)
        return x_t_minus_1, mu, sigma, key

    def forward(self, x_0: jnp.ndarray, t: jnp.ndarray, key=None) -> tuple:
        if key is None:
            key = self.key
        B, C, H, W = x_0.shape
        timesteps = t
        x_t, key = self.kernel.forward(x_0, timesteps, key)
        return x_t, key
    
    def get_mu_sigma_original(self, x_0: jnp.ndarray, t: jnp.ndarray, x_t: jnp.ndarray) -> tuple:
        B, C, H, W = x_0.shape
        t = t-1 # Adjusting t to be zero-indexed
        alpha_cumprod_t = jnp.take(self.kernel.scheduler.alpha_cumprod, t)[:, None, None, None]
        alpha_cumprod_t_minus_1 = jnp.take(self.kernel.scheduler.alpha_cumprod, t-1)[:, None, None, None]
        beta_t = jnp.take(self.kernel.scheduler.betas, t)[:, None, None, None]
        print(f"alpha_cumprod_t: {alpha_cumprod_t.shape}, alpha_cumprod_t_minus_1: {alpha_cumprod_t_minus_1.shape}, beta_t: {beta_t.shape}")
        mu = (jnp.sqrt(alpha_cumprod_t_minus_1) * beta_t / (1-alpha_cumprod_t)) * x_0 + jnp.sqrt(alpha_cumprod_t) * (1-alpha_cumprod_t_minus_1) * x_t / (1-alpha_cumprod_t)
        sigma_squared = (1 - alpha_cumprod_t_minus_1) / (1 - alpha_cumprod_t) * beta_t

        return mu, sigma_squared
    
    def sample(self, shape: tuple) -> jnp.ndarray:
        pass


# Can work on predefined toy problems:
# 1. Make Blob dataset
# 2. Swiss Roll as done originally in paper (should be baseline)
# 3. Scaling up to MNIST
# 4. Can work with CIFAR-10 or 3D datasets as well 

# if __name__ == "__main__":
    # gs = GaussianScheduler()
    # alpha_cumprod = gs.forward()
    # print(alpha_cumprod.shape)              
    # print(alpha_cumprod.device) 
    
    # gk = GaussianKernel()
    # x_0 = jnp.ones((8, 2))  # Example input (batch_size=8, feature_dim=2)
    # print(f"x_0: {x_0}")
    # timesteps = jnp.broadcast_to(jnp.array([1]), (8,))
    # print(f"timesteps: {timesteps}")
    # key = jax.random.PRNGKey(0)  # Random key for JAX
    # x_t, key = gk.forward(x_0, timesteps, key)
    # print(f"x_t: {x_t}")
    # x_t, key = gk.forward(x_0, timesteps, key)
    # print(f"x_t: {x_t}")
    
    # diff = Diffusion(model=None, diffusion_steps=10)
    # x_0 = jnp.ones((8, 2))  
    # key = jax.random.PRNGKey(0)  # Random key for JAX
    # forward_trajectory = diff.forward(x_0, key)
    # print(f"Forward trajectory shape: {forward_trajectory.shape}")
    # print(f"Forward trajectory: {forward_trajectory}")  
    
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)  # Random key for JAX
    model = ReverseDiffusion(features=16, channels=3, diffusion_steps=10)
    diff = Diffusion(model=model, input_shape=(2, 3, 4, 8), key=key, diffusion_steps=10)
    x_0 = jnp.ones((2, 3, 4, 8))  # (B, C, H, W)
    forward_trajectory, key = diff.forward(x_0)
    print(f"Forward trajectory shape: {forward_trajectory.shape}")

    reverse_trajectory, key = diff.reverse(jnp.ones((2, 3, 4, 8)), key)
    print(f"Reverse trajectory shape: {reverse_trajectory.shape}")


