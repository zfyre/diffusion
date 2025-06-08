import jax
import jax.numpy as jnp

class GaussianScheduler:
    def __init__(self,
                type:str='linear',
                beta_bounds:tuple[float]=(0.0001,0.02), # Taking the schedule limits from DDPM -> (beta_1, beta_T)
                diffusion_steps:int=40,# 40 steps taken in DPM,
                batch_size:int=16,
                is_trainable:bool=False,
            ):
        self.type = type
        self.beta_bounds = beta_bounds
        self.steps = diffusion_steps
        self.batch_size = batch_size
        self.is_trainable = is_trainable
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

    def __init__(self, diffusion_steps, batch_size, scheduler=None):
        if scheduler is None:
            self.scheduler = GaussianScheduler(diffusion_steps=diffusion_steps, batch_size=batch_size)
            print(f"[INFO] Using default GaussianScheduler: {self.scheduler}")
        else:
            print(f"[INFO] Using provided Scheduler: {scheduler}")
            self.scheduler = scheduler
    
    # TODO: currently the dtype of the timesteps in int but it'll be float when used with models, and eventually it'll be between 0 and 1
    def forward(self, x_0: jnp.ndarray, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        N, D = x_0.shape
        assert t.shape == (N,), "t should be a 1D array with the same batch size as x_0"
        assert jnp.all(t <= self.scheduler.steps) and jnp.all(t >= 1), "t should be less than or equal to the number of diffusion steps"
        
        t = t-1  # Adjusting t to be zero-indexed
        
        # Since the N(0,I) has identiy covariance hence each noise can be sampled independently, hence multivariate normal is not needed
        key, subkey = jax.random.split(key) # Important to split key to ensure diverse random numbers
        epsilon = jax.random.normal(subkey, shape=x_0.shape)
        
        alpha_cumprod_t = jnp.take(self.scheduler.alpha_cumprod, t)  # Shape: (batch_size,)
        alpha_cumprod_t = jnp.expand_dims(alpha_cumprod_t, axis=-1)  # Shape: (batch_size, 1)
        assert alpha_cumprod_t.shape == (N, 1), "alpha_cumprod_t should have shape (batch_size, 1)"
        assert not jnp.any(jnp.isnan(alpha_cumprod_t)), "NaN values detected in alpha_cumprod_t"

        x_t = jnp.sqrt(alpha_cumprod_t) * x_0 + jnp.sqrt(1 - alpha_cumprod_t) * epsilon
        assert not jnp.any(jnp.isnan(x_t)), "NaN values detected in x_t"
        
        return x_t, key  # Returning the noisy sample and the key for further operations

    def reverse(self, x_t: jnp.ndarray, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        assert self.scheduler is not None
        pass

    def sample(self, shape: tuple, t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        assert self.scheduler is not None
        pass
     
class Diffusion:
    def __init__(self, model, diffusion_steps=1000, diffusion_kernel=None, device=None):
        self.device = jax.default_backend() if device is None else device
        self.model = model
        self.steps = diffusion_steps
        self.kernel = GaussianKernel(diffusion_steps=diffusion_steps, batch_size=16) if diffusion_kernel is None else diffusion_kernel

    def forward(self, x_0: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        # Returns the `forward trajectory` in form of jnp.ndarray
        forward_trajectory = []
        N, D = x_0.shape
        for step in range(self.steps):
            timestep = jnp.broadcast_to(jnp.array([step+1]), shape=(N,))
            x_t, key = self.kernel.forward(x_0, timestep, key)
            print(f"[INFO] Step {step+1}/{self.steps}, x_t shape: {x_t.shape}, key: {key}, device: {self.device}, dtype: {x_t.dtype}")
            forward_trajectory.append(x_t) # [T, N, D]
        return jnp.stack(forward_trajectory, axis=1)  # [N, T, D]

    def reverse(self, x_t: jnp.ndarray, key:jax.Array) -> jnp.ndarray:
        # returns the 'reverse trjectory` in form of jnp.ndarray
        return self.model.forward(x_t, key)

    def sample(self, shape: tuple, key:jax.Array) -> jnp.ndarray:
        x_T = self.kernel.sample(shape, t=self.steps, key=key)
        return self.reverse(x_T, key=key)  # The first instance is the result of sampling from the trained model


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
    

