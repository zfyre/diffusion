import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.random as random

class PixelWiseDense(nn.Module):
    """Pixel-wise dense block with leaky ReLU as in the reference implementation."""
    features: int
    kernel_size: tuple = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        B, C, H, W = x.shape
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        x_nhwc = nn.Conv(self.features, kernel_size=self.kernel_size, padding='SAME')(x_nhwc)
        # x = nn.leaky_relu(x, negative_slope=0.02)  # Using leaky ReLU as in reference
        pixel_wise_dense_out = nn.tanh(x_nhwc)
        return jnp.transpose(pixel_wise_dense_out, (0, 3, 1, 2))

class DenseBlock(nn.Module):
    """Dense block with leaky ReLU activation."""
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x_flat = x.reshape((x.shape[0], -1))
        dense_output = nn.Dense(self.features)(x_flat)
        output = nn.tanh(dense_output)
        return output    

class MultiScaleConv(nn.Module):
    """
    Multi-scale convolutional processing as in the original implementation.
    """
    features: int               # number of output channels
    num_scales: int = 3         # how many scales (e.g., 3 = [1x, 1/2x, 1/4x])
    kernel_size: tuple = (3, 3) # kernel size for each conv
        
    @nn.compact
    def __call__(self, x_t):
        # x: shape (B, C, H, W)
        B, C, H, W = x_t.shape
        outputs = []
        # Convert to NHWC format for JAX
        x_nhwc = jnp.transpose(x_t, (0, 2, 3, 1))  # (B, H, W, C)
        
        for s in range(self.num_scales):
            scale_factor = 2**s
            
            # Downsampling by pooling according to the given scale
            if scale_factor > 1:
                x_scaled = nn.avg_pool(x_nhwc,
                    window_shape=(scale_factor, scale_factor),
                    strides=(scale_factor, scale_factor),
                    padding='SAME',
                )
            else:
                x_scaled = x_nhwc
            
            # Convolution at this scale using softplus as in the paper
            conv = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                name=f"conv_scale_{s}"
            )(x_scaled)
            conv = nn.softplus(conv)  # Using softplus as mentioned in the paper
            
            # Upsampling back to original resolution
            if scale_factor > 1:
                conv = jax.image.resize(
                    conv,
                    shape=(B, H, W, self.features),
                    method='nearest'
                )
            outputs.append(conv)
            
        # Sum across all scales
        output = jnp.array(outputs).sum(axis=0)
        return jnp.transpose(output, (0, 3, 1, 2))  # Back to (B, C, H, W)
class ReverseDiffusion(nn.Module):
    """Reverse diffusion model as in the paper."""
    features: int # features per channel
    channels: int # number of channels
    diffusion_steps: int # number of diffusion steps
    num_multi_scale_conv_layers: int = 3
    num_pixel_wise_dense_layers: int = 3
    use_dense_block: bool = True
    num_scales: int = 3 # number of scales for multi-scale convolution
    alpha: float = 1.5 # (1.0 for minimum overlap, 2.0 for maximum overlap) 

    # For storing variables
    variables: dict = None

    def setup(self):
        # Precompute bump weights for all timesteps (0, ..., diffusion_steps-1)
        T = self.diffusion_steps
        bump_J = self.features 
        bump_w = self.alpha * T/bump_J # The width bump_w controls how much each bump overlaps with its neighbors.

        tau = jnp.linspace(0, T, bump_J + 2)[1:-1]  # (J,)
        t_grid = jnp.arange(T)[:, None]  # (T, 1)
        tau_grid = tau[None, :]          # (1, J)
        exps = jnp.exp(-0.5 / (bump_w ** 2) * (t_grid - tau_grid) ** 2)  # (T, J)
        bump_weights_all = exps / jnp.sum(exps, axis=-1, keepdims=True)  # (T, J)
        # print("bump_weights_all shape:", bump_weights_all.shape)
        self.bump_weights_all = bump_weights_all  # (T, J)

    @nn.compact
    def backbone(self, x: jnp.ndarray):
        for _ in range(self.num_multi_scale_conv_layers):
            x_multiscale = MultiScaleConv(features=2 * self.features * self.channels)(x)
            if self.use_dense_block:
                x_dense = DenseBlock(features=2 * self.features * self.channels)(x) # (B, self.features)
                x_dense = x_dense[:, :, None, None]
                x = x_multiscale + jnp.broadcast_to(x_dense, x_multiscale.shape)
            else:
                x = x_multiscale
        for _ in range(self.num_pixel_wise_dense_layers):
            x = PixelWiseDense(features=2 * self.features * self.channels)(x)
        return x # (B, 2*features*C, H, W)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, beta_t: jnp.ndarray, key: jnp.ndarray):
        B, C, H, W = x.shape
        assert C == self.channels, "Number of channels in input must match model's channels"
        x_init = x
        x = self.backbone(x) # (B, 2*features*C, H, W)
        y_mu, y_sigma = jnp.split(x, 2, axis=1) # (B, features*C, H, W)
        y_mu = y_mu.reshape(B, self.features, self.channels, H, W) # (B, features, C, H, W)
        y_sigma = y_sigma.reshape(B, self.features, self.channels, H, W) # (B, features, C, H, W)

        # print(t.shape, self.bump_weights_all.shape)
        g = self.bump_weights_all[t] # (1, features)
        # for broadcasting
        g = g[:, :, None, None, None] # (1, features, 1, 1, 1)
        z_mu =  jnp.sum(y_mu * g, axis=1) # (B, C, H, W)
        z_sigma = jnp.sum(y_sigma * g, axis=1) # (B, C, H, W)

        beta_t = beta_t[:, None, None, None]
        sigma = nn.sigmoid(z_sigma + jnp.log(jnp.exp(beta_t) - 1))
        mu = (x_init - z_mu) * (1 - sigma) + z_mu

        # Sampling from the mean and sigma
        key, subkey = jax.random.split(key)
        x_t = mu + sigma * jax.random.normal(subkey, mu.shape)

        return x_t, mu, sigma, key




if __name__ == "__main__":
    # Testing MultiScaleConv
    model = MultiScaleConv(features=16)
    x = jnp.ones((1, 3, 4, 8))
    variables = model.init(random.key(0), x)
    y1 = model.apply(variables, x)
    print("MultiScaleConv output shape:", y1.shape)

    # Testing DenseBlock
    model = DenseBlock(features=16)
    x = jnp.ones((1, 3, 4, 8))
    variables = model.init(random.key(0), x)
    y2 = model.apply(variables, x)
    print("DenseBlock output shape:", y2.shape)

    # Combining MultiScaleConv and DenseBloc2k
    y2 = y2[:, :, None, None]
    # print("y2 shape:", y2.shape)
    y = y1 + jnp.broadcast_to(y2, y1.shape)
    print("Combined output shape:", y.shape)

    # Testing PixelWiseDense using convolution with kernel size 1
    model = PixelWiseDense(features=16)
    x = jnp.ones((1, 3, 4, 8))
    variables = model.init(random.key(0), x)
    y3 = model.apply(variables, x)
    print("PixelWiseDense output shape:", y3.shape)

    # Testing ReverseDiffusion
    model = ReverseDiffusion(features=16, channels=3)
    x = jnp.ones((2, 3, 4, 8))
    t = jnp.array([0, 0])
    beta_t = jnp.array([0.0, 0.0])
    key = random.key(0)
    variables = model.init(key, x, t=t, beta_t=beta_t, key=key)
    x_t, mu, sigma, key = model.apply(variables, x, t=t, beta_t=beta_t, key=key)
    print("ReverseDiffusion output shape:", "mu:", mu.shape, "sigma:", sigma.shape, "key:", key)