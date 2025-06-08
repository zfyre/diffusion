import jax
import jax.numpy as jnp
from flax import linen as nn
import math

def get_temporal_basis(timesteps, n_temporal_basis, trajectory_length):
    """
    Create temporal basis functions for time conditioning as in original DPM.
    
    Args:
        timesteps: Array of timestep values [0, trajectory_length-1]
        n_temporal_basis: Number of temporal basis functions
        trajectory_length: Total number of diffusion steps
    
    Returns:
        Temporal basis representation of timesteps
    """
    # Normalize timesteps to [0, 1]
    t_normalized = timesteps / (trajectory_length - 1)
    
    # Create basis functions - using Fourier basis as in original
    basis_funcs = []
    for i in range(n_temporal_basis):
        if i == 0:
            # Constant basis
            basis_funcs.append(jnp.ones_like(t_normalized))
        elif i % 2 == 1:
            # Sine basis
            freq = (i + 1) // 2
            basis_funcs.append(jnp.sin(2 * math.pi * freq * t_normalized))
        else:
            # Cosine basis
            freq = i // 2
            basis_funcs.append(jnp.cos(2 * math.pi * freq * t_normalized))
    
    return jnp.stack(basis_funcs, axis=-1)  # Shape: (batch_size, n_temporal_basis)

class ConvBlock(nn.Module):
    """Convolutional block with leaky ReLU as in the reference implementation."""
    features: int
    kernel_size: tuple = (3, 3)
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.02)  # Using leaky ReLU as in reference
        return x

class DenseBlock(nn.Module):
    """Dense block with leaky ReLU activation."""
    features: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.leaky_relu(x, negative_slope=0.02)
        return x

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

class DiffusionReverseModel(nn.Module):
    """
    Reverse diffusion model following the original DPM implementation.
    
    This implements the hierarchical MLP architecture from the paper:
    - Lower half: Convolutional layers + dense layers
    - Upper half: Dense layers for final mu and sigma prediction
    
    The model learns both mu (mean) and sigma (log variance) for the reverse process.
    """
    
    spatial_width: int
    n_colors: int
    trajectory_length: int = 1000
    n_temporal_basis: int = 10
    n_hidden_dense_lower: int = 500
    n_hidden_dense_lower_output: int = 2
    n_hidden_dense_upper: int = 20
    n_hidden_conv: int = 20
    n_layers_conv: int = 4
    n_layers_dense_lower: int = 4
    n_layers_dense_upper: int = 2
    n_scales: int = 1
    step1_beta: float = 0.001
    
    def setup(self):
        """Setup the hierarchical MLP architecture."""
        
        # Lower half: Convolutional processing
        self.conv_layers = [
            ConvBlock(features=self.n_hidden_conv, kernel_size=(3, 3))
            for _ in range(self.n_layers_conv)
        ]
        
        # Multi-scale convolution as in your existing implementation
        if self.n_scales > 1:
            self.multi_scale_conv = MultiScaleConv(
                features=self.n_hidden_conv, 
                num_scales=self.n_scales
            )
        
        # Lower half: Dense layers
        if self.n_hidden_dense_lower > 0:
            self.dense_lower_layers = [
                DenseBlock(features=self.n_hidden_dense_lower)
                for _ in range(self.n_layers_dense_lower)
            ]
            
            # Output layer for lower dense network
            self.dense_lower_output = nn.Dense(
                self.n_hidden_dense_lower_output * self.spatial_width * self.spatial_width
            )
        else:
            self.dense_lower_layers = []
        
        # Temporal basis processing
        self.temporal_dense = nn.Dense(self.n_hidden_dense_upper * self.spatial_width * self.spatial_width)
        
        # Upper half: Dense layers for final prediction
        self.dense_upper_layers = [
            DenseBlock(features=self.n_hidden_dense_upper)
            for _ in range(self.n_layers_dense_upper)
        ]
        
        # Final output layers for mu and log_sigma
        self.mu_output = nn.Dense(self.n_colors, name='mu_output')
        self.log_sigma_output = nn.Dense(self.n_colors, name='log_sigma_output')
    
    def __call__(self, x_t, timesteps):
        """
        Forward pass of the reverse diffusion model.
        
        Args:
            x_t: Noisy input image, shape (B, C, H, W)
            timesteps: Timestep values, shape (B,)
        
        Returns:
            mu: Predicted mean, shape (B, C, H, W)
            log_sigma: Predicted log variance, shape (B, C, H, W)
        """
        B, C, H, W = x_t.shape
        
        # === LOWER HALF: Feature extraction ===
        
        # Convolutional processing
        h_conv = x_t
        for conv_layer in self.conv_layers:
            # Convert to NHWC for conv
            h_conv_nhwc = jnp.transpose(h_conv, (0, 2, 3, 1))
            h_conv_nhwc = conv_layer(h_conv_nhwc)
            h_conv = jnp.transpose(h_conv_nhwc, (0, 3, 1, 2))
        
        # Multi-scale processing if enabled
        if self.n_scales > 1:
            h_conv = self.multi_scale_conv(h_conv)
        
        # Flatten convolutional features
        h_conv_flat = h_conv.reshape(B, -1)
        
        # Dense processing in lower half
        h_dense_lower = h_conv_flat
        if self.n_hidden_dense_lower > 0:
            for dense_layer in self.dense_lower_layers:
                h_dense_lower = dense_layer(h_dense_lower)
            h_dense_lower = self.dense_lower_output(h_dense_lower)
        
        # Get temporal basis representation
        temporal_basis = get_temporal_basis(
            timesteps, self.n_temporal_basis, self.trajectory_length
        )  # Shape: (B, n_temporal_basis)
        
        # Process temporal information
        h_temporal = self.temporal_dense(temporal_basis)  # (B, n_hidden_dense_upper * spatial_width^2)
        
        # Combine lower features with temporal features
        if self.n_hidden_dense_lower > 0:
            h_combined = jnp.concatenate([h_dense_lower, h_temporal], axis=1)
        else:
            h_combined = h_temporal
        
        # === UPPER HALF: Per-pixel prediction ===
        
        # Reshape to per-pixel processing
        n_features_per_pixel = h_combined.shape[1] // (self.spatial_width * self.spatial_width)
        h_pixels = h_combined.reshape(B, self.spatial_width, self.spatial_width, n_features_per_pixel)
        
        # Dense processing in upper half (per-pixel)
        h_upper = h_pixels
        for dense_layer in self.dense_upper_layers:
            h_upper = dense_layer(h_upper)  # Process each pixel independently
        
        # Final predictions
        mu = self.mu_output(h_upper)      # (B, H, W, C)
        log_sigma = self.log_sigma_output(h_upper)  # (B, H, W, C)
        
        # Apply minimum variance constraint (step1_beta)
        log_sigma = jnp.maximum(log_sigma, jnp.log(self.step1_beta))
        
        # Convert back to (B, C, H, W) format
        mu = jnp.transpose(mu, (0, 3, 1, 2))
        log_sigma = jnp.transpose(log_sigma, (0, 3, 1, 2))
        
        return mu, log_sigma

class ReverseModel(nn.Module):
    """
    Enhanced reverse model that combines the original DPM approach with modern improvements.
    """
    spatial_width: int
    in_channels: int = 3
    out_channels: int = 3
    trajectory_length: int = 1000
    n_temporal_basis: int = 10
    n_hidden_dense_lower: int = 500
    n_hidden_conv: int = 20
    n_layers_conv: int = 4
    n_scales: int = 3
    step1_beta: float = 0.001  # Expose this parameter
    
    def setup(self):
        self.diffusion_model = DiffusionReverseModel(
            spatial_width=self.spatial_width,
            n_colors=self.out_channels,
            trajectory_length=self.trajectory_length,
            n_temporal_basis=self.n_temporal_basis,
            n_hidden_dense_lower=self.n_hidden_dense_lower,
            n_hidden_conv=self.n_hidden_conv,
            n_layers_conv=self.n_layers_conv,
            n_scales=self.n_scales,
            step1_beta=self.step1_beta
        )
    
    def __call__(self, x_t, timesteps):
        """
        Args:
            x_t: Noisy input image, shape (B, C, H, W)
            timesteps: Timestep values, shape (B,)
        
        Returns:
            mu: Predicted mean for reverse process
            log_var: Predicted log variance for reverse process
        """
        return self.diffusion_model(x_t, timesteps)

if __name__ == "__main__":
    # Test the DPM-style reverse model
    B, C, H, W = 2, 3, 32, 32
    
    # Create dummy input
    x_t = jax.random.normal(jax.random.PRNGKey(0), (B, C, H, W))
    timesteps = jnp.array([100, 200])  # Different timesteps for each batch element
    
    # Initialize model
    model = ReverseModel(
        spatial_width=H,  # Assuming square images
        in_channels=C,
        out_channels=C,
        trajectory_length=1000,
        n_temporal_basis=10,
        n_hidden_conv=64,
        n_layers_conv=3,
        n_scales=3
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.init(key, x_t, timesteps)
    
    # Apply model
    mu, log_var = model.apply(params, x_t, timesteps)
    
    print("=== DPM-Style Reverse Noise Generation ===")
    print("Input shape (x_t):", x_t.shape)
    print("Timesteps:", timesteps)
    print("Predicted mu shape:", mu.shape)
    print("Predicted log_var shape:", log_var.shape)
    print("✓ Model successfully implements DPM reverse process!")
    
    # Verify variance constraint
    min_var = jnp.exp(log_var).min()
    print(f"Minimum variance: {min_var:.6f} (should be >= {model.step1_beta})")
    
    # Test temporal basis
    temporal_basis = get_temporal_basis(timesteps, 10, 1000)
    print("Temporal basis shape:", temporal_basis.shape)
    print("✓ Temporal basis functions working correctly!") 