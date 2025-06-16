import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from diffusion import Diffusion
from model import ReverseDiffusion
import jax.debug

# Hyperparameters
BATCH_SIZE = 1000
N_EPOCHS = 500
LR = 5e-5
DIFFUSION_STEPS = 25
FEATURES = 32
C, H, W = 1, 1, 2
save_dir = Path('dpm/checkpoints')
save_dir.mkdir(exist_ok=True)

# Data preparation (Swiss Roll, 2D)
X, _ = make_swiss_roll(n_samples=100000, noise=0.1)
X = X[:, [0, 2]]
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = X.reshape(-1, 1, 1, 2)  # (N, 1, 1, 2)
X = jnp.array(X)
num_batches = X.shape[0] // BATCH_SIZE
print(f"Number of batches: {num_batches}")

# Model and diffusion initialization
key = jax.random.PRNGKey(0)
model = ReverseDiffusion(features=FEATURES, channels=C, diffusion_steps=DIFFUSION_STEPS)
diff = Diffusion(model=model, input_shape=(BATCH_SIZE, C, H, W), key=key, diffusion_steps=DIFFUSION_STEPS)

# Optimizer
optimizer = optax.adam(LR)
variables = diff.variables
opt_state = optimizer.init(variables)

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two Gaussian distributions.
    KL(N(mu1, sigma1) || N(mu2, sigma2))
    
    Args:
        mu1, mu2: Means of the distributions
        sigma1, sigma2: Standard deviations of the distributions
    
    Returns:
        KL divergence value
    """
    # Ensure sigma values are positive
    sigma1 = jnp.abs(sigma1) + 1e-8
    sigma2 = jnp.abs(sigma2) + 1e-8
    
    # Compute KL divergence
    kl_div = jnp.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5
    
    return kl_div


# Training step
@jax.jit
def train_step(variables, opt_state, batch, key):
    def loss_fn(params, key):
        # Sample a random timestep for the batch
        key, subkey = jax.random.split(key)
        t = jax.random.randint(subkey, (BATCH_SIZE,), 2, DIFFUSION_STEPS+1)  # [B]
        # Forward: get noisy sample and original parameters
        key, subkey = jax.random.split(key)
        x_t, key = diff.forward(batch, t, key)
        print(f"INSIDE TRAIN STEP")
        mu_original, sigma_squared_original = diff.get_mu_sigma_original(batch, t, x_t)
        
        # Get model predictions using the reverse method with the current parameters
        x_t_minus_1, mu_predicted, sigma_predicted, key = diff.reverse(params, x_t, t, key)
        
        # Convert sigma_squared to sigma (standard deviation)
        sigma_original = jnp.sqrt(sigma_squared_original)
        sigma_predicted = jnp.sqrt(sigma_predicted)
        
        # Compute KL divergence loss for each element in the batch and take the mean
        kl_div = kl_divergence_gaussian(mu_predicted, sigma_predicted, mu_original, sigma_original)
        total_loss = jnp.mean(kl_div)  # Take mean across all dimensions to get a scalar

        return total_loss, (key)
    
    (loss, (key)), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables, key)
    updates, opt_state = optimizer.update(grads, opt_state, variables)
    variables = optax.apply_updates(variables, updates)

    return variables, opt_state, loss, key

# Visualization
def visualize(diff: Diffusion, X: jnp.ndarray, key: jnp.ndarray, epoch: int, save_dir: Path):
    forward_trajectory, key = diff.forward_trajectory(X, key)
    n_steps = forward_trajectory.shape[1]
    
    # Select specific timesteps: first 4 at 5-step intervals, middle, and last
    plot_steps = np.array([
        0,              # First step
        5,              # 5th step
        10,             # 10th step
        15,             # 15th step
        n_steps // 2,   # Middle step
        n_steps - 1     # Last step
    ])
    
    # Sample x_T from pure Gaussian for reverse diffusion
    x_T = jax.random.normal(key, X.shape)
    reverse_trajectory, key = diff.reverse_trajectory(x_T, key)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, step in enumerate(plot_steps):
        # Forward trajectory
        pts = np.array(forward_trajectory[:, step, 0, 0, :])
        axes[0, i].scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5)
        axes[0, i].set_title(f"Forward t={step}")
        axes[0, i].set_xlim(-4, 4)
        axes[0, i].set_ylim(-4, 4)
        
        # Reverse trajectory
        pts = np.array(reverse_trajectory[:, step, 0, 0, :])
        axes[1, i].scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5)
        axes[1, i].set_title(f"Reverse t={step}")
        axes[1, i].set_xlim(-4, 4)
        axes[1, i].set_ylim(-4, 4)
    
    plt.suptitle(f"Epoch {epoch}: Forward (top) and Reverse (bottom) Diffusion Trajectories")
    plt.tight_layout()
    plt.savefig(save_dir / f"vis_epoch{epoch}.png")
    plt.close()

# Training loop
losses = []
for epoch in range(1, N_EPOCHS + 1):
    perm = np.random.permutation(X.shape[0])
    epoch_losses = []   
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch}/{N_EPOCHS}"):
        idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch = X[idx]
        variables, opt_state, loss, key = train_step(variables, opt_state, batch, key)
        # Update the model variables
        diff.model.variables = variables
        key, _ = jax.random.split(key)  # Update key after each batch
        if jnp.isnan(loss) or jnp.isinf(loss):
            exit()
        epoch_losses.append(loss)
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)
    print(f"Epoch {epoch}: Total Loss = {mean_loss:.6f}")
    # Visualization every 10 epochs
    if epoch % 10 == 0:
        key, _ = jax.random.split(key)  # Update key before visualization
        visualize(diff, X[:BATCH_SIZE], key, epoch, save_dir)
    # Save checkpoint every 50 epochs
    if epoch % 250 == 0:
        with open(save_dir / f"checkpoint_epoch{epoch}.pkl", "wb") as f:
            pickle.dump({
                'variables': variables,
                'opt_state': opt_state,
                'epoch': epoch,
                'loss': mean_loss,
                'model_config': {
                    'features': FEATURES,
                    'channels': C,
                    'diffusion_steps': DIFFUSION_STEPS
                }
            }, f)
        print(f"Checkpoint saved at epoch {epoch}")
# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Training Loss')
plt.tight_layout()
plt.savefig(save_dir / 'training_loss_curves.png')
plt.close()


# if __name__ == "__main__":
#     key = jax.random.PRNGKey(0)
#     t = jax.random.randint(key, (BATCH_SIZE,), 1, DIFFUSION_STEPS+1)
#     print(t)