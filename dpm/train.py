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

# Hyperparameters
batch_size = 128
n_epochs = 200
learning_rate = 1e-4
diffusion_steps = 256
features = 64
channels = 1
save_dir = Path('dpm/checkpoints')
save_dir.mkdir(exist_ok=True)

# Data preparation (Swiss Roll, 2D)
X, _ = make_swiss_roll(n_samples=10000, noise=0.1)
X = X[:, [0, 2]]
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = X.reshape(-1, 1, 1, 2)  # (N, 1, 1, 2)
X = jnp.array(X)
num_batches = X.shape[0] // batch_size

# Model and diffusion initialization
key = jax.random.PRNGKey(0)
model = ReverseDiffusion(features=features, channels=channels, diffusion_steps=diffusion_steps)
diff = Diffusion(model=model, input_shape=(batch_size, 1, 1, 2), key=key, diffusion_steps=diffusion_steps)

# Optimizer
optimizer = optax.adam(learning_rate)
variables = diff.variables
opt_state = optimizer.init(variables)

# Training step
@jax.jit
def train_step(variables, opt_state, batch, key):
    def loss_fn(params):
        # Sample a random timestep for the batch
        t = jax.random.randint(key, (), 1, diffusion_steps)  # single t for the whole batch
        # Forward: get x_t at timestep t
        x_t, key2 = diff.forward(batch, t, key)
        # Reverse: predict x_{t-1} from x_t
        x_t_pred, key3 = diff.reverse(x_t, t, key2)
        # Loss: MSE between predicted and true x_{t-1} (or x_0 if t==1)
        # For simplicity, compare to batch (x_0)
        loss = jnp.mean((x_t_pred - batch) ** 2)
        return loss, x_t_pred
    (loss, x_t_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables)
    updates, opt_state = optimizer.update(grads, opt_state)
    variables = optax.apply_updates(variables, updates)
    return variables, opt_state, loss

# Visualization
# def visualize(diff, X, key, epoch, save_dir):
#     forward_trajectory, key = diff.forward(X, key)
#     n_steps = forward_trajectory.shape[1]
#     plot_steps = np.linspace(0, n_steps - 1, 10, dtype=int)
#     # Sample x_T from pure Gaussian for reverse diffusion
#     x_T = jax.random.normal(key, X.shape)
#     reverse_trajectory, key = diff.reverse(x_T, key)
#     fig, axes = plt.subplots(2, 10, figsize=(30, 6))
#     for i, step in enumerate(plot_steps):
#         pts = np.array(forward_trajectory[:, step, 0, 0, :])
#         axes[0, i].scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5)
#         axes[0, i].set_title(f"Forward t={step}")
#         axes[0, i].set_xlim(-4, 4)
#         axes[0, i].set_ylim(-4, 4)
#         pts = np.array(reverse_trajectory[:, step, 0, 0, :])
#         axes[1, i].scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5)
#         axes[1, i].set_title(f"Reverse t={step}")
#         axes[1, i].set_xlim(-4, 4)
#         axes[1, i].set_ylim(-4, 4)
#     plt.suptitle(f"Epoch {epoch}: Forward (top) and Reverse (bottom) Diffusion Trajectories")
#     plt.tight_layout()
#     plt.savefig(save_dir / f"vis_epoch{epoch}.png")
#     plt.close()

# Training loop
losses = []
for epoch in range(1, n_epochs + 1):
    perm = np.random.permutation(X.shape[0])
    epoch_losses = []
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch}/{n_epochs}"):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        batch = X[idx]
        variables, opt_state, loss = train_step(variables, opt_state, batch, key)
        key, _ = jax.random.split(key)  # Update key after each batch
        epoch_losses.append(loss)
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)
    print(f"Epoch {epoch}: Loss = {mean_loss:.6f}")
    # Visualization every 10 epochs
    if epoch % 10 == 0:
        key, _ = jax.random.split(key)  # Update key before visualization
        visualize(diff, X[:batch_size], key, epoch, save_dir)
    # Save checkpoint every 50 epochs
    if epoch % 50 == 0:
        with open(save_dir / f"checkpoint_epoch{epoch}.pkl", "wb") as f:
            pickle.dump({
                'variables': variables,
                'opt_state': opt_state,
                'epoch': epoch,
                'loss': mean_loss,
                'model_config': {
                    'features': features,
                    'channels': channels,
                    'diffusion_steps': diffusion_steps
                }
            }, f)
        print(f"Checkpoint saved at epoch {epoch}")
# Plot training curve
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig(save_dir / 'training_loss_curve.png')
plt.close()
