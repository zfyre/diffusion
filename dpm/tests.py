def test_gaussian_forward_diffusion():
    """
    Test the Gaussian forward diffusion process on a Swiss Roll dataset.
    This function generates a Swiss Roll dataset, applies the forward diffusion process,
    and visualizes the results at different timesteps.
    """
    import jax
    import jax.numpy as jnp
    from diffusion import Diffusion
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_swiss_roll
    
    # ========== Generate Swiss Roll Data ========== #

    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)
    X = X[:, [0, 2]]  # Take x and z for 2D
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize

    # ========== Apply Forward Diffusion ========== #

    key = jax.random.PRNGKey(0)
    diff = Diffusion(model=None, diffusion_steps=50)
    X_jax = jnp.array(X)
    forward_trajectory = diff.forward(X_jax, key)  # (1000, 100, 2)

    # ========== Plot Results ========== #

    fig, axes = plt.subplots(1, 10, figsize=(40, 4))
    plot_steps = np.linspace(0, forward_trajectory.shape[1] - 1, 10, dtype=int)

    for i, step in enumerate(plot_steps):
        axes[i].scatter(
            np.array(forward_trajectory[:, step, 0]),
            np.array(forward_trajectory[:, step, 1]),
            alpha=0.5,
            s=5
        )
        axes[i].set_title(f"t = {step}")
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)

    plt.tight_layout()
    plt.show()

def test_mnist_forward_diffusion():
    """
    Test the Gaussian forward diffusion process on MNIST dataset.
    This function loads a subset of MNIST data, applies the forward diffusion process,
    and visualizes the results at different timesteps.
    """
    import jax
    import jax.numpy as jnp
    from diffusion import Diffusion
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras.datasets import mnist

    # ========== Load MNIST Data ========== #
    
    (X_train, _), _ = mnist.load_data()
    # Take first 1000 samples and reshape to (N, 784)
    X = X_train[:1000].reshape(-1, 28*28).astype('float32')
    # Normalize to [-1, 1]
    X = (X / 127.5) - 1.0

    # ========== Apply Forward Diffusion ========== #

    key = jax.random.PRNGKey(0)
    diff = Diffusion(model=None, diffusion_steps=50)
    X_jax = jnp.array(X)
    forward_trajectory = diff.forward(X_jax, key)  # (1000, 100, 784)

    # ========== Plot Results ========== #

    fig, axes = plt.subplots(2, 10, figsize=(40, 8))
    plot_steps = np.linspace(0, forward_trajectory.shape[1] - 1, 10, dtype=int)
    
    # Plot first row - single image evolution
    sample_idx = 0
    for i, step in enumerate(plot_steps):
        img = np.array(forward_trajectory[sample_idx, step]).reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"t = {step}")

    # Plot second row - average of all images
    for i, step in enumerate(plot_steps):
        avg_img = np.mean(np.array(forward_trajectory[:, step]), axis=0).reshape(28, 28)
        axes[1, i].imshow(avg_img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"t = {step} (avg)")

    plt.suptitle("Forward Diffusion Process on MNIST\nTop: Single Image, Bottom: Average")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_gaussian_forward_diffusion()
    test_mnist_forward_diffusion()