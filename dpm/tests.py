import argparse

def test(func):
    """Decorator to mark a function as a test and add test metadata."""
    func.__test__ = True  # Mark the function as a test
    def wrapper(*args, verbose=False, **kwargs):
        print(f"Running test: {func.__name__}")
        return func(*args, verbose=verbose, **kwargs)
    wrapper.__test__ = True  # Also mark the wrapper as a test
    wrapper.__name__ = func.__name__  # Preserve the original function name
    return wrapper

@test
def gaussian_forward_diffusion(verbose=False):
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
    diff = Diffusion(model=None, diffusion_steps=50, verbose=verbose)
    X_jax = jnp.array(X)
    forward_trajectory = diff.forward(X_jax, key)

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

@test
def mnist_forward_diffusion(verbose=False):
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
    diff = Diffusion(model=None, diffusion_steps=100, verbose=verbose)
    
    # Apply the uniform noise for images data to make the initial data more suitable for continous diffusion -> (-1.0/255, 1.0/255) for Image data
    X = jax.random.uniform(key, shape=X.shape, minval=-1.0, maxval=1.0) * 1.0/255 + X
    
    X_jax = jnp.array(X)
    forward_trajectory = diff.forward(X_jax, key)

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
    parser = argparse.ArgumentParser(description="Run diffusion model tests")
    parser.add_argument("--test", type=str, help="Run a specific test by name")
    parser.add_argument("-a", dest="run_all", action="store_true", help="Run all tests")
    parser.add_argument("-A", dest="run_all", action="store_true", help="Run all tests (same as -a)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get all test functions - look for both the decorator and the wrapper
    test_functions = [name for name, func in globals().items() 
                     if callable(func) and 
                     (hasattr(func, '__test__') or 
                      (hasattr(func, '__wrapped__') and hasattr(func.__wrapped__, '__test__')))]
    
    if not test_functions:
        print("No test functions found! Make sure test functions are decorated with @test")
        exit(1)

    if args.verbose:
        print(f"Found test functions: {', '.join(test_functions)}")
    
    if args.run_all:
        print("Running all tests...")
        for test_name in test_functions:
            test_func = globals()[test_name]
            try:
                test_func(verbose=args.verbose)
                print(f"✓ {test_name} passed")
            except Exception as e:
                print(f"✗ {test_name} failed with error: {str(e)}")
    elif args.test:
        if args.test not in test_functions:
            print(f"Error: Test '{args.test}' not found. Available tests: {', '.join(test_functions)}")
            exit(1)
        test_func = globals()[args.test]
        try:
            test_func(verbose=args.verbose)
            print(f"✓ {args.test} passed")
        except Exception as e:
            print(f"✗ {args.test} failed with error: {str(e)}")
            exit(1)
    else:
        print("Please specify either --test <test_name> or -a/-A to run all tests")
        print("Available tests:", ", ".join(test_functions))
        exit(1)
    