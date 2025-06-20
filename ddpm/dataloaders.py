import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.random
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_swiss_roll
from tensorflow.keras.datasets import mnist

class MNISTDataset(Dataset):
    def __init__(self, batch_size: int = 128, num_samples: int = 10000):
        self.batch_size = batch_size
        
        # Load MNIST data
        (X_train, _), _ = mnist.load_data()
        if num_samples < X_train.shape[0]:
            print(f"Warning: num_samples used for MNIST dataset is less than the total number of samples.")

        self.num_samples = min(num_samples, X_train.shape[0])
        self.data = X_train[:self.num_samples].astype('float32')

        # Normalize data to [-1, 1]
        min_vals = self.data.min(axis=0)
        max_vals = self.data.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0 # To pevent divide by zero
        self.data = (self.data - min_vals) / ranges * 2 - 1

        # Applying Noise to image data for better training
        self.data = np.random.uniform(low=-1.0, high=1.0, size=self.data.shape) * 1.0/255 + self.data
        # Converting to torch.FloatTensor
        self.data = torch.FloatTensor(self.data)
        print(f"MNISTDataset: {self.data.shape}")
    
    def show_samples(self, samples: list[torch.Tensor], epoch: int):
        """
        Visualize samples from the dataset or generated samples
        """
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
        if len(samples) == 1:
            axes = [axes]
        
        for i, sample_batch in enumerate(samples):
            if isinstance(sample_batch, torch.Tensor):
                sample_batch = sample_batch.detach().cpu().numpy()
            
            axes[i].imshow(sample_batch[0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1} - Epoch {epoch}')
        
        plt.tight_layout()
        save_path = f'samples/mnist_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples to {save_path} for epoch {epoch}")
        plt.close()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_mnist_dataloader(batch_size: int = 32, num_samples: int = 10000, 
                            shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = MNISTDataset(batch_size=batch_size, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class SwissRollDataset(Dataset):
    def __init__(self, batch_size: int = 128, num_samples: int = 10000, noise: float = 0.0):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.noise = noise
        
        # Generate Swiss roll data
        self.data, _ = make_swiss_roll(n_samples=num_samples, noise=noise, random_state=42)
        # Keep only the first two dimensions (x, y) for 2D visualization
        self.data = self.data[:, [0, 2]]  # x and z coordinates
        # Normalize data to [-1, 1] range
        self.data = (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0)) * 2 - 1
        self.data = torch.FloatTensor(self.data)
        print(f"SwissRollDataset: {self.data.shape}")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

    def show_samples(self, samples: list[torch.Tensor], epoch: int):
        """
        Visualize samples from the dataset or generated samples
        
        Args:
            samples: List of tensors, each of shape (batch_size, 2)
            epoch: Current epoch number for title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
        if len(samples) == 1:
            axes = [axes]
        
        for i, sample_batch in enumerate(samples):
            if isinstance(sample_batch, torch.Tensor):
                sample_batch = sample_batch.detach().cpu().numpy()
            
            axes[i].scatter(sample_batch[:, 0], sample_batch[:, 1], alpha=0.6, s=10)
            axes[i].set_xlim(-1, 1)
            axes[i].set_ylim(-1, 1)
            axes[i].set_aspect('equal')
            axes[i].set_title(f'Sample {i+1} - Epoch {epoch}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f'samples/epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples to {save_path} for epoch {epoch}")
        plt.close()

def get_swiss_roll_dataloader(batch_size: int = 128, num_samples: int = 10000, 
                            noise: float = 0.0, shuffle: bool = True, 
                            num_workers: int = 0) -> DataLoader:
    
    dataset = SwissRollDataset(batch_size=batch_size, num_samples=num_samples, noise=noise)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    from ddpm import DDPMSampler
    dataloader = get_swiss_roll_dataloader(batch_size=200, num_samples=10000, noise=0.0)
    for batch in dataloader:
        # Create a sampler for forward diffusion
        sampler = DDPMSampler(
            generator=torch.Generator(device='cpu'),
            num_training_timesteps=100,
            beta_schedule='linear',
            beta_bounds=(1e-4, 2e-2)
        )
        sampler.set_inference_timesteps(num_inference_timesteps=10)

        noisy_samples = []
        print(sampler.timesteps)
        # Apply forward diffusion for each timestep
        for t in sampler.timesteps:
            t_batch = torch.full((batch.shape[0],), t, dtype=torch.int64)
            noisy_batch, _ = sampler.add_noise(batch, t_batch)
            noisy_samples.append(noisy_batch)

        # Plot original data and noisy versions
        plt.figure(figsize=(15, 3))
        # Reverse noisy_samples and timesteps for left-to-right increasing t
        reversed_noisy_samples = noisy_samples[::-1]
        reversed_timesteps = list(sampler.timesteps)[::-1]
        all_samples = reversed_noisy_samples
        for i, sample in enumerate(all_samples):
            plt.subplot(1, len(all_samples), i + 1)
            sample_np = sample.numpy()
            plt.scatter(sample_np[:, 0], sample_np[:, 1], alpha=0.6, s=10)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.title(f't={reversed_timesteps[i]}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved forward diffusion plot to forward_diffusion.png")


        break