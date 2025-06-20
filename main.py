"""
TODO: Implement a general config file for the training process
"""

from ddpm.trainer import Trainer
from ddpm.ddpm import DDPMSampler
from ddpm.model import SimpleModel, MNISTModel
from ddpm.dataloaders import get_swiss_roll_dataloader, get_mnist_dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim

# Hyperparameters
batch_size = 256
num_samples = 100000
noise = 0.05
num_workers = 0
num_epochs = 500
learning_rate = 3e-4
num_training_timesteps = 100
num_inference_timesteps = 100

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create sampler
sampler = DDPMSampler(
    generator=torch.Generator(device=device),
    num_training_timesteps=num_training_timesteps,
    beta_schedule='linear',
    beta_bounds=(1e-4, 2e-2),
)
# Set the inference timesteps
sampler.set_inference_timesteps(num_inference_timesteps=num_inference_timesteps)

print(f"Using sampler: {sampler}")

# Create model
# simple_swissroll_model = SimpleModel(in_channels=2, out_channels=2, hidden_size=64, num_training_timesteps=num_training_timesteps).to(device)
mnist_model = MNISTModel(num_training_timesteps=num_training_timesteps).to(device)

# print(f"Using model: {simple_swissroll_model}")
print(f"Using model: {mnist_model}")

# Create optimizer
optimizer = optim.Adam(mnist_model.parameters(), lr=learning_rate)

# Create dataloader
swiss_roll_dataloader = get_swiss_roll_dataloader(
    batch_size=batch_size,
    num_samples=num_samples,
    noise=noise,
    shuffle=True,
    num_workers=num_workers
)
mnist_dataloader = get_mnist_dataloader(
    batch_size=batch_size,
    num_samples=num_samples,
    shuffle=True,
    num_workers=num_workers
)

# Get the dataset to access the show_samples function
swiss_roll_dataset = swiss_roll_dataloader.dataset
mnist_dataset = mnist_dataloader.dataset

# Create trainer
trainer = Trainer(
    model=mnist_model, 
    sampler=sampler, 
    dataloader=mnist_dataloader, 
    optimizer=optimizer,
    num_epochs=num_epochs,
    show_samples_fn=mnist_dataset.show_samples,
    num_show_samples=1,
)

# Start training
print("Starting training...")
trainer.train()
print("Training completed!")