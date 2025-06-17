from ddpm.trainer import Trainer
from ddpm.ddpm import DDPMSampler
from ddpm.model import SimpleModel
from ddpm.dataloaders import get_swiss_roll_dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim

# Hyperparameters
batch_size = 250
num_samples = 100000
noise = 0.05
num_workers = 0
num_epochs = 500
learning_rate = 5e-4
num_training_timesteps = 100
num_inference_timesteps = 50

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
sampler.set_inference_timesteps(num_inference_timesteps=num_inference_timesteps)

# Create model
model = SimpleModel(in_channels=2, out_channels=2, hidden_size=64, num_training_timesteps=num_training_timesteps)
model.to(device)

# Create dataloader
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataloader = get_swiss_roll_dataloader(
    batch_size=batch_size,
    num_samples=num_samples,
    noise=noise,
    shuffle=True,
    num_workers=num_workers
)

# Get the dataset to access the show_samples function
dataset = dataloader.dataset

# Create trainer
trainer = Trainer(
    model=model, 
    sampler=sampler, 
    dataloader=dataloader, 
    optimizer=optimizer,
    num_epochs=num_epochs,
    show_samples=dataset.show_samples,
)

# Start training
print("Starting training...")
trainer.train()
print("Training completed!")