import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ddpm.ddpm import DDPMSampler
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt
import os
class Trainer:
    def __init__(self,
        model: nn.Module,
        sampler: DDPMSampler,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        num_epochs: int = 200,
        show_samples_fn: Callable = None, # TODO: Handle this more elegantly
        num_show_samples: int = 1, # TODO: Handle this more elegantly
        save_path: str = 'checkpoints'
    ):
        self.model = model
        self.sampler = sampler
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.show_samples_fn = show_samples_fn
        self.num_show_samples = num_show_samples
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Ensure sampler's generator is on the correct device
        if hasattr(self.sampler, 'generator') and self.sampler.generator.device != self.device:
            self.sampler.generator = torch.Generator(device=self.device)

        self.batch_size = self.dataloader.batch_size

    
    def train(self):
        # Set the model to train mode

        # Initialize the losses list
        losses = []
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            for (batch_idx, batch) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                if batch_idx == 0 and epoch == 1:
                    print(f"Batch shape: {batch.shape}")
                    self.input_shape = batch.shape
                loss = self.train_step(batch)
                epoch_loss = epoch_loss + loss
                num_batches += 1
            average_loss = epoch_loss / num_batches  # Average by number of batches, not batch size
            losses.append(average_loss.item())

            print(f"Epoch {epoch}/{self.num_epochs}, Loss: {average_loss}")

            if epoch % 10 == 0: # TODO: Handle this more elegantly
                self.model.eval()
                samples = self.sample(num_samples=self.num_show_samples)
                if self.show_samples_fn is not None:
                    self.show_samples_fn(samples, epoch)

        plt.plot(losses)
        plt.savefig(os.path.join(self.save_path, 'losses.png'), dpi=150, bbox_inches='tight')
        print(f"Saved losses to {os.path.join(self.save_path, 'losses.png')}")
        plt.close()

        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        print(f"Saved model to {os.path.join(self.save_path, 'model.pth')}")        
    
    def sample(self, num_samples: int = 1) -> list[torch.Tensor]:
        new_shape = list(self.input_shape)
        new_shape[0] = num_samples
        x_T = torch.randn(tuple(new_shape), generator=self.sampler.generator, device=self.device)
        x_t = x_T
        samples = [x_T] 
        for t in tqdm(self.sampler.timesteps):
            timesteps = torch.tensor([t]*num_samples, device=self.device, dtype=torch.int64)
            model_outs = self.model(x_t, timesteps)
            x_t, mean_pred, variance_pred, z = self.sampler.denoise_step(
                timestep=t,
                model_outs=model_outs,
                x_t=x_t
            )
            samples.append(x_t)
        return samples

    def train_step(self, batch: torch.Tensor) -> torch.Tensor:
        # Shift the batch to GPU from CPU (if needed)
        batch = batch.to(self.device)

        # Sample timesteps randomly from [1, T) of shape (batch_size,), note that our timesteps are 0-indexed
        # Reverse Process starts from t=1, according to DDPM paper. -> L1:T-1
        # And for t=0, the paper trains an encoder to predict the noise.
        timesteps = torch.randint(
            low=0,
            high=self.sampler.num_training_timesteps,
            size=(batch.shape[0],),
            device=self.device,
            dtype=torch.int64
        )
        """
        TODO: Add the case where the L_0 loss is modeled differently, right now we are using the same model.
        The reverse model should incorporate the discrete log-likelihood prediction for images.
        """

        # Add noise to the batch
        noisy_samples, noise = self.sampler.add_noise(batch, timesteps)
        # Get the model output noise epsilon_theta
        predited_noise = self.model(noisy_samples, timesteps)
        # Compute the loss simple -> according to the DDPM paper
        loss_simple = nn.functional.mse_loss(predited_noise, noise, reduction='mean')

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss_simple.backward()
        self.optimizer.step()

        return loss_simple
