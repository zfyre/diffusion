import torch
import numpy as np
import torch.nn as nn

class SinusoidPosEmb(nn.Module):
    def __init__(self, dim, num_training_timesteps, theta=10000, device='cuda'):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.device = device
        self.T_max = num_training_timesteps

    def forward(self, t: torch.IntTensor): 
        """
        Args:   
            t: torch.FloatTensor; (batch_size, )
        Returns:s
            emb: torch.FloatTensor; (batch_size, dim)
        """
        # Normalize the timesteps from [0, T) to [0, 1], converts to FloatTensor
        t = t.float().unsqueeze(-1) / self.T_max # (batch_size, 1)

        device = t.device
        self.device = device

        # To handle the PE(pos, 2i) = sin(pos/theta^(i/(d/2))) and PE(pos, 2i+1) = cos(pos/theta^(i/(d/2))), hence we take half_dim
        half_dim = self.dim //2
        emb = np.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(0, half_dim, device=device) * -emb) # (half_dim, )
        emb = t * emb[None, :] # (batch_size, half_dim) -> our final embedding matrix

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # Concatenates along final dimension -> [sin(f1​x),cos(f1​x),sin(f2​x),cos(f2​x),…]

        return emb # (batch_size, dim)

class TimeEmbedding(nn.Module):
    pass 
    # TODO: Implement a general class for time embedding
    """
    Tasks: 
        - implemeet the Learned Embeddings scenario
        - SinusoidPosEmb
        - RotatoryPosEmb
        - implement hybrid scenarios
    """
    

class SimpleModel(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, hidden_size=128, num_training_timesteps=1000):
        super().__init__()
        self.num_training_timesteps = num_training_timesteps
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.sinusoid_pos_emb = SinusoidPosEmb(
            dim=hidden_size,
            num_training_timesteps=num_training_timesteps
        )
        # For Swiss roll data: (batch_size, 2) -> 2D points
        # First layer: concatenate input (2 features) with time embedding (hidden_size features)
        self.fc1 = nn.Linear(in_channels + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, timesteps: torch.IntTensor):
        """
        Args:
            x: (batch_size, 2) - for Swiss roll: 2D points
            timesteps: torch.IntTensor; (batch_size,)
        """
        
        # Embed timesteps

        #TODO: Tests using learned time embedding
        # time_emb = self.time_embedding(timesteps)  # (batch_size, hidden_size)
        
        #TODO: Tests using only SinusoidPosEmb
        time_emb = self.sinusoid_pos_emb(timesteps)  # (batch_size, hidden_size)

        #TODO: Tests using both SinusoidPosEmb and learned time embedding
        
        # Concatenate input with time embedding
        x = torch.cat([x, time_emb], dim=1)
        
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 
