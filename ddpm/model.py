import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_size=128, num_training_timesteps=1000):
        super().__init__()
        self.num_training_timesteps = num_training_timesteps
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # For Swiss roll data: (batch_size, 2) -> 2D points
        # First layer: concatenate input (2 features) with time embedding (hidden_size features)
        self.fc1 = nn.Linear(in_channels + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, timesteps):
        # x: (batch_size, 2) - for Swiss roll: 2D points
        # timesteps: (batch_size,)
        
        # Embed timesteps
        timesteps = timesteps.float().unsqueeze(-1) / self.num_training_timesteps  # Normalize to [0, 1]
        time_emb = self.time_embedding(timesteps)  # (batch_size, hidden_size)
                
        # Concatenate input with time embedding
        x = torch.cat([x, time_emb], dim=1)
        
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

