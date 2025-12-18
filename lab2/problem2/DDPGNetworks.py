import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim1=400, latent_dim2 = 200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim1)
        self.fc2 = nn.Linear(latent_dim1, latent_dim2)
        self.fc3 = nn.Linear(latent_dim2, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim, latent_dim1=400, latent_dim2 = 200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim1)
        self.fc2 = nn.Linear(latent_dim1 + action_dim, latent_dim2)
        self.fc3 = nn.Linear(latent_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, s, a):
        x = self.relu(self.fc1(s))
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc2(x))
        return self.fc3(x)