import torch.nn as nn
import torch

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim1=400, latent_dim2 = 200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim1)
        self.fc2 = nn.Linear(latent_dim1, latent_dim2)
        self.fc3 = nn.Linear(latent_dim2, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class Actor(nn.Module):
    def __init__(self, state_dim, output_dim=1, latent_dim1=400, latent_dim2 = 200):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, latent_dim1)

        self.mu_input = nn.Linear(latent_dim1, latent_dim2)
        self.mu_output = nn.Linear(latent_dim2, output_dim)
        self.tanh = nn.Tanh()
        
        self.var_input = nn.Linear(latent_dim1, latent_dim2)
        self.var_output = nn.Linear(latent_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()

    def forward(self, s):
        x = self.relu(self.fc1(s))
        mu = self.relu(self.mu_input(x))
        var = self.relu(self.var_input(x))
        mu = self.tanh(self.mu_output(mu))
        var = self.sigmoid(self.var_output(var))
        return mu, var