
import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.hidden_layer1 = nn.Linear(latent_dim, latent_dim)
        self.output_layer = nn.Linear(latent_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x)) 
        x = self.activation(self.hidden_layer1(x))
        return self.output_layer(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.hidden_layer = nn.Linear(latent_dim, latent_dim)
        self.Adv_output_layer = nn.Linear(latent_dim, output_dim)
        self.V_output_layer = nn.Linear(latent_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))

        value = self.V_output_layer(x)
        advantage = self.Adv_output_layer(x) 

        q = value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        return q