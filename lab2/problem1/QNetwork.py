
import torch.nn as nn

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
    