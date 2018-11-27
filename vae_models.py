"""
    VAE model/architecture definitions
    Maurits Diephuis
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    """
    Simple VAE model
    """

    def __init__(self):
        super(VAE, self).__init__()
        # tensor defs here
        self.fc1 = nn.Linear(784, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2_1 = self.fc2_1(h1)
        h2_2 = self.fc2_2(h1)
        return h2_1, h2_2

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            draw = eps.mul(std).add_(mu)
            return draw
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        output = F.sigmoid(self.fc4(h3))
        return output

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class CVAE(nn.Module):
    """
    Simple Conditional VAE
    """

    def __init__(self):
        super(CVAE, self).__init__()
        # tensor defs
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        x_cond = torch.cat((x, y), 1)
        h1 = F.relu(self.fc1(x_cond))
        mu = self.fc2_1(h1)
        log_var = self.fc2_2(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            draw = eps.mul(std).add_(mu)
            return draw
        else:
            return mu

    def decode(self, z, y):
        z_cond = torch.cat((z, y), 1)
        h3 = F.relu(self.fc3(z_cond))
        output = torch.sigmoid(self.fc4(h3))
        return output

    def forward(self, x, y):
        mu, log_var = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var


class CVAE_C10(nn.Module):
    """
    Simple Conditional VAE for CIFAR10
    """

    def __init__(self):
        super(CVAE, self).__init__()
        # tensor defs
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        x_cond = torch.cat((x, y), 1)
        h1 = F.relu(self.fc1(x_cond))
        mu = self.fc2_1(h1)
        log_var = self.fc2_2(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            draw = eps.mul(std).add_(mu)
            return draw
        else:
            return mu

    def decode(self, z, y):
        z_cond = torch.cat((z, y), 1)
        h3 = F.relu(self.fc3(z_cond))
        output = torch.sigmoid(self.fc4(h3))
        return output

    def forward(self, x, y):
        mu, log_var = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var
