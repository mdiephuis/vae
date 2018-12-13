"""
    VAE model/architecture definitions
    Maurits Diephuis
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from nn_helpers.layers import DCGAN_Encoder, DCGAN_Decoder


class VAE(nn.Module):
    """
    Simple VAE model
    """

    def __init__(self, input_shape, encoder_size, latent_size):
        super(VAE, self).__init__()
        self.input_shape = np.prod(list(input_shape))
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        # tensor defs here
        self.fc1 = nn.Linear(self.input_shape, self.encoder_size)
        self.fc2_1 = nn.Linear(self.encoder_size, self.latent_size)
        self.fc2_2 = nn.Linear(self.encoder_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.encoder_size)
        self.fc4 = nn.Linear(self.encoder_size, self.input_shape)

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
        mu, log_var = self.encode(x.view(-1, self.input_shape))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class CVAE(nn.Module):
    """
    Simple Conditional VAE
    """

    def __init__(self, input_shape, encoder_size, latent_size, num_class):
        super(CVAE, self).__init__()
        self.input_shape = np.prod(list(input_shape))
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        self.num_class = num_class

        # tensor defs
        self.fc1 = nn.Linear(self.input_shape + self.num_class, self.encoder_size)
        self.fc2_1 = nn.Linear(self.encoder_size, self.latent_size)
        self.fc2_2 = nn.Linear(self.encoder_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size + self.num_class, self.encoder_size)
        self.fc4 = nn.Linear(self.encoder_size, self.input_shape)

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
        mu, log_var = self.encode(x.view(-1, self.input_shape), y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var


class INFO_VAE(nn.Module):
    '''
    Info VAE model, using DC-GAN Encoders and Decoders
    '''

    def __init__(self, input_shape, out_channels, encoder_size, decoder_size, latent_size):
        super(INFO_VAE, self).__init__()
        self.encoder = DCGAN_Encoder(input_shape, out_channels, encoder_size, latent_size)
        H_conv_out = self.encoder.H_conv_out
        self.decoder = DCGAN_Decoder(H_conv_out, out_channels, decoder_size, latent_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(x)
        return z, x_hat
