"""
    VAE model/architecture definitions
    Maurits Diephuis
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from nn_helpers.layers import DCGAN_Encoder, DCGAN_Decoder, DCGAN2_Encoder, DCGAN2_Decoder


class VAE(nn.Module):
    """
    Simple VAE model
    """

    def __init__(self, input_shape, encoder_size, latent_size, num_class=None):
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
        output = torch.sigmoid(self.fc4(h3))
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
        self.decoder = DCGAN_Decoder(self.encoder.H_conv_out, out_channels, decoder_size, latent_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class INFO_VAE2(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(INFO_VAE2, self).__init__()
        self.encoder = DCGAN2_Encoder(input_shape, out_channels, encoder_size, latent_size)
        self.decoder = DCGAN2_Decoder(self.encoder.H_conv_out, out_channels, encoder_size, latent_size)

    def encode(self, x):
        mu_z, std_z = self.encoder(x)
        return mu_z, std_z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu_z, std_z):
        eps = torch.randn_like(std_z)
        return mu_z + eps * std_z

    def forward(self, x):
        mu_z, std_z = self.encode(x)
        mu_z = self.reparameterize(mu_z, std_z)
        x_hat = self.decode(mu_z)
        return x_hat, mu_z, std_z


class PlanarFlow(nn.Module):
    def __init__(self, input_shape):
        super(PlanarFlow, self).__init__()
        self.input_shape = np.prod(input_shape)
        self.scale = nn.Parameter(torch.Tensor(1, input_shape))
        self.weight = nn.Parameter(torch.Tensor(1, input_shape))
        self.bias = nn.Parameter(torch.Tensor(1))

    def forward(self, z):
        f_z = nn.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = nn.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        log_det_grad = torch.log(det_grad.abs() + 1e-9)
        return log_det_grad


class RadialFlow(nn.Module):
    def __init__(self, input_shape):
        super(RadialFlow, self).__init__()
        self.input_shape = np.prod(input_shape)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.z0 = nn.Parameter(torch.Tensor(1, input_shape))

    def forward(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1. / (self.alpha + r)
        return z + (self.beta * h * (z - self.z0))

    def log_abs_det_jacobian(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1. / (self.alpha + r)
        dh = -1. / (self.alpha + r) ** 2
        base = 1 + self.beta * h
        det_grad = (base ** self.dim - 1) * (base + self.beta * dh * r)
        log_det_grad = torch.log(det_grad.abs() + 1e-9)
        return log_det_grad


class NormalizingFlow(nn.Module):
    def __init__(self, input_shape, f_blocks, length_flow):
        super(NormalizingFlow, self).__init__()
        self.input_shape = np.prod(input_shape)
        bijector_list = []
        for f in range(length_flow):
            for f_block in f_blocks:
                bijector_list.append(f_block(input_shape))

        self.bijectors = nn.ModuleList(bijector_list)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det


class FVAE(nn.Module):
    def __init__(self, input_shape, encoder_size, decoder_size, latent_size):
        super(FVAE, self).__init__()
        self.input_shape = np.prod(list(input_shape))
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.latent_size = latent_size

        self.encoder = nn.ModuleList([
            nn.Linear(self.input_shape, encoder_size),
            nn.ReLU(True),
            nn.Linear(self.encoder_size, self.encoder_size // 2),
            nn.ReLU(True),
            nn.Linear(self.encoder_size // 2, self.encoder_size // 2)
        ])

        self.decoder = nn.ModuleList([
            nn.Linear(self.latent_size, self.decoder_size // 2),
            nn.ReLU(True),
            nn.Linear(self.decoder_size // 2, self.decoder_size),
            nn.ReLU(True),
            nn.Linear(self.decoder_size, self.input_shape),
            nn.Sigmoid()
        ])

        self.encoder_mu = nn.Linear(self.encoder_size // 2, latent_size)
        self.encoder_std = nn.ModuleList([
            nn.Linear(self.encoder_size // 2, latent_size),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        ])

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)

        std = x
        for layer in self.encoder_std:
            std = layer(std)

        return mu, std

    def reparameterize(self, mu, std):
        if self.training:
            # std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def decode(self, z):
        x_hat = z
        for layer in self.decoder:
            x_hat = layer(x_hat)
        return x_hat

    def forward(self, x):
        mu, std = self.encode(x.view(-1, self.input_shape))
        z = self.reparameterize(mu, std)
        x_hat = self.decode(z)
        return x_hat, mu, std
