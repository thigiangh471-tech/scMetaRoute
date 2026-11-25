"""Continuous RNA encoder, decoder and negative-binomial loss."""

# The adapted encoder, decoder and loss retain their original notice:
# MIT License
#
# Copyright (c) 2024 XLearning Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_genes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_genes):
        super().__init__()
        # Keep the historical key spelling for loading published checkpoints.
        self.docoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.decoder_mean = nn.Linear(512, n_genes)
        self.decoder_disp = nn.Sequential(nn.Linear(512, n_genes), nn.Softplus())

    def forward(self, z):
        hidden = self.docoder(z)
        mean = torch.clamp(torch.exp(self.decoder_mean(hidden)), 1e-5, 1e6)
        dispersion = torch.clamp(self.decoder_disp(hidden), 1e-4, 1e4)
        return mean, dispersion


class CountVAE(nn.Module):
    def __init__(self, n_genes, latent_dim=32):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(n_genes, latent_dim)])
        # The original runner initialized unused modules between these steps.
        # Consume the same CPU RNG draws; do not retain or run those modules.
        nn.Embedding(500, latent_dim)
        self.decoders = nn.ModuleList([Decoder(latent_dim, n_genes)])
        Decoder(latent_dim, n_genes)


def negative_binomial_loss(mean, disp, scale_factor, x):
    eps = 1e-12
    mean = mean * scale_factor
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
        x * (torch.log(disp + eps) - torch.log(mean + eps))
    )
    return t1 + t2
