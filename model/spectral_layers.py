import torch
import einops
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.fft as fft

class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=1,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = 'silu'
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = dropout
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = nn.Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = 2*spacial_dim*freq_dim
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 return_freq=False,
                 activation='silu',
                 debug=False):
        super(SpectralConv1d, self).__init__()

        '''
        Modified Zongyi Li's Spectral1dConv code
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
        '''

        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = 'silu'
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = nn.Parameter(
            torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1/(in_dim*out_dim))
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid, in_features)
        Output: (-1, n_grid, out_features)
        '''
        seq_len = x.size(1)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = self.complex_matmul_1d(
            x_ft[:, :, :self.modes], self.fourier_weight)

        pad_size = seq_len//2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft(out_ft, n=seq_len, norm="ortho")

        x = x.permute(0, 2, 1)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x