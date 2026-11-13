"""
models.py — CNN3D and Spherical Transformer architectures
(Methods, Section 2.5; Eq. 11).

Both operate directly on the raw (N_SPH_PHI, N_SPH_THETA, N_SPH_RADII)
spherical intensity tensor from spherical_transform.py -- no PCA projection.
Extracted verbatim from deep_learning_experiment.py; no logic changed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spherical_transform import N_SPH_PHI, N_SPH_THETA, N_SPH_RADII


class ThetaCircularConv3d(nn.Conv3d):
    """3x3x3 convolution over (phi, theta, r) that pads the azimuth axis
    theta circularly (theta = 0 and 2*pi are neighbours) and the elevation
    and radial axes with zeros. Parameter layout is identical to nn.Conv3d."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(in_ch, out_ch, kernel_size=3, padding=(1, 0, 1))

    def forward(self, x):
        x = F.pad(x, (0, 0, 1, 1, 0, 0), mode="circular")   # theta is dim -2
        return super().forward(x)


class CNN3D(nn.Module):
    """
    3-D convolutional network (Methods Section 2.5).

    Input  : (B, 1, 36, 72, 48)
    After pool x2: (B, 32, 9, 18, 12)
    AdaptiveAvgPool3d -> (B, 64) -> classifier head -> single logit

    circular_theta=False gives the identical architecture with ordinary zero
    padding on all axes, used for MNI-registered volumes (Methods 2.9), which
    have no periodic axis. Parameter layout is the same either way.
    """

    def __init__(self, circular_theta: bool = True):
        super().__init__()

        def conv(i, o):
            return ThetaCircularConv3d(i, o) if circular_theta else nn.Conv3d(i, o, kernel_size=3, padding=1)

        self.encoder = nn.Sequential(
            conv(1, 16), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),                                    # (B,16,18,36,24)

            conv(16, 32), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),                                    # (B,32, 9,18,12)

            conv(32, 64), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),                            # (B,64, 1, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x.unsqueeze(1))).squeeze(1)


class SphericalTransformer(nn.Module):
    """
    Spherical Transformer (Methods Section 2.5, Eq. 11).

    Each (phi, theta) angular cell (after a stride-2 angular decimation,
    Section 2.5) is one token; its 48-D radial intensity profile is linearly
    projected to the token embedding (Eq. 11). A learned positional
    embedding (Eq. 11) and class token are added, followed by an
    L-layer pre-norm Transformer encoder and a linear
    classification head on the final class-token representation.
    """

    DS = 2  # angular downsample factor: 2 -> 18x36 = 648 tokens (Methods 2.5)

    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        n_phi   = N_SPH_PHI   // self.DS
        n_theta = N_SPH_THETA // self.DS
        n_tok   = n_phi * n_theta

        self.n_phi, self.n_theta = n_phi, n_theta
        self.proj      = nn.Linear(N_SPH_RADII, d_model)   # Eq. 11 (W_proj)
        self.pos_embed = nn.Embedding(n_tok, d_model)      # Eq. 11 (p_m)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # class token

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-norm (Methods 2.5)
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, 36, 72, 48)
        if self.DS > 1:
            x = x[:, ::self.DS, ::self.DS, :]          # stride-2 decimation, (B, 18, 36, 48)
        B, P, T, _ = x.shape
        tokens = self.proj(x.reshape(B, P * T, -1))    # Eq. 11: (B, n_tok, d_model)

        pos    = torch.arange(P * T, device=x.device)
        tokens = tokens + self.pos_embed(pos)           # Eq. 11

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)        # class token: (B, n_tok+1, d_model)

        out = self.transformer(tokens)                  # encoder: (B, n_tok+1, d_model)
        return self.head(self.norm(out[:, 0])).squeeze(1)   # CLS logit
