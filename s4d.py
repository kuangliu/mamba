"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.

Reference:
  - https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    Note: lr is set to min(0.001, args.lr)

    Reference:
      - https://github.com/state-spaces/s4/blob/main/example.py#L177
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        Args:
          L: (int) sequence length.

        returns: 
          (..., c, L) where c is number of channels (default 1)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt)  # [H,]
        C = torch.view_as_complex(self.C)  # [H,N]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # [H, N]

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # [H,N]
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # [H,N,L]
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real
        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()  # NOTE: bugged in PyTorch 1.11

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        """
        Args:
          u: (tensor) input tensor, sized [B,L,H].

        Returns:
          (tensor) output tensor, sized [B,L,H].
        """
        u = u.transpose(-1, -2)  # [B,L,H] -> [B,H,L]
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # [H,L]

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # [H,L]
        u_f = torch.fft.rfft(u, n=2*L)  # [B,H,L]
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # [B,H,L]

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        y = y.transpose(-1, -2)  # [B,H,L] -> [B,L,H]
        return y


if __name__ == "__main__":
    N, L, D = 2, 3, 4
    m = S4D(D)
    x = torch.randn(N, L, D)
    y = m(x)
    print(y.shape)
