"""
Reference: 
  - Simple Hardware-Efficient Long Convolutions for Sequence Modeling
  - https://hazyresearch.stanford.edu/blog/2023-02-15-long-convs
  - https://github.com/HazyResearch/safari/blob/main/src/models/sequence/long_conv.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class LongConvBlock(nn.Module):
    def __init__(self, d_model, seq_len, channels=1):
        """
        Args:
          d_model: (int) input feature dimension.
          seq_len: (int) input sequence length.
          channels: (int) number of kernels. SSM is a map from [L,D] to [L,C,D] sequence.
        """
        super().__init__()
        k = torch.randn(channels, d_model, seq_len)
        self.kernel = nn.Parameter(k)
        self.D = nn.Parameter(torch.randn(channels, d_model))

    def forward(self, x):
        """
        Args:
          x: [B,L,D].

        Returns:
          [B,L,D].
        """
        L = x.size(1)
        x = rearrange(x, "B L D -> B D L")
        k = self.kernel  # [C,D,L]
        k = F.relu(torch.abs(k) - 0.1) * torch.sign(k)
        k_f = torch.fft.rfft(k, n=2*L)  # [C,D,L]
        x_f = torch.fft.rfft(x, n=2*L)  # [B,D,L]
        y_f = torch.einsum("bhl,chl->bchl", [x_f, k_f])
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]  # [B,C,D,L]
        y = y + torch.einsum("bhl,ch->bchl", [x, self.D])
        y = rearrange(y, "b c h l -> b l (c h)")
        return y


class LongConvSequenceEncoder(nn.Module):
    def __init__(self, d_model, seq_len, channels=1):
        super().__init__()
        self.conv = LongConvBlock(d_model, seq_len, channels)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
          x: [B,L,D].

        Returns: 
          [B,L,D].
        """
        x = x + self.norm(self.conv(x))
        return x


class LongConvSequenceDecoder(nn.Module):
    def __init__(self, d_model, channels):
        super().__init__()
        self.conv = LongConvBlock(d_model, 1, channels)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
          x: [B,D].

        Returns: 
          [B,C,D].
        """
        B, D = x.size()
        x = x.unsqueeze(1)  # [B,D] -> [B,1,D]
        x = self.conv(x).reshape(B, -1, D)  # [B,1,C*D] -> [B,C,D]
        x = x + self.norm(x)
        return x


def test_encoder():
    N, L, D = 2, 3, 4
    x = torch.randn(N, L, D)
    m = LongConvSequenceEncoder(D, L)
    y = m(x)
    print(y.shape)


def test_decoder():
    N, D = 2, 4
    x = torch.randn(N, D)
    m = LongConvSequenceDecoder(D, channels=3)
    y = m(x)
    print(y)


if __name__ == "__main__":
    # test_encoder()
    test_decoder()
