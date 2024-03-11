import torch
import torch.nn.functional as F

from einops import rearrange


# ---------------------------
# General SSM utils.
# ---------------------------
def random_ssm(N):
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    C = torch.randn(N, N)
    return A, B, C


def discretize(A, B, C, seq_len):
    """
    Args:
      A: [N,N]
      B: [N,N]
      C: [N,N]
      seq_len: int

    Returns:
      dA: [N,N]
      dB: [N,N]
      C:  [N,N]
    """
    N = A.size(0)
    I = torch.eye(N)
    step = 1.0 / seq_len
    dB = (I - 0.5 * step * A).inverse()  # [N,N]
    dA = dB @ (I + 0.5 * step * A)  # [N,N]
    dB = (dB * step) @ B  # [N,1]
    return dA, dB, C


# -----------------------
# Conv-style SSM
# -----------------------
def ssm_conv_kernel(dA, dB, dC, L):
    """
    Args:
      dA: [N,N]
      dB: [N,N]
      dC: [N,N]
      L: int

    Returns:
      [N,N,L]
    """
    kernel = [dC @ torch.matrix_power(dA, l) @ dB for l in reversed(range(L))]
    kernel = torch.stack(kernel, dim=-1)
    return kernel


def naive_conv(u, kernel):
    """
    Args:
      u: [B,L,N]
      kernel: [L,N,N]

    Returns: 
      [B,L,N]
    """
    seq_len = u.size(1)
    u = F.pad(u, (0, 0, seq_len-1, 0))
    u = rearrange(u, "B L N -> B N L")
    y = F.conv1d(u, kernel)
    y = rearrange(y, "B N L -> B L N")
    return y


def ssm_conv(A, B, C, u):
    """
    Args:
      A: [N,N]
      B: [N,N]
      C: [N,N]
      u: [B,L,N]

    Returns:
      [B,L,N]
    """
    seq_len = u.size(1)
    dA, dB, C = discretize(A, B, C, seq_len)
    K = ssm_conv_kernel(dA, dB, C, seq_len)
    ys = naive_conv(u, K)
    return ys


# -----------------------
# RNN-style SSM
# -----------------------
def ssm_scan(A, B, C, u):
    """
    Args:
      A: [N,N]
      B: [N,N]
      C: [N,N]
      u: [B,L,N]

    Returns:
      [B,L,N]
    """
    bs, seq_len, d_state = u.shape
    dA, dB, C = discretize(A, B, C, seq_len)

    x = torch.zeros((bs, d_state), device=u.device)
    ys = []
    for i in range(seq_len):
        dA_x = torch.einsum("nd,bd->bn", [dA, x])  # [B,N]
        dB_u = torch.einsum("nd,bd->bn", [dB, u[:, i]])  # [B,N]
        x = dA_x + dB_u  # [B,N]
        y = torch.einsum("nd,bd->bn", [C, x])  # [B,N]
        ys.append(y)
    ys = torch.stack(ys, dim=1)  # [B,L,N]
    return ys


# -----------------------
# Hippo
# -----------------------
def make_hippo(d_state):
    P = torch.sqrt(1 + 2 * torch.arange(d_state))  # [N,]
    A = P[:, None] * P[None, :]  # [N,N]
    A = torch.tril(A) - torch.diag(torch.arange(N))  # [N,N]
    return -A


if __name__ == "__main__":
    N = 5
    A, B, C = random_ssm(5)
    dA, dB, C = discretize(A, B, C, seq_len=10)
    u = torch.randn(2, 3, N)
    # y = ssm_scan(A, B, C, u)
    # print(y)
    # y = ssm_conv(A, B, C, u)
    # print(y)
    make_hippo(N)
