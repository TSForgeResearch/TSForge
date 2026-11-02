import warnings
import torch
from torch import nn
import math


class IdentityLayer(nn.Module):
    """Drop-in replacement when encoder, decoder, or output_layer is set to 'none'."""
    def forward(self, x, **kwargs):
        return x
    
class MLP(nn.Module):
    """Multi-Layer Perceptron Class

    **Parameters:**<br>
    `in_features`: int, dimension of input.<br>
    `out_features`: int, dimension of output.<br>
    `activation`: str, activation function to use.<br>
    `hidden_size`: int, dimension of hidden layers.<br>
    `num_layers`: int, number of hidden layers.<br>
    `dropout`: float, dropout rate.<br>
    """

    def __init__(
        self, in_features, out_features, activation, hidden_size, num_layers, dropout
    ):
        super().__init__()
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        self.activation = getattr(nn, activation)()

        # MultiLayer Perceptron
        # Input layer
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.activation,
            nn.Dropout(dropout),
        ]
        # Hidden layers
        for i in range(num_layers - 2):
            layers += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                self.activation,
                nn.Dropout(dropout),
            ]
        # Output layer
        layers += [nn.Linear(in_features=hidden_size, out_features=out_features)]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PositionalEncoding(nn.Module):
    """
    Unified positional encoding module supporting:
      - Sin/Cos encodings (1D or 2D)
      - Linear / Exponential coordinate encodings (1D or 2D)
      - Random (normal/uniform/zeros) encodings
      - Learnable or fixed encodings
    """

    def __init__(
        self,
        pe_type: str = "sincos",   # ['sincos', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'gauss', 'uniform', 'zeros', 'zero', None]
        q_len: int = 5000,
        hidden_size: int = 768,
        learn_pe: bool = False,
        normalize: bool = True,
    ):
        super().__init__()

        self.pe_type = pe_type
        self.q_len = q_len
        self.hidden_size = hidden_size
        self.learn_pe = learn_pe
        self.normalize = normalize

        # Build encoding tensor
        W_pos = self._build_encoding()
        if self.learn_pe:
            self.W_pos = nn.Parameter(W_pos)
        else:
            self.register_buffer("W_pos", W_pos)

    def _build_encoding(self):
        pe = self.pe_type

        if pe is None:
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.uniform_(W_pos, -0.02, 0.02)

        elif pe in ["zero", "zeros"]:
            W_pos = torch.zeros((self.q_len, self.hidden_size))

        elif pe in ["gauss", "normal"]:
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.normal_(W_pos, mean=0.0, std=0.1)

        elif pe == "uniform":
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)

        elif pe == "sincos":
            W_pos = self._sin_cos_encoding(self.q_len, self.hidden_size)

        elif pe in ["lin1d", "exp1d"]:
            W_pos = self._coord1d_encoding(self.q_len, exponential=("exp" in pe))

        elif pe in ["lin2d", "exp2d"]:
            W_pos = self._coord2d_encoding(self.q_len, self.hidden_size, exponential=("exp" in pe))

        else:
            raise ValueError(
                f"{pe} is not a valid positional encoding type. "
                f"Available: None, 'zeros', 'zero', 'normal', 'uniform', 'lin1d', "
                f"'exp1d', 'lin2d', 'exp2d', 'sincos'."
            )

        if self.normalize:
            W_pos = (W_pos - W_pos.mean()) / (W_pos.std() * 10)

        return W_pos

    def _sin_cos_encoding(self, q_len, hidden_size):
        """Classic sinusoidal encoding"""
        pe = torch.zeros(q_len, hidden_size)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _coord1d_encoding(self, q_len, exponential=False):
        """1D coordinate encoding (linear or exponential)"""
        exponent = 0.5 if exponential else 1.0
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** exponent) - 1
        return cpe

    def _coord2d_encoding(self, q_len, hidden_size, exponential=False, eps=1e-3):
        """2D coordinate encoding (linear or exponential)"""
        x = 0.5 if exponential else 1
        for _ in range(100):
            cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
            )
            mean = cpe.mean()
            if abs(mean) <= eps:
                break
            x += -0.001 if mean > eps else 0.001
        return cpe

    def forward(self, x, positions: torch.Tensor = None):
        """
        Returns positional encodings broadcastable to input tensor `x`.
        Accepts:
          - [batch_size, n_channels, seq_len, d_model]
        """
        seq_len = x.size(2)
        pe = self.W_pos[:seq_len]
        return pe.unsqueeze(0).unsqueeze(0) #[batch_size, n_channels, seq_len, d_model]

    
class Transpose(nn.Module):
    """
    Transpose
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

def _causal_len_token_num(config) -> int:
    """
    context_len in patch units: the causal lookback band shared by attention
    (`_make_causal_token_mask`) and the MICA memory. None when context_len == -1
    (full context) → unrestricted causal.
    """
    if config.context_len > 0:
        return int((config.context_len - config.patch_len) / config.stride + 1)
    return None


def _banded_causal_sum(x: torch.Tensor, dim: int, causal_len_token_num: int = None) -> torch.Tensor:
    """
    out_i = Σ_{j ∈ [i - causal_len_token_num + 1, i]} x_j along `dim`, computed as a
    cumulative sum minus the cumulative sum causal_len_token_num steps back. Same band
    as `_make_causal_token_mask`; None → plain cumulative (unrestricted causal) sum.
    """
    out = x.cumsum(dim)
    n = x.shape[dim]
    if causal_len_token_num is None or causal_len_token_num >= n:
        return out
    shifted = torch.cat([
        torch.zeros_like(out.narrow(dim, 0, causal_len_token_num)),
        out.narrow(dim, 0, n - causal_len_token_num),
    ], dim=dim)
    return out - shifted

def _make_causal_token_mask(
    key_padding_mask: torch.Tensor,  # [B, C, L] — 1/True = VALID, 0/False = INVALID
    device: torch.device,
    causal_len_token_num: int = None,
) -> torch.Tensor:
    """
    causal_len_token_num : int, optional
        Caps each query's lookback to at most this many keys (itself
        included) instead of the full causal history. Set to context_len
        expressed in patch units so a shared encoder pass — training block
        or the whole eval series — gives each forecast origin exactly
        context_len worth of history, regardless of fcd_samples. None
        keeps unrestricted causal attention.
    """
    B, C, L = key_padding_mask.shape

    # Causal mask: [1, 1, 1, L, L] — 1 where attention is ALLOWED
    causal_mask = torch.ones(L, L, dtype=torch.float, device=device).tril()
    if causal_len_token_num is not None and causal_len_token_num < L:
        causal_mask = causal_mask - causal_mask.tril(diagonal=-causal_len_token_num)
    causal_mask = causal_mask.view(1, 1, 1, L, L)

    # Token validity: [B, C, 1, 1, L] — 1 = valid key, 0 = invalid key
    token_mask = key_padding_mask.float().unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1, L]

    # Combine: 0 if EITHER is blocked
    combined = causal_mask * token_mask  # [B, C, 1, L, L]

    return combined  # [B, C, 1, L, L] — 1=attend, 0=block