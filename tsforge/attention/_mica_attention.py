import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ..common._modules import MLP, _causal_len_token_num, _banded_causal_sum


class MICAScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with MICA.
    Based on "Attention is All You Need" (Vaswani et al., 2017) and 
    "Leave No Context Behind" (Munkhdalai et al., 2024).
    """
    
    def __init__(
        self,
        config,
        d_k,
        d_v,
        beta: Optional[torch.tensor] = None,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.scale = d_k ** -0.5
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.res_attention = config.res_attention
        self.inner_dim = config.n_heads * d_v

        self.elu = nn.ELU()
        # Memory is causal with the same context_len band as the attention mask:
        # patch i's memory only holds keys/values of patches [i-W+1, i].
        self.causal_len_token_num = _causal_len_token_num(config)
        
        # Select memory update/retrieval methods based on channel exclusion
        if config.mica_channel_exclusion:
            self._update_memory_matrix = self._update_memory_matrix_channelexl
        else:
            self._update_memory_matrix = self._update_memory_matrix_allchannels

        # Select channel weight type:
        if config.mica_channel_weight_type == 'uniform':
            self._compute_channel_weights = self._compute_uniform_channel_weights
        elif config.mica_channel_weight_type == 'static':
            self.channel_weights = nn.Parameter(torch.ones(1, config.n_channels, config.n_heads, 1, 1))
            self._compute_channel_weights = self._compute_static_channel_weights
        elif config.mica_channel_weight_type == 'dynamic':
            self.channel_attn = nn.Linear(d_k, 1)
            self._compute_channel_weights = self._compute_query_channel_weights
        else:
            raise ValueError(f"mica_channel_weight_type '{config.mica_channel_weight_type}' not recognized. "
                    f"Use 'uniform', 'static', or 'dynamic'.")

        # Select gate mechanism
        if config.mica_mixer_type.lower() == 'betas':
            self.mixing_gate = self.beta_mixing_gate
            if beta is not None:
                self.beta = beta
            else:
                if config.channelwise_beta:
                    self.beta = nn.Parameter(torch.rand((1, config.n_channels, config.n_heads, 1, 1))*1e-2)
                else:
                    self.beta = nn.Parameter(torch.rand((1, 1, config.n_heads, 1, 1))*1e-2)
                # Center values around 0
                with torch.no_grad():
                    self.beta -= self.beta.mean(dim=2, keepdim=True)
    
        elif config.mica_mixer_type.lower() == 'mlp':
            self.mixing_gate = self.mlp_mixing_gate
            self.mlp = MLP(
                in_features=d_v * 2,
                out_features=d_v,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        elif config.mica_mixer_type.lower() == 'mlp_query':
            self.mixing_gate = self.mlp_query_mixing_gate
            self.mlp = MLP(
                in_features=d_v * 3,
                out_features=d_v,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        else:
            raise ValueError(f"mica_mixer_type '{config.mica_mixer_type}' not recognized. "
                    f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")
    
    def _compute_uniform_channel_weights(self, query_states, channel_mask):
        return channel_mask.float().view(channel_mask.shape[0], channel_mask.shape[1], 1, 1, 1)
    
    def _compute_static_channel_weights(self, query_states, channel_mask):
        weights = self.channel_weights  # [1, C, H, 1, 1]
        mask = channel_mask.float().view(channel_mask.shape[0], channel_mask.shape[1], 1, 1, 1)
        weights = weights.expand(mask.shape[0], -1, -1, -1, -1).clone()
        weights = weights.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(weights, dim=1)
    
    def _compute_query_channel_weights(self, query_states, channel_mask):
        # query_states: [B, C, H, P, D] — pooled causally over each patch's band
        P = query_states.shape[3]
        ones = torch.ones(P, 1, device=query_states.device, dtype=query_states.dtype)
        count = _banded_causal_sum(ones, dim=0, causal_len_token_num=self.causal_len_token_num)  # [P, 1]
        q_pooled = _banded_causal_sum(query_states, dim=3, causal_len_token_num=self.causal_len_token_num) / count  # [B, C, H, P, D]
        scores = self.channel_attn(q_pooled)  # [B, C, H, P, 1]
        mask = channel_mask.float().view(channel_mask.shape[0], channel_mask.shape[1], 1, 1, 1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        return torch.softmax(scores, dim=1)  # [B, C, H, P, 1]

    def _banded_memory(self, key_states, value_states, key_padding_mask):
        """Per-channel causal memory: M_i = Σ_{j in band(i)} σ(k_j) v_jᵀ, z_i = Σ σ(k_j)."""
        pad_mask = key_padding_mask.unsqueeze(2).unsqueeze(-1)  # [B, C, 1, P, 1]
        sigma_k = (self.elu(key_states) + 1.0) * pad_mask  # [B, C, H, P, D]
        kv = sigma_k.unsqueeze(-1) * (value_states * pad_mask).unsqueeze(-2)  # [B, C, H, P, D, D]

        memory_matrix = _banded_causal_sum(kv, dim=3, causal_len_token_num=self.causal_len_token_num)  # [B, C, H, P, D, D]
        z = _banded_causal_sum(sigma_k, dim=3, causal_len_token_num=self.causal_len_token_num)  # [B, C, H, P, D]
        return memory_matrix, z

    def _update_memory_matrix_allchannels(self, key_states, value_states, query_states, key_padding_mask, channel_mask):
        w = self._compute_channel_weights(query_states, channel_mask)  # [B, C, H, 1|P, 1]
        per_channel_mm, per_channel_z = self._banded_memory(key_states, value_states, key_padding_mask)

        memory_matrix = (w.unsqueeze(-1) * per_channel_mm).sum(dim=1, keepdim=True)  # [B, 1, H, P, D, D]
        z = (w * per_channel_z).sum(dim=1, keepdim=True)                           # [B, 1, H, P, D]

        return memory_matrix, z

    def _update_memory_matrix_channelexl(self, key_states, value_states, query_states, key_padding_mask, channel_mask):
        w = self._compute_channel_weights(query_states, channel_mask)  # [B, C, H, 1|P, 1]
        per_channel_mm, per_channel_z = self._banded_memory(key_states, value_states, key_padding_mask)

        C = key_states.shape[1]
        weighted_mm = w.unsqueeze(-1) * per_channel_mm                                 # [B, C, H, P, D, D]
        memory_matrix = weighted_mm.sum(dim=1, keepdim=True).expand(-1, C, -1, -1, -1, -1) - weighted_mm

        weighted_z = w * per_channel_z                                                 # [B, C, H, P, D]
        z = weighted_z.sum(dim=1, keepdim=True).expand(-1, C, -1, -1, -1) - weighted_z

        return memory_matrix, z

    def retrieve_from_memory(self, query_states, memory_matrix, z, key_padding_mask, channel_mask):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = (sigma_q.unsqueeze(-2) @ memory_matrix).squeeze(-2)  # [B, C, H, P, D]
        denominator = (sigma_q * z).sum(dim=-1, keepdim=True) + 1e-6     # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]

        # Zero out padded timesteps
        pad_mask = key_padding_mask.unsqueeze(2).unsqueeze(-1)  # [B, C, 1, L, 1]
        A_mem = A_mem * pad_mask

        # Handle masked channels
        A_mem = A_mem * channel_mask.float().view(
                    query_states.shape[0], 
                    query_states.shape[1], 1, 1, 1
                )
    
        return A_mem
    
    def beta_mixing_gate(self, a_mem, attn_output, query_states):
        """Learned interpolation between memory and attention using per-head betas."""
        attn_output = torch.sigmoid(self.beta) * a_mem + (1 - torch.sigmoid(self.beta)) * attn_output 
        return attn_output
    
    def mlp_mixing_gate(self, a_mem, attn_output, query_states):
        """Context-aware mixing via MLP on concatenated memory and attention."""
        attn_output = torch.cat([a_mem, attn_output], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
        attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim]
        return attn_output
    
    def mlp_query_mixing_gate(self, a_mem, attn_output, query_states):
        """Context-aware mixing via MLP with query, memory, and attention."""
        attn_output = torch.cat([a_mem, attn_output, query_states], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*3]
        attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim]
        return attn_output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_channels: int,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention with memory mechanism.
        
        Input shape (with channels):
            q: [bs * n_channels x seq_len x n_heads x d_k]
            k: [bs * n_channels x seq_len x n_heads x d_k]
            v: [bs * n_channels x seq_len x n_heads x d_v]
            n_channels: int
            prev: [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attention_mask: [bs*n_channels x 1 x seq_len x seq_len] — 1=attend, 0=block.
            channel_mask  : [bs x n_channels] — 1=real, 0=padded.

            
        Output shape:
            output: [bs x n_channels x n_heads x seq_len x d_v]
            attn_weights: [bs x n_channels x n_heads x seq_len x seq_len]
        """

        batch_size = q.shape[0] // n_channels
        seq_len = q.shape[1]

        q = q.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        q = q.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_k]
            
        k = k.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        k = k.permute(0, 1, 3, 4, 2).contiguous()  # [bs x n_channels x n_heads x d_k x seq_len]
            
        v = v.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        v = v.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_v]
        
        # Scaled MatMul (q, k) - compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # Vaswani et al. scaling

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            prev = prev.view(batch_size, n_channels, self.n_heads, seq_len, seq_len)
            attn_scores = attn_scores + prev
        
        attention_mask = attention_mask * channel_mask.view(-1, 1, 1, 1).float()
        attention_mask = attention_mask.reshape(batch_size, n_channels, 1, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)  # padded channels → 0 attention, not NaN
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output (v is [B, C, H, P, D])
        output = torch.matmul(attn_weights, v)
        
        # Infini-attention: retrieve from memory
        # k_for_memory should be [B, C, H, P, D] (not transposed)
        k_for_memory = k.transpose(-2, -1)
        memory_matrix, z = self._update_memory_matrix(
            key_states=k_for_memory, 
            value_states=v, 
            query_states=q, 
            key_padding_mask=key_padding_mask,
            channel_mask=channel_mask,
        )
        A_mem = self.retrieve_from_memory(
            query_states=q,
            memory_matrix=memory_matrix,
            z=z,
            key_padding_mask=key_padding_mask,
            channel_mask=channel_mask,
        )

        # Channel mixing
        output = self.mixing_gate(
            a_mem=A_mem, 
            attn_output=output, 
            query_states=q,
        ) # [bs x n_channels x n_heads x seq_len x d_v]


        output = output.transpose(2, 3).contiguous()  # [bs x n_channels x seq_len x n_heads x d_v]
        output = output.view(batch_size*n_channels, -1, self.inner_dim)  # [bs*n_channels x seq_len x n_heads*d_v]
        attn_weights = attn_weights.view(batch_size*n_channels, self.n_heads, seq_len, seq_len)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
        
