import torch 
import torch.nn as nn
from typing import Optional

from ..attention._attention_layer import MultiheadAttention
from ..common._modules import Transpose, _make_causal_token_mask, _causal_len_token_num

def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )

class TSTEncoder(nn.Module):
    """
    TSTEncoder
    """
    def __init__(
        self,
        config,
    ):
        super().__init__()

        if config.mica_mixer_type == 'betas':
            if config.layerwise_beta:
                beta = None
            else:
                beta = nn.Parameter(torch.rand((1, 1, config.n_heads, 1, 1))*1e-2)
                # Adjust the values to ensure they sum to 0
                with torch.no_grad():
                    beta -= beta.mean(dim=2, keepdim=True)
        else:
            beta=None

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    config=config,
                    beta=beta,
                )
                for i in range(config.n_layers)
            ]
        )
        self.res_attention = config.res_attention
        # context_len > 0 → cap lookback to exactly context_len (in patch
        # units) so fcd_samples never changes what a single FCD can see.
        # context_len == -1 is this codebase's "full context" convention
        # (ForkingSequences, Transformer's assert) → unrestricted causal.
        #
        # KNOWN LIMITATION: this bounds the lookback of a single attention
        # layer, not the whole stack's effective receptive field. With
        # n_layers > 1, each additional layer can extend the true receptive
        # field by up to (causal_len_token_num - 1) more patches (same
        # reason a deeper CNN has a bigger receptive field per layer) — so
        # a forecast's final representation can draw on somewhat more than
        # context_len of history, roughly 1 + n_layers*(causal_len_token_num-1)
        # patches in the worst case. This is bounded (unlike the pre-fix,
        # unbounded-by-block/series-length behavior) but not exact. An
        # exact fix would require either shrinking each layer's band so the
        # composed reach equals context_len (a training-relevant
        # architecture change, not just an inference tweak), or folding
        # eval windows into the batch dimension in chunks sized to match
        # training's own block_len (context_len + (fcd_samples-1)*stride)
        # instead of running the whole series through one shared pass —
        # see discussion in the PR/commit introducing this comment.
        self.causal_len_token_num = _causal_len_token_num(config)

    def forward(
        self,
        x: torch.Tensor,
        n_channels: int,
        key_padding_mask: torch.Tensor,
        channel_mask: torch.Tensor,
    ):
        output = x
        scores = None

        attention_mask = _make_causal_token_mask(
            key_padding_mask=key_padding_mask,
            device=x.device,
            causal_len_token_num=self.causal_len_token_num,
        ).reshape(x.shape[0], 1, x.shape[1], x.shape[1])
    
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    inputs_embeds=output,
                    n_channels=n_channels,
                    prev=scores,
                    attention_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    channel_mask=channel_mask,
                )
    
            return output
        else:
            for mod in self.layers:
                output = mod(
                    inputs_embeds=output, 
                    n_channels=n_channels,
                    attention_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    channel_mask=channel_mask,
                )

            return output

class TSTEncoderLayer(nn.Module):
    """
    TSTEncoderLayer
    """
    def __init__(
        self,
        config,
        beta: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Multi-Head attention
        self.res_attention = config.res_attention
        self.self_attn = MultiheadAttention(
            config=config,
            beta=beta,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(config.dropout)
        if "batch" in config.norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(config.hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.linear_hidden_size, bias=True), # bias hard-coded in PatchTST
            get_activation_fn(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_hidden_size, config.hidden_size, bias=True), # bias hard-coded in PatchTST
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(config.dropout)
        if "batch" in config.norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(config.hidden_size)

        self.pre_norm = config.pre_norm
        self.store_attn = config.store_attn

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        n_channels: int,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:

        # Multi-Head attention sublayer
        if self.pre_norm:
            inputs_embeds = self.norm_attn(inputs_embeds)
        ## Multi-Head attention
        if self.res_attention:
            inputs_embeds2, attn, scores = self.self_attn(
                n_channels=n_channels,
                Q=inputs_embeds,
                K=inputs_embeds,
                V=inputs_embeds,
                prev=prev,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                channel_mask=channel_mask,
            )
        else:
            inputs_embeds2, attn = self.self_attn(
                n_channels=n_channels,
                Q=inputs_embeds,
                K=inputs_embeds,
                V=inputs_embeds,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                channel_mask=channel_mask,
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        inputs_embeds = inputs_embeds + self.dropout_attn(
            inputs_embeds2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            inputs_embeds = self.norm_attn(inputs_embeds)

        # Feed-forward sublayer
        if self.pre_norm:
            inputs_embeds = self.norm_ffn(inputs_embeds)
        ## Position-wise Feed-Forward
        inputs_embeds2 = self.ff(inputs_embeds)
        ## Add & Norm
        inputs_embeds = inputs_embeds + self.dropout_ffn(
            inputs_embeds2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            inputs_embeds = self.norm_ffn(inputs_embeds)

        if self.res_attention:
            return inputs_embeds, scores
        else:
            return inputs_embeds
