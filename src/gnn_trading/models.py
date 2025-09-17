"""Neural network building blocks used across the trading agents.

This module collects the latest iteration of the graph-attention encoder along
with the prediction heads that power the supervised training pipeline.  The
focus of the refactor in this iteration is to make the encoder more expressive
while keeping it modular so that future agent specialisations (price action,
news, macro, etc.) can easily share the same components.

The module favours explicit annotations and inline documentation so that the
reasoning behind each architectural choice is easy to follow.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def _make_skip_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Return an identity module when the dimensions match, otherwise a Linear.

    Residual/skip connections are critical for stable training when stacking
    many attention blocks.  When the dimensionality changes between layers we
    simply inject a small projection so that the residual path can still be
    added to the main branch.  This helper keeps the block definitions tidy.
    """
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim)


def _resolve_heads(dim: int, requested: int) -> int:
    """Pick a head count that divides the embedding dimension.

    Torch's multi-head attention prefers a clean division between the embedding
    size and the number of heads.  Instead of forcing the caller to get this
    right we adaptively pick the largest divisor not exceeding the requested
    count.
    """
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Attention and channel mixing helpers


class DropPath(nn.Module):
    """Stochastic depth / DropPath regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        if keep_prob <= 0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)


class SqueezeExcite(nn.Module):
    """Simple squeeze-excite block to re-weight channel responses."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from the graph encoder arrives as [steps, channels].  Convert to
        # [batch, channels, steps] so that the adaptive pool can operate.
        if x.dim() == 2:
            x_reshaped = x.transpose(0, 1).unsqueeze(0)
        elif x.dim() == 3:
            x_reshaped = x.transpose(1, 2)
        else:
            raise ValueError("Unexpected tensor rank for squeeze-excite block")

        pooled = self.pool(x_reshaped).squeeze(-1)
        weights = self.fc(pooled).unsqueeze(-1)
        scaled = x_reshaped * weights

        if x.dim() == 2:
            return scaled.squeeze(0).transpose(0, 1)
        return scaled.transpose(1, 2)


class GraphAttentionBlock(nn.Module):
    """Transformer-style attention block with residual MLP and SE gating."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.2,
        concat: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.concat = concat
        self.heads = heads
        self.out_dim = out_dim * heads if concat else out_dim
        self.conv = GATv2Conv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout)
        self.skip = _make_skip_proj(in_dim, self.out_dim)
        self.norm1 = nn.LayerNorm(self.out_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.ffn = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        self.channel_gate = SqueezeExcite(self.out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Attention path
        residual = x
        attn = self.conv(x, edge_index)
        attn = self.act(attn)
        attn = self.dropout(attn)
        x = self.norm1(self.skip(residual) + self.drop_path(attn))

        # Feed-forward refinement with squeeze-excite gating
        ff_res = x
        ff = self.ffn(x)
        ff = self.dropout(ff)
        ff = self.channel_gate(ff)
        x = self.norm2(ff_res + self.drop_path(ff))
        return x


class TemporalConvRefiner(nn.Module):
    """Temporal convolutional block supporting multi-scale receptive fields."""

    def __init__(
        self,
        channels: int,
        kernels: Sequence[int] | int = (3,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(kernels, int):
            kernels = (kernels,)
        if not kernels:
            raise ValueError("TemporalConvRefiner requires at least one kernel size")
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.mix = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels * len(self.convs), channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        res = seq
        seq_t = seq.transpose(1, 2)
        mixes = [conv(seq_t) for conv in self.convs]
        stacked = torch.cat(mixes, dim=1)
        refined = self.mix(stacked).transpose(1, 2)
        return self.norm(refined + res)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding injected prior to recurrent modelling."""

    def __init__(self, dim: int, max_len: int = 10_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        steps = seq.size(1)
        return seq + self.pe[:steps].unsqueeze(0).to(seq.device)


class TemporalAttentionPooling(nn.Module):
    """Lightweight attention pooling head with a learnable query token."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        heads = _resolve_heads(dim, heads)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = seq.size(0)
        query = self.query.expand(batch, -1, -1)
        pooled, weights = self.attn(query, seq, seq)
        pooled = pooled.squeeze(1)
        weights = weights.squeeze(1)
        return self.dropout(pooled), weights


class GAT_GRU_Encoder(nn.Module):
    """Graph attention encoder with temporal refinement and optional attention.

    Compared to the previous iteration we add:

    * Positional encodings prior to the GRU so that the recurrent model can
      rely on explicit ordering information.
    * A lightweight temporal convolutional refiner that sharpens local
      dynamics before the sequence is collapsed by the GRU.
    * Richer post-GRU processing including layer-normalised projections and a
      squeeze-excite gate.
    """

    def __init__(
        self,
        in_dim: int,
        hid: int = 64,
        heads: int = 2,
        gru_hid: int = 64,
        proj_dim: int = 64,
        dropout: float = 0.2,
        num_gru_layers: int = 2,
        bidirectional: bool = True,
        num_gat_layers: int = 2,
        use_temporal_attn: bool = True,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_positional: bool = True,
        temporal_kernels: Sequence[int] | int = (3, 5),
        stochastic_depth: float = 0.1,
        use_temporal_pooling: bool = True,
        pooling_heads: int = 4,
    ) -> None:
        super().__init__()
        blocks = []
        cur_dim = in_dim
        drop_rates = torch.linspace(0, float(stochastic_depth), steps=max(num_gat_layers, 1))
        for layer_idx in range(num_gat_layers):
            layer_heads = heads if layer_idx == 0 else 1
            block = GraphAttentionBlock(
                in_dim=cur_dim,
                out_dim=hid,
                heads=layer_heads,
                dropout=dropout,
                concat=True,
                drop_path=float(drop_rates[layer_idx]) if num_gat_layers > 1 else 0.0,
            )
            blocks.append(block)
            cur_dim = block.out_dim
        self.graph_layers = nn.ModuleList(blocks)
        self.graph_norm = nn.LayerNorm(cur_dim)

        self.graph_proj = (
            nn.Sequential(
                nn.Linear(cur_dim, hid),
                nn.GELU(),
                nn.LayerNorm(hid),
            )
            if cur_dim != hid
            else None
        )

        gru_input_dim = hid if self.graph_proj is not None else cur_dim
        self.positional = PositionalEncoding(gru_input_dim) if use_positional else None
        self.temporal_refiner = TemporalConvRefiner(gru_input_dim, kernels=temporal_kernels, dropout=dropout)

        gru_dropout = dropout if num_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hid,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = gru_hid * self.num_directions
        self.post_gru = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )
        self.final_dropout = nn.Dropout(dropout)
        self.final_gate = SqueezeExcite(self.output_dim)
        self.temporal_pool = (
            TemporalAttentionPooling(self.output_dim, pooling_heads, attn_dropout)
            if use_temporal_pooling
            else None
        )
        self._last_attention: Optional[torch.Tensor] = None

        self.temporal_attn: Optional[nn.MultiheadAttention] = None
        if use_temporal_attn:
            attn_heads = _resolve_heads(gru_input_dim, attn_heads)
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=gru_input_dim,
                num_heads=attn_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.temporal_norm = nn.LayerNorm(gru_input_dim)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        """Return the most recent temporal attention weights (if pooling enabled)."""

        return self._last_attention

    def forward(self, data, return_proj: bool = True):
        x, edge_index = data.x, data.edge_index
        for block in self.graph_layers:
            x = block(x, edge_index)
        x = self.graph_norm(x)
        if self.graph_proj is not None:
            x = self.graph_proj(x)
        seq = x.unsqueeze(0)  # [1, T, H]
        if self.positional is not None:
            seq = self.positional(seq)
        seq = self.temporal_refiner(seq)
        if self.temporal_attn is not None:
            attn_out, _ = self.temporal_attn(seq, seq, seq)
            seq = self.temporal_norm(attn_out + seq)
        out, _ = self.gru(seq)
        if self.temporal_pool is not None:
            pooled, weights = self.temporal_pool(out)
            self._last_attention = weights.detach()
            h = pooled
        else:
            h = out[:, -1, :]
            self._last_attention = None
        h = self.post_gru(h)
        h = self.final_gate(h)
        h = self.final_dropout(h)
        if return_proj:
            z = self.proj(h)
            z = nn.functional.normalize(z, dim=-1)
            return z.squeeze(0)
        return h.squeeze(0)


class MultiTaskHead(nn.Module):
    """Shared head that emits classification, return and volatility signals."""

    def __init__(self, in_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )

        def head_block(out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, out),
            )

        self.cls_head = head_block(2)
        self.reg_head = head_block(1)
        self.vol_head = head_block(1)

    def forward(self, h: torch.Tensor):
        h = self.dropout(h)
        base = self.shared(h)
        return self.cls_head(base), self.reg_head(base), self.vol_head(base)


class NewsFusion(nn.Module):
    """Cross-modal fusion between price embeddings and external news vectors."""

    def __init__(self, in_price: int, in_news: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_price + in_news),
            nn.Linear(in_price + in_news, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, price_h: torch.Tensor, news_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([price_h, news_vec], dim=-1)
        return self.fc(x)


class SupervisedModel(nn.Module):
    """Supervised wrapper that supports MC-dropout for uncertainty estimates."""

    def __init__(self, encoder: GAT_GRU_Encoder, news_dim: int | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.news_dim = news_dim
        hid = getattr(encoder, "output_dim", 64)
        if news_dim is not None:
            self.fusion = NewsFusion(hid, news_dim, hid)
        else:
            self.fusion = None
        self.heads = MultiTaskHead(hid, dropout=dropout)

    def forward(self, data, mc_dropout: bool = False, samples: int = 1):
        def single_pass():
            h = self.encoder(data, return_proj=False)
            if hasattr(data, "news") and self.fusion is not None:
                news_vec = data.news
                h = self.fusion(h, news_vec)
            logits, ret, vol = self.heads(h)
            return logits, ret, vol

        if not mc_dropout or samples == 1:
            return single_pass()
        # MC dropout for uncertainty
        prev_training = self.training
        self.train()  # enable dropout
        probs = []
        rets = []
        vols = []
        for _ in range(samples):
            lg, rg, vg = single_pass()
            probs.append(torch.softmax(lg, dim=-1)[1].unsqueeze(0))
            rets.append(rg)
            vols.append(vg)
        if not prev_training:
            self.eval()
        probs = torch.cat(probs, 0).mean(0)
        return probs, torch.stack(rets, 0).mean(0), torch.stack(vols, 0).mean(0)
