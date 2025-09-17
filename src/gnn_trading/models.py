import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def _make_skip_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Return an identity module when the dimensions match, otherwise a Linear."""
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim)


def _resolve_heads(dim: int, requested: int) -> int:
    """Pick a head count that divides the embedding dimension."""
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


class GraphAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.2, concat: bool = True):
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
        self.ffn = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm1(x + self.skip(residual))

        ff_res = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + ff_res)
        return x


class GAT_GRU_Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
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
    ):
        super().__init__()
        blocks = []
        cur_dim = in_dim
        for layer_idx in range(num_gat_layers):
            layer_heads = heads if layer_idx == 0 else 1
            block = GraphAttentionBlock(
                in_dim=cur_dim,
                out_dim=hid,
                heads=layer_heads,
                dropout=dropout,
                concat=True,
            )
            blocks.append(block)
            cur_dim = block.out_dim
        self.graph_layers = nn.ModuleList(blocks)
        self.graph_norm = nn.LayerNorm(cur_dim)

        self.graph_proj = nn.Sequential(
            nn.Linear(cur_dim, hid),
            nn.GELU(),
            nn.LayerNorm(hid),
        ) if cur_dim != hid else None

        gru_input_dim = hid if self.graph_proj is not None else cur_dim

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
        )
        self.final_dropout = nn.Dropout(dropout)

        self.temporal_attn = None
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

    def forward(self, data, return_proj=True):
        x, edge_index = data.x, data.edge_index
        for block in self.graph_layers:
            x = block(x, edge_index)
        x = self.graph_norm(x)
        if self.graph_proj is not None:
            x = self.graph_proj(x)
        seq = x.unsqueeze(0)  # [1,T,H]
        if self.temporal_attn is not None:
            attn_out, _ = self.temporal_attn(seq, seq, seq)
            seq = self.temporal_norm(attn_out + seq)
        out, _ = self.gru(seq)
        h = out[:, -1, :]     # [1,H]
        h = self.post_gru(h)
        h = self.final_dropout(h)
        if return_proj:
            z = self.proj(h)
            z = nn.functional.normalize(z, dim=-1)
            return z.squeeze(0)
        return h.squeeze(0)


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
        )
        def head_block():
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.reg_head = head_block()
        self.vol_head = head_block()

    def forward(self, h):
        h = self.dropout(h)
        base = self.shared(h)
        return self.cls_head(base), self.reg_head(base), self.vol_head(base)


class NewsFusion(nn.Module):
    def __init__(self, in_price, in_news, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_price + in_news),
            nn.Linear(in_price + in_news, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, price_h, news_vec):
        x = torch.cat([price_h, news_vec], dim=-1)
        return self.fc(x)


class SupervisedModel(nn.Module):
    def __init__(self, encoder: GAT_GRU_Encoder, news_dim: int|None = None, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.news_dim = news_dim
        hid = getattr(encoder, "output_dim", 64)
        if news_dim is not None:
            self.fusion = NewsFusion(hid, news_dim, hid)
        else:
            self.fusion = None
        self.heads = MultiTaskHead(hid, dropout=dropout)

    def forward(self, data, mc_dropout=False, samples=1):
        def single_pass():
            h = self.encoder(data, return_proj=False)
            if hasattr(data, 'news') and self.fusion is not None:
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
