import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def _make_skip_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Return an identity module when the dimensions match, otherwise a Linear."""
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim)


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
    ):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hid, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hid * heads, hid, heads=1, concat=True, dropout=dropout)

        self.skip1 = _make_skip_proj(in_dim, hid * heads)
        self.skip2 = _make_skip_proj(hid * heads, hid)
        self.norm1 = nn.LayerNorm(hid * heads)
        self.norm2 = nn.LayerNorm(hid)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        gru_dropout = dropout if num_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hid,
            hidden_size=gru_hid,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = gru_hid * self.num_directions

        self.proj = nn.Sequential(
            nn.Linear(self.output_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, data, return_proj=True):
        x, edge_index = data.x, data.edge_index
        residual = x
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm1(x + self.skip1(residual))

        residual = x
        x = self.gat2(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm2(x + self.skip2(residual))
        seq = x.unsqueeze(0)  # [1,T,H]
        out, _ = self.gru(seq)
        h = out[:, -1, :]     # [1,H]
        h = self.dropout(h)
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
