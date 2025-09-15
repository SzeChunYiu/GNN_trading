import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

class GAT_GRU_Encoder(nn.Module):
    def __init__(self, in_dim, hid=64, heads=2, gru_hid=64, proj_dim=64, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hid, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hid*heads, hid, heads=1, concat=True, dropout=dropout)
        self.gru  = nn.GRU(input_size=hid, hidden_size=gru_hid, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(gru_hid, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, return_proj=True):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        seq = x.unsqueeze(0)  # [1,T,H]
        out, _ = self.gru(seq)
        h = out[:, -1, :]     # [1,H]
        h = self.dropout(h)
        if return_proj:
            z = self.proj(h); z = nn.functional.normalize(z, dim=-1)
            return z.squeeze(0)
        return h.squeeze(0)

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
        self.reg = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.vol = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, h):
        h = self.dropout(h)
        return self.cls(h), self.reg(h), self.vol(h)

class NewsFusion(nn.Module):
    def __init__(self, in_price, in_news, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_price + in_news, out_dim),
            nn.ReLU(),
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
        hid = 64  # encoder gru_hid by default
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

        if not mc_dropout or samples==1:
            return single_pass()
        # MC dropout for uncertainty
        probs = []; rets = []; vols = []
        self.train()  # enable dropout
        for _ in range(samples):
            lg, rg, vg = single_pass()
            probs.append(torch.softmax(lg, dim=-1)[1].unsqueeze(0))
            rets.append(rg); vols.append(vg)
        self.eval()
        return torch.cat(probs,0).mean(0), torch.stack(rets,0).mean(0), torch.stack(vols,0).mean(0)
