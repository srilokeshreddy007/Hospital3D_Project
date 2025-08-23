
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k=16):
    # x: (B,N,C) -> pairwise distances (B,N,N)
    # Use cdist; on CPU keep N<=4096 for speed
    with torch.no_grad():
        d = torch.cdist(x, x)  # (B,N,N)
        idx = d.topk(k, largest=False).indices  # (B,N,k)
    return idx

def group_features(x, idx):
    # x: (B,N,C), idx: (B,N,k) -> (B,N,k,C)
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_expand = idx.unsqueeze(-1).expand(B, N, k, C)
    return torch.gather(x.unsqueeze(1).expand(B, N, N, C), 2, idx_expand)

class DWConv1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=(1,1), groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=(1,1))
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        # x: (B,C,N,1)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class LocalBlock(nn.Module):
    """PointNeXt-style local aggregation with residual MLP + depthwise conv"""
    def __init__(self, in_c, out_c, k=16):
        super().__init__()
        self.k = k

        # Use LayerNorm (works on (B,N,C)); BatchNorm1d expects (B,C,L) and caused the crash
        self.fc1 = nn.Linear(in_c, out_c)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(out_c)

        self.fc2 = nn.Linear(out_c, out_c)
        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(out_c)

        self.conv = DWConv1d(out_c)
        self.proj = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, pts, feats):
        # pts: (B,N,3), feats: (B,N,C_in)
        idx = knn(pts, self.k)                 # (B,N,k)
        neigh = group_features(feats, idx)     # (B,N,k,C_in)
        x = neigh.mean(dim=2)                  # (B,N,C_in)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.ln2(x)

        # depthwise along N
        x2 = x.transpose(1, 2).unsqueeze(-1)   # (B,C_out,N,1)
        x2 = self.conv(x2).squeeze(-1).transpose(1, 2)  # (B,N,C_out)
        return x2 + self.proj(feats)


class PointNeXtTiny(nn.Module):
    """Minimal CPU-friendly PointNeXt-ish backbone"""
    def __init__(self, out_dim=512, width=64, k=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.BatchNorm1d(width),
        )
        self.block1 = LocalBlock(width, width, k=k)
        self.block2 = LocalBlock(width, width*2, k=k)
        self.block3 = LocalBlock(width*2, width*2, k=k)
        self.head = nn.Sequential(
            nn.Linear(width*2, out_dim), nn.GELU(), nn.BatchNorm1d(out_dim)
        )

    def forward(self, pts):
        # pts: (B,N,3)
       B, N, _ = pts.shape
       pts = pts.contiguous()  # ensure contiguous before flattening
       x = self.stem(pts.reshape(B * N, 3)).reshape(B, N, -1)
       x = self.block1(pts, x)
       x = self.block2(pts, x)
       x = self.block3(pts, x)
       x = x.max(dim=1).values  # global max pooling
       x = self.head(x)
       return x  # (B,out_dim)
