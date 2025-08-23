
import torch
import torch.nn as nn

class IdentityVotes(nn.Module):
    def forward(self, feats):
        # placeholder for compatibility; feats -> (B,C) global
        return feats

class VotesLite(nn.Module):
    """Single-object 'VoteNet-lite' head: predict per-seed center/size/yaw offsets and aggregate"""
    def __init__(self, feat_c=512, seeds=128):
        super().__init__()
        self.seeds = seeds
        self.seed_proj = nn.Linear(feat_c, feat_c)
        self.vote = nn.Sequential(
            nn.Linear(feat_c, feat_c//2), nn.GELU(),
            nn.Linear(feat_c//2, 8)  # dx,dy,dz, sx,sy,sz, sin(yaw), cos(yaw)
        )
        self.score = nn.Linear(feat_c, 1)

    def forward(self, feats, seed_feats):
        # feats: (B,C) global, seed_feats: (B,S,C)
        B,S,C = seed_feats.shape
        v = self.vote(seed_feats)                     # (B,S,8)
        score = self.score(seed_feats).squeeze(-1)    # (B,S)
        w = torch.softmax(score, dim=-1).unsqueeze(-1)# (B,S,1)
        v_avg = (v * w).sum(dim=1)                    # (B,8)
        yaw = torch.atan2(v_avg[...,6], v_avg[...,7]) # (-pi,pi)
        center = v_avg[...,:3]
        size = torch.relu(v_avg[...,3:6]) + 1e-3
        return center, size, yaw

class TwoHeadSOTA(nn.Module):
    def __init__(self, num_classes=38, backbone="pointnext_tiny", box_head="residual", out_dim=512):
        super().__init__()
        if backbone == "pointnext_tiny":
            from backbones.pointnext_tiny import PointNeXtTiny as Backbone
            self.backbone = Backbone(out_dim=out_dim)
        else:
            raise ValueError("Unknown backbone")
        self.cls = nn.Linear(out_dim, num_classes)
        self.box_head_type = box_head
        if box_head == "residual":
            self.box = nn.Linear(out_dim, 7)
            self.vote_head = None
        elif box_head == "votes":
            self.box = None
            self.vote_head = VotesLite(feat_c=out_dim, seeds=128)
        else:
            raise ValueError("box_head must be 'residual' or 'votes'")

    def forward(self, pts):
        # pts: (B,N,3)
        feats = self.backbone(pts)     # (B,C)
        logits = self.cls(feats)
        if self.box_head_type == "residual":
            box = self.box(feats)
            return logits, box
        else:
            # dummy seed features: tile global feats as seeds (CPU-friendly)
            seed_feats = feats.unsqueeze(1).expand(-1,128,-1)
            center, size, yaw = self.vote_head(feats, seed_feats)
            box = torch.cat([center, size, yaw.unsqueeze(-1)], dim=-1)
            return logits, box
