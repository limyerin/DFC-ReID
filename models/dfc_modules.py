import torch
from torch import nn
from torch.nn import functional as F


class IdentityGuidedCompensationGate(nn.Module):
    """Input-dependent residual scale for the existing NAF compensation;
    identity guidance is provided indirectly by the proposed DCL, not by
    label inputs to the gate.

    Returns naf_feat * tanh(z), which the caller ADDS to the baseline
    compensation (naf_feat * base_scale), i.e. the effective per-channel scale
    is base_scale + tanh(z). With z == 0 at init the network is exactly the
    baseline, tanh'(0) == 1 (no saturation), and the gate receives gradient
    even when base_scale is 0 (NAF1's delta is zero-initialized).
    """
    def __init__(self, channels=1024, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, main_feat, naf_feat):
        if main_feat.shape != naf_feat.shape:
            raise ValueError('ICG main and NAF feature shapes differ')
        pooled = torch.cat([main_feat, naf_feat], dim=1).mean((2, 3), keepdim=True)
        return naf_feat * self.net(pooled)


def discriminative_compensation_loss(feat_pre, feat_post, labels, margin=0.02):
    """Encourage improvement in hardest-positive / hardest-negative cosine margin.
    Hypothesis-level proposed loss; not part of the published IRL.
    """
    if feat_pre.shape != feat_post.shape:
        raise ValueError('DCL features must have matching shapes')
    pre = F.normalize(feat_pre.float(), dim=1)
    post = F.normalize(feat_post.float(), dim=1)
    sim_pre, sim_post = pre @ pre.T, post @ post.T
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = same & ~eye
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return feat_post.sum() * 0.0
    pos_pre = sim_pre.masked_fill(~positive, 2.0).min(1).values
    neg_pre = sim_pre.masked_fill(~negative, -2.0).max(1).values
    pos_post = sim_post.masked_fill(~positive, 2.0).min(1).values
    neg_post = sim_post.masked_fill(~negative, -2.0).max(1).values
    target = (pos_pre - neg_pre).detach() + margin
    return F.relu(target[valid] - (pos_post - neg_post)[valid]).mean()
