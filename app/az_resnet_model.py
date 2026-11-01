"""Search-free residual CNN for 5x5 / 4-in-line (v22).

Difference vs v21: the policy head is fully convolutional (one logit per cell from a
3x3 conv + 1x1 conv) instead of a flattened FC layer, so tactical patterns learned in
one place transfer to every other place. v21 checkpoints still load (policy_head='fc').
No search logic lives here: forward() is a plain network pass.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(groups, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.relu(x + y)


class AZResNet(nn.Module):
    def __init__(self, channels=64, blocks=5, policy_head='conv'):
        super().__init__()
        self.channels, self.blocks_count, self.policy_head = int(channels), int(blocks), policy_head
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, channels), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels, 8) for _ in range(blocks)])
        if policy_head == 'conv':
            mid = max(8, channels // 2)
            self.policy_conv = nn.Conv2d(channels, mid, 3, padding=1, bias=False)
            self.policy_norm = nn.GroupNorm(8, mid)
            self.policy_out = nn.Conv2d(mid, 1, 1)
            self.policy_cell_bias = nn.Parameter(torch.zeros(25))
        else:  # v21 layout
            self.policy_conv = nn.Conv2d(channels, 2, 1)
            self.policy_norm = nn.GroupNorm(1, 2)
            self.policy_fc = nn.Linear(2 * 25, 25)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_norm = nn.GroupNorm(1, 1)
        self.value_fc1 = nn.Linear(25, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        z = self.blocks(self.stem(x))
        if self.policy_head == 'conv':
            p = F.relu(self.policy_norm(self.policy_conv(z)))
            logits = self.policy_out(p).flatten(1) + self.policy_cell_bias
        else:
            p = F.relu(self.policy_norm(self.policy_conv(z))).flatten(1)
            logits = self.policy_fc(p)
        v = F.relu(self.value_norm(self.value_conv(z))).flatten(1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))[:, 0]
        return logits, v


def encode(board, player):
    b = np.asarray(board).reshape(5, 5)
    return np.stack([(b == player), (b == (3 - player)), (b == 0)]).astype(np.float32)


def load_checkpoint(path, map_location='cpu'):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = payload['architecture']
    model = AZResNet(cfg['channels'], cfg['blocks'], cfg.get('policy_head', 'fc'))
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload
