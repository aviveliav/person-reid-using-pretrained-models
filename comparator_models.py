import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineComparator(nn.Module):
    """
    Affine function on the cosine similarity of the two features vectors
    """
    def __init__(self, init_scale: float = 10.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0))


    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        cos = (f1 * f2).sum(dim=1, keepdim=True)
        return (self.scale * cos + self.bias).squeeze(1)



class MLPComparator(nn.Module):
    """
    FC network
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.LazyLinear(hidden_dim)   # adapts to input dim
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, f1, f2):
        z1 = self.proj(f1)
        z2 = self.proj(f2)

        # elementwise product only
        h = z1 * z2
        h = self.dropout(F.relu(h))
        return self.fc(h).squeeze(1)


class AttentionComparator(nn.Module):
    """
    Comparator with attention over |f1 - f2| and f1*f2 interactions.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.proj = nn.LazyLinear(hidden_dim) # linear layer for projection
        self.dropout = nn.Dropout(0.3)

        # attention network
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())

        # final classifier
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, f1, f2):
        # projection
        z1 = self.proj(f1)
        z2 = self.proj(f2)

        # multiplicative + difference
        mul = z1 * z2
        diff = torch.abs(z1 - z2)
        inter = torch.cat([mul, diff], dim=1)

        # attention
        attn_weights = self.attn(inter)

        # apply attention to the multiply result of the two features projections
        h = mul * attn_weights

        # final score
        h = self.dropout(F.relu(h))
        return self.fc(h).squeeze(1)



class LowRankBilinearHead(nn.Module):
    """
    logits = <P@f1, P@f2> + w.dot(|f1-f2|) + b
    """
    def __init__(self, rank: int = 16, l2_normalize: bool = True):
        super().__init__()
        self.P = nn.LazyLinear(rank, bias=False)
        self.w = nn.LazyLinear(1, bias=False)
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.l2_normalize = l2_normalize
        self.dropout = nn.Dropout(0.5)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        z1 = self.P(f1)
        z2 = self.P(f2)

        if self.l2_normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        bil = (z1 * z2).sum(dim=1, keepdim=True)

        sad = self.w(self.dropout(torch.abs(f1 - f2)))
        return (bil + sad + self.bias).squeeze(1)


_COMPARATOR_REGISTRY = {
    "cosine_similarity": CosineComparator,
    "mlp": MLPComparator,
    "attention": AttentionComparator,
    "lowrank_bilinear": LowRankBilinearHead,
}

def make_comparator(name: str, device: torch.device, **kwargs):
    key = name.lower()
    if key not in _COMPARATOR_REGISTRY:
        raise ValueError(f"Unknown comparator '{name}'. Available: {list(_COMPARATOR_REGISTRY.keys())}")
    return _COMPARATOR_REGISTRY[key](**kwargs).to(device)