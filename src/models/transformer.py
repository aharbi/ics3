import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.RMSNorm(dim, eps=eps)
        hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        q = self.norm1(x)
        attn_out = self.attn(q, q, q, attn_mask=attn_mask)[0]
        attn_out = self.dropout1(attn_out)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        mlp_out = self.dropout2(mlp_out)
        x = x + mlp_out

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio, dropout, eps)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return x


if __name__ == "__main__":
    B, N, d = 2, 16, 64
    x = torch.randn(B, N, d)
    attn_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()  # True = masked

    model = Transformer(dim=d, depth=2, heads=4, mlp_ratio=2.0, dropout=0.1)
    y = model(x, attn_mask=attn_mask)
    print(y.shape)
