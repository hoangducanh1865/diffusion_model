import torch.nn as nn
from src.attention import SelfAttention
from src.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, mlp_ratio=2, proj_p=0, attn_p=0, mlp_p=0):
        super().__init__()
        self.norm_1 = nn.LayerNorm(in_channels, eps=1e-6)  # @QUESTION: what does eps mean?
        self.attn = SelfAttention(
            in_channels=in_channels, num_heads=num_heads, attn_p=attn_p, proj_p=proj_p
        )
        self.norm_2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.mlp = MLP(in_channels=in_channels, mlp_ratio=mlp_ratio, mlp_p=mlp_p)

    def forward(self, x):
        b, c, h, w = x.shape  # batch_size, channels, height, weight
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # Swap dim 1 anf dim 2
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x
