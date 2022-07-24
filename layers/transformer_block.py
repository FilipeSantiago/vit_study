import torch.nn as nn
from layers import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_mult=4, dropout=0.2):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_mult * embed_size),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = self.norm1(x)

        attention = self.dropout(self.attention(_x))

        x = attention + x

        _x = self.norm2(x)
        _x = self.dropout(self.ff(_x))

        return _x + x
