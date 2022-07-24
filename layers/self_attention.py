import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), f"Embed size needs to be divisible by heads ({heads})"

        self.tovalues = nn.Linear(embed_size, embed_size, bias=False)
        self.tokeys = nn.Linear(embed_size, embed_size, bias=False)
        self.toqueries = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        N, batch, embedding = x.size()
        assert embedding == self.embed_size, f'Input embedding dim ({embedding}) should match layer embedding dim ({self.emb})'

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        keys = keys.view(N, batch, self.heads, self.head_dim)
        queries = queries.view(N, batch, self.heads, self.head_dim)
        values = values.view(N, batch, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape(N:n, query_len: q, heads: h, heads_dim: d) => nqhd
        # keys shape(N:n, key_len: k, heads: h, heads_dim: d) => nkhd
        # energy shape(N:n, heads: h, query_len: q, key_len:k) => nhqk

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        # attetion shape(N:n, heads: h, query_len: q, key_len: k) => nhqk
        # values shape(N:n, values_len: v, heads: h, heads_dim: d) => nvhd (v = k) nkhd
        # out shape(N:n, query_len: q, heads: h, heads_dim: d) => nqhd

        out = out.reshape(N, batch, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out
