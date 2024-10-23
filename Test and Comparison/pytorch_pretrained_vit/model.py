"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    def __init__(
        self, 
        dim,
        seq_len,
        num_heads,
        num_layers,
        dropout_rate = 0.1,
    ):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_len += 1
        self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=4*dim, dropout=dropout_rate)
        # Classifier head
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        #nn.init.constant_(self.fc.weight, 0)
        #nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d 
        x = self.positional_embedding(x)  # b,gh*gw+1,d （+1：Patch + Position Embedding
        x = self.transformer(x)  # b,gh*gw+1,d 有数层transformer block
        x = self.norm(x)[:, 0]  # b,d:每个图片仅使用class embedding作为特征
        return x

