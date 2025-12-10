import torch
from torch import nn
from config import *

class MetaViewXGen(nn.Module):
    def __init__(self, meta_dim=3):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)

        self.meta_embed = nn.Embedding(16, HIDDEN_SIZE)

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_HEADS,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, x, meta):
        b,t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        meta_vec = self.meta_embed(meta).unsqueeze(1)
        h = self.token_embed(x) + self.pos_embed(pos) + meta_vec
        h = self.tr(h)
        return self.out(h)
