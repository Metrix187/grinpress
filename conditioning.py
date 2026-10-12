"""Character-level caption conditioning for cross-attention.

Choice: we encode the WHOLE caption ("a navy sad face with 'DAILY' next to it"), max_len=64.
Longest possible caption is ~50 chars ("a purple smiley face with 'ABCDEFGH' next to it" = 48),
so 64 covers it with headroom and lets the model also handle free-form-ish prompts; the
template words are cheap tokens and the encoder learns to attend to the informative spans.
(Encoding only parsed parts would fit in 32 but bakes the parser into training.)
"""
import math
import torch, torch.nn as nn

MAX_LEN = 64
VOCAB = 128  # pure ASCII bytes; id 0 (NUL) doubles as PAD since captions never contain it


def tokenize(texts, max_len=MAX_LEN):
    """List[str] -> (ids LongTensor [B,L], mask BoolTensor [B,L], True = real token)."""
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        b = [c if 0 < c < 128 else ord("?") for c in t.encode("ascii", "replace")][:max_len]
        ids[i, :len(b)] = torch.tensor(b, dtype=torch.long)
    return ids, ids != 0


def sinusoidal_1d(L, d):
    # PE[p,2i] = sin(p / 10000^(2i/d)), PE[p,2i+1] = cos(...)
    pos = torch.arange(L).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TextConditioner(nn.Module):
    """Embeds chars -> per-token sequence [B,L,D] + mask [B,L]. No pooling.

    CFG dropout: with prob p_uncond a sample's whole sequence is replaced by a learned null
    sequence (all positions marked valid so attention is never fully masked).
    """
    def __init__(self, dim=256, layers=2, heads=4, max_len=MAX_LEN, p_uncond=0.1):
        super().__init__()
        self.max_len, self.p_uncond = max_len, p_uncond
        self.emb = nn.Embedding(VOCAB, dim)
        self.register_buffer("pe", sinusoidal_1d(max_len, dim), persistent=False)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.null_seq = nn.Parameter(torch.randn(max_len, dim) * 0.02)

    def forward(self, ids, mask, force_uncond=False):
        x = self.emb(ids) + self.pe[None, :ids.shape[1]]
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        B = ids.shape[0]
        if force_uncond:
            drop = torch.ones(B, dtype=torch.bool, device=ids.device)
        elif self.training and self.p_uncond > 0:
            drop = torch.rand(B, device=ids.device) < self.p_uncond
        else:
            return x, mask
        null = self.null_seq[None, :ids.shape[1]].expand(B, -1, -1).to(x.dtype)
        x = torch.where(drop[:, None, None], null, x)
        mask = torch.where(drop[:, None], torch.ones_like(mask), mask)
        return x, mask
