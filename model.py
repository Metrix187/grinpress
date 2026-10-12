"""Pixel-space U-Net velocity model with self+cross attention to char tokens."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from conditioning import TextConditioner


def timestep_embedding(t, dim):
    # t in [0,1] scaled to [0,1000]; emb = [cos(t*w_i), sin(t*w_i)], w_i = 10000^(-i/half)
    half = dim // 2
    w = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = (t.float() * 1000.0)[:, None] * w[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


def gn(c): return nn.GroupNorm(32, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout * 2)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.t(F.silu(temb))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b                                 # FiLM / AdaGN on timestep
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Self-attention over pixels, then cross-attention pixels->char tokens, then MLP."""
    def __init__(self, c, ctx_dim, heads=8):
        super().__init__()
        self.h = heads
        self.n1, self.qkv, self.o1 = gn(c), nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.n2, self.q, self.kv, self.o2 = nn.LayerNorm(c), nn.Linear(c, c), nn.Linear(ctx_dim, 2 * c), nn.Linear(c, c)
        self.n3, self.mlp = nn.LayerNorm(c), nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        for o in (self.o1, self.o2, self.mlp[2]): nn.init.zeros_(o.weight); nn.init.zeros_(o.bias)

    def split(self, x):  # [B,N,C] -> [B,h,N,C/h]
        B, N, C = x.shape; return x.view(B, N, self.h, C // self.h).transpose(1, 2)

    def forward(self, x, ctx, mask):
        B, C, H, W = x.shape
        h = self.n1(x).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(h).chunk(3, -1)
        a = F.scaled_dot_product_attention(self.split(q), self.split(k), self.split(v))  # softmax(QK^T/sqrt(d))V
        t = x.flatten(2).transpose(1, 2) + self.o1(a.transpose(1, 2).reshape(B, -1, C))
        k, v = self.kv(ctx).chunk(2, -1)
        am = mask[:, None, None, :]                                  # key padding mask: True = attend
        a = F.scaled_dot_product_attention(self.split(self.q(self.n2(t))), self.split(k), self.split(v), attn_mask=am)
        t = t + self.o2(a.transpose(1, 2).reshape(B, -1, C))
        t = t + self.mlp(self.n3(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class UNet(nn.Module):
    """Levels at 128px input: 128,64,32,16 (attn at the 32 and 16 levels, i.e. levels 2,3;
    at 256px input the same levels sit at 64/32)."""
    def __init__(self, ch=64, mults=(1, 2, 3, 4), attn_levels=(2, 3), nres=2, ctx_dim=256, tdim=512):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.ch = ch
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.down, chans, c = nn.ModuleList(), [ch], ch
        for i, m in enumerate(mults):
            for _ in range(nres):
                blk = nn.ModuleList([ResBlock(c, ch * m, tdim)]); c = ch * m
                if i in attn_levels: blk.append(AttnBlock(c, ctx_dim))
                self.down.append(blk); chans.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)])); chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), AttnBlock(c, ctx_dim), ResBlock(c, c, tdim)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(nres + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, tdim)]); c = ch * m
                if i in attn_levels: blk.append(AttnBlock(c, ctx_dim))
                self.up.append(blk)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))
        nn.init.zeros_(self.out[2].weight); nn.init.zeros_(self.out[2].bias)

    def _run(self, blk, h, temb, ctx, mask):
        for m in blk:
            if isinstance(m, ResBlock): h = m(h, temb)
            elif isinstance(m, AttnBlock): h = m(h, ctx, mask)
            else: h = m(h)
        return h

    def forward(self, x, t, ctx, mask):
        temb = self.tmlp(timestep_embedding(t, self.ch))
        h = self.inp(x); hs = [h]
        for blk in self.down:
            h = self._run(blk, h, temb, ctx, mask); hs.append(h)
        h = self._run(self.mid, h, temb, ctx, mask)
        for blk in self.up:
            if isinstance(blk[0], ResBlock): h = torch.cat([h, hs.pop()], 1)
            h = self._run(blk, h, temb, ctx, mask)
        return self.out(h)


class FlowModel(nn.Module):
    """Text conditioner + U-Net, trained jointly. forward returns predicted velocity."""
    def __init__(self, ctx_dim=256, **kw):
        super().__init__()
        self.cond = TextConditioner(dim=ctx_dim)
        self.unet = UNet(ctx_dim=ctx_dim, **kw)

    def forward(self, x, t, ids, mask, force_uncond=False):
        ctx, m = self.cond(ids, mask, force_uncond=force_uncond)
        return self.unet(x, t, ctx, m)


PARAM_RANGE = (25e6, 40e6)


def build_model(verbose=True):
    m = FlowModel()
    n = sum(p.numel() for p in m.parameters())
    if verbose:
        nu = sum(p.numel() for p in m.unet.parameters())
        print(f"params: total {n/1e6:.2f}M (unet {nu/1e6:.2f}M, text {(n-nu)/1e6:.2f}M)")
    assert PARAM_RANGE[0] <= n <= PARAM_RANGE[1], f"param count {n} out of range"
    return m


if __name__ == "__main__":
    from conditioning import tokenize
    m = build_model()
    ids, mask = tokenize(["a red smiley face with 'HI' next to it"] * 2)
    print(m(torch.randn(2, 3, 128, 128), torch.rand(2), ids, mask).shape)
