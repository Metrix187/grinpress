"""Euler ODE sampler for the rectified-flow model with classifier-free guidance.

Integrates dx/dt = v(x,t) from t=1 (noise) to t=0 (data):  x_{t-dt} = x_t - dt * v,
with CFG  v = v_u + s * (v_c - v_u).
"""
import argparse, os, re, sys
import numpy as np, torch
from PIL import Image
from conditioning import tokenize, MAX_LEN
from model import build_model
from generate_data import PALETTE, EXPRESSIONS

PROMPT_RE = re.compile(r"^\s*an?\s+(?P<color>[a-z]+)\s+(?P<expr>[a-z]+)\s+face\s+with\s+"
                       r"['\"](?P<text>[^'\"]*)['\"]\s+next\s+to\s+it\s*\.?\s*$", re.I)


def parse_prompt(p):
    m = PROMPT_RE.match(p)
    if not m:
        raise ValueError(f"Prompt must look like: a red smiley face with 'HELLO' next to it  (got: {p!r})")
    color, expr, text = m["color"].lower(), m["expr"].lower(), m["text"]
    errs = []
    if color not in PALETTE: errs.append(f"unknown color {color!r}; choose from {sorted(PALETTE)}")
    if expr not in EXPRESSIONS: errs.append(f"unknown expression {expr!r}; choose from {EXPRESSIONS}")
    if not (1 <= len(text) <= 8): errs.append(f"text must be 1-8 characters (got {len(text)})")
    if not text.isascii(): errs.append("text must be ASCII")
    if errs: raise ValueError("; ".join(errs))
    # canonical caption, exactly the training template
    return color, expr, text, f"a {color} {expr} face with '{text}' next to it"


@torch.no_grad()
def sample(model, caption, res, steps, cfg, seed, device, n=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, res, res, generator=g).to(device)
    ids, mask = tokenize([caption] * n, MAX_LEN); ids, mask = ids.to(device), mask.to(device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        tb = t.expand(n)
        # batch cond + uncond in one pass
        xin, tin = torch.cat([x, x]), torch.cat([tb, tb])
        ctx_c, m_c = model.cond(ids, mask)
        ctx_u, m_u = model.cond(ids, mask, force_uncond=True)
        v = model.unet(xin, tin, torch.cat([ctx_c, ctx_u]), torch.cat([m_c, m_u]))
        v_c, v_u = v.chunk(2)
        v = v_u + cfg * (v_c - v_u)
        x = x - dt * v                       # Euler step toward t=0 (data)
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--out", default="sample.png")
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--num_images", type=int, default=1, help=">1 saves a horizontal strip")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    try:
        color, expr, text, caption = parse_prompt(a.prompt)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"parsed: color={color} expression={expr} text={text!r}")
    model = build_model().to(a.device).eval()
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    sd = ck["model"] if (a.no_ema and "model" in ck) else ck.get("ema", ck.get("model"))
    model.load_state_dict(sd)
    x = sample(model, caption, a.resolution, a.steps, a.cfg_scale, a.seed, a.device, a.num_images)
    imgs = ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray(np.concatenate(list(imgs), axis=1)).save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
