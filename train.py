"""Rectified-flow training.

Convention: t=0 is data, t=1 is noise.
  x_t = (1 - t) * x0 + t * eps,   target velocity v = dx_t/dt = eps - x0,   loss = ||v_theta(x_t,t,c) - v||^2
t ~ logit-normal(0,1) (sigmoid of a standard normal) by default, concentrating on mid t.
"""
import argparse, copy, json, math, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from conditioning import tokenize
from model import build_model


class FaceDataset(Dataset):
    def __init__(self, data_dir, resolution=None):
        self.dir = data_dir
        with open(os.path.join(data_dir, "metadata.jsonl")) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.res = resolution

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.dir, r["image_path"])).convert("RGB")
        if self.res and img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1  # [-1,1]
        ids, mask = tokenize([r["caption"]])
        return x, ids[0], mask[0]


def sample_t(n, mode, device):
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device))
    return torch.rand(n, device=device)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p.detach(), 1 - decay)  # ema = decay*ema + (1-decay)*p


def lr_at(step, total, warmup, base):
    if step < warmup: return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine to 5% of base


def save(path, model, ema, opt, scaler, step, epoch, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None, "step": step, "epoch": epoch,
                "args": vars(args)}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def save_sample_grid(ema, prompts, res, steps, path, device):
    """One row per prompt: 4 EMA samples (CFG 5) with fixed seeds, for tracking progress."""
    from sample import sample, parse_prompt
    rows = []
    for j, p in enumerate(prompts):
        x = sample(ema, parse_prompt(p)[3], res, steps, 5.0, 1000 + j, device, n=4)
        rows.append(torch.cat(list(x), dim=2))                 # [3,res,4*res]
    g = ((torch.cat(rows, dim=1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(g).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N total steps (0 = no limit)")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--t_sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    ap.add_argument("--resolution", type=int, default=0, help="resize images (0 = as stored)")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--resume", default="auto", help="'auto' (latest.pt if present), a path, or 'none'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_every", type=int, default=0, help="save EMA sample grid every N steps (0 = off)")
    ap.add_argument("--sample_dir", default="", help="where sample grids go (default: <checkpoint_dir>/samples)")
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--sample_prompts", nargs="+", default=[
        "a red smiley face with 'HELLO' next to it", "a blue sad face with 'NEWS' next to it",
        "a green meh face with 'OK' next to it", "a yellow smiley face with 'XQ7' next to it"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    prec = args.precision
    if prec == "auto":
        prec = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if dev.type == "cuda" else "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[prec]
    scaler = torch.amp.GradScaler("cuda") if (prec == "fp16" and dev.type == "cuda") else None
    if dev.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    ds = FaceDataset(args.data_dir, args.resolution or None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=len(ds) >= args.batch_size,
                    num_workers=args.num_workers, pin_memory=dev.type == "cuda",
                    persistent_workers=args.num_workers > 0)
    model = build_model().to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "null_seq" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "null_seq" not in n)]
    opt = torch.optim.AdamW([{"params": decay_params, "weight_decay": args.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.99))
    step, start_epoch = 0, 0
    rp = os.path.join(args.checkpoint_dir, "latest.pt") if args.resume == "auto" else args.resume
    if rp != "none" and os.path.exists(rp):
        ck = torch.load(rp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        if scaler and ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        step, start_epoch = ck["step"], ck["epoch"]
        print(f"resumed from {rp} at step {step}, epoch {start_epoch}")
    total = args.epochs * len(dl)
    if args.max_steps: total = min(total, args.max_steps)
    print(f"device={dev} precision={prec} samples={len(ds)} steps/epoch={len(dl)} total_steps={total}")

    t0, done, s0 = time.time(), False, step
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for x0, ids, mask in dl:
            x0, ids, mask = x0.to(dev, non_blocking=True), ids.to(dev), mask.to(dev)
            for g in opt.param_groups: g["lr"] = lr_at(step, total, args.warmup_steps, args.lr)
            eps = torch.randn_like(x0)
            t = sample_t(x0.shape[0], args.t_sampling, dev)
            tb = t[:, None, None, None]
            xt = (1 - tb) * x0 + tb * eps          # straight-line interpolation
            target = eps - x0                       # constant velocity along the line
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(xt, t, ids, mask)
            loss = F.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            # EMA warm-up: effective decay min(ema_decay, (1+s)/(10+s)) so early EMA isn't stuck at init
            ema_update(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"ep {epoch} step {step}/{total} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time()-t0)/max(1, step-s0):.2f}s/it", flush=True)
            if step % args.save_every == 0:
                save(os.path.join(args.checkpoint_dir, "latest.pt"), model, ema, opt, scaler, step, epoch, args)
                print(f"saved checkpoint at step {step}")
            if args.sample_every and step % args.sample_every == 0:
                sd = args.sample_dir or os.path.join(args.checkpoint_dir, "samples"); os.makedirs(sd, exist_ok=True)
                save_sample_grid(ema, args.sample_prompts, x0.shape[-1], args.sample_steps,
                                 os.path.join(sd, f"step_{step:07d}.png"), dev)
                print(f"saved sample grid at step {step}", flush=True)
            if step >= total: done = True; break
        save(os.path.join(args.checkpoint_dir, "latest.pt"), model, ema, opt, scaler, step,
             epoch if done else epoch + 1, args)
        if done: break
    torch.save({"ema": ema.state_dict(), "step": step}, os.path.join(args.checkpoint_dir, "ema_final.pt"))
    print(f"done at step {step}; checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
