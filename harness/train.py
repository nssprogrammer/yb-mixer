"""
Unified training loop for the YB-Mixer harness (long-range bidirectional classification).

Example (GPU):
  python train.py --task scifar --model yb        --dim 128 --depth 6 --epochs 60 --amp
  python train.py --task scifar --model transformer --dim 128 --depth 6 --epochs 60 --amp
  python train.py --task scifar --model s4d       --dim 128 --depth 6 --epochs 60 --amp

Sanity (CPU, no download):
  python train.py --smoke
"""
import argparse, json, os, time, math, csv
import torch, torch.nn as nn
from models import build_model
import data as datamod


def cosine_warmup(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="scifar",
                    choices=["pmnist","smnist","scifar","imdb","listops","pathfinder","induction"])
    ap.add_argument("--model", default="yb",
                    choices=["yb","yb_relaxed","transformer","s4d","s4dlin","lru","fnet","mamba","rglru","scornn","rwkv"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_basis", type=int, default=32)
    ap.add_argument("--mlp_ratio", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=1024)   # imdb only
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"

    # ---- data ----
    if args.smoke:
        in_type = "discrete" if args.task == "imdb" else "continuous"
        train_loader, val_loader, meta = datamod.get_smoke(bs=8, in_type=in_type, L=64)
        args.epochs, args.dim, args.depth = 2, 32, 2
    else:
        pin = device == "cuda"
        train_loader, val_loader, meta = datamod.get_data(
            args.task, args.batch_size, args.workers, pin, args.seq_len)

    # ---- model ----
    model = build_model(args.model, meta, dim=args.dim, depth=args.depth,
                        n_heads=args.n_heads, n_basis=args.n_basis,
                        dropout=args.dropout, pos_emb=not args.no_pos).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model} on {args.task}] params={n_params:,}  device={device}  amp={amp}  "
          f"seq_len={meta['seq_len']}  classes={meta['n_classes']}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_frac * total_steps)

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.task}_{args.model}_d{args.dim}_L{args.depth}_s{args.seed}"
    csv_path = os.path.join(args.out, tag + ".csv")
    best = 0.0; step = 0; t0 = time.time()
    with open(csv_path, "w", newline="") as fcsv:
        wr = csv.writer(fcsv); wr.writerow(["epoch", "train_loss", "val_acc", "elapsed_s"])
        for ep in range(args.epochs):
            model.train(); run = 0.0; nb = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                for g in opt.param_groups:
                    g["lr"] = cosine_warmup(step, total_steps, warmup, args.lr)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device, enabled=amp):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
                run += loss.item(); nb += 1; step += 1
            acc = evaluate(model, val_loader, device)
            best = max(best, acc); el = time.time() - t0
            wr.writerow([ep, run / max(1, nb), acc, round(el, 1)]); fcsv.flush()
            print(f"  epoch {ep:3d}  train_loss={run/max(1,nb):.4f}  val_acc={acc:.4f}  "
                  f"best={best:.4f}  ({el:.0f}s)")

    summary = {"task": args.task, "model": args.model, "params": n_params,
               "best_val_acc": best, "epochs": args.epochs, "dim": args.dim,
               "depth": args.depth, "seed": args.seed, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE {tag}  best_val_acc={best:.4f}  ({summary['seconds']}s)")
    if args.smoke:
        print("SMOKE PASS" if best >= 0.0 else "SMOKE FAIL")


if __name__ == "__main__":
    main()
