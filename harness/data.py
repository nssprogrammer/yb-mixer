"""
Data loaders for the YB-Mixer harness. All tasks are long-range BIDIRECTIONAL classification.

  smoke   : random tensors (no download) -- for sanity tests on CPU
  pmnist  : permuted sequential MNIST  (L=784, continuous, 10 classes)
  smnist  : sequential MNIST           (L=784, continuous, 10 classes)
  scifar  : sequential CIFAR-10 (gray) (L=1024, continuous, 10 classes)  == LRA-Image
  imdb    : byte-level IMDB sentiment  (L=cfg, discrete vocab 256, 2 classes) ~ LRA-Text

Returns (train_loader, val_loader, meta) with
  meta = {seq_len, in_type in {continuous, discrete}, vocab, n_classes}.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader


def _loaders(Xtr, ytr, Xva, yva, bs, workers, pin):
    pw = workers > 0
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                    num_workers=workers, pin_memory=pin, drop_last=True, persistent_workers=pw)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=pin, persistent_workers=pw)
    return tr, va


def get_smoke(bs=8, workers=0, pin=False, in_type="continuous", L=64, n_classes=4, n=64):
    if in_type == "discrete":
        Xtr = torch.randint(0, 256, (n, L)); Xva = torch.randint(0, 256, (n, L))
        vocab = 256
    else:
        Xtr = torch.rand(n, L); Xva = torch.rand(n, L); vocab = 0
    ytr = torch.randint(0, n_classes, (n,)); yva = torch.randint(0, n_classes, (n,))
    tr, va = _loaders(Xtr, ytr, Xva, yva, bs, workers, pin)
    return tr, va, {"seq_len": L, "in_type": in_type, "vocab": vocab, "n_classes": n_classes}


def _mnist(root, permute, seed):
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    Xtr = tr.data.float().div(255.).view(-1, 784); ytr = tr.targets
    Xte = te.data.float().div(255.).view(-1, 784); yte = te.targets
    if permute:
        g = torch.Generator().manual_seed(seed); perm = torch.randperm(784, generator=g)
        Xtr = Xtr[:, perm]; Xte = Xte[:, perm]
    return Xtr, ytr, Xte, yte


def get_pmnist(root="./data", bs=64, workers=4, pin=True, seed=0):
    Xtr, ytr, Xte, yte = _mnist(root, permute=True, seed=seed)
    tr, va = _loaders(Xtr, ytr, Xte, yte, bs, workers, pin)
    return tr, va, {"seq_len": 784, "in_type": "continuous", "vocab": 0, "n_classes": 10}


def get_smnist(root="./data", bs=64, workers=4, pin=True, seed=0):
    Xtr, ytr, Xte, yte = _mnist(root, permute=False, seed=seed)
    tr, va = _loaders(Xtr, ytr, Xte, yte, bs, workers, pin)
    return tr, va, {"seq_len": 784, "in_type": "continuous", "vocab": 0, "n_classes": 10}


def get_scifar(root="./data", bs=64, workers=4, pin=True):
    from torchvision import datasets
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)
    def gray(d):
        x = torch.tensor(d.data).float().div(255.)          # (N,32,32,3)
        x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])  # luminance
        return x.view(-1, 1024)
    Xtr, ytr = gray(tr), torch.tensor(tr.targets)
    Xte, yte = gray(te), torch.tensor(te.targets)
    tr_, va_ = _loaders(Xtr, ytr, Xte, yte, bs, workers, pin)
    return tr_, va_, {"seq_len": 1024, "in_type": "continuous", "vocab": 0, "n_classes": 10}


def get_imdb(bs=32, workers=4, pin=True, L=1024, n_train=None):
    from datasets import load_dataset
    ds = load_dataset("imdb")
    def enc(split):
        texts = ds[split]["text"]; labels = ds[split]["label"]
        if split == "train" and n_train:
            texts, labels = texts[:n_train], labels[:n_train]
        X = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            b = t.encode("utf-8", "ignore")[:L]
            X[i, : len(b)] = torch.tensor(list(b), dtype=torch.long)
        return X, torch.tensor(labels)
    Xtr, ytr = enc("train"); Xte, yte = enc("test")
    tr, va = _loaders(Xtr, ytr, Xte, yte, bs, workers, pin)
    return tr, va, {"seq_len": L, "in_type": "discrete", "vocab": 256, "n_classes": 2}



# ----------------------------- ListOps (LRA) -----------------------------
# Tokens: 0-9 digits; 10='[MIN' 11='[MAX' 12='[MED' 13='[SM'(sum mod 10); 14=']'; 15=PAD
import numpy as _np
def _listops_expr(rng, depth, max_depth, max_args):
    leaf_p = 0.25 + 0.65 * depth / max_depth        # likelier to stop deeper -> bounded length
    if depth >= max_depth or rng.random() < leaf_p:
        v = int(rng.integers(0, 10)); return [v], v
    op = int(rng.integers(0, 4)); n = int(rng.integers(2, max_args + 1))
    toks = [10 + op]; vals = []
    for _ in range(n):
        t, v = _listops_expr(rng, depth + 1, max_depth, max_args); toks += t; vals.append(v)
    toks.append(14)
    if   op == 0: val = min(vals)
    elif op == 1: val = max(vals)
    elif op == 2: val = int(_np.median(_np.array(vals)))      # floor of median
    else:         val = sum(vals) % 10
    return toks, val

def _listops_set(n, max_len, max_depth, max_args, seed):
    rng = _np.random.default_rng(seed); X = []; y = []
    while len(X) < n:
        toks, val = _listops_expr(rng, 0, max_depth, max_args)
        if 4 <= len(toks) <= max_len:
            toks = toks + [15] * (max_len - len(toks)); X.append(toks); y.append(val)
    return torch.tensor(X), torch.tensor(y)

def get_listops(bs=32, workers=2, pin=True, max_len=1024, n_train=8000, n_val=2000,
                max_depth=5, max_args=4, seed=0):
    Xtr, ytr = _listops_set(n_train, max_len, max_depth, max_args, seed)
    Xva, yva = _listops_set(n_val, max_len, max_depth, max_args, seed + 1)
    tr, va = _loaders(Xtr, ytr, Xva, yva, bs, workers, pin)
    return tr, va, {"seq_len": max_len, "in_type": "discrete", "vocab": 16, "n_classes": 10}

# ------------------- Pathfinder-style synthetic connectivity -------------------
# NOTE: a self-contained REIMPLEMENTATION of Pathfinder's connectivity challenge
# (binary: are the two endpoints joined by a path among distractors), NOT the official
# pixel dataset; numbers are not directly comparable to published LRA-Pathfinder.
def _rand_walk(rng, G, start, steps):
    path=[start]; y,x=start
    for _ in range(steps):
        y=int(_np.clip(y+rng.integers(-1,2),0,G-1)); x=int(_np.clip(x+rng.integers(-1,2),0,G-1)); path.append((y,x))
    return path

def _guided_walk(rng, G, a, b, maxsteps):
    """Walk from a toward b with noise; returns path and whether it reached b."""
    path=[a]; y,x=a
    for _ in range(maxsteps):
        sy=_np.sign(b[0]-y); sx=_np.sign(b[1]-x)
        y=int(_np.clip(y+(sy if rng.random()<0.8 else rng.integers(-1,2)),0,G-1))
        x=int(_np.clip(x+(sx if rng.random()<0.8 else rng.integers(-1,2)),0,G-1))
        path.append((y,x))
        if (y,x)==b: return path,True
    return path,False

def _mark(grid,cells,val,G,thick=True):
    for (y,x) in cells:
        grid[y,x]=val
        if thick:
            for dy,dx in ((0,1),(1,0)):
                yy,xx=min(y+dy,G-1),min(x+dx,G-1); 
                if grid[yy,xx]==0: grid[yy,xx]=val

def _pathfinder_one(rng, G):
    grid=_np.zeros((G,G),dtype=_np.float32)
    connected=bool(rng.integers(0,2))
    p1=(int(rng.integers(0,G)),int(rng.integers(0,G)))
    p2=(int(rng.integers(0,G)),int(rng.integers(0,G)))
    while abs(p1[0]-p2[0])+abs(p1[1]-p2[1])<G//2:
        p2=(int(rng.integers(0,G)),int(rng.integers(0,G)))
    if connected:
        path,_=_guided_walk(rng,G,p1,p2,G*4); _mark(grid,path,0.5,G)
    else:
        mid=(int(rng.integers(0,G)),int(rng.integers(0,G)))
        path,_=_guided_walk(rng,G,p1,mid,G*2); _mark(grid,path,0.5,G)
        _mark(grid,_rand_walk(rng,G,p2,3),0.5,G)        # short isolated stub at p2
    for _ in range(rng.integers(1,3)):                  # few distractors
        s=(int(rng.integers(0,G)),int(rng.integers(0,G))); _mark(grid,_rand_walk(rng,G,s,G//3),0.5,G)
    grid[p1]=1.0; grid[p2]=1.0
    return grid.reshape(-1), int(connected)

def _pathfinder_set(n, G, seed):
    rng = _np.random.default_rng(seed); X = []; y = []
    for _ in range(n):
        g, lab = _pathfinder_one(rng, G); X.append(g); y.append(lab)
    return torch.tensor(_np.stack(X)), torch.tensor(y)

def get_pathfinder(bs=32, workers=2, pin=True, G=32, n_train=8000, n_val=2000, seed=0):
    Xtr, ytr = _pathfinder_set(n_train, G, seed); Xva, yva = _pathfinder_set(n_val, G, seed + 1)
    tr, va = _loaders(Xtr, ytr, Xva, yva, bs, workers, pin)
    return tr, va, {"seq_len": G * G, "in_type": "continuous", "vocab": 0, "n_classes": 2}

# ----------------------------- Induction Heads -----------------------------
# Predict the token immediately following a unique SPECIAL marker (token V-1).
# Data tokens are 0..V-2 (=> n_classes = V-1). Supports eval at longer L (extrapolation).
def _induction_set(n, L, V, seed):
    rng = _np.random.default_rng(seed)
    X = rng.integers(0, V - 1, size=(n, L))                  # data tokens 0..V-2
    pos = rng.integers(0, L - 1, size=n)                     # marker position (p+1 valid)
    y = X[_np.arange(n), pos + 1].copy()                     # target = token after marker
    X[_np.arange(n), pos] = V - 1                            # place unique SPECIAL marker
    return torch.tensor(X), torch.tensor(y)

def get_induction(bs=32, workers=2, pin=True, L=256, V=16, n_train=8000, n_val=2000, seed=0):
    Xtr, ytr = _induction_set(n_train, L, V, seed); Xva, yva = _induction_set(n_val, L, V, seed + 1)
    tr, va = _loaders(Xtr, ytr, Xva, yva, bs, workers, pin)
    # pos_emb must be OFF at build time for length extrapolation (see notebook).
    return tr, va, {"seq_len": L, "in_type": "discrete", "vocab": V, "n_classes": V - 1}

def induction_eval_set(L, V=16, n=2000, seed=123):
    """Return (X, y) tensors for evaluating induction-heads extrapolation at length L."""
    return _induction_set(n, L, V, seed)

# ----------------------------- Selective Copying -----------------------------
# Scatter n_data data tokens (1..V-2) among noise (0) in the first L-n_data slots; the last
# n_data slots are QUERY markers (V-1) where the model must emit the data tokens IN ORDER.
# Uses per-position outputs (SequenceClassifier.forward_tokens) + masked CE at the query slots.
def selective_copy_batch(B, L, V, n_data, seed=None):
    rng = _np.random.default_rng(seed)
    content = L - n_data
    X = _np.zeros((B, L), dtype=_np.int64)                   # noise = 0
    Y = _np.zeros((B, n_data), dtype=_np.int64)
    for i in range(B):
        vals = rng.integers(1, V - 1, size=n_data)           # data tokens 1..V-2
        posn = _np.sort(rng.choice(content, n_data, replace=False))
        X[i, posn] = vals; Y[i] = vals
    X[:, content:] = V - 1                                    # QUERY markers
    out_pos = torch.arange(content, L)
    return torch.tensor(X), torch.tensor(Y), out_pos

def get_data(task, bs, workers, pin, seq_len=1024):
    if task == "pmnist": return get_pmnist(bs=bs, workers=workers, pin=pin)
    if task == "smnist": return get_smnist(bs=bs, workers=workers, pin=pin)
    if task == "scifar": return get_scifar(bs=bs, workers=workers, pin=pin)
    if task == "imdb":   return get_imdb(bs=bs, workers=workers, pin=pin, L=seq_len)
    if task == "listops":    return get_listops(bs=bs, workers=workers, pin=pin, max_len=seq_len)
    if task == "pathfinder": return get_pathfinder(bs=bs, workers=workers, pin=pin)
    if task == "induction":  return get_induction(bs=bs, workers=workers, pin=pin, L=min(seq_len,256))
    raise ValueError(task)
