"""
YB-Mixer Step 7: rigorous empirical case (multi-seed) + attention baseline + honest length-gen.

A. Competitiveness: transport task @ L=16, 3 seeds, matched-ish params.
   models: FlowYB (integrable) | Attention | LinMix | NoMix(control).
B. The anytime advantage, with error bars: on each trained FlowYB show
   (i) order-independence of split inference (~machine eps) and (ii) the budget curve.
   Attention/standard nets have NO consistent variable-budget/reorderable mode.
C. Length generalization (HONEST): translation-invariant flow trained @ L=16, tested longer.
   Reports the limitation: dispersive transport does not cleanly length-generalize.
"""
import torch, torch.nn as nn, numpy as np, time
torch.set_default_dtype(torch.float64)

def data(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, L), generator=g); return x, x[:, -1].clone()

# ---------- models ----------
class FlowYB(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Embedding(2, C); self.K = nn.Parameter(0.1*torch.randn(L, L))
        self.head = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, 2))
    def Kanti(self): return self.K - self.K.t()
    def mix(self, h, s): return torch.einsum('blc,kl->bkc', h, torch.matrix_exp(s*self.Kanti()))
    def forward(self, x, s=1.0): return self.head(self.mix(self.emb(x), s)[:, 0, :])

class Attn(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Embedding(2, C); self.pos = nn.Parameter(0.02*torch.randn(L, C))
        self.enc = nn.TransformerEncoderLayer(C, nhead=2, dim_feedforward=2*C, batch_first=True, dropout=0.0)
        self.head = nn.Linear(C, 2)
    def forward(self, x): return self.head(self.enc(self.emb(x) + self.pos)[:, 0, :])

class LinMix(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Embedding(2, C); self.W = nn.Parameter(torch.eye(L) + 0.02*torch.randn(L, L))
        self.head = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, 2))
    def forward(self, x):
        h = torch.einsum('blc,kl->bkc', self.emb(x), self.W); return self.head(h[:, 0, :])

class NoMix(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb = nn.Embedding(2, C); self.head = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, 2))
    def forward(self, x): return self.head(self.emb(x)[:, 0, :])

def train(model_fn, seed, L=16, C=24, n=5000, epochs=20, bs=256):
    torch.manual_seed(seed)
    Xtr, ytr = data(n, L, seed); Xte, yte = data(3000, L, seed+999)
    net = model_fn(L, C); npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3); lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            out = net(Xtr[idx]) if not isinstance(net, FlowYB) else net(Xtr[idx], 1.0)
            lf(out, ytr[idx]).backward(); opt.step()
    with torch.no_grad():
        out = net(Xte) if not isinstance(net, FlowYB) else net(Xte, 1.0)
        acc = (out.argmax(1) == yte).float().mean().item()
    return net, acc, npar

print("="*70); print("STEP 7  (3 seeds)"); print("="*70)
seeds = [0, 1, 2]
print("A. transport @ L=16 -- test accuracy (mean ± std), matched-ish params")
flow_nets = []
for name, fn in [("FlowYB (integrable)", FlowYB), ("Attention", Attn), ("LinMix", LinMix), ("NoMix (control)", NoMix)]:
    accs, npar = [], None
    for s in seeds:
        net, acc, npar = train(fn, s); accs.append(acc)
        if isinstance(net, FlowYB): flow_nets.append(net)
    print(f"   {name:<22} acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}   params={npar}")

print("\nB. FlowYB anytime properties (mean ± std over the 3 trained models)")
spreads, curves = [], []
for net in flow_nets:
    Xte, yte = data(2000, 16, 12345)
    with torch.no_grad():
        # order-independence: split s=1 into 5 random increments, 6 random orders
        inc = torch.rand(5); inc = inc / inc.sum(); K = net.Kanti()
        h0 = net.emb(Xte[:64]); orders = [torch.randperm(5).tolist() for _ in range(6)]
        def run(order):
            h = h0.clone()
            for j in order: h = torch.einsum('blc,kl->bkc', h, torch.matrix_exp(inc[j]*K))
            return h
        outs = [run(o) for o in orders]; ref = outs[0]
        spreads.append(max((torch.linalg.norm(o-ref)/torch.linalg.norm(ref)).item() for o in outs[1:]))
        curves.append([ (net(Xte, s=s).argmax(1)==yte).float().mean().item() for s in [0.25,0.5,0.75,1.0] ])
curves = np.array(curves)
print(f"   order-independence spread (split & reorder) = {np.mean(spreads):.2e} ± {np.std(spreads):.0e}")
print(f"   anytime budget curve acc @ s=[.25 .5 .75 1.0] = "
      + " ".join(f"{m:.2f}" for m in curves.mean(0)))
print("   (attention/standard nets: fixed compute, no consistent reorderable variable-budget mode)")

print("\nC. length generalization (HONEST) -- translation-invariant flow, train L=16")
def toeplitz_anti(coeffs, L):
    K = torch.zeros(L, L)
    for r, c in enumerate(coeffs, start=1):
        idx = torch.arange(L-r); K[idx, idx+r] = c; K[idx+r, idx] = -c
    return K
class FlowTI(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.emb = nn.Embedding(2, C); self.coeffs = nn.Parameter(torch.tensor([1.0, 0.2]))
        self.logs = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.head = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, 2))
    def forward(self, x, L, scale=1.0):
        U = torch.matrix_exp(torch.exp(self.logs)*scale * toeplitz_anti(self.coeffs, L))
        return self.head(torch.einsum('blc,kl->bkc', self.emb(x), U)[:, 0, :])
torch.manual_seed(0); net = FlowTI(24); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lf = nn.CrossEntropyLoss()
Xtr, ytr = data(8000, 16, 0)
for ep in range(35):
    perm = torch.randperm(8000)
    for i in range(0, 8000, 256):
        idx = perm[i:i+256]; opt.zero_grad(); lf(net(Xtr[idx], 16), ytr[idx]).backward(); opt.step()
print(f"   {'L':>3} | {'budget-scaled acc':>18}")
for L in [16, 24, 32]:
    Xte, yte = data(2000, L, L); 
    with torch.no_grad(): a = (net(Xte, L, L/16).argmax(1) == yte).float().mean().item()
    print(f"   {L:>3} | {a:>18.3f}")
print("   -> dispersive free-fermion transport does NOT cleanly length-generalize (documented limitation)")
print("="*70)
