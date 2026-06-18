"""
YB-Mixer Step 6: end-to-end ANYTIME inference in an integrable-flow model.

Model (FlowYB): emb -> integrable orthogonal flow U(s)=exp(s*K) over positions
(K = learned antisymmetric generator; the whole sequence-mixing is ONE one-parameter
group) -> nonlinear head on position-0 readout. Trained at s=1 on the transport task.

Because the mixing is a one-parameter group exp(sK):
  (i)  rapidity is additive & splittable: U(s)=U(s1)...U(sk) for ANY split / order
       -> order-independent, cacheable, parallelizable inference (the integrable payoff)
  (ii) a variable "rapidity budget" gives a consistent coarse->fine answer (anytime).
Contrast: increments built from DIFFERENT generators (non-integrable) are order-DEPENDENT.
Dependencies: torch.
"""
import torch, torch.nn as nn, time
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def make_data(n, L):
    x = torch.randint(0, 2, (n, L)); return x, x[:, -1].clone()

class FlowYB(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.emb  = nn.Embedding(2, C)
        self.K    = nn.Parameter(0.1*torch.randn(L, L))
        self.head = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, 2))
    def Kanti(self):  return self.K - self.K.t()                 # antisymmetric generator
    def flow(self, s): return torch.matrix_exp(s*self.Kanti())   # L x L orthogonal
    def mix(self, h, s):  return torch.einsum('blc,kl->bkc', h, self.flow(s))
    def forward(self, x, s=1.0):
        return self.head(self.mix(self.emb(x), s)[:, 0, :])

def train(L=16, C=24, n=6000, epochs=25, bs=256):
    Xtr, ytr = make_data(n, L); Xte, yte = make_data(3000, L)
    net = FlowYB(L, C); opt = torch.optim.Adam(net.parameters(), lr=3e-3); lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad(); lf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
    with torch.no_grad(): acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return net, acc, (Xte, yte)

print("="*66); print("STEP 6: anytime inference in an integrable-flow model"); print("="*66)
t0 = time.time()
net, acc, (Xte, yte) = train()
print(f"(1) trained FlowYB  test acc @ s=1 = {acc:.3f}   (MLP-free mixing learns transport)")

# (2) anytime "rapidity budget" curve
print("\n(2) variable rapidity budget (anytime: stop at any s, consistent coarse->fine)")
print(f"      {'budget s':>8} | {'test acc':>8}")
with torch.no_grad():
    for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        a = (net(Xte, s=s).argmax(1) == yte).float().mean().item()
        print(f"      {s:>8.2f} | {a:>8.3f}")

# (3) order-independence: split s=1 into random increments, apply in many random orders
print("\n(3) split rapidity into 5 random increments; apply in 6 random orders")
torch.manual_seed(1)
inc = torch.rand(5); inc = (inc / inc.sum())            # increments summing to 1
Kanti = net.Kanti().detach()
gen_same = [Kanti]*5                                    # integrable: ONE generator
gen_diff = [(lambda M: M - M.t())(torch.randn(16, 16)) for _ in range(5)]   # non-integrable: different generators

def mixed_under_order(h0, increments, gens, order):
    h = h0.clone()
    for j in order:
        h = torch.einsum('blc,kl->bkc', h, torch.matrix_exp(increments[j]*gens[j]))
    return h

with torch.no_grad():
    h0 = net.emb(Xte[:64])
    orders = [torch.randperm(5).tolist() for _ in range(6)]
    def spread(gens):
        outs = [mixed_under_order(h0, inc, gens, o) for o in orders]
        ref = outs[0]
        return max((torch.linalg.norm(o-ref)/torch.linalg.norm(ref)).item() for o in outs[1:])
    s_int  = spread(gen_same)
    s_gen  = spread(gen_diff)
print(f"      integrable (one generator)      : max relative spread over orders = {s_int:.2e}")
print(f"      non-integrable (diff generators): max relative spread over orders = {s_gen:.2e}")

print("="*66)
ok = acc > 0.95 and s_int < 1e-9 and s_gen > 1e-2
print(f"verdict: learns={acc:.2f}  order-indep(integrable)={s_int:.1e}  order-dep(generic)={s_gen:.1e}  ({time.time()-t0:.0f}s)")
print("RESULT:", "PASS ✅  integrable flow => trainable + exactly order-independent (anytime)"
      if ok else "FAIL ❌  inspect numbers")
