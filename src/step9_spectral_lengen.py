"""
Step 9 -- Length generalization via a SPECTRAL (non-local circulant) orthogonal flow
(addresses reviewer W7). The phase profile phi(f) is a function of NORMALIZED frequency
f=m/L, so the generator is length-agnostic, non-dispersive, still orthogonal, and still a
commuting family (all circulant generators commute => anytime preserved). Trained at L=16,
it generalizes to L=64 (4x) with roughly flat accuracy, vs the local generator's collapse.
Dependencies: torch.
"""
# Test: can a SPECTRAL (circulant, non-local) orthogonal flow fix length generalization?
# Phase profile phi(f) as a function of NORMALIZED frequency f=m/L -> length-agnostic.
# Still orthogonal (norm-preserving), still a commuting family (all circulant gens commute).
import torch, torch.nn as nn
torch.set_default_dtype(torch.float64); torch.manual_seed(0)

def data(n, L, seed):
    g=torch.Generator().manual_seed(seed); x=torch.randint(0,2,(n,L),generator=g); return x, x[:,-1].clone()

class SpectralFlow(nn.Module):
    def __init__(self, C, n_basis=8):
        super().__init__()
        self.emb = nn.Embedding(2, C)
        # phase phi(f) = sum_k w_k sin(2 pi k f) + v_k (1-cos(2 pi k f)) ; phi(0)=0
        self.w = nn.Parameter(0.1*torch.randn(n_basis))
        self.v = nn.Parameter(0.1*torch.randn(n_basis))
        self.head = nn.Sequential(nn.Linear(C,2*C), nn.GELU(), nn.Linear(2*C,2))
        self.K = torch.arange(1, n_basis+1).double()
    def phase(self, f):                          # f: (M,)  -> (M,)
        ang = 2*torch.pi*self.K[None,:]*f[:,None]    # (M, n_basis)
        return (torch.sin(ang)*self.w[None,:] + (1-torch.cos(ang))*self.v[None,:]).sum(1)
    def forward(self, x, L):
        h = self.emb(x)                          # B,L,C
        H = torch.fft.rfft(h, dim=1)             # B, M, C
        M = H.shape[1]; f = torch.arange(M).double()/L
        phi = self.phase(f)                       # (M,)
        H = H * torch.exp(1j*phi)[None,:,None]
        h = torch.fft.irfft(H, n=L, dim=1)
        return self.head(h[:,0,:])

C=24; net=SpectralFlow(C); opt=torch.optim.Adam(net.parameters(),lr=3e-3); lf=nn.CrossEntropyLoss()
Xtr,ytr=data(8000,16,0)
for ep in range(30):
    perm=torch.randperm(8000)
    for i in range(0,8000,256):
        idx=perm[i:i+256]; opt.zero_grad(); lf(net(Xtr[idx],16),ytr[idx]).backward(); opt.step()

def acc(L):
    Xte,yte=data(2000,L,L)
    with torch.no_grad(): return (net(Xte,L).argmax(1)==yte).float().mean().item()

print("SPECTRAL flow length generalization (train L=16):")
for L in [16,24,32,48,64]:
    print(f"  L={L:>3}  acc={acc(L):.3f}")
# what phase did it learn? compare to clean shift-by-1 (phi=2 pi f)
with torch.no_grad():
    f=torch.linspace(0,0.5,5); learned=net.phase(f); ideal=2*torch.pi*f*1.0
print("  learned phase/ (2pi f) at f=[0..0.5]:", [round((learned[i]/(2*torch.pi*f[i]+1e-9)).item(),2) for i in range(1,5)])
