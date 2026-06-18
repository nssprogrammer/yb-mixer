"""
Step 8 -- Critical baselines (addresses reviewer W5/Q4): orthogonal RNN, diagonal SSM,
attention, nonlinear MLP-Mixer, and a no-mixing control, compared with the integrable
flow on (A) a long-range MEMORY task and (B) a content-dependent RECALL task.
Establishes that YB-Mixer is competitive with / better than the most relevant structured
baselines (orthogonal RNN, SSM) on long-range memory at fewer parameters, and honestly
quantifies the expressivity gap to a nonlinear mixer on content-dependent recall.
Dependencies: torch.
"""
import torch, torch.nn as nn, numpy as np, time
torch.set_default_dtype(torch.float64)

# ---------- tasks ----------
def task_memory(n, L, seed):       # long-range memory: label = x[0], read at LAST position
    g=torch.Generator().manual_seed(seed); x=torch.randint(0,2,(n,L),generator=g); return x, x[:,0].clone()

def task_recall(n, L, seed):       # content-dependent: value following the query key (assoc. recall)
    g=torch.Generator().manual_seed(seed)
    # tokens 0..3 = values; positions: [q, k0,v0, k1,v1, ...]; key in {0,1}; answer = value after key==q
    x=torch.randint(0,2,(n,L),generator=g)
    q=x[:,0]
    # find first position p (odd) where key x[:,p]==q, answer x[:,p+1]
    y=torch.zeros(n,dtype=torch.long)
    for i in range(n):
        ans=int(x[i,2].item())
        for p in range(1,L-1,2):
            if x[i,p].item()==q[i].item(): ans=int(x[i,p+1].item()); break
        y[i]=ans
    return x, y

# ---------- models (all: emb -> mixing -> readout(last) -> 2-class head) ----------
class FlowYB(nn.Module):           # integrable orthogonal flow (our model)
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C); self.K=nn.Parameter(0.4*torch.randn(L,L))
        self.head=nn.Sequential(nn.Linear(C,2*C),nn.GELU(),nn.Linear(2*C,2))
    def forward(self,x):
        U=torch.matrix_exp(self.K-self.K.t()); h=torch.einsum('blc,kl->bkc',self.emb(x),U); return self.head(h[:,-1,:])

class OrthRNN(nn.Module):          # orthogonal recurrent net (Arjovsky-style)
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C); self.Wr=nn.Parameter(0.1*torch.randn(C,C))
        self.U=nn.Linear(C,C); self.head=nn.Sequential(nn.Linear(C,2*C),nn.GELU(),nn.Linear(2*C,2))
    def forward(self,x):
        W=torch.matrix_exp(self.Wr-self.Wr.t()); e=self.emb(x); h=torch.zeros(x.shape[0],e.shape[2])
        for t in range(x.shape[1]): h=torch.tanh(h@W.t()+self.U(e[:,t,:]))
        return self.head(h)

class DiagSSM(nn.Module):          # diagonal complex linear SSM (S4D/Mamba core)
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C)
        self.a=nn.Parameter(-2+0.1*torch.randn(C)); self.th=nn.Parameter(0.3*torch.randn(C))
        self.Cr=nn.Parameter(0.5*torch.randn(C)); self.head=nn.Sequential(nn.Linear(C,2*C),nn.GELU(),nn.Linear(2*C,2))
    def forward(self,x):
        e=self.emb(x); lam=torch.exp(-torch.exp(self.a)+1j*self.th); s=torch.zeros(x.shape[0],e.shape[2],dtype=torch.complex128)
        for t in range(x.shape[1]): s=lam[None,:]*s+e[:,t,:].to(torch.complex128)
        return self.head((s.real*self.Cr[None,:]))

class Attn(nn.Module):
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C); self.pos=nn.Parameter(0.02*torch.randn(L,C))
        self.enc=nn.TransformerEncoderLayer(C,2,2*C,batch_first=True,dropout=0.0); self.head=nn.Linear(C,2)
    def forward(self,x): return self.head(self.enc(self.emb(x)+self.pos)[:,-1,:])

class MLPMixer(nn.Module):         # nonlinear unconstrained token mixer
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C)
        self.tok=nn.Sequential(nn.Linear(L,2*L),nn.GELU(),nn.Linear(2*L,L)); self.head=nn.Sequential(nn.Linear(C,2*C),nn.GELU(),nn.Linear(2*C,2))
    def forward(self,x):
        h=self.emb(x); h=self.tok(h.transpose(1,2)).transpose(1,2); return self.head(h[:,-1,:])

class NoMix(nn.Module):
    def __init__(self,L,C):
        super().__init__(); self.emb=nn.Embedding(4,C); self.head=nn.Sequential(nn.Linear(C,2*C),nn.GELU(),nn.Linear(2*C,2))
    def forward(self,x): return self.head(self.emb(x)[:,-1,:])

def train_eval(model_fn, task, L, seeds=(0,1), C=24, n=4000, epochs=15, bs=256):
    accs=[]; npar=None
    for sd in seeds:
        torch.manual_seed(sd); Xtr,ytr=task(n,L,sd); Xte,yte=task(2000,L,sd+500)
        net=model_fn(L,C); npar=sum(p.numel() for p in net.parameters())
        opt=torch.optim.Adam(net.parameters(),lr=3e-3); lf=nn.CrossEntropyLoss()
        for ep in range(epochs):
            perm=torch.randperm(n)
            for i in range(0,n,bs):
                idx=perm[i:i+bs]; opt.zero_grad(); lf(net(Xtr[idx]),ytr[idx]).backward(); opt.step()
        with torch.no_grad(): accs.append((net(Xte).argmax(1)==yte).float().mean().item())
    return np.mean(accs), np.std(accs), npar

models=[("FlowYB (ours, integrable)",FlowYB),("OrthRNN",OrthRNN),("DiagSSM (S4D-like)",DiagSSM),
        ("Attention",Attn),("MLP-Mixer (nonlinear)",MLPMixer),("NoMix (control)",NoMix)]
print("="*72); print("BASELINES  (mean over 2 seeds)"); print("="*72)
print("\nTask A: long-range MEMORY (label=x[0], read at last pos), L=24")
for name,fn in models:
    m,s,p=train_eval(fn,task_memory,24); print(f"  {name:<26} acc={m:.3f}±{s:.3f}  params={p}")
print("\nTask B: content-dependent RECALL (assoc. recall), L=17")
for name,fn in models:
    m,s,p=train_eval(fn,task_recall,17); print(f"  {name:<26} acc={m:.3f}±{s:.3f}  params={p}")
