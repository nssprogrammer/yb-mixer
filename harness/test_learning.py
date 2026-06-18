import torch, torch.nn as nn, time, sys
from models import build_model
torch.manual_seed(0)
L=32
meta={"seq_len":L,"in_type":"continuous","vocab":0,"n_classes":2}
def make(n):
    x=torch.rand(n,L); y=(x[:,0]>0.5).long()   # only first pos matters; mean-pool => needs mixing to route it
    return x,y
Xtr,ytr=make(2000); Xva,yva=make(1000)
def run(m,epochs=30):
    torch.manual_seed(0); net=build_model(m,meta,dim=48,depth=3,dropout=0.0)
    npar=sum(p.numel() for p in net.parameters())
    opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01); lf=nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm=torch.randperm(2000)
        for i in range(0,2000,128):
            idx=perm[i:i+128]; opt.zero_grad(); lf(net(Xtr[idx]),ytr[idx]).backward(); opt.step()
    with torch.no_grad(): acc=(net(Xva).argmax(1)==yva).float().mean().item()
    return acc,npar
m=sys.argv[1]; t0=time.time(); acc,npar=run(m)
print(f"{m:<12} val_acc={acc:.3f} params={npar} ({time.time()-t0:.0f}s) {'OK' if acc>0.8 else 'CHECK'}")
