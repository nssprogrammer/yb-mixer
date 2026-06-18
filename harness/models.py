"""
Models for the YB-Mixer GPU harness.

A single shared block scaffold (pre-norm: x = x + mixer(norm(x)); x = x + mlp(norm(x)))
is used for every architecture, so comparisons differ ONLY in the token mixer:

  - 'yb'        : Spectral YB-Mixer (orthogonal, phase-only)         <- our model
  - 'yb_relaxed': Spectral mixer with learnable magnitude (NOT orthogonal; a bidirectional
                  global-conv / linear-SSM ablation that quantifies the cost of orthogonality)
  - 'transformer': bidirectional multi-head self-attention
  - 's4d'       : diagonal complex SSM (S4D-style), bidirectional, via FFT convolution
  - 'mamba'     : real Mamba block if `mamba_ssm` is importable; else a gated diagonal SSM

All spectral / phase profiles are parameterized as functions of the NORMALIZED frequency
f = m / L, so the mixers are length-agnostic (key for length generalization).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- token mixers -----------------------------

class SpectralYBMixer(nn.Module):
    """Bidirectional orthogonal token mixer: per-channel unit-magnitude spectral multiplier.
    multiplier(m, c) = exp( i * phi_c(f_m) )  [+ log-magnitude if not orthogonal],
    with phi_c(f) = sum_k W[c,k] sin(2 pi k f) + V[c,k] (1 - cos(2 pi k f)),  f = m/L.
    """
    def __init__(self, dim, n_basis=24, orthogonal=True, init=0.05):
        super().__init__()
        self.dim, self.n_basis, self.orthogonal = dim, n_basis, orthogonal
        self.Wp = nn.Parameter(init * torch.randn(dim, n_basis))   # phase basis (sin)
        self.Vp = nn.Parameter(init * torch.randn(dim, n_basis))   # phase basis (1-cos)
        if not orthogonal:
            self.Wm = nn.Parameter(init * torch.randn(dim, n_basis))   # log-mag basis
            self.Vm = nn.Parameter(init * torch.randn(dim, n_basis))
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer("K", torch.arange(1, n_basis + 1).float())

    def _profiles(self, M, L, device, dtype):
        f = torch.arange(M, device=device, dtype=dtype) / L          # (M,)
        ang = 2 * math.pi * self.K.to(dtype)[None, :] * f[:, None]   # (M, n_basis)
        s, c = torch.sin(ang), 1.0 - torch.cos(ang)                  # (M, n_basis)
        phase = (s @ self.Wp.t().to(dtype) + c @ self.Vp.t().to(dtype)) * self.scale  # (M, dim)
        if self.orthogonal:
            logmag = torch.zeros_like(phase)
        else:
            logmag = -F.softplus(s @ self.Wm.t().to(dtype) + c @ self.Vm.t().to(dtype))  # <=0
        return phase, logmag

    def forward(self, x):                       # x: (B, L, C)
        B, L, C = x.shape
        # entire spectral op in float32 with autocast OFF (avoids Half/ComplexHalf under AMP)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            Xf = torch.fft.rfft(x32, dim=1)                              # (B, M, C) complex64
            M = Xf.shape[1]
            phase, logmag = self._profiles(M, L, x.device, torch.float32)   # (M, C) float32
            mult = torch.exp(torch.complex(logmag, phase))              # complex64
            Xf = Xf * mult.unsqueeze(0)
            y = torch.fft.irfft(Xf, n=L, dim=1)                         # (B, L, C) float32
        return y.to(x.dtype)


class AttentionMixer(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


class S4DMixer(nn.Module):
    """Bidirectional diagonal complex SSM via FFT convolution (S4D-style)."""
    def __init__(self, dim, bidirectional=True):
        super().__init__()
        self.dim, self.bi = dim, bidirectional
        self.log_a = nn.Parameter(torch.log(0.5 + torch.rand(dim)))   # decay rate (>0)
        self.theta = nn.Parameter(math.pi * torch.rand(dim))          # frequency
        self.Bc = nn.Parameter(0.5 * torch.randn(dim))
        self.Cc = nn.Parameter(0.5 * torch.randn(dim))
        self.D = nn.Parameter(torch.zeros(dim))
        if bidirectional:
            self.Bc2 = nn.Parameter(0.5 * torch.randn(dim))
            self.Cc2 = nn.Parameter(0.5 * torch.randn(dim))

    def _kernel(self, L, Bc, Cc, device):
        t = torch.arange(L, device=device, dtype=torch.float32)         # (L,)
        a = torch.exp(self.log_a).clamp(max=20.0)                       # (C,)
        lam_mag = torch.exp(-a)                                         # |lambda| < 1
        ang = self.theta[None, :] * t[:, None]                          # (L, C)
        # Re( B*C* lambda^t ) with lambda = lam_mag e^{i theta}
        k = (Bc * Cc)[None, :] * (lam_mag[None, :] ** t[:, None]) * torch.cos(ang)
        return k                                                        # (L, C)

    def _causal_conv(self, x, k):                # x:(B,L,C) k:(L,C)
        B, L, C = x.shape
        n = 2 * L
        Xf = torch.fft.rfft(x.float(), n=n, dim=1)
        Kf = torch.fft.rfft(k.float(), n=n, dim=0).unsqueeze(0)        # (1, n//2+1, C)
        y = torch.fft.irfft(Xf * Kf, n=n, dim=1)[:, :L, :]
        return y

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            L = xf.shape[1]
            k = self._kernel(L, self.Bc, self.Cc, xf.device)
            y = self._causal_conv(xf, k)
            if self.bi:
                k2 = self._kernel(L, self.Bc2, self.Cc2, xf.device)
                y = y + torch.flip(self._causal_conv(torch.flip(xf, [1]), k2), [1])
            y = y + self.D[None, None, :] * xf
        return y.to(x.dtype)


class MambaMixer(nn.Module):
    """Real Mamba block if available, else a gated diagonal SSM fallback (labeled)."""
    def __init__(self, dim, bidirectional=True):
        super().__init__()
        self.using_real = False
        try:
            from mamba_ssm import Mamba
            self.m = Mamba(d_model=dim)
            self.using_real = True
        except Exception:
            self.ssm = S4DMixer(dim, bidirectional=bidirectional)
            self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        if self.using_real:
            return self.m(x)
        return self.ssm(x) * torch.sigmoid(self.gate(x))   # input-dependent gate (selective-ish)



class FNetMixer(nn.Module):
    """FNet token mixing (Lee-Thorp et al.): parameter-free 2D FFT, keep the real part.
    The closest cousin to the spectral YB-Mixer; isolates the value of a *learned orthogonal*
    spectral map over a *fixed* FFT."""
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            y = torch.fft.fft(torch.fft.fft(xf, dim=-1), dim=1).real
        return y.to(x.dtype)


class LRUMixer(nn.Module):
    """Linear Recurrent Unit (Orvieto et al. 2023): diagonal complex linear recurrence with
    stable magnitude/phase parameterization and gamma normalization. One complex state per
    channel; bidirectional via forward+backward FFT convolution for classification."""
    def __init__(self, dim, r_min=0.9, r_max=0.999, max_phase=6.283, bidirectional=True):
        super().__init__()
        self.bi = bidirectional
        u1 = torch.rand(dim); u2 = torch.rand(dim)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        self.B_re = nn.Parameter(torch.randn(dim) / dim**0.5); self.B_im = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.C_re = nn.Parameter(torch.randn(dim) / dim**0.5); self.C_im = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.D = nn.Parameter(torch.zeros(dim))
        if bidirectional:
            self.C_re2 = nn.Parameter(torch.randn(dim) / dim**0.5); self.C_im2 = nn.Parameter(torch.randn(dim) / dim**0.5)

    def _kernel(self, L, device, C_re, C_im):
        nu = torch.exp(self.nu_log); mag = torch.exp(-nu)                  # |lambda| in (0,1)
        phase = torch.exp(self.theta_log)
        lam = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))   # (C,)
        gamma = torch.sqrt(torch.clamp(1 - mag**2, min=1e-6))                  # normalization
        B = torch.complex(self.B_re, self.B_im); C = torch.complex(C_re, C_im)
        coeff = C * gamma * B                                                  # (C,)
        t = torch.arange(L, device=device, dtype=torch.float32)
        lam_pow = lam[None, :] ** t[:, None]                                   # (L, C)
        return (coeff[None, :] * lam_pow).real                                 # (L, C)

    def _conv(self, x, k):
        L = x.shape[1]; n = 2 * L
        Xf = torch.fft.rfft(x.float(), n=n, dim=1)
        Kf = torch.fft.rfft(k, n=n, dim=0).unsqueeze(0)
        return torch.fft.irfft(Xf * Kf, n=n, dim=1)[:, :L, :]

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float(); L = xf.shape[1]
            y = self._conv(xf, self._kernel(L, xf.device, self.C_re, self.C_im))
            if self.bi:
                k2 = self._kernel(L, xf.device, self.C_re2, self.C_im2)
                y = y + torch.flip(self._conv(torch.flip(xf, [1]), k2), [1])
            y = y + self.D[None, None, :] * xf
        return y.to(x.dtype)


class S4DLinMixer(nn.Module):
    """Properly initialized S4D (Gu et al., 'On the Parameterization and Initialization of
    Diagonal SSMs'): N complex states per channel with the S4D-Lin/HiPPO init A_n=-1/2+i*pi*n,
    log-spaced timescales dt. Bidirectional via forward+backward FFT convolution. This is the
    *fair* SSM baseline."""
    def __init__(self, dim, n_states=64, dt_min=1e-3, dt_max=1e-1, bidirectional=True):
        super().__init__()
        self.bi = bidirectional; H, N = dim, n_states
        log_dt = torch.rand(H) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) \
                 + torch.log(torch.tensor(dt_min))
        self.log_dt = nn.Parameter(log_dt)                                  # (H,)
        n = torch.arange(N).float()
        # parameterize Re(A) = -exp(A_re_log) < 0  => |discrete pole| < 1 ALWAYS (stable).
        self.A_re_log = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))     # init Re(A) = -0.5 (S4D-Lin)
        self.A_im = nn.Parameter((math.pi * n)[None, :].repeat(H, 1).clone())  # S4D-Lin imag part
        self.C_re = nn.Parameter(torch.randn(H, N) / N**0.5); self.C_im = nn.Parameter(torch.randn(H, N) / N**0.5)
        self.D = nn.Parameter(torch.zeros(H))
        if bidirectional:
            self.C_re2 = nn.Parameter(torch.randn(H, N) / N**0.5); self.C_im2 = nn.Parameter(torch.randn(H, N) / N**0.5)

    def _kernel(self, L, device, C_re, C_im):
        dt = torch.exp(self.log_dt)[:, None]                                 # (H,1) > 0
        A = torch.complex(-torch.exp(self.A_re_log), self.A_im)              # (H,N) with Re(A) < 0
        dtA = dt * A                                                        # (H,N) complex, Re < 0
        C = torch.complex(C_re, C_im)                                       # (H,N)
        t = torch.arange(L, device=device, dtype=torch.float32)
        # lambda^t = exp(t * dt * A); magnitude = exp(t*Re(dtA)) <= 1  -> no explosion
        lam_pow = torch.exp(t[:, None, None] * dtA[None])                    # (L,H,N) complex
        k = torch.einsum('lhn,hn->lh', lam_pow, C).real                      # (L,H)
        return k

    def _conv(self, x, k):
        L = x.shape[1]; n = 2 * L
        Xf = torch.fft.rfft(x.float(), n=n, dim=1)
        Kf = torch.fft.rfft(k, n=n, dim=0).unsqueeze(0)
        return torch.fft.irfft(Xf * Kf, n=n, dim=1)[:, :L, :]

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float(); L = xf.shape[1]
            y = self._conv(xf, self._kernel(L, xf.device, self.C_re, self.C_im))
            if self.bi:
                k2 = self._kernel(L, xf.device, self.C_re2, self.C_im2)
                y = y + torch.flip(self._conv(torch.flip(xf, [1]), k2), [1])
            y = y + self.D[None, None, :] * xf
        return y.to(x.dtype)


# ----------------------------- block + model -----------------------------

class MLP(nn.Module):
    def __init__(self, dim, ratio=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ratio * dim, dim))

    def forward(self, x):
        return self.net(x)


class RGLRUMixer(nn.Module):
    """Real-Gated Linear Recurrent Unit (Hawk/Griffin, De et al. 2024).
    Input-dependent gates make the recurrence weight a_t = sigma(Lambda)^(c*r_t) data-dependent,
    so this CANNOT use the FFT-conv fast path; it is computed with a stable log-space sequential
    scan. Bidirectional (forward + backward) for the classification scaffold.
    NOTE: sequential scan -> slowest mixer; faithful to eq. (1)-(4)."""
    def __init__(self, dim, c=8.0, bidirectional=True):
        super().__init__()
        self.bi = bidirectional; self.c = c
        self.W_a = nn.Linear(dim, dim); self.W_x = nn.Linear(dim, dim)
        self.Lambda = nn.Parameter(self._init_lambda(dim, c))
        if bidirectional:
            self.W_a2 = nn.Linear(dim, dim); self.W_x2 = nn.Linear(dim, dim)
            self.Lambda2 = nn.Parameter(self._init_lambda(dim, c))

    @staticmethod
    def _init_lambda(dim, c):
        # init so that a^c ~ U(0.9, 0.999) at start (a = sigmoid(Lambda)), as in Griffin
        ac = torch.empty(dim).uniform_(0.9, 0.999); a = ac ** (1.0 / c)
        return torch.log(a / (1 - a))                                   # logit(a)

    def _scan(self, x, W_a, W_x, Lambda):
        r = torch.sigmoid(W_a(x)); i = torch.sigmoid(W_x(x))            # gates (B,L,C)
        log_a = self.c * r * F.logsigmoid(Lambda)[None, None, :]        # log a_t <= 0  (stable)
        a = torch.exp(log_a)
        b = torch.sqrt(torch.clamp(1 - a * a, min=1e-6)) * (i * x)      # input term
        B, L, C = x.shape; h = x.new_zeros(B, C); outs = []
        for t in range(L):
            h = a[:, t] * h + b[:, t]; outs.append(h)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            y = self._scan(xf, self.W_a, self.W_x, self.Lambda)
            if self.bi:
                y = y + torch.flip(self._scan(torch.flip(xf, [1]), self.W_a2, self.W_x2, self.Lambda2), [1])
        return y.to(x.dtype)


class SCORNNMixer(nn.Module):
    """Scaled Cayley Orthogonal RNN (scoRNN, Helfrich et al. 2018).
    Recurrent matrix W = (I+A)^{-1}(I-A)D with A skew-symmetric and D diagonal +/-1 (rho = dim//2),
    so W is orthogonal to machine precision. Nonlinear (modReLU) sequential RNN -> no conv/scan
    fast path. This is the canonical orthogonal-RNN baseline for our orthogonality claim.
    NOTE: sequential + dense recurrence -> slow."""
    def __init__(self, dim, rho=None, bidirectional=True):
        super().__init__()
        self.bi = bidirectional; self.dim = dim
        rho = dim // 2 if rho is None else rho
        self._make_dir("f", dim, rho)
        if bidirectional: self._make_dir("b", dim, rho)

    def _make_dir(self, tag, dim, rho):
        setattr(self, f"A_{tag}", nn.Parameter(0.05 * torch.randn(dim, dim)))
        setattr(self, f"U_{tag}", nn.Linear(dim, dim, bias=False))
        setattr(self, f"bias_{tag}", nn.Parameter(torch.zeros(dim)))
        D = torch.ones(dim); D[:rho] = -1.0
        self.register_buffer(f"D_{tag}", D)

    def _W(self, tag):
        A = getattr(self, f"A_{tag}"); A = A - A.t()                    # skew-symmetric
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I + A, I - A) * getattr(self, f"D_{tag}")[None, :]   # (I+A)^-1(I-A) D

    @staticmethod
    def _modrelu(z, b):
        return torch.sign(z) * F.relu(torch.abs(z) + b)                 # real modReLU

    def _scan(self, x, tag):
        W = self._W(tag); U = getattr(self, f"U_{tag}"); b = getattr(self, f"bias_{tag}")
        B, L, C = x.shape; h = x.new_zeros(B, C); Ux = U(x); outs = []
        for t in range(L):
            h = self._modrelu(Ux[:, t] + h @ W.t(), b); outs.append(h)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            y = self._scan(xf, "f")
            if self.bi: y = y + torch.flip(self._scan(torch.flip(xf, [1]), "b"), [1])
        return y.to(x.dtype)


class RWKVMixer(nn.Module):
    """RWKV-4 time-mixing (Peng et al. 2023): token-shift + R/K/V, with the numerically stable
    WKV decay-scan (running-max). Bidirectional for classification.
    NOTE: decay scan is sequential -> slow. (RWKV-4 variant; channel-mix is handled by the
    shared scaffold MLP, so this module is the time-mixing operator.)"""
    def __init__(self, dim, bidirectional=True):
        super().__init__()
        self.bi = bidirectional
        self._make_dir("f", dim)
        if bidirectional: self._make_dir("b", dim)

    def _make_dir(self, tag, dim):
        setattr(self, f"time_decay_{tag}", nn.Parameter(torch.randn(dim) * 0.1))   # w = -exp(.)
        setattr(self, f"time_first_{tag}", nn.Parameter(torch.randn(dim) * 0.1))   # bonus u
        for nm in ("mix_k", "mix_v", "mix_r"):
            setattr(self, f"{nm}_{tag}", nn.Parameter(torch.rand(dim)))
        setattr(self, f"Wk_{tag}", nn.Linear(dim, dim, bias=False))
        setattr(self, f"Wv_{tag}", nn.Linear(dim, dim, bias=False))
        setattr(self, f"Wr_{tag}", nn.Linear(dim, dim, bias=False))
        setattr(self, f"Wo_{tag}", nn.Linear(dim, dim, bias=False))

    def _wkv(self, k, v, w, u):
        # stable RWKV-4 scan; k,v: (B,L,C); w,u: (C,)
        B, L, C = k.shape
        a = k.new_zeros(B, C); b = k.new_zeros(B, C); p = k.new_full((B, C), -1e30)
        outs = []
        for t in range(L):
            kt = k[:, t]; vt = v[:, t]
            q = torch.maximum(p, u + kt); e1 = torch.exp(p - q); e2 = torch.exp(u + kt - q)
            outs.append((e1 * a + e2 * vt) / (e1 * b + e2))
            q2 = torch.maximum(p + w, kt); e1 = torch.exp(p + w - q2); e2 = torch.exp(kt - q2)
            a = e1 * a + e2 * vt; b = e1 * b + e2; p = q2
        return torch.stack(outs, dim=1)

    def _scan(self, x, tag):
        mk = getattr(self, f"mix_k_{tag}"); mv = getattr(self, f"mix_v_{tag}"); mr = getattr(self, f"mix_r_{tag}")
        xs = F.pad(x, (0, 0, 1, 0))[:, :-1]                              # token shift (prev token)
        k = getattr(self, f"Wk_{tag}")(x * mk + xs * (1 - mk))
        v = getattr(self, f"Wv_{tag}")(x * mv + xs * (1 - mv))
        r = torch.sigmoid(getattr(self, f"Wr_{tag}")(x * mr + xs * (1 - mr)))
        w = -torch.exp(getattr(self, f"time_decay_{tag}")); u = getattr(self, f"time_first_{tag}")
        wkv = self._wkv(k, v, w, u)
        return getattr(self, f"Wo_{tag}")(r * wkv)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            y = self._scan(xf, "f")
            if self.bi: y = y + torch.flip(self._scan(torch.flip(xf, [1]), "b"), [1])
        return y.to(x.dtype)


def make_mixer(name, dim, n_heads, n_basis, n_states=64):
    if name == "yb":          return SpectralYBMixer(dim, n_basis=n_basis, orthogonal=True)
    if name == "yb_relaxed":  return SpectralYBMixer(dim, n_basis=n_basis, orthogonal=False)
    if name == "transformer": return AttentionMixer(dim, n_heads=n_heads)
    if name == "s4d":         return S4DMixer(dim, bidirectional=True)        # untuned (ablation)
    if name == "s4dlin":      return S4DLinMixer(dim, n_states=n_states, bidirectional=True)  # HiPPO/S4D-Lin (fair)
    if name == "lru":         return LRUMixer(dim, bidirectional=True)        # Linear Recurrent Unit
    if name == "fnet":        return FNetMixer(dim)                            # fixed-FFT cousin
    if name == "mamba":       return MambaMixer(dim, bidirectional=True)
    if name == "rglru":       return RGLRUMixer(dim, bidirectional=True)       # Hawk/Griffin RG-LRU
    if name == "scornn":      return SCORNNMixer(dim, bidirectional=True)      # scaled-Cayley orthogonal RNN
    if name == "rwkv":        return RWKVMixer(dim, bidirectional=True)        # RWKV-4 time mixing
    raise ValueError(name)


class Block(nn.Module):
    def __init__(self, name, dim, n_heads, n_basis, mlp_ratio, dropout, n_states=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = make_mixer(name, dim, n_heads, n_basis, n_states)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class SequenceClassifier(nn.Module):
    """Shared scaffold: embed -> [Block]*depth -> mean-pool -> linear head."""
    def __init__(self, name, *, in_type, vocab, n_classes, seq_len,
                 dim=128, depth=4, n_heads=4, n_basis=24, mlp_ratio=2,
                 dropout=0.0, pos_emb=True, n_states=64):
        super().__init__()
        self.in_type = in_type
        if in_type == "discrete":
            self.embed = nn.Embedding(vocab, dim)
        else:  # continuous scalar per position
            self.embed = nn.Linear(1, dim)
        self.pos = nn.Parameter(0.02 * torch.randn(seq_len, dim)) if pos_emb else None
        self.blocks = nn.ModuleList([Block(name, dim, n_heads, n_basis, mlp_ratio, dropout, n_states)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        if self.in_type == "discrete":
            h = self.embed(x)                          # (B,L,dim)
        else:
            h = self.embed(x.unsqueeze(-1).float())    # (B,L,1)->(B,L,dim)
        if self.pos is not None:
            h = h + self.pos[: h.shape[1]].unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)

    def forward_tokens(self, x):                  # per-position logits (for copy/seq-output tasks)
        if self.in_type == "discrete":
            h = self.embed(x)
        else:
            h = self.embed(x.unsqueeze(-1).float())
        if self.pos is not None:
            h = h + self.pos[: h.shape[1]].unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))            # (B, L, n_classes)


def build_model(name, meta, dim, depth, n_heads=4, n_basis=24, dropout=0.0, pos_emb=True, n_states=64):
    return SequenceClassifier(
        name, in_type=meta["in_type"], vocab=meta.get("vocab", 0),
        n_classes=meta["n_classes"], seq_len=meta["seq_len"],
        dim=dim, depth=depth, n_heads=n_heads, n_basis=n_basis,
        dropout=dropout, pos_emb=pos_emb, n_states=n_states)
