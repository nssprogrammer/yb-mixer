"""
Step 3 -- Brick-wall YB-Mixer layer and depth stability with honest controls.

Free-fermion integrable gates act as ORTHOGONAL maps on token-feature channels
(single-particle picture). We compare three gates across depth on a brick-wall mixer:
  (A) integrable / free-fermion : per-channel 2x2 rotations (orthogonal, integrable)
  (B) random orthogonal         : full Q from QR (orthogonal, NOT integrable)  <- control
  (C) random generic            : gaussian matrix (neither)                    <- control
Metrics vs depth: output/input norm ratio, Jacobian condition number.
The (B) control shows depth stability comes from orthogonality (which integrable gates
supply); integrability's extra payoff (commuting charges) is Step 4.
Dependencies: torch.
"""
import torch
torch.set_default_dtype(torch.float64)

L, c = 8, 4
dim2 = 2 * c


def integrable_gate(theta=0.7):
    g = torch.eye(dim2)
    ct, st = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    for k in range(c):
        g[k, k] = ct; g[k, c + k] = -st
        g[c + k, k] = st; g[c + k, c + k] = ct
    return g


def random_orthogonal_gate(seed):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim2, dim2))
    return Q


def random_generic_gate(seed):
    torch.manual_seed(seed)
    return torch.randn(dim2, dim2) / (dim2 ** 0.5)


def brickwall_once(X, g):
    X = X.clone()
    for offset in (0, 1):
        for i in range(offset, L - 1, 2):
            pair = torch.cat([X[:, i, :], X[:, i + 1, :]], dim=-1)
            out = pair @ g.T
            X[:, i, :] = out[:, :c]
            X[:, i + 1, :] = out[:, c:]
    return X


def run_stack(X, g, depth):
    for _ in range(depth):
        X = brickwall_once(X, g)
    return X


def linear_map_matrix(g, depth):
    n = L * c
    cols = []
    for j in range(n):
        e = torch.zeros(1, n); e[0, j] = 1.0
        cols.append(run_stack(e.view(1, L, c), g, depth).reshape(-1))
    return torch.stack(cols, dim=1)


def main():
    print("=" * 70)
    print(f"STEP 3: brick-wall YB-Mixer  (L={L}, channels={c}, feature dim={L * c})")
    print("=" * 70)
    gates = {
        "A integrable (free-fermion)": integrable_gate(0.7),
        "B random orthogonal (ctrl) ": random_orthogonal_gate(1),
        "C random generic    (ctrl) ": random_generic_gate(1),
    }
    depths = [1, 2, 4, 8, 16, 32]
    torch.manual_seed(0)
    X0 = torch.randn(16, L, c)
    n0 = torch.linalg.norm(X0).item()
    for name, g in gates.items():
        print(f"\n{name}")
        print(f"   {'depth':>5} | {'||Y||/||X||':>12} | {'Jacobian cond #':>16}")
        for Dp in depths:
            Y = run_stack(X0, g, Dp)
            ratio = torch.linalg.norm(Y).item() / n0
            s = torch.linalg.svdvals(linear_map_matrix(g, Dp))
            cond = (s.max() / s.min()).item()
            print(f"   {Dp:>5} | {ratio:>12.4e} | {cond:>16.4e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
