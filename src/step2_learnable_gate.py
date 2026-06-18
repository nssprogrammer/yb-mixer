"""
Step 2 -- Differentiable YBE and learning an integrable gate.

Fact (proved by hand, verified in Step 1): for R(lambda)=I+tan(lambda)M the braided
parameter-dependent YBE residual is zero  <=>  M^2=I  AND  {M12,M23}=0.

Part A: validate a differentiable YBE residual against the free-fermion anchor M=X(x)Y.
Part B: learn an integrable gate RELIABLY via the algebraic surrogate -- build M as an
        involution by construction (M = U D U^dagger, D=diag(1,1,-1,-1)) so M^2=I, tr M=0
        exactly, then minimize only the neighbour anticommutator; verify the full YBE
        residual vanishes at the solution.
Dependencies: torch.
"""
import torch
torch.set_default_dtype(torch.float64)
cdt = torch.complex128

I2 = torch.eye(2, dtype=cdt)
X = torch.tensor([[0, 1], [1, 0]], dtype=cdt)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=cdt)
I8 = torch.eye(8, dtype=cdt)
I4 = torch.eye(4, dtype=cdt)
D = torch.diag(torch.tensor([1., 1., -1., -1.], dtype=cdt))


def kron(a, b): return torch.kron(a, b)
def embed(M4):  return kron(M4, I2), kron(I2, M4)
def Rgate(M_emb, lam): return I8 + torch.tan(lam) * M_emb


def ybe_residual(M4, lam, mu):
    M12, M23 = embed(M4)
    lhs = Rgate(M12, lam) @ Rgate(M23, lam + mu) @ Rgate(M12, mu)
    rhs = Rgate(M23, mu) @ Rgate(M12, lam + mu) @ Rgate(M23, lam)
    return torch.linalg.norm(lhs - rhs)


def involution_from_generator(A_raw):
    A = A_raw - A_raw.conj().T
    U = torch.linalg.matrix_exp(A)
    return U @ D @ U.conj().T


def anticomm_loss(M4):
    M12, M23 = embed(M4)
    return torch.linalg.norm(M12 @ M23 + M23 @ M12) ** 2


def main():
    print("=" * 64)
    print("STEP 2: differentiable YBE + learnable integrable gate")
    print("=" * 64)

    # Part A -- anchor validation
    torch.manual_seed(0)
    M_anchor = kron(X, Y)
    res_anchor = [ybe_residual(M_anchor, torch.rand(()) * 2 - 1, torch.rand(()) * 2 - 1).item()
                  for _ in range(20)]
    Mr = torch.randn(4, 4, dtype=cdt); Mr = (Mr + Mr.conj().T) / 2
    res_rand = [ybe_residual(Mr, torch.rand(()) * 2 - 1, torch.rand(()) * 2 - 1).item()
                for _ in range(20)]
    print("Part A -- residual validation")
    print(f"  anchor M=X(x)Y  YBE residual (mean) = {sum(res_anchor) / 20:.3e}")
    print(f"  random Hermitian residual (mean)    = {sum(res_rand) / 20:.3e}  (should be LARGE)")

    # Part B -- reliable learning via the algebraic surrogate
    results = []
    for restart in range(6):
        torch.manual_seed(200 + restart)
        A = (0.3 * torch.randn(4, 4, dtype=cdt)).requires_grad_(True)
        opt = torch.optim.Adam([A], lr=0.05)
        for _ in range(800):
            opt.zero_grad()
            anticomm_loss(involution_from_generator(A)).backward()
            opt.step()
        with torch.no_grad():
            M4 = involution_from_generator(A)
            sq = torch.linalg.norm(M4 @ M4 - I4).item()
            M12, M23 = embed(M4)
            anti = torch.linalg.norm(M12 @ M23 + M23 @ M12).item()
            ybe = sum(ybe_residual(M4, torch.rand(()) * 2 - 1, torch.rand(()) * 2 - 1).item()
                      for _ in range(50)) / 50
        results.append((anti, sq, ybe, M4.detach().clone()))
    anti, sq, ybe, M_best = min(results, key=lambda r: r[0])
    print("\nPart B -- learn integrable gate (best of 6 restarts)")
    print(f"  anticommutator |{{M12,M23}}|         = {anti:.3e}")
    print(f"  involution     |M^2-I|              = {sq:.3e}")
    print(f"  full YBE residual at solution       = {ybe:.3e}")
    print("=" * 64)
    print("RESULT:", "PASS" if (sum(res_anchor) / 20 < 1e-10 and anti < 1e-4 and ybe < 1e-3) else "FAIL")


if __name__ == "__main__":
    main()
