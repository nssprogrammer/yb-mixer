"""
Step 1 -- Numerical verification of the generalized Yang-Baxter (gYBE) primitive
from Sinha, Maity, Padmanabhan, Korepin (arXiv:2605.30007).

Builds Majorana operators via Jordan-Wigner, the multi-site M operators (Eq. 13/57),
the Baxterized R-matrix R(lambda)=1+tan(lambda) M (Eq. 58), and verifies:
  (1) CAR  {gamma_i, gamma_j} = 2 delta_ij
  (2) M^2 = 1
  (3) adjacent M's anticommute, distant M's commute  (Eq. 15)
  (4) the (d,6,3)-gYBE residual (Eq. 59)  -- the integrability signature
Dependencies: numpy.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def majoranas(n_qubits):
    """gamma_1..gamma_{2n} via Jordan-Wigner (strict-prefix Z string)."""
    g = []
    for k in range(1, n_qubits + 1):
        odd, even = [], []
        for j in range(1, n_qubits + 1):
            if j < k:
                odd.append(Z); even.append(Z)
            elif j == k:
                odd.append(X); even.append(Y)
            else:
                odd.append(I2); even.append(I2)
        g.append(kron_list(odd))   # gamma_{2k-1}
        g.append(kron_list(even))  # gamma_{2k}
    return g


def build_M(g, start, omega):
    """M = (w1 g_s + w2 g_{s+1} + w3 g_{s+2}) . g_{s+3} g_{s+4} g_{s+5}, 1-based 'start'."""
    gi = lambda i: g[i - 1]
    a = omega[0] * gi(start) + omega[1] * gi(start + 1) + omega[2] * gi(start + 2)
    b = gi(start + 3) @ gi(start + 4) @ gi(start + 5)
    return a @ b


def comm(A, B):  return A @ B - B @ A
def acomm(A, B): return A @ B + B @ A
def mx(A):       return float(np.max(np.abs(A)))


def main():
    NQ = 6
    dim = 2 ** NQ
    g = majoranas(NQ)
    Id = np.eye(dim, dtype=complex)

    rng = np.random.default_rng(0)
    omega = rng.standard_normal(3)
    omega = omega / np.linalg.norm(omega)         # ||Omega|| = 1  (Eq. 14)

    MA = build_M(g, 1, omega)   # gamma_1..gamma_6
    MB = build_M(g, 4, omega)   # gamma_4..gamma_9 (shift m=3, adjacent/overlapping)
    MC = build_M(g, 7, omega)   # gamma_7..gamma_12 (distant from M_A)

    print("=" * 60)
    print("STEP 1: gYBE primitive verification")
    print(f"qubits={NQ}  Hilbert dim={dim}  ||omega||={np.linalg.norm(omega):.6f}")
    print("=" * 60)

    car_err = 0.0
    for i in range(2 * NQ):
        for j in range(2 * NQ):
            target = 2.0 * Id if i == j else 0.0 * Id
            car_err = max(car_err, mx(acomm(g[i], g[j]) - target))
    print(f"(1) CAR  max|{{g_i,g_j}} - 2d|       = {car_err:.3e}")
    print(f"(2) |M_A^2 - I|                      = {mx(MA @ MA - Id):.3e}")
    print(f"    |M_B^2 - I|                      = {mx(MB @ MB - Id):.3e}")
    print(f"(3) |{{M_A, M_B}}|  (adjacent ->0)    = {mx(acomm(MA, MB)):.3e}")
    print(f"    |[M_A, M_C]|  (distant ->0)       = {mx(comm(MA, MC)):.3e}")

    def R(M, lam): return Id + np.tan(lam) * M
    resids = []
    for _ in range(20):
        lam, mu = rng.uniform(-1.0, 1.0, size=2)
        lhs = R(MA, lam) @ R(MB, lam + mu) @ R(MA, mu)
        rhs = R(MB, mu) @ R(MA, lam + mu) @ R(MB, lam)
        resids.append(mx(lhs - rhs))
    resids = np.array(resids)
    print(f"(4) gYBE residual  max|LHS-RHS|      = {resids.max():.3e}")
    print(f"    gYBE residual  mean over 20 runs = {resids.mean():.3e}")

    bad = []
    for _ in range(20):
        lam, mu = rng.uniform(-1.0, 1.0, size=2)
        lhs = R(MA, lam) @ R(MB, lam - mu) @ R(MA, mu)   # wrong addition law
        rhs = R(MB, mu) @ R(MA, lam - mu) @ R(MB, lam)
        bad.append(mx(lhs - rhs))
    print(f"    control (wrong addition law)     = {np.mean(bad):.3e}   (should be LARGE)")
    print("=" * 60)
    ok = (car_err < 1e-10 and mx(MA @ MA - Id) < 1e-10 and mx(acomm(MA, MB)) < 1e-10
          and mx(comm(MA, MC)) < 1e-10 and resids.max() < 1e-9 and np.mean(bad) > 1e-3)
    print("RESULT:", "PASS  integrable primitive verified" if ok else "FAIL")


if __name__ == "__main__":
    main()
