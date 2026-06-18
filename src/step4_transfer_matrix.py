"""
YB-Mixer Step 4: commuting transfer matrices -- the integrability payoff
that orthogonality alone (Step 3) does NOT provide.

Build the QISM transfer matrix tau(lambda) from our free-fermion gate and show:
  Part A: [tau(λ), tau(μ)] ≈ 0 for the integrable gate, and the commutator GROWS
          as we perturb the gate off the extraspecial-2-group algebra (M^2=I,
          anticommuting neighbors). A random gate gives a large commutator too.
  Part B (ML payoff): because the tau(λ) family commutes, applying a schedule of
          spectral-parameter passes in ANY order gives the same output
          ("anytime / schedule-invariant" inference) -- generic mixers do not.
Dependencies: numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)  # permutation P

def two_site_op(R4, pa, pb, nf, d=2):
    rest = nf - 2
    K = np.kron(R4, np.eye(d**rest, dtype=complex))
    spectators = [f for f in range(nf) if f not in (pa, pb)]
    order = [pa, pb] + spectators
    perm = np.argsort(order)
    T = K.reshape([d]*nf + [d]*nf)
    T = np.transpose(T, list(perm) + [nf + p for p in perm])
    return T.reshape(d**nf, d**nf)

def transfer(R_of_l, lam, L, d=2):
    """tau(λ) = tr_aux  prod_n R_{aux,n}(λ);  aux=factor0, sites=1..L."""
    nf = L + 1
    Rl = R_of_l(lam)
    T = np.eye(d**nf, dtype=complex)
    for n in range(1, L+1):
        T = T @ two_site_op(Rl, 0, n, nf, d)
    dL = d**L
    return np.einsum('aiaj->ij', T.reshape(d, dL, d, dL))

def comm(A, B): return np.max(np.abs(A@B - B@A))

L = 4
M = np.kron(X, Y)                                   # our learned/anchor free-fermion gate
R_int = lambda l: (np.eye(4, dtype=complex) + np.tan(l)*M) @ SWAP   # integrable (non-braided)

print("="*68)
print("STEP 4: commuting transfer matrices  (the integrability differentiator)")
print("="*68)

# ---------- Part A: commuting family + algebra-perturbation sweep ----------
pairs = list(zip(rng.uniform(-0.8,0.8,8), rng.uniform(-0.8,0.8,8)))

def max_comm(R_of_l):
    return max(comm(transfer(R_of_l,a,L), transfer(R_of_l,b,L)) for a,b in pairs)

print("Part A -- [tau(λ),tau(μ)] vs distance from the integrability algebra")
print(f"   {'perturbation ε':>16} | {'|M²-I|':>10} | {'max|[tau,tau]|':>16}")
Hpert = rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))
Hpert = (Hpert + Hpert.conj().T)/2                  # random Hermitian direction
for eps in [0.0, 0.05, 0.1, 0.2, 0.4]:
    Me = M + eps*Hpert
    Me = Me / np.linalg.norm(Me) * 2.0              # keep scale comparable to M (||M||_F=2)
    sq = np.max(np.abs(Me@Me - np.eye(4)))
    R_e = (lambda Mloc: (lambda l: (np.eye(4,dtype=complex)+np.tan(l)*Mloc) @ SWAP))(Me)
    print(f"   {eps:>16.2f} | {sq:>10.2e} | {max_comm(R_e):>16.3e}")

# fully random gate control
Rr = rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))
R_rand = lambda l: np.eye(4,dtype=complex) + np.tan(l)*Rr
print(f"   {'random gate':>16} | {'   --':>10} | {max_comm(R_rand):>16.3e}")

# ---------- Part B: schedule-invariant (anytime) inference ----------
print("\nPart B -- apply a 4-rapidity schedule in different orders to one input")
lambdas = [0.2, 0.4, 0.6, 0.8]
x0 = rng.standard_normal(2**L) + 1j*rng.standard_normal(2**L)
x0 = x0 / np.linalg.norm(x0)

def apply_schedule(R_of_l, order):
    x = x0.copy()
    for k in order:
        x = transfer(R_of_l, lambdas[k], L) @ x
    return x

def schedule_spread(R_of_l):
    orders = [[0,1,2,3],[3,2,1,0],[1,3,0,2],[2,0,3,1],[0,2,1,3]]
    outs = [apply_schedule(R_of_l, o) for o in orders]
    ref = outs[0]
    return max(np.linalg.norm(o-ref)/np.linalg.norm(ref) for o in outs[1:])

# integrable vs perturbed(ε=0.2) vs random
Mp = M + 0.2*Hpert; Mp = Mp/np.linalg.norm(Mp)*2.0
R_pert = lambda l: (np.eye(4,dtype=complex)+np.tan(l)*Mp) @ SWAP
print(f"   integrable gate   : max relative output spread over 5 orders = {schedule_spread(R_int):.3e}")
print(f"   perturbed (ε=0.2) : max relative output spread over 5 orders = {schedule_spread(R_pert):.3e}")
print(f"   random gate       : max relative output spread over 5 orders = {schedule_spread(R_rand):.3e}")

print("="*68)
c_int = max_comm(R_int); c_rand = max_comm(R_rand)
ok = c_int < 1e-10 and c_rand > 1e-2 and schedule_spread(R_int) < 1e-10 and schedule_spread(R_pert) > 1e-2
print(f"integrable [tau,tau]={c_int:.2e}   random [tau,tau]={c_rand:.2e}")
print("RESULT:", "PASS ✅  integrable gate => commuting tau(λ) => schedule-invariant inference"
      if ok else "FAIL ❌  inspect numbers")
