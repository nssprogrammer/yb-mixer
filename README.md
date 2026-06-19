# YB-Mixer: An Integrable Token-Mixing Layer from the Generalized Yang–Baxter Equation

**Author:** Snigdha Chandan Khilar (Independent Researcher) · `snkhilar@gmail.com`

YB-Mixer is a sequence token-mixing layer derived from the free-fermion / generalized
Yang–Baxter (gYBE) structure of *hidden* transverse-field Ising models.

The design rests on one transferable principle from integrable systems:

> A **local** algebraic constraint on adjacent operations can certify **global**
> computational guarantees, independent of representation.

Concretely, the *Ising exchange algebra* (an extraspecial-2-group relation) certifies
(i) a free-fermionic structure → the mixer is an exactly **norm-preserving orthogonal map**,
and (ii) **commuting transfer matrices** → inference is **order-free** and **variable-budget
("anytime")**.


## Results at a glance (all reproducible)

| # | Result | Evidence |
|---|--------|----------|
| 1 | gYBE primitive verified | residual `~1e-15`; wrong-addition-law control `~1.7` |
| 2 | Integrable gates learnable via algebraic surrogate | surrogate → full YBE residual `~1e-7` (best `~1e-14`) |
| 3 | Norm-preserving, depth-stable mixer | Jacobian cond. `=1` ∀depth; generic gate `~1e18` @depth 32 |
| 4 | Commuting transfer matrices | `‖[τ(λ),τ(μ)]‖ ~1e-15`, grows monotonically off-algebra |
| 5 | Trainable + competitive + init recipe | `1.000` w/ near-π/4 init; `0.500` w/ small init (ablation) |
| 6 | End-to-end anytime / order-free inference | reorder spread `~1e-16`; coarse→fine budget curve |
| 7 | Matches attention at ~3.3× fewer params (3 seeds) | `1.000±0.000` (1,602 vs 5,354 params) |
| 8 | **Beats orthogonal-RNN/SSM on long-range memory; trails nonlinear mixer on recall** | memory `1.000` vs OrthRNN `0.75`, SSM `0.83`; recall gap quantified |
| 9 | **Length generalization fixed via spectral generator** | flat `~0.92` from L=16 → L=64 (4×); local generator collapses to chance |

### Scaled GPU benchmarks (~0.5M params, controlled harness — only the mixer differs)

| task (seq len) | **YB (ours)** | Transformer | S4D | Mamba |
|---|---|---|---|---|
| permuted-MNIST (784) | 0.983 | 0.985 | 0.980 | 0.979 |
| **seq-CIFAR / LRA-Image (1024)** | **0.849** | 0.493 | 0.661 | 0.651 |
| **byte-IMDB / LRA-Text (1024)** | **0.775** | 0.631 | 0.733 | 0.744 |

YB ties on pMNIST and **leads on the two harder long-range tasks** (+18 pts over the best
baseline on LRA-Image) at fewer params than Transformer/Mamba. Honest scope: SSM baselines are
minimal (no HiPPO init; tuned S4/S5 reach ~88% on LRA-Image), single-seed, ~0.5M params — the
comparison is *within the harness*, not vs maximally-tuned published SSMs.

Honest scope: a **local** generator does not length-generalize (dispersion); the **spectral**
generator (Step 9) fixes this on the transport task to 4× length. Real-benchmark validation
(LRA, language modeling) requires GPU-scale training beyond these CPU scripts and is the
essential next step — the layer is written to drop into a standard block.

## Repository layout

```
yb-mixer/
├── paper/
│   ├── yb_mixer.tex          # arXiv-ready LaTeX source
│   └── yb_mixer.pdf          # compiled paper (11 pp.)
├── src/
│   ├── step1_gybe_primitive.py        # verify the (d,6,3)-gYBE  (numpy)
│   ├── step2_learnable_gate.py        # differentiable YBE + surrogate learning (torch)
│   ├── step3_brickwall_stability.py   # brick-wall mixer; depth stability (torch)
│   ├── step4_transfer_matrix.py       # commuting transfer matrices (numpy)
│   ├── step5_trainable_transport.py   # trainable layer + init recipe (torch)
│   ├── step6_anytime_flow.py          # integrable flow; anytime inference (torch)
│   ├── step7_baselines_lengen.py      # attention baseline + length-gen (torch)
│   ├── step8_baselines.py             # orthogonal RNN, SSM, attention, nonlinear mixer (torch)
│   └── step9_spectral_lengen.py       # spectral generator -> length generalization (torch)
├── harness/                  # GPU-ready training harness (LRA-Image, pMNIST, IMDB)
│   ├── models.py             # shared block scaffold + all 5 token mixers
│   ├── data.py               # pMNIST / sMNIST / sCIFAR / IMDB loaders
│   ├── train.py              # unified AMP training loop (--smoke for CPU test)
│   ├── test_learning.py      # per-mixer long-range learnability check
│   ├── aggregate.py          # collate runs/*.json into a table
│   ├── run_gpu.sh            # 12-hour single-GPU experiment plan
│   ├── requirements_gpu.txt
│   └── README.md
├── run_all.sh                # reproduce the paper's CPU experiments (steps 1-9)
├── requirements.txt
└── LICENSE                   # MIT
```

## Quickstart

```bash
pip install -r requirements.txt
bash run_all.sh                       # runs steps 1–7 (CPU, a few minutes)
# or individually:
python src/step1_gybe_primitive.py
```

Each script prints a self-checking report ending in `PASS`/`FAIL`. Steps 1 and 4 need only
`numpy`; the rest use CPU `torch`.

## GPU benchmark harness

The `harness/` directory scales the spectral YB-Mixer to real long-range benchmarks against
Transformer, S4D-SSM, and Mamba baselines in a controlled scaffold (only the token mixer differs):

```bash
cd harness
pip install -r requirements_gpu.txt
python train.py --smoke                 # CPU plumbing test (~3s)
bash run_gpu.sh                         # ~12h on one GPU: pMNIST, sCIFAR(=LRA-Image), IMDB
python aggregate.py runs                # results table
```

See `harness/README.md` for details. Note the mixer is bidirectional (global FFT), so the harness
targets LRA-style classification, not causal language modeling.

## The architecture in one paragraph

A free-fermion gate is quadratic in Majoranas, so it acts on token features as an **orthogonal**
map (e.g. `X⊗Y = −i γ₂γ₄`). A brick-wall of such gates is a norm-preserving token mixer
(`step3`). The Baxterized gate `R(λ)=1+tan(λ)M` is integrable **iff** `M²=1` and neighbours
anticommute (Lemma 1), which makes integrable gates learnable via a well-conditioned surrogate
(`step2`). The QISM transfer matrix `τ(λ)` then commutes across `λ` (`step4`), and in the
continuous-time **integrable flow** `U(s)=exp(sK)` the whole mixing is a one-parameter group
`U(s)U(s')=U(s+s')` — giving exact order-free, variable-budget inference (`step6`).

## Citation

```bibtex
@misc{khilar2026ybmixer,
  title  = {YB-Mixer: An Integrable Token-Mixing Layer from the Generalized Yang--Baxter Equation},
  author = {Khilar, Snigdha Chandan},
  year   = {2026},
  note   = {Independent Researcher}
}
```

Built on the hidden-Ising / gYBE construction of Sinha, Maity, Padmanabhan, and Korepin
(arXiv:2605.30007).

## License
MIT © 2026 Snigdha Chandan Khilar
