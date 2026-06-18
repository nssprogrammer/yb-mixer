# YB-Mixer GPU Harness

A controlled, GPU-ready comparison harness. Every architecture uses the **same pre-norm block
scaffold** — `x = x + mixer(norm(x)); x = x + mlp(norm(x))` — so results differ **only in the
token mixer**. This is the apples-to-apples setup reviewers asked for.

## Models (`--model`)
| name | mixer | notes |
|------|-------|-------|
| `yb` | **Spectral YB-Mixer** (orthogonal, phase-only) | our model; norm-preserving, commuting/anytime, length-agnostic |
| `yb_relaxed` | spectral with learnable magnitude | **not** orthogonal — ablation: the cost of orthogonality (≈ a bidirectional global-conv SSM) |
| `transformer` | bidirectional multi-head attention | |
| `s4dlin` | **S4D with HiPPO/S4D-Lin init** | the *fair* SSM baseline (W1 fix) |
| `lru` | **Linear Recurrent Unit** (Orvieto et al. 2023) | diagonal orthogonal-ish linear recurrence |
| `fnet` | **FNet** fixed-FFT mixing | parameter-free spectral cousin of YB |
| `s4d` | diagonal complex SSM, *untuned* | kept as an initialization ablation |
| `rglru` | **RG-LRU** (Hawk/Griffin) gated linear recurrence | modern gated recurrent baseline (input-dependent gate; sequential scan) |
| `scornn` | **scoRNN** scaled-Cayley orthogonal RNN | canonical orthogonal-RNN baseline (W orthogonal to machine precision) |
| `rwkv` | **RWKV-4** time-mixing (stable WKV scan) | linear-attention/RNN hybrid baseline |
| `mamba` | real `mamba_ssm.Mamba` if installed, else gated diagonal-SSM fallback | |

The spectral mixers parameterize their per-mode phase (and, for `yb_relaxed`, log-magnitude) as a
function of the **normalized** frequency `f = m/L`, so the same generator instantiates at any
length — this is what enables length generalization.

## Tasks (`--task`) — all long-range **bidirectional classification**
| task | seq len | classes | source |
|------|--------:|--------:|--------|
| `pmnist` | 784 | 10 | torchvision MNIST (fixed permutation) |
| `smnist` | 784 | 10 | torchvision MNIST |
| `scifar` | 1024 | 10 | torchvision CIFAR-10, grayscale = **LRA-Image** |
| `imdb`   | `--seq_len` | 2 | HF `datasets` IMDB, byte-level ≈ **LRA-Text** |

> The mixer is bidirectional (global FFT), which suits LRA-style classification. It is **not**
> causal, so this harness deliberately does not include autoregressive char-LM.

## Quickstart

```bash
pip install -r requirements_gpu.txt          # install the CUDA torch build for your driver
python train.py --smoke                        # CPU plumbing test (random data, ~3s)
python test_learning.py yb                      # confirm a mixer learns a long-range pattern

# a real run:
python train.py --task scifar --model yb --dim 128 --depth 6 --epochs 80 --amp
```

## Reproduce the full study

```bash
bash run_gpu.sh           # ~12h on one A100/3090-class GPU; edit phases to fit budget
python aggregate.py runs  # prints the results table from runs/*.json
```

`run_gpu.sh` runs three phases: permuted-MNIST (sanity), sequential-CIFAR/LRA-Image (flagship,
all five mixers + a second YB seed), and byte-level IMDB/LRA-Text. Comment out phases if short on
time; Phase 2 (sCIFAR) is the one to keep.

## Outputs
Each run writes `runs/<task>_<model>_d<dim>_L<depth>_s<seed>.csv` (per-epoch curve) and `.json`
(summary: params, `best_val_acc`, time). `aggregate.py` collates them.

## Key knobs
`--dim --depth --n_basis` (spectral basis size) `--epochs --batch_size --lr --weight_decay`
`--dropout --warmup_frac --grad_clip --label_smoothing --seq_len` (imdb) `--no_pos` `--seed --amp`.

## What to look for
1. **Flagship (sCIFAR):** does `yb` reach the SSM/attention ballpark? Does `yb_relaxed` (non-orthogonal)
   beat `yb` (orthogonal)? That gap = the measured cost of orthogonality.
2. **`yb` vs `s4d`/`mamba`:** the direct structured-mixer comparison at matched params.
3. **Length generalization:** train `yb` at one `--seq_len`, evaluate at a longer one (spectral
   generator should hold up far better than a local one).


### Extended downstream tasks
| task | type | notes |
|---|---|---|
| `listops` | LRA ListOps, 10-way | nested MIN/MAX/MED/SUM-mod over digits; faithful generator |
| `induction` | Induction Heads, (V-1)-way | recall token after a unique marker; tests retrieval + **length extrapolation** (build with `pos_emb=False`) |
| `selective_copy` | copy n_data tokens among noise | per-position output via `forward_tokens` + masked CE (notebook) |
| `pathfinder` | binary connectivity (L=1024) | **synthetic** Pathfinder-style reimplementation; hard tracing task, *not* comparable to official LRA-Pathfinder; for reportable numbers use the official dataset |
