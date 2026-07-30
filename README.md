# Stress Field Prediction for 2D Topology Optimisation using U-Net

## Overview

This repository contains a U-Net-based neural network for predicting the stress
fields (σx, σy, τxy) of 2D topology-optimisation problems, together with the
machinery to **use those predictions as a surrogate for finite elements inside a
topology-optimisation loop**. Replacing the FEM stress solve by a fast network
prediction is what makes the optimisation cheaper; the FEM is kept as an optional
corrector.

This repository is still a work in progress. Some parts of the code may not be 
fully implemented, cleaned up, or documented yet.

Two architectures are provided:

- **UNetTopo** — standard U-Net that takes the boundary loads as two image
  channels `(tx, ty)` alongside the density.
- **BE_UNetTopo** — U-Net with a dedicated **Boundary Embedding** module that
  encodes the 8 nodal traction vectors directly into the bottleneck latent space
  (density-only image input).

Both models are trained on FEM-generated datasets and integrated into the loop.

Beyond the plain surrogate, the repository implements two accelerators:

1. **Hybrid U-Net / FEM optimisation** — several strategies that interleave fast
   U-Net steps with occasional FEM corrections (`run_topology_optimization`).
2. **Energy-distance warm start** — a database method that, given an *unoptimised*
   problem, retrieves the *converged* geometry of the most similar case (compared
   by strain-energy field) and starts the optimisation from it, cutting the number
   of iterations to convergence.

This work uses the I2MB MATLAB codes in `OT_Functions/` and `OT_Software/`, called
from Python in `OT_NN/`. My contributions live in `OT_NN/Pytorch_NN/`.

---

## Repository Structure
```text
├── Software/OT_NN/Pytorch_NN/
│   ├── model.py                    # UNetTopo, BE_UNetTopo, CBAM, BoundaryEmbedding
│   ├── train.py                    # Training loop + input/target tensor builders
│   ├── evaluate.py                 # Evaluation and visualisation
│   ├── dataset.py                  # Datasets, D4 data augmentation, energy warm start
│   ├── main.py                     # Training entry point
│   ├── topology_utils.py           # Stress prediction (U-Net / FEM) + optimisation loop
│   ├── TopOpt_process.py           # Run one hybrid U-Net/FEM optimisation
│   ├── TopOpt_accelerated.py       # Energy-distance warm start + optimisation
│   ├── TopOpt_benchmark_*.py       # Benchmarks (architectures, models, hybrid strategy)
│   ├── training_benchmark.py       # Train several models and log results
│   ├── tune_Unet.py                # Hyper-parameter search
│   ├── verify_rotation_consistency.py  # Checks the D4 augmentation is consistent
│   │
│   ├── results/{U-Net,BE_Unet}/{dataset}/   # checkpoints: *_best.pth, *_checkpoint.pth
│   └── illustrations/{U-Net,BE_Unet}/{dataset}/
│
├── Software/OT_Functions/          # I2MB FEM MATLAB code (stiffness, stress, tractions)
├── Software/OT_Software/           # I2MB TopOpt MATLAB code (SolveFE, GenTopology, mesh)
├── PlotNeuralNet/                  # PDF/TikZ representation of the networks
└── Bibliography/                   # Reference documents
```

---

## Problem set-up

A unit square `[-1, 1]²` meshed with 32×32 quad elements.

- **Loads (Neumann)** — a distributed traction on all four edges, defined by
  **8 boundary nodes** (2 per edge). Each node stores **`(Tn, Tt)` = (normal,
  tangential)** components in the edge-local frame — **not** global `(tx, ty)`.
  This convention is dictated by the MATLAB solver (`VectorF_Line.m`,
  `GenerateTractions.m`); the datasets are generated with self-equilibrated
  tractions (ΣF = 0, ΣM = 0).
- **Supports (Dirichlet)** — a **pin** at `(-1, -1)` (x and y fixed) and a
  **roller** at `(1, -1)` (y fixed). Three DOFs, removing the three rigid-body
  modes; the configuration has mirror symmetry.

---

## Models

### UNetTopo
Standard U-Net, 4 encoder/decoder levels, configurable convolutions per block,
optional CBAM attention at the bottleneck.

| Input | Output |
|---|---|
| ρ + tx + ty — `[B, 3, 32, 32]` | σx, σy, τxy — `[B, 3, 32, 32]` |

The `tx, ty` channels are built from the stored `(Tn, Tt)` by converting them to
**global** components and rasterising them at their **physical edge positions**
(`get_traction_distribution` / `train._tractions_to_maps`, kept identical so
training and inference feed the network the same representation).

### BE_UNetTopo
U-Net with a Boundary Embedding module: the 16 nodal traction scalars are encoded
by an MLP + `ConvTranspose2d` and concatenated at the bottleneck before CBAM.

| Input | Output |
|---|---|
| ρ — `[B, 1, 32, 32]` + nodes `[B, 16]` | σx, σy, τxy — `[B, 3, 32, 32]` |

---

## Data augmentation (D4, ×8)

Elasticity is equivariant under the symmetries of the square, so each sample can
be replicated under the **dihedral group D4** — the 4 rotations (0/90/180/270°),
each optionally mirrored — giving **8×** more physically-consistent data.

`dataset.py` transforms density, tractions **and** stress consistently:

- **density / stress** rotate as image / tensor fields (`torch.rot90`, with the
  σx↔σy swap and τxy sign change on odd rotations);
- **tractions** are transformed in the *global* frame: convert `(Tn, Tt)`→global,
  relocate + rotate the nodal vectors, convert back to the new edge's `(Tn, Tt)`
  (`transform_tractions`).

Entry points: `random_augment` (on-the-fly, per epoch) and
`AcceleratedDataset.augment` (materialised ×8 expansion).

> **Note on verification.** The augmentation rotates the *stored* stress `g·σ`
> (the true physical answer) — it never re-solves the FEM. Do **not** validate it
> with a FEM round-trip (`FEM(g·inputs) == g·FEM(inputs)`): the discrete FEM/mesh
> is only reflection-covariant, so that test fails for a general load even when the
> augmentation is correct. The right, load-independent check is
> `verify_rotation_consistency.py` (bending + shear, all 8 D4 elements ≈ 1e-11).

---

## Topology-optimisation loop

`run_topology_optimization` (in `topology_utils.py`) runs SIMP-style optimisation
where each iteration's stress comes from either the U-Net or the FEM, selected by
the `RULE` string:

| `RULE` | Behaviour |
|---|---|
| `'Only FEM'` | every step uses the FEM (baseline / ground truth) |
| `'<n> Unet - <m> FEM'` | periodic: `n` fast U-Net steps then `m` FEM corrections |
| `'Decreasing compliance'` | U-Net until the compliance stops decreasing, then FEM |
| anything else | U-Net only |

`TopOpt_process.py` drives one such run and produces the iteration montage
(U-Net-driven vs FEM-driven geometries).

---

## Energy-distance warm start (accelerated method)

Instead of optimising from a uniform block, `AcceleratedDataset.closest_point`
retrieves a good **starting geometry** from a database of already-solved cases:

1. For every stored case, the **strain-energy field** of its *first (unoptimised)*
   image is kept (`Ener`, per element).
2. Given a new problem, its own first-image energy field is compared to every
   stored case by **squared error**, and the **closest** case is selected.
3. The new sample is initialised with that case's **converged density**, then the
   optimisation is run from there.

Because similar loads produce similar optimal topologies, this warm start lands
close to the optimum and converges in far fewer iterations. The database can be
grown ×8 with the same D4 augmentation (`AcceleratedDataset.augment`) so a single
solved case covers all 8 orientations. See `TopOpt_accelerated.py`.

---

## Benchmarks

The `TopOpt_benchmark_*.py` scripts provide a **common yardstick for any accelerated
topology-optimisation method**. Each method is run against a set of test problems
and compared to a full-FEM optimisation on two axes:

- **Cost** — `Ratio of FEM iterations (Hybrid / full-FEM)`: how many expensive FEM
  solves the accelerated method still needs, relative to pure FEM. Lower = faster.
- **Quality** — `Relative compliance error`: how far the accelerated design's
  compliance is from the full-FEM optimum. Lower = more accurate.

The ideal method sits in the bottom-left corner (few FEM iterations, low error).

| Script | Compares | Output CSV |
|---|---|---|
| `TopOpt_benchmark_hybrid_strategy.py` | hybrid `RULE` strategies (Only UNet, `n Unet - m FEM`, Decreasing compliance, …) for a fixed model | `results/benchmark_hybrid_strategy.csv` |
| `TopOpt_benchmark_architecture.py` | network configurations (NIF, N_conv, CBAM, augmentation, dataset portion, …) | `results/benchmark_architecture.csv` |
| `TopOpt_benchmark_model.py` | full methods head-to-head, incl. the **energy-distance warm start** | `results/benchmark_model.csv` |

Each CSV row is one `(method, test problem)` pair, storing the configuration plus
the two metrics above (and the raw FEM-iteration counts), so results can be
aggregated per method. New accelerated methods only need to report these same two
numbers to be directly comparable. (`training_benchmark.py` and
`benchmark_smape_architecture.csv` additionally log the pure stress-prediction
accuracy — sMAPE — per architecture.)

---

## Installation

```bash
conda create -n env_stage python=3.11
conda activate env_stage
pip install torch torchvision scipy matplotlib tensorboard
# MATLAB Engine for Python (required for the optimisation loop / FEM):
#   in your MATLAB install: cd extern/engines/python && python setup.py install
```

---

## Training

Edit `user` / `name_file` in `main.py`, adapt the `BASE` path to your folder, then:

```bash
python main.py
```

Monitor training:

```bash
tensorboard --logdir results/runs
```

---

## Loss function

Symmetric Mean Absolute Percentage Error (sMAPE):

$$\mathcal{L} = \frac{1}{N} \sum_i \frac{2|\sigma_i - \hat{\sigma}_i|}{|\sigma_i| + |\hat{\sigma}_i| + \varepsilon}$$

---

## Requirements

- Python 3.11, PyTorch, NumPy, SciPy, Matplotlib, TensorBoard
- MATLAB Engine for Python (topology-optimisation loop / FEM)

---

## Acknowledgements

The MATLAB codes used for finite-element computation and topology optimisation were
provided by the [I2MB laboratory](https://i2mb.upv.es/).

### AI & Tools Assistance

Portions of this codebase were developed with the assistance of:

- **Claude** (Anthropic) — generating portions of PyTorch code, debugging the
  data-augmentation / traction conventions, harmonising comments and docstrings,
  and resolving Git issues.
- **Grammarly** — spelling and grammar correction in documentation.
