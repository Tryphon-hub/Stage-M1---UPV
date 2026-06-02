# Stress Field Prediction for 2D Topology optimisation using U-Net

## Overview

This repository contains the implementation of a U-Net-based neural network for predicting stress fields (σx, σy, τxy) in 2D topology optimisation. Two architectures are provided:

- **UNetTopo** — standard U-Net with boundary conditions (tx, ty) as input channels
- **BE_UNetTopo** — U-Net with a dedicated Boundary Embedding module that encodes nodal traction vectors directly into the bottleneck latent space

Both models are trained on FEM-generated datasets and integrated into a topology optimisation loop as surrogates for finite-element stress computation.

This work uses I2MB MATLAB codes from OT_Functions/ and OT_Software/, which are called by Python files in OT_NN/.

My contributions to this work are implemented in Pytorch_NN/.

---

## Repository Structure
```text
├── Software/OT_NN/Pytorch_NN/
│   ├── model.py              # UNetTopo and BE_UNetTopo architectures
│   ├── train.py              # Training loop (shared U-Net / BE_UNet)
│   ├── evaluate.py           # Evaluation and visualisation
│   ├── dataset.py            # Dataset classes
│   └── main.py               # Entry point
│
├── results/
│   ├── U-Net/
│   │   └── {name_dataset}/
│   │       ├── unet_{name_dataset}_best.pth
│   │       └── unet_{name_dataset}_checkpoint.pth
│   └── BE_Unet/
│       └── {name_dataset}/
│           ├── unet_{name_dataset}_best.pth
│           └── unet_{name_dataset}_checkpoint.pth
│
├── illustrations/
│   ├── U-Net/
│       └── {name_dataset}/
│   └── BE_Unet/
│       └── {name_dataset}/
│
├── topology_utils.py         # Topology optimisation utilities
└── TopOpt_process.py         # Main topology optimisation script
```

---

## Models

### UNetTopo
Standard U-Net with 4 encoder/decoder levels, triple convolutions per block, and optional CBAM attention at the bottleneck.

| Input | Output |
|---|---|
| ρ + tx + ty — `[B, 3, 32, 32]` | σx, σy, τxy — `[B, 3, 32, 32]` |

### BE_UNetTopo
U-Net with a Boundary Embedding module. Traction vectors are encoded via a MLP + ConvTranspose2d and concatenated at the bottleneck before CBAM.

| Input | Output |
|---|---|
| ρ — `[B, 1, 32, 32]` + nodes `[B, 16]` | σx, σy, τxy — `[B, 3, 32, 32]` |

---

## Installation

```bash
conda create -n env_stage python=3.10
conda activate env_stage
pip install torch torchvision scipy matplotlib tensorboard
```

---

## Training

Edit `user` and `name_file` in `main.py`, adapt the BASE string to your folder, then:

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

- Python 3.10
- PyTorch
- MATLAB Engine for Python (for topology optimisation loop)
- SciPy, NumPy, Matplotlib


## Acknowledgements

The MATLAB codes used for finite element computation and topology optimisation
were provided by the [I2MB laboratory](https://i2mb.upv.es/).

### AI & Tools Assistance

Portions of this codebase were developed with the assistance of the following tools:

- **Claude Sonnet** (Anthropic) — used for generating portions of PyTorch code,
  harmonising comments and docstrings across files, and resolving Git issues
- **Grammarly** — used for spelling and grammar correction in documentation
