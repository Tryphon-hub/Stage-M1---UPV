#%% Verify data augmentation against the FEM
"""
Sanity-check the data augmentation (rotations + symmetries) on a single sample
by comparing it against a genuine finite-element solve.

Golden test — FEM round-trip
----------------------------
For every square symmetry g (the map that transforms density, tractions AND
the stress field together), elasticity is equivariant, so:

        FEM( g(rho, T) )   ==   g( FEM(rho, T) )

    left  side : run the FEM on the *transformed* inputs      -> "FEM truth"
    right side : apply g to the *original* FEM stress field   -> "augmented"

If the augmentation maps the stress correctly (node permutations, tx/ty sign
flips, sigma_x/sigma_y swap on odd rotations, tau_xy sign change, rotation
direction), the two must match to solver precision. Any visible discrepancy
localises a bug in the corresponding transform.

Run cell-by-cell (#%%) or as a script:
    python verify_augmentation_FEM.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matlab.engine

BASE = Path(__file__).parents[3]

sys.path.append(str(BASE / 'Software' / 'OT_NN' / 'Pytorch_NN'))
sys.path.append(str(BASE / 'Software' / 'OT_Functions'))
sys.path.append(str(BASE / 'Software' / 'OT_Software'))

from dataset import (
    load_mat, Dataset_TopOpt, IterationDataset, IterationSample,
    rotation_90, symmetry_x, symmetry_y,
)
from topology_utils import predict_stress_FEM


#%% Constants / paths
IMG_SIZE = 32
PENAL    = 3
NGPpS    = 9
NGPpL    = 2
E        = 1000
NU       = 0.3

DATA_NAME  = 'dataset_test'          # dataset to draw the sample from
SAMPLE_IDX = 30                      # global index in the IterationDataset
TOL        = 1e-2                    # max relative error to call a transform PASS

GMSH_EXE  = BASE.parent / 'gmsh' / 'gmsh.exe'
DATA_PATH = BASE / 'HeavyFiles' / 'data' / (DATA_NAME + '.mat')
GEO_FILE  = BASE / 'Software' / 'OT_Software' / 'Square.geo'
MESH_FILE = BASE / 'Software' / 'OT_Software' / 'Square.msh'


#%% Start MATLAB engine, regenerate the mesh and build the Hooke matrix
print("Starting MATLAB engine ...")
eng = matlab.engine.start_matlab()
eng.addpath(str(BASE / 'Software' / 'OT_Functions'))
eng.addpath(str(BASE / 'Software' / 'OT_Software'))

eng.workspace['GmshExe']     = str(GMSH_EXE)
eng.workspace['GeoFileName'] = str(GEO_FILE)
eng.workspace['Mesh_File']   = str(MESH_FILE)

print(f"GmshExe   exists: {GMSH_EXE.exists()}")
print(f"GeoFile   exists: {GEO_FILE.exists()}")

eng.eval(rf"""
if isfile(Mesh_File)
    delete(Mesh_File)
end
CallString = ['"' GmshExe '" "' GeoFileName '" -setnumber numLayers {IMG_SIZE} -o "' Mesh_File '" -'];
status = system(CallString);
disp(['system() status: ' num2str(status)])
[MeshData] = ReadGMSH(Mesh_File);
""", nargout=0)

print(f"NumEls after regen: {eng.eval('length(MeshData.Surf.Elements)')}")
eng.eval(f"D = DHooks2D({E}, {NU}, 'Plane Stress');", nargout=0)


#%% Load a sample and compute its reference FEM stress
data      = load_mat(DATA_PATH)
ds_base   = Dataset_TopOpt(data)
data_iter = IterationDataset(ds_base)

sample = IterationSample(data_iter, SAMPLE_IDX)
print(f"\nLoaded sample #{SAMPLE_IDX} from {DATA_NAME}.mat")
print(sample)

# Recompute the reference stress with THIS solver so the round-trip stays
# internally consistent (independent of whatever was stored in the .mat).
predict_stress_FEM(eng, sample)   # fills sample.FEM_Stress -> ground-truth base

# Sorted per-element stress magnitude of the untransformed problem. Any faithful
# symmetry of the inputs must reproduce this exact multiset after a fresh solve.
BASE_MAG = np.sort(np.sqrt(
    sample.FEM_Stress.numpy()[:, 0]**2
    + sample.FEM_Stress.numpy()[:, 1]**2
    + sample.FEM_Stress.numpy()[:, 3]**2
))


#%% Helpers
COLS  = [0, 1, 3]                       # sigma_x, sigma_y, tau_xy
NAMES = ['sigma_x', 'sigma_y', 'tau_xy']


def relative_error(pred, truth):
    """Max abs component-wise diff on (sigma_x, sigma_y, tau_xy), normalised."""
    diff = np.abs(pred[:, COLS] - truth[:, COLS])
    ref  = np.abs(truth[:, COLS]).max() + 1e-12
    return diff.max() / ref


def per_component_error(pred, truth):
    """Relative error for each of sigma_x, sigma_y, tau_xy separately."""
    ref = np.abs(truth[:, COLS]).max() + 1e-12
    return {NAMES[k]: np.abs(pred[:, c] - truth[:, c]).max() / ref
            for k, c in enumerate(COLS)}


def stress_magnitude(stress):
    """Per-element magnitude sqrt(sigma_x^2 + sigma_y^2 + tau_xy^2), sorted.

    Invariant (as a multiset) under any D4 symmetry of the *inputs* alone,
    regardless of how the stress components are relabelled. Lets us test the
    density+traction transform independently of the stress transform.
    """
    m = np.sqrt(stress[:, 0]**2 + stress[:, 1]**2 + stress[:, 3]**2)
    return np.sort(m)


def fem_roundtrip(sample, transform, name):
    """
    Compare g(FEM(x))  vs  FEM(g(x)) for one transform.

    Returns dict with augmented stress, FEM-truth stress, and error metrics.
    """
    # right side: g applied to the reference stress (what augmentation claims)
    aug        = transform(sample.copy())
    stress_aug = aug.FEM_Stress.numpy()          # (NumEls, 6)

    # left side: a real FEM solve on the transformed density + tractions
    aug_fem = aug.copy()
    predict_stress_FEM(eng, aug_fem)             # overwrites aug_fem.FEM_Stress
    stress_fem = aug_fem.FEM_Stress.numpy()      # (NumEls, 6)

    err      = relative_error(stress_aug, stress_fem)
    per_comp = per_component_error(stress_aug, stress_fem)

    # input-transform check: |sigma| multiset must match the untransformed base
    mag_err = np.abs(stress_magnitude(stress_fem) - BASE_MAG).max() / (BASE_MAG.max() + 1e-12)

    status = 'PASS' if err < TOL else 'FAIL'
    comp_str = "  ".join(f"{k}={v:.2e}" for k, v in per_comp.items())
    print(f"[{name:9s}] round-trip={err:.2e} ({status})   "
          f"input(|sigma|)-err={mag_err:.2e}   [{comp_str}]")
    return {'aug': stress_aug, 'fem': stress_fem,
            'err': err, 'per_comp': per_comp, 'mag_err': mag_err}


def plot_comparison(stress_aug, stress_fem, name):
    """3x3 grid: rows = sigma_x/sigma_y/tau_xy, cols = augmented / FEM / diff."""
    n = IMG_SIZE
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for r, c in enumerate(COLS):
        a = stress_aug[:, c].reshape(n, n)
        f = stress_fem[:, c].reshape(n, n)
        d = a - f
        vmax = max(np.abs(a).max(), np.abs(f).max(), 1e-12)
        for col, (img, lim, ttl) in enumerate([
            (a, vmax,          f'{NAMES[r]}  augmented'),
            (f, vmax,          f'{NAMES[r]}  FEM truth'),
            (d, np.abs(d).max() + 1e-12, f'{NAMES[r]}  diff'),
        ]):
            ax = axes[r, col]
            im = ax.imshow(img, cmap='RdBu', origin='lower', vmin=-lim, vmax=lim)
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Augmentation "{name}"  —  augmented vs FEM(transformed inputs)',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


#%% Run the round-trip test for every transform
transforms = [
    # Identity control: k=0 must round-trip to ~0. If it FAILS, the harness /
    # solver / convention is at fault, NOT the augmentation transforms.
    ('Identity', lambda s: rotation_90(s, N_rot=0)),
    ('Rot 90',   lambda s: rotation_90(s, N_rot=1)),
    ('Rot 180',  lambda s: rotation_90(s, N_rot=2)),
    ('Rot 270',  lambda s: rotation_90(s, N_rot=3)),
    ('Flip X',   symmetry_x),
    ('Flip Y',   symmetry_y),
]

print("\n" + "=" * 78)
print("FEM round-trip:  g(FEM(x)) vs FEM(g(x))   +   input-transform |sigma| check")
print("=" * 78)

results = {}
for name, fn in transforms:
    results[name] = fem_roundtrip(sample, fn, name)


#%% Localise the bug from the two independent signals
print("\n" + "=" * 78)
print("Localisation")
print("=" * 78)

identity_ok = results['Identity']['err'] < TOL
if not identity_ok:
    print("Identity control FAILED -> the test harness / FEM solve / element\n"
          "ordering is inconsistent (re-solving the SAME inputs already differs).\n"
          "Fix the harness before trusting the transform verdicts below.")
else:
    print("Identity control PASSED -> harness is sound; failures below are real\n"
          "augmentation bugs.\n")
    for name, r in results.items():
        if name == 'Identity':
            continue
        input_ok = r['mag_err'] < TOL          # density+tractions transform faithful?
        stress_ok = r['err'] < TOL             # full round-trip
        if not input_ok:
            verdict = "INPUT transform wrong (density and/or tractions map)"
        elif not stress_ok:
            verdict = "STRESS transform wrong (sigma_x/sigma_y swap, tau sign, rot dir)"
        else:
            verdict = "OK"
        print(f"  {name:9s}: {verdict}")


#%% FEM-free isolation: which D4 element does each field actually realise?
#
# For a faithful augmentation the density IMAGE and the traction VECTOR FIELD
# must both correspond to the *same* element g of the dihedral group D4. Here we
# search all 8 elements independently for density and for tractions; if the two
# best matches disagree, that IS the bug (e.g. density rotates one way, tractions
# the other). No MATLAB needed, so this iterates fast while fixing dataset.py.

def _rot_matrix(r):
    c, s = int(round(np.cos(r * np.pi / 2))), int(round(np.sin(r * np.pi / 2)))
    return np.array([[c, -s], [s, c]], dtype=float)

_FLIPX = np.array([[-1., 0.], [0., 1.]])   # mirror across the vertical axis

def _d4_ops():
    """Yield (label, scalar_op, vector_matrix) for the 8 elements of D4."""
    for f in (0, 1):
        for r in (0, 1, 2, 3):
            label = (f"rot{r*90}" if not f else f"flipx.rot{r*90}")
            def scalar_op(img, r=r, f=f):
                out = np.fliplr(img) if f else img
                return np.rot90(out, r)
            M = _rot_matrix(r) @ (_FLIPX if f else np.eye(2))
            yield label, scalar_op, M

def _best_scalar_match(target, base):
    best = (None, np.inf)
    ref_max = np.abs(base).max() + 1e-12
    for label, op, _ in _d4_ops():
        err = np.abs(op(base) - target).max() / ref_max
        if err < best[1]:
            best = (label, err)
    return best

def _best_vector_match(tx_t, ty_t, tx0, ty0):
    best = (None, np.inf)
    ref_max = max(np.abs(tx0).max(), np.abs(ty0).max()) + 1e-12
    for label, op, M in _d4_ops():
        sx, sy = op(tx0), op(ty0)
        tx_ref = M[0, 0] * sx + M[0, 1] * sy
        ty_ref = M[1, 0] * sx + M[1, 1] * sy
        err = max(np.abs(tx_ref - tx_t).max(), np.abs(ty_ref - ty_t).max()) / ref_max
        if err < best[1]:
            best = (label, err)
    return best

rho0        = sample.Densities.squeeze().numpy().reshape(IMG_SIZE, IMG_SIZE)
tx0, ty0    = sample.get_traction_distribution()

print("\n" + "=" * 78)
print("FEM-free D4 isolation:  which element does density vs tractions realise?")
print("=" * 78)
for name, fn in transforms:
    if name == 'Identity':
        continue
    aug   = fn(sample.copy())
    rho_a = aug.Densities.squeeze().numpy().reshape(IMG_SIZE, IMG_SIZE)
    tx_a, ty_a = aug.get_traction_distribution()

    d_lbl, d_err = _best_scalar_match(rho_a, rho0)
    t_lbl, t_err = _best_vector_match(tx_a, ty_a, tx0, ty0)
    agree = (d_lbl == t_lbl) and d_err < TOL and t_err < TOL
    flag  = "OK" if agree else "<-- MISMATCH"
    print(f"  {name:9s}: density -> {d_lbl:14s}(err {d_err:.1e})   "
          f"tractions -> {t_lbl:14s}(err {t_err:.1e})   {flag}")


#%% Visual inspection (one figure per transform)
for name, r in results.items():
    plot_comparison(r['aug'], r['fem'], name)


#%% Optional: also verify the INPUTS visually (density + tractions transform)
for name, fn in transforms:
    aug = fn(sample.copy())
    print(f"\n--- inputs after '{name}' ---")
    aug.plot_inputs(TITLE=name)
# %%
