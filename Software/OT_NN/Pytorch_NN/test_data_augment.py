#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
# ============================================================================
# 1. Round-trip FEM: FEM(augmented inputs) ≈ augmented FEM(original inputs)
# ============================================================================

def check_FEM_roundtrip(sample, augment_fn, transform_name, eng, tol=1e-2):
    """
    Verify that applying augmentation to inputs and then running FEM gives
    the same result as running FEM first and then augmenting the output.
    """
    # Path A: augment inputs first, then FEM
    aug_sample = augment_fn(sample.copy())
    stress_FEM_aug = predict_stress_FEM(eng, aug_sample)  # (NumEls, 6)

    # Path B: FEM first (already in sample.FEM_Stress), then augment outputs
    aug_after_FEM = augment_fn(sample.copy())
    stress_FEM_after_aug = aug_after_FEM.FEM_Stress.numpy()

    # Compare only the components actually transformed: σx, σy, τxy (cols 0, 1, 3)
    cols = [0, 1, 3]
    diff = np.abs(stress_FEM_aug[:, cols] - stress_FEM_after_aug[:, cols])
    ref  = np.abs(stress_FEM_after_aug[:, cols]).max() + 1e-9
    rel_diff = diff.max() / ref

    passed = rel_diff < tol
    print(f"[{transform_name}] FEM round-trip: max relative diff = {rel_diff:.2e}  → {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# 2. Inverse round-trip: g⁻¹(g(x)) = x
# ============================================================================

def check_inverse_roundtrip(sample, augment_fn, inverse_fn, transform_name, tol=1e-2):
    """
    Verify that applying a transformation and then its inverse returns the
    original sample.
    """
    transformed = augment_fn(sample.copy())
    restored    = inverse_fn(transformed)

    diff_rho       = (restored.Densities - sample.Densities).abs().max().item()
    diff_tractions = (restored.Tractions - sample.Tractions).abs().max().item()
    diff_stress    = (restored.FEM_Stress - sample.FEM_Stress).abs().max().item()

    max_diff = max(diff_rho, diff_tractions, diff_stress)
    passed = max_diff < tol
    print(f"[{transform_name}] Inverse round-trip: max abs diff = {max_diff:.2e}  → {'PASS' if passed else 'FAIL'}")
    print(f"    rho: {diff_rho:.2e}  tractions: {diff_tractions:.2e}  stress: {diff_stress:.2e}")
    return passed


# ============================================================================
# 3. Visual inspection
# ============================================================================

def visual_inspection(sample, augmentations, scale_force=10):
    """
    Plot the original sample and all its augmented versions side-by-side
    for visual inspection of consistency.
    """
    n = len(augmentations) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))

    def plot_sample(ax, s, title):
        topo = s.Densities.squeeze().numpy()
        img_size = int(np.sqrt(len(topo)))
        img = topo.reshape(img_size, img_size)

        cadre = int(scale_force)
        ax.imshow(img, cmap='gray_r', origin='lower',
                  extent=[0, img_size, 0, img_size], vmin=0, vmax=1)
        ax.set_xlim(-cadre, img_size + cadre)
        ax.set_ylim(-cadre, img_size + cadre)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12)

        T_scale = s.Tractions.squeeze().numpy() * scale_force
        T_scale = T_scale.T

        Points = np.array([
            [0, img_size], [img_size, img_size],
            [img_size, img_size], [img_size, 0],
            [img_size, 0], [0, 0],
            [0, 0], [0, img_size],
        ], dtype=float)

        for k in range(8):
            sx, sy = Points[k]
            tx, ty = T_scale[k]
            ax.quiver(sx, sy, tx, ty, angles='xy', scale_units='xy', scale=1,
                      color='r', linewidth=1, headwidth=2)

    plot_sample(axes[0], sample, 'Original')
    for i, (name, fn) in enumerate(augmentations):
        plot_sample(axes[i+1], fn(sample.copy()), name)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Main test runner
# ============================================================================

def run_augmentation_tests(sample, eng):
    """
    Run all consistency checks on a given sample.

    Sample must have:
      - Densities (1, 1024)
      - Tractions (1, 2, 8)
      - FEM_Stress (1024, 6) — must be precomputed
    """
    augmentations = [
        ('Rot 90°',   lambda s: rotation_90(s, N_rot=1)),
        ('Rot 180°',  lambda s: rotation_90(s, N_rot=2)),
        ('Rot 270°',  lambda s: rotation_90(s, N_rot=3)),
        ('Flip X',    symmetry_x),
        ('Flip Y',    symmetry_y),
    ]
    inverses = [
        ('Rot 90°',   lambda s: rotation_90(s, N_rot=3)),  # inverse of 90° is 270°
        ('Rot 180°',  lambda s: rotation_90(s, N_rot=2)),  # self-inverse
        ('Rot 270°',  lambda s: rotation_90(s, N_rot=1)),  # inverse of 270° is 90°
        ('Flip X',    symmetry_x),  # self-inverse
        ('Flip Y',    symmetry_y),  # self-inverse
    ]

    print("=" * 60)
    print("Test 1: FEM round-trip — FEM(aug(x)) ≈ aug(FEM(x))")
    print("=" * 60)
    for name, fn in augmentations:
        check_FEM_roundtrip(sample, fn, name, eng)

    print()
    print("=" * 60)
    print("Test 2: Inverse round-trip — g⁻¹(g(x)) = x")
    print("=" * 60)
    for (name, fn), (_, inv_fn) in zip(augmentations, inverses):
        check_inverse_roundtrip(sample, fn, inv_fn, name)

    print()
    print("=" * 60)
    print("Test 3: Visual inspection")
    print("=" * 60)
    visual_inspection(sample, augmentations)


# Usage example:
# sample = IterationSample(ds_iter, 0)
# run_augmentation_tests(sample, eng)



#%% Test

if __name__ == '__main__':
    path = (Path.cwd().parents[2] / 'HeavyFiles/data/dataset_test.mat').resolve()

    data = load_mat(path)
    dataset = Dataset_TopOpt(data)

    data_iter = IterationDataset(dataset)

    sample = IterationSample(data_iter, 30)
    
    print('original sample')
    sample.plot()
    sample.plot_inputs()
    sample.plot_outputs('FEM')

    # new_sample=symmetry_y(sample)
    new_sample = rotation_90(sample, 1)

    print('new sample')   
    new_sample.plot()
    new_sample.plot_inputs()
    new_sample.plot_outputs('FEM')