#%% data_augmentation.py
import torch
import numpy as np

IMG_SIZE = 32

# ── Permutations des 8 nœuds de bordure pour chaque opération élémentaire ──────
# Ordre des nœuds (cf. get_traction_distribution) :
#   side1(haut) 0->1 | side2(droite) 2->3 | side3(bas) 4->5 | side4(gauche) 6->7
# ⚠️ À VÉRIFIER avec le test FEM (round-trip).
PERM = {
    'rot90' : [6, 7, 0, 1, 2, 3, 4, 5],
    'flip_h': [1, 0, 7, 6, 5, 4, 3, 2],
    'flip_v': [5, 4, 3, 2, 1, 0, 7, 6],
}


def apply_op_maps(rho, sx, sy, txy, op):
    """rho, sx, sy, txy : [H, W]. Retourne les cartes transformees + remapping."""
    if op == 'rot90':                       # rotation 90 deg (sens trigo)
        rho = torch.rot90(rho, 1, dims=(0, 1))
        sx  = torch.rot90(sx,  1, dims=(0, 1))
        sy  = torch.rot90(sy,  1, dims=(0, 1))
        txy = torch.rot90(txy, 1, dims=(0, 1))
        return rho, sy, sx, -txy            # sx'=sy, sy'=sx, txy'=-txy

    if op == 'flip_h':                      # miroir gauche-droite (colonnes)
        rho = torch.flip(rho, dims=(1,))
        sx  = torch.flip(sx,  dims=(1,))
        sy  = torch.flip(sy,  dims=(1,))
        txy = torch.flip(txy, dims=(1,))
        return rho, sx, sy, -txy

    if op == 'flip_v':                      # miroir haut-bas (lignes)
        rho = torch.flip(rho, dims=(0,))
        sx  = torch.flip(sx,  dims=(0,))
        sy  = torch.flip(sy,  dims=(0,))
        txy = torch.flip(txy, dims=(0,))
        return rho, sx, sy, -txy


def apply_op_tractions(T, op):
    """T : [2, 8] (ligne 0 = tx, ligne 1 = ty)."""
    perm   = PERM[op]
    T      = T[:, perm]                     # reindexation des noeuds
    tx, ty = T[0].clone(), T[1].clone()
    if op == 'rot90':
        return torch.stack([ty,  -tx])
    if op == 'flip_h':
        return torch.stack([-tx,  ty])
    if op == 'flip_v':
        return torch.stack([ tx, -ty])


def random_ops():
    """
    Tire une sequence d'operations aleatoire :
      - une rotation de 90, 180 ou 270 deg  (soit 1, 2 ou 3 applications de rot90)
      - ou une symetrie (horizontale ou verticale)
      - ou les deux combinees
    """
    choice = np.random.choice(['rot', 'flip', 'both'])
    ops = []

    if choice in ('flip', 'both'):
        ops.append(np.random.choice(['flip_h', 'flip_v']))

    if choice in ('rot', 'both'):
        k = np.random.randint(1, 4)         # 1->90, 2->180, 3->270
        ops += ['rot90'] * k

    return ops


def augment_one(batch, b, ops):
    """Applique la sequence d'operations au sample b du batch (in place)."""
    rho = batch['Densities'][b].squeeze().reshape(IMG_SIZE, IMG_SIZE)
    S   = batch['Stress'][b]                          # [1024, 6]
    sx  = S[:, 0].reshape(IMG_SIZE, IMG_SIZE)
    sy  = S[:, 1].reshape(IMG_SIZE, IMG_SIZE)
    txy = S[:, 3].reshape(IMG_SIZE, IMG_SIZE)
    T   = batch['Tractions'][b].squeeze()             # [2, 8]

    for op in ops:
        rho, sx, sy, txy = apply_op_maps(rho, sx, sy, txy, op)
        T = apply_op_tractions(T, op)

    batch['Densities'][b] = rho.reshape(-1).unsqueeze(0)
    S_new = S.clone()
    S_new[:, 0] = sx.reshape(-1)
    S_new[:, 1] = sy.reshape(-1)
    S_new[:, 3] = txy.reshape(-1)
    batch['Stress'][b]    = S_new
    batch['Tractions'][b] = T.unsqueeze(0)


def Data_Augmentation(batch: dict, p: float = 0.2) -> dict:
    """
    Augmente chaque sample du batch independamment avec probabilite p.

    Transformations possibles (combinees) : rotation de 90/180/270 deg,
    symetrie horizontale ou verticale, ou les deux.
    Les composantes du tenseur de contraintes (sx, sy, txy) et les vecteurs
    de traction (tx, ty) sont transformes de facon physiquement coherente.

    A appeler UNIQUEMENT sur le train, jamais sur la validation.
    """
    B = batch['Densities'].shape[0]
    for b in range(B):
        if torch.rand(1).item() < p:
            ops = random_ops()
            if ops:
                augment_one(batch, b, ops)
    return batch
