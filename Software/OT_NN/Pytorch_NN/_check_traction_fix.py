#%%
"""Pure-Python sanity check of the fixed traction transforms (no MATLAB needed)."""
import numpy as np
import torch
import dataset as ds

tt = ds.transform_tractions
Rx, Ry, Rr = ds._R_FLIP_X, ds._R_FLIP_Y, ds._R_rot

T = torch.randn(2, 8)

print("flipX involution :", torch.allclose(tt(tt(T, Rx), Rx), T, atol=1e-5))
print("flipY involution :", torch.allclose(tt(tt(T, Ry), Ry), T, atol=1e-5))

T4 = T.clone()
for _ in range(4):
    T4 = tt(T4, Rr(1))
print("rot90 x4 = id    :", torch.allclose(T4, T, atol=1e-5))
print("rot180==rot90^2  :", torch.allclose(tt(T, Rr(2)), tt(tt(T, Rr(1)), Rr(1)), atol=1e-5))

for name, R in [("flipX", Rx), ("flipY", Ry), ("rot1", Rr(1)), ("rot2", Rr(2)), ("rot3", Rr(3))]:
    pi = ds._node_permutation(R)
    print(f"perm {name:6s} bijection:", sorted(pi.tolist()) == list(range(8)), pi.tolist())

# DIRECTION: density uses torch.rot90(img, k_image=(-k)%4); must match _R_rot(k).
N = 4
img = np.arange(N * N).reshape(N, N)   # rho[row, col], origin lower -> x=col, y=row
cen = (N - 1) / 2
for k in (1, 2, 3):
    k_image = (-k) % 4
    rot = torch.rot90(torch.tensor(img), k_image, dims=(0, 1)).numpy()
    r, c = np.argwhere(rot == img[0, 3])[0]          # marker pixel row0,col3 -> (x=3,y=0)
    new_xy = np.array([c - cen, r - cen])
    pred = Rr(k) @ np.array([3 - cen, 0 - cen])
    print(f"k={k}: density {new_xy.tolist()} vs _R_rot {pred.tolist()}  match={np.allclose(new_xy, pred)}")
