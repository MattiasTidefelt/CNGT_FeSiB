#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:24:33 2025

@author: ag0406
"""

#%%
from scipy.stats import qmc
import numpy as np
import h5py
from os import makedirs
from os.path import exists

# Htype = "Heating"
# Htype = "Quenching"
# Htype = "AM"
Htype = "Iso"

DirPathOut = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/"

if not exists(DirPathOut):
    makedirs(DirPathOut)

# bounds = {
#     'Fe': (0.7, 0.90),
#     'Si': (0.01, 0.24),
#     'B' : (0.01, 0.24),
# }

bounds = {
    'Fe': (0.7, 0.98),
    'Si': (0.01, 0.29),
    'B' : (0.01, 0.29),
}

# n_samples = 32                         # samples per heating-rate slice

n_samples = 128                         # samples per heating-rate slice

if Htype == "Heating":
    # nHRs = 7
    # HRs = 10**np.linspace(3,9,nHRs)
    nHRs = 6
    HRs = 10**np.linspace(np.log10(5e6), np.log10(1e8), nHRs)
elif Htype == "Quenching":
    # nHRs = 5
    # HRs = -10**np.linspace(4,9,nHRs)
    nHRs = 6
    HRs = -10**np.linspace(4,7,nHRs)
elif Htype == "AM":
    # nHRs = 5
    # HRs = 10**np.linspace(7,10,nHRs)
    nHRs = 6
    HRs = 10**np.linspace(np.log10(5e7), np.log10(1e9), nHRs)
elif Htype == "Iso":
    nHRs = 1
    HRs = np.linspace(1000+273,1100+273,1)

# n_samples = 128  # n = 2^x

# # nHRs = 3
# # HRs = np.linspace(50,200,nHRs)

# # n_samples = 32  # n = x**2

# CompsAndHR = np.zeros([n_samples*nHRs,4]) 

# lb = np.array([0.7, 0.01, 0.01])        # lower bounds for matrix composition

# # Remaining amount after accounting for lower bounds
# remaining = 1.0 - np.sum(lb)

# if remaining <= 0:
#     raise ValueError("Lower bounds are too restrictive to sum to 1!")

# for j in range(nHRs):
#     # Create a Sobol sampler for 3 components
#     sampler = qmc.Sobol(d=3, scramble=True)
#     raw_samples = sampler.random(n=n_samples)
    
#     # Normalize to sum to 1 (simplex)
#     normed = raw_samples / raw_samples.sum(axis=1, keepdims=True)
    
#     # Scale to remaining "free" composition
#     scaled = normed * remaining
    
#     # Add back the lower bounds
#     compositions = scaled + lb
    
#     # Check validity
#     mask = np.all(compositions >= lb, axis=1) & np.isclose(compositions.sum(axis=1), 1.0)
#     compositions = compositions[mask]
    
#     CompsAndHR[n_samples*j:n_samples*(j+1),:3] = compositions 
#     CompsAndHR[n_samples*j:n_samples*(j+1),3] = HRs[j]

#     print("Compositions:")
#     print(compositions)
#     print("Row sums:", compositions.sum(axis=1))

# with h5py.File(f'{DirPathOut}/CompData_{Htype}.hdf5', 'w') as fh:
#     fh.create_dataset("CompData", data=CompsAndHR)


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# -------- Parameters (edit these as needed) --------
seed = 42

# Control how large a candidate pool we build before thinning.
# Larger pools -> more even spacing but more compute.
pool_min_size = max(5000, 50 * n_samples)   # number of ACCEPTED candidates before thinning

# ==============================
# Uniform sampling + Poisson-like thinning (farthest-point sampling)
# ==============================
def sample_uniform_simplex_truncated_pool(min_n, bounds, seed=None):
    """Collect at least min_n uniform points from simplex within box bounds using rejection.
    Returns (points[min_n_or_more,3], accept_ratio).
    """
    rng = np.random.default_rng(seed)
    accepted = []
    total_proposed = 0
    # adapt batch size to reach target fairly quickly
    batch = 50000
    while (sum(a.shape[0] for a in accepted) < min_n):
        x = rng.dirichlet([1.0, 1.0, 1.0], size=batch)  # Fe, Si, B
        total_proposed += x.shape[0]
        mask = (
            (x[:, 0] >= bounds['Fe'][0]) & (x[:, 0] <= bounds['Fe'][1]) &
            (x[:, 1] >= bounds['Si'][0]) & (x[:, 1] <= bounds['Si'][1]) &
            (x[:, 2] >= bounds['B'][0])  & (x[:, 2] <= bounds['B'][1])
        )
        y = x[mask]
        if y.size:
            accepted.append(y)
        # safety
        if total_proposed > 20_000_000 and sum(a.shape[0] for a in accepted) == 0:
            raise RuntimeError("No feasible points found — please loosen bounds.")
    pts = np.vstack(accepted)
    accept_ratio = pts.shape[0] / total_proposed
    return pts, accept_ratio

# Ternary plotting helpers
SQ3_2 = np.sqrt(3.0) / 2.0
V_Fe = np.array([0.0, 0.0])
V_Si = np.array([1.0, 0.0])
V_B  = np.array([0.5, SQ3_2])

def barycentric_to_cart(fe, si, b):
    return fe * V_Fe + si * V_Si + b * V_B

# Farthest-Point Sampling (greedy k-center) on 2D projected points
def farthest_point_sampling(xy, k, rng=None):
    """Select k indices from xy (N,2) by greedy farthest-point sampling.
    Returns array of selected indices.
    """
    N = xy.shape[0]
    if k >= N:
        return np.arange(N)
    if rng is None:
        rng = np.random.default_rng()
    # Start from a random seed point, then iteratively add farthest point
    start = rng.integers(0, N)
    selected = [start]
    # Maintain distance to nearest selected point for each candidate
    d2 = np.sum((xy - xy[start])**2, axis=1)
    for _ in range(1, k):
        # pick farthest remaining index
        idx = int(np.argmax(d2))
        selected.append(idx)
        # update distances
        d2 = np.minimum(d2, np.sum((xy - xy[idx])**2, axis=1))
    return np.array(selected, dtype=int)

# ------------------------------
# Build candidate pool then thin
# ------------------------------
rng = np.random.default_rng(seed)
pool, acc = sample_uniform_simplex_truncated_pool(pool_min_size, bounds, seed=seed)
# Project to 2D for spacing
xy_pool = np.column_stack([barycentric_to_cart(pool[i,0], pool[i,1], pool[i,2]) for i in range(pool.shape[0])]).T

sel_idx = farthest_point_sampling(xy_pool, n_samples, rng=rng)
compositions = pool[sel_idx]

# --------------
# Assemble output
# --------------
CompsAndHR = np.zeros((n_samples * len(HRs), 4), float)
for j, hr in enumerate(HRs):
    sl = slice(j*n_samples, (j+1)*n_samples)
    CompsAndHR[sl, :3] = compositions
    CompsAndHR[sl, 3] = hr

# --------------
# Plot for sanity
# --------------
fig, ax = plt.subplots(figsize=(6.8, 6.2))
tri = np.array([V_Fe, V_Si, V_B])
ax.add_patch(Polygon(tri, closed=True, fill=False, edgecolor='black', lw=1.5))

# Optional: show faded candidate pool to illustrate thinning effect
ax.scatter(xy_pool[:,0], xy_pool[:,1], s=6, c='lightgray', alpha=0.35, label=f'Pool (n={xy_pool.shape[0]})')

xy_sel = np.column_stack([barycentric_to_cart(*row) for row in compositions]).T
ax.scatter(xy_sel[:,0], xy_sel[:,1], s=22, c='tab:blue', alpha=0.95, label=f'Selected (n={n_samples})')

# Draw bound lines (lower & upper for each component)
colors = {'Fe': 'tab:red', 'Si': 'tab:green', 'B': 'tab:purple'}

def const_component_segment(component, value):
    t = 1.0 - value
    if component == 'Fe':
        p1 = barycentric_to_cart(value, 0.0, t)
        p2 = barycentric_to_cart(value, t, 0.0)
    elif component == 'Si':
        p1 = barycentric_to_cart(0.0, value, t)
        p2 = barycentric_to_cart(t, value, 0.0)
    else:  # 'B'
        p1 = barycentric_to_cart(0.0, t, value)
        p2 = barycentric_to_cart(t, 0.0, value)
    return np.vstack([p1, p2])

for comp in ['Fe','Si','B']:
    for v in bounds[comp]:
        seg = const_component_segment(comp, v)
        ax.plot(seg[:,0], seg[:,1], color=colors[comp], lw=1.2, alpha=0.9, label=f"{comp} = {v:.2f}")

ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, SQ3_2 + 0.05)
ax.axis('off')
ax.text(*(V_Fe + np.array([-0.04, -0.04])), 'Fe', ha='right', va='top', fontsize=11)
ax.text(*(V_Si + np.array([0.04, -0.04])), 'Si', ha='left', va='top', fontsize=11)
ax.text(*(V_B  + np.array([0.0,  0.04])),  'B',  ha='center', va='bottom', fontsize=11)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(), loc='upper right', frameon=True)

fig.suptitle('Uniform → farthest-point (Poisson-like) thinning in bounded ternary (Fe–Si–B)')
fig.tight_layout()

print({
    'candidate_pool_size': int(xy_pool.shape[0]),
    'acceptance_ratio_estimate': float(np.round(acc, 5)),
    'selected': int(compositions.shape[0]),
    'HRs': HRs.tolist()
})

plt.show()



with h5py.File(f'{DirPathOut}/CompData_{Htype}.hdf5', 'w') as fh:
    fh.create_dataset("CompData", data=CompsAndHR)
    


    
print(f'# {nHRs*n_samples} compositions have been created') 
    