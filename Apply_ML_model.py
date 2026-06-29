#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:44:50 2025

@author: ag0406

Starting to write final script 20/10-25

Final product of Fe-Si-B paper; do cool things with trained ML model.
Should handle three models: heating, queching, and cyclic temperature history

These models are optimized in ML_interpolation_FeSiB.py script where data generated
from MPI_Driver_NucFeSiB.py is used (randomly sampled compositions from SampleComp.py)

MPI_Driver_NucFeSiB.py uses scripts much similar to NucMulti_FeSiB_NoSolve.py and AM_FeSiB.py

Due to bad installations, by me, scripts using MPI4py, pytorch, or mpltern, have to be executed by terminal.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import mpltern
import json
import joblib
from scipy.special import logit
from scipy.stats import qmc
from scipy.interpolate import LinearNDInterpolator

from paramFeSiB import sysParam

from ReadTempData import initialize_Tspline_AML

from ML_utilities import load_or_generate_Tg_Tm_grid_simplex, load_or_generate_tend_grid_AM_TgTm, plot_ternary_data, Network, ReverseTransform, load_data_Kfold
    
# Tag = "Single"
Tag = "Double"

# Htype = "Heating"
Htype = "Quenching"
# Htype = "AM"
# Htype = "Iso"

# Type = "Uniform"
# Type = "Normal"
Type = "NormalBounded"
# Type = "NormalSigmoid"
# Type = "StandardScaler"
# Type = "Power"

MLdata = load_data_Kfold(Htype,Type)

print(f"--------- {Htype} ---------")

num_samples = 2**18  # Number of samples to generate, was 16

bounds = [
    (0.70, 0.98),  # xFe bounds
    (0.01, 0.29),  # xSi
    (0.01, 0.29)   # xB
]
# bounds = [
#     (0.70, 0.9),  # xFe bounds
#     (0.01, 0.24),  # xSi
#     (0.01, 0.24)   # xB
# ]

# --- User-defined settings ---
top_k = 10

Scale = MLdata["scaler"]
    
# --- Heating rate bounds ---
if Htype == "Heating":
    hr_min, hr_max = np.log(5e6), np.log(1e9)
    # hr_min, hr_max = np.log(10**5), np.log(10**9)
    # hr_min, hr_max = np.log(10**3), np.log(10**8)
elif Htype == "Quenching":
    # hr_min, hr_max = np.log(10**4), np.log(10**7)
    hr_min, hr_max = np.log(10**4), np.log(5e7)
    # hr_min, hr_max = np.log(10**3), np.log(10**7)
elif Htype == "AM":
    hr_min, hr_max = np.log(5e7), np.log(5e9)
    # hr_min, hr_max = np.log(10**6), np.log(10**10)
    # hr_min, hr_max = np.log(10**7), np.log(10**9)
    # hr_min, hr_max = np.log(5*10**7), np.log(10**9)
elif Htype == "Iso":
    hr_min, hr_max = np.log(800), np.log(900)
  
# Ref Rc1
xFe_Ref_1_Q = np.array([0.7, 0.7, 0.725, 0.75, 0.75, 0.75, 0.75, 0.75, 0.775, 0.8])
xSi_Ref_1_Q = np.array([0.15, 0.18, 0.125, 0.05, 0.075, 0.1, 0.125, 0.15, 0.075, 0.05])
xB_Ref_1_Q = np.array([0.15, 0.12, 0.15, 0.2, 0.175, 0.15, 0.125, 0.1, 0.15, 0.15])
Rc_Ref_1_Q = np.array([3e5, 2.1e6, 6.3e4, 1.2e6, 6e5, 3.1e5, 1.9e5, 7.6e6, 5.8e5, 5.8e6])

dx = 1e-2
# Ref Rc2
xFe_Ref_2_Q = np.array([0.89, 0.85, 0.83, 0.8, 0.75, 0.85, 0.85, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.725, 0.725])
xSi_Ref_2_Q = np.array([dx, dx, dx, dx, dx, 0.05, 0.1, 0.15, 0.1, 0.05, 0.2, 0.175, 0.15, 0.1, 0.05, 0.15, 0.1])
xB_Ref_2_Q = np.array([0.11-dx,  0.15-dx, 0.17-dx, 0.2-dx, 0.25-dx, 0.1, 0.05, 0.05, 0.1, 0.15, 0.05, 0.075, 0.1, 0.15, 0.2, 0.125, 0.175])
Rc_Ref_2_Q = np.array([1.7e6, 2.8e5, 8.3e5, 8.8e5, 6.8e6, 1.9e6, 1e6, 8e5, 1.2e4, 9.6e5, 4.4e7, 2.5e7, 4.3e5, 9.8e5, 1.6e7, 7e5, 3.1e6])
########################
########################

target_phase = "bcc"

DirPathIn = f"/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/{Htype}/"

simt = time.time()

def read_colIdx(Htype, filename="colIdx_data.txt"):
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(Htype + ":"):
                colIdx_str = line.strip().split(":")[1]
                return list(map(int, colIdx_str.split(",")))
    return []

##############
x0Blank = np.array([0.8,0.1,0.1])
Parameters=sysParam(x0Blank)
PD=Parameters[0]  

colIdx = read_colIdx(Htype)   
PD = np.delete(PD, colIdx, axis=1)     
##############

# create the network
input_dimension = 3
output_dimension = 4

###################

with open(f"{DirPathIn}best_params.json", "r") as f:
    config = json.load(f)

best_params = config["best_params"]
print("Loaded best params:", best_params)

model = Network(
    input_dimension,
    [best_params[f"n_units_layer{i}"] for i in range(best_params["n_layers"])],
    output_dimension,
    # activation=best_params["activation"],
    activation="leaky_relu",
    dropout=best_params["dropout"]
)

###################
model.load_state_dict(torch.load(f"{DirPathIn}TrainedModel.pth", weights_only=True))
model.eval()

######################## Filter based on conditions
######################## 
######################## 

# --- Phase index mapping ---
TargetIdx = np.where(PD[0,:] == target_phase)[0][0]

lb = [b[0] for b in bounds]  # Fe_min, Si_min, B_min
ub = [b[1] for b in bounds] # Fe_max, Si_max, B_max

#%% ################### Get samples centered

# # Parameters

# # Remaining amount after accounting for lower bounds
# remaining = 1.0 - np.sum(lb)
# if remaining <= 0:
#     raise ValueError("Lower bounds are too restrictive to sum to 1!")

# # Sobol sampler for 3 components
# sampler = qmc.Sobol(d=3, scramble=True)
# raw_samples = sampler.random(n=num_samples)

# # Normalize to sum to 1 (simplex)
# normed = raw_samples / raw_samples.sum(axis=1, keepdims=True)

# # Scale to remaining "free" composition
# scaled = normed * remaining

# # Add back the lower bounds
# compositions = scaled + lb

# # Check validity (should always hold if lb is feasible)
# mask = np.all(compositions >= lb, axis=1) & np.isclose(compositions.sum(axis=1), 1.0)
# Comps = torch.tensor(compositions[mask], dtype=torch.float32)

#%%################## Get samples Uniform

def sample_uniform_simplex_with_box_bounds(
    n,
    lb,
    ub,
    seed=None,
    dtype=torch.float32,
    device=None,
    batch=None,
    return_acceptance=False,
):
    """
    Uniformly sample n compositions x on the simplex with box constraints:
        x >= lb (elementwise), x <= ub (elementwise), sum(x) = 1
    where x = [x_Fe, x_Si, x_B].

    Method:
    - Draw y ~ Dirichlet(1,1,1) (uniform on the full simplex)
    - Reject rows violating any bound lb_i <= y_i <= ub_i
    This preserves exact uniformity on the feasible intersection.

    Args
    ----
    n : int
        Number of accepted samples to return.
    lb : array-like (3,)
        Lower bounds [Fe_min, Si_min, B_min].
    ub : array-like (3,)
        Upper bounds [Fe_max, Si_max, B_max].
    seed : int or None
        RNG seed for reproducibility.
    dtype : torch.dtype
        Output tensor dtype.
    device : torch.device or None
        Target device (e.g., 'cpu' or 'cuda').
    batch : int or None
        Batch size for proposal draws. Defaults to max(10*n, 50_000).
    return_acceptance : bool
        If True, also return acceptance ratio (accepted / proposed).

    Returns
    -------
    comps : torch.Tensor, shape (n, 3)
        Accepted compositions in (Fe, Si, B) order, each row sums to ~1.
    acceptance_ratio : float (optional)
        Estimated acceptance ratio over the last proposal loop.
    """
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)

    if lb.shape != (3,) or ub.shape != (3,):
        raise ValueError("lb and ub must be length-3 arrays for [Fe, Si, B].")
    if not np.all(lb <= ub + 1e-15):
        raise ValueError("Each lower bound must be <= upper bound (lb_i <= ub_i).")
    if np.any(lb < 0.0) or np.any(ub > 1.0):
        raise ValueError("Bounds must lie within [0,1].")

    # Feasibility checks:
    # 1) Sum of lowers must be <= 1, and sum of uppers must be >= 1
    if lb.sum() - 1.0 > 1e-12:
        raise ValueError(f"Infeasible: sum(lb)={lb.sum():.6f} > 1.0")
    if 1.0 - ub.sum() > 1e-12:
        raise ValueError(f"Infeasible: sum(ub)={ub.sum():.6f} < 1.0")

    # 2) Quick constructive feasibility check:
    #    Start from lb, remaining = 1 - sum(lb), see if we can fill up to ub
    remaining = 1.0 - lb.sum()
    capacity = (ub - lb).sum()
    if remaining > capacity + 1e-12:
        raise ValueError(
            "Infeasible: not enough capacity between lb and ub to reach sum=1. "
            f"remaining={remaining:.6f}, capacity={capacity:.6f}"
        )

    rng = np.random.default_rng(seed)
    if batch is None:
        batch = max(10 * n, 50_000)

    accepted = []
    proposed = 0

    while sum(chunk.shape[0] for chunk in accepted) < n:
        # Uniform on simplex via Dirichlet(1,1,1)
        y = rng.dirichlet([1.0, 1.0, 1.0], size=batch)  # rows sum to 1
        mask = np.all((y >= lb - 1e-6) & (y <= ub + 1e-6), axis=1)
        if np.any(mask):
            accepted.append(y[mask])
        proposed += batch

        # Safety valve
        if proposed > 50_000_000 and sum(c.shape[0] for c in accepted) == 0:
            raise RuntimeError("No feasible points found — bounds may be too tight.")

    x = np.vstack(accepted)[:n]
    comps = torch.tensor(x, dtype=dtype, device=device)

    if return_acceptance:
        acc_ratio = x.shape[0] / proposed
        return comps, acc_ratio
    return comps




Comps, acc = sample_uniform_simplex_with_box_bounds(
    num_samples, lb, ub, seed=42, return_acceptance=True
)
#%%############################
    
# Concatenate and trim to desired number of samples
hr_raw = torch.tensor(np.random.uniform(hr_min, hr_max, size=(num_samples, 1)), dtype=torch.float32)
X_samples = torch.cat([Comps, hr_raw], dim=1)

X_samples = torch.hstack((X_samples[:,:2],X_samples[:,-1].reshape(-1,1)))

############### Extracting times, should be used to have time of sim for comparision

############# Tg Tm
X_int, Tg_int, Tm_int = load_or_generate_Tg_Tm_grid_simplex(
    filename="FeSiB_TgTm_grid_simplex_50.npz",
    bounds=bounds,
    PD = PD,
    n_grid=50,                 # 50×50 grid (filtered by bounds)
    independent=("Fe", "Si"),  # treat Fe & Si as independent; B computed
)


X2 = X_int[:, :2]  # (xFe, xSi) since independent=("Fe","Si")

Tg_interp = LinearNDInterpolator(X2, Tg_int)
Tm_interp = LinearNDInterpolator(X2, Tm_int)

Tgs = np.empty(num_samples)
Tms = Tgs.copy()

############# t_end, AM
Q_tend, tend_vals = load_or_generate_tend_grid_AM_TgTm(
    filename="FeSiB_tend_AM_TgTmLogHr.npz",
    Tg_vals=Tg_int,     # from Tg/Tm grid generation (true)
    Tm_vals=Tm_int,     # from Tg/Tm grid generation (true)
    logHr_bounds=(hr_min, hr_max),
    n_hr=50,
    as_grid_hr=True
)

# fig77, ax77 = plt.subplots(1,1)
# ax77.hist(tend_vals,100)
# ax77.set_title("t_ends_vals")

Q = np.hstack((Q_tend[:,0].reshape(-1,1),Q_tend[:,-1].reshape(-1,1)))
tend_interp = LinearNDInterpolator(Q, tend_vals)
# tend_interp = LinearNDInterpolator(Q_tend, tend_vals)

# print(Q_tend)
# print(np.unique(tend_vals))

# print(tend_interp([Tg_int[0], Tm_int[0], np.log(1e8)]))

print("--- Data loaded ---")
#############
nanTic = 0
time_end = np.empty(num_samples)
RealHrs = time_end.copy()
for i in range(num_samples):
    # x_i = X_samples[i,:3].numpy()
    x_i = X_samples[i,:2].numpy()
    if Htype == "Quenching":
        Hr_i = -np.exp(np.abs(X_samples[i,-1].numpy()))
    else:
        Hr_i = np.exp(X_samples[i,-1].numpy())
        
    RealHrs[i] = abs(Hr_i)
        
    Tg = Tg_interp(X_samples[i,:2].numpy())[0]
    Tm = Tm_interp(X_samples[i,:2].numpy())[0]
        
    Tgs[i] = Tg
    Tms[i] = Tm
    if np.isnan(Tg):
        nanTic += 1
        Tgs[i] = Tgs[i-1] 
        Tms[i] = Tms[i-1] 
       
    if Htype == "AM":
        # q = np.array([Tgs[i], Tms[i], np.log(Hr_i)], dtype=float)
        q = np.array([Tgs[i], np.log(Hr_i)], dtype=float)
        t_end = tend_interp(q)[0]

        # Fallback if out of convex hull:
        if np.isnan(t_end):
            t_end = initialize_Tspline_AML(Tgs[i], Tms[i], Hr_i)
            print("Falling back on regular, slow extraction of time")
        
        time_end[i] = t_end

    else:
        if Htype == "Heating":
            time_end[i] = (Tm-Tg)/Hr_i
        elif Htype == "Quenching":
            time_end[i] = (Tm-Tg)/np.abs(Hr_i)
        elif Htype == "Iso":
            time_end[i] = 0.0

# fig66, ax66 = plt.subplots(1,1)
# ax66.hist(time_end,100)
# ax66.set_title("t_ends")

################## Plot Tg/Tm
# fig5 = plt.figure(figsize=(7, 4))

# v = Tgs
# t = X_samples[:, 0]
# l = X_samples[:, 1]
# r = 1-t-l

# ax = fig5.add_subplot(1, 1, 1, projection='ternary')

# cs = ax.tripcolor(t, l, r, v, shading='flat', cmap='viridis')
# ax.set_tlabel('Fe')
# ax.set_llabel('Si')
# ax.set_rlabel('B')

# ax.set_tlim(bounds[0][0], bounds[0][1])
# ax.set_llim(bounds[1][0], bounds[1][1])
# ax.set_rlim(bounds[2][0], bounds[2][1])

# # Colorbar with log ticks
# cax = ax.inset_axes([1.25, 0.1, 0.05, 0.9], transform=ax.transAxes)
# colorbar = fig5.colorbar(cs, cax=cax)
# colorbar.set_label('$T_g$ (K/)', rotation=270, va='baseline')

# fig6 = plt.figure(figsize=(7, 4))

# v = Tms
# t = X_samples[:, 0]
# l = X_samples[:, 1]
# r = 1-t-l

# ax = fig6.add_subplot(1, 1, 1, projection='ternary')

# cs = ax.tripcolor(t, l, r, v, shading='flat', cmap='viridis')
# ax.set_tlabel('Fe')
# ax.set_llabel('Si')
# ax.set_rlabel('B')

# ax.set_tlim(bounds[0][0], bounds[0][1])
# ax.set_llim(bounds[1][0], bounds[1][1])
# ax.set_rlim(bounds[2][0], bounds[2][1])

# # Colorbar with log ticks
# cax = ax.inset_axes([1.25, 0.1, 0.05, 0.9], transform=ax.transAxes)
# colorbar = fig6.colorbar(cs, cax=cax)
# colorbar.set_label('$T_m$ (K)', rotation=270, va='baseline')

###############
print(f"Number of nans in Tg; {nanTic}")

bundle = joblib.load(f"deploy_bundle_{Htype}.joblib")
model.load_state_dict(bundle["model_state_dict"])
R_scaler = bundle["R_scaler"]
X4_scaler = bundle["X4_scaler"]
f_scaler = bundle["f_scaler"]
ftot_scaler = bundle["ftot_scaler"]
Hrs_saved = bundle["Hrs"]
t_scaler = bundle["t_scaler"]

X_samples[:,-1] = torch.from_numpy(X4_scaler.fit_transform(X_samples[:,-1].reshape(-1, 1)))[:,0]
# --- Predict using trained model ---
    
with torch.no_grad():
    Y_preds = model(X_samples)

Y_preds = Y_preds.detach().cpu().numpy()
X_samples = X_samples.detach().cpu().numpy()

# --- Extract outputs ---

###### Get back parameters

fig4, ax4 = plt.subplots(1,4)
ax4[0].hist(Y_preds[:,-1],30)
ax4[0].set_title("Before")

Y_preds = ReverseTransform(Y_preds,Scale,Type,MLdata)
Y_preds[:,-1] /= RealHrs

ax4[1].hist(Y_preds[:,-1],50)
ax4[1].set_title("After")

ax4[2].hist(time_end,50)
ax4[2].set_title("Sim end")

ax4[3].hist(np.abs((Y_preds[:,-1]-time_end)/Y_preds[:,-1]),50)
ax4[3].set_title("diff")
ax4[3].set_ylim([0,1000])

X_samples[:,-1] = X4_scaler.inverse_transform(X_samples[:,-1].reshape(-1, 1))[:,0]
########## Sanity check
######################## total volume
# xFe = X_samples[:,0]
# xSi = X_samples[:,1]
# xB = 1 - xFe - xSi

# HR = X_samples[:,-1]
# HR = np.exp(np.abs(HR))

# data = Y_preds

# Hrs = np.array([9.9e7])
# sanity_mask = np.isclose(HR, Hrs[0], atol=1e5)
# print(f"len sanity mask: {len(sanity_mask[sanity_mask==True])}")
# data = data[sanity_mask,:]
# HR = Hrs*np.ones(len(sanity_mask[sanity_mask==True]))
# xFe = xFe[sanity_mask]
# xSi = xSi[sanity_mask]
# xB = xB[sanity_mask]
# time_end_s = time_end[sanity_mask]

# Fe_Lim = np.array([lb[0],ub[0]])
# Si_Lim = np.array([lb[1],ub[1]])
# B_Lim = np.array([lb[2],ub[2]])

# # 0: vol, -1: times
# fig6, ax6 = plot_ternary_data(
#     data,
#     xFe, xSi, xB, HR,
#     Hrs, Htype, time_end_s,
#     Fe_Lim, Si_Lim, B_Lim,
#     v_index=0,
#     vmin=0,
#     vmax=20,
#     isLog=False,
#     suptitle="Total vol, Target 20%",
#     cbar_label="Total vol (%)",
#     per_panel_colorbars=False,   # every subplot has one
#     cmap = 'viridis'
# )
    
########## Filter data (important plots)
total_fraction = Y_preds[:, 0]
target_fraction = Y_preds[:, 1]
target_rm = Y_preds[:, 2]

print(min(target_rm),max(target_rm),len(target_rm[target_rm > 1]))
# phase_rm = Y_preds[:, len(PD[0]):2*len(PD[0])]#*Rscale
# total_fraction = torch.sum(phase_fractions, dim=1)
# target_fraction = phase_fractions[:, TargetIdx]
# target_rm = phase_rm[:, TargetIdx]
# print(f"{np.mean(target_rm.numpy())}, {np.max(target_rm.numpy())}, {np.min(target_rm.numpy())}")
time_vals = Y_preds[:, -1]

if Tag == "Single":
    ftot_tol = 1 # vol frac
    ftarget_tol = 1 # vol frac
    target_phase_fraction = 3
    target_total_fraction = 3
elif Tag == "Double":
    ftot_tol = 2 # vol frac
    ftarget_tol = 2 # vol frac
    target_phase_fraction = 10
    target_total_fraction = 10

rm_tol_l = 1 #nm
rm_tol_u = 50 #nm

time_tol = 0.05 #(time_sim-time_end)/time_end

# --- Stepwise filtering ---
mask_total = np.isclose(total_fraction, target_total_fraction, atol=ftot_tol)
mask_target = np.isclose(target_fraction, target_phase_fraction, atol=ftarget_tol)
mask_rm = (target_rm > rm_tol_l) & (target_rm < rm_tol_u)

eps_t = (time_vals - time_end)/time_end
mask_time =  eps_t >= -time_tol

plt.figure()
plt.scatter(target_rm, target_fraction, s=1, alpha=0.1)
plt.xlabel("rm (nm)")
plt.ylabel("target fraction (%)")

# valid = mask_total & mask_target & mask_rm
valid = mask_target & mask_rm

print("valid before time:", np.sum(valid))

valid_time = valid & mask_time

print("valid after time:", np.sum(valid_time))

rm_in_valid = target_rm[mask_total & mask_target]

print("min rm in valid region:", np.min(rm_in_valid))
print("max rm in valid region:", np.max(rm_in_valid))
print("mean rm in valid region:", np.mean(rm_in_valid))


if Tag == "Single":
    combined_mask = mask_total & mask_time 
elif Tag == "Double":
    combined_mask = mask_total & mask_time & mask_target 
    # combined_mask = mask_total & mask_time & mask_target & mask_rm

print("-"*20)
print(f"n f_tot: {len(mask_total[mask_total == True])}")
print(f"n f_target: {len(mask_target[mask_target == True])}")
print(f"n rm: {len(mask_rm[mask_rm == True])}")
print(f"n time: {len(mask_time[mask_time == True])}")
print(f"n all: {len(combined_mask[combined_mask == True])}")
print("-"*20)

# ##########################


# # --- Stack masks ---
# mask_names = ["f_total", "f_target", "rm", "time"]
# masks = [mask_total, mask_target, mask_rm, mask_time]

# # Convert to 2D array (n_masks x n_samples)
# mask_array = np.vstack(masks)

# n_masks, n_samples = mask_array.shape

# # X-axis = sample index
# x = np.arange(n_samples)

# fig, ax = plt.subplots(figsize=(14, 5))

# colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# for i in range(n_masks):
#     y_true = np.ones(n_samples) * i
#     y_false = np.ones(n_samples) * i

#     # True points
#     ax.scatter(
#         x[mask_array[i]],
#         y_true[mask_array[i]],
#         s=3,
#         color=colors[i],
#         label=mask_names[i] if i == 0 else None,
#         alpha=0.8
#     )

#     # False points (optional, lighter)
#     ax.scatter(
#         x[~mask_array[i]],
#         y_false[~mask_array[i]],
#         s=1,
#         color="lightgray",
#         alpha=0.2
#     )

# # Formatting
# ax.set_yticks(range(n_masks))
# ax.set_yticklabels(mask_names)
# ax.set_xlabel("Sample index")
# ax.set_ylabel("Mask condition")
# ax.set_title("Boolean mask overlap visualization")

# plt.tight_layout()
# plt.show()


# ##########################

# # --- Extract ternary composition ---
# t = X_samples[:, 0]
# l = X_samples[:, 1]
# r = 1-t-l

# # --- Create figure ---
# fig = plt.figure()

# # --- Masks to plot ---
# masks = [
#     ("f_total", mask_total),
#     ("f_target", mask_target),
#     ("rm", mask_rm),
#     ("time", mask_time),
# ]

# # --- Loop over masks ---
# for i, (name, mask) in enumerate(masks, 1):
#     ax = fig.add_subplot(2, 2, i, projection="ternary")

#     # All samples (background)
#     ax.scatter(t, l, r, s=2, alpha=0.05, color="gray")

#     # Filtered samples
#     ax.scatter(t[mask], l[mask], r[mask], s=5, alpha=0.7, color="tab:red")

#     ax.set_title(name)
#     ax.set_tlabel("X1")
#     ax.set_llabel("X2")
#     ax.set_rlabel("X3")

#     ax.set_tlim(bounds[0][0], bounds[0][1])
#     ax.set_llim(bounds[1][0], bounds[1][1])
#     ax.set_rlim(bounds[2][0], bounds[2][1])     
    
# plt.tight_layout()
# plt.show()

# ###################


# --- Apply filtering ---
filtered_X = X_samples[combined_mask]
filtered_Y = Y_preds[combined_mask]
filtered_time = time_vals[combined_mask]
filtered_time_end = time_end[combined_mask]

# --- Select sample(s) with maximum time ---
if len(filtered_X) > 0:
    top_indices = np.argsort(filtered_time)[::-1][:top_k]
    for i in range(len(top_indices)):
        idx = top_indices[i]
        x = filtered_X[idx]
        y = filtered_Y[idx]
        hr = -np.exp(np.abs(x[-1].item())) if Htype == "Quenching" else np.exp(x[-1].item())
        print(f"Sample {i+1}:")
        print(f"  Fe: {x[0].item():.4f}, Si: {x[1].item():.4f}, B: {1-x[0].item()-x[1].item():.4f}")
        print(f"  Heating Rate: {hr:.2e} [K/s]")
        print(f"  V tot: {y[0]:1.3f}")
        print("  Phase: bcc")
        print(f"  Vol:   {y[1]}")
        print(f"  rm:    {y[2]} [nm]")
        print(f"  Time: {y[-1].item():.2e} of {filtered_time_end[idx]:.2e}  [s]")
        print("-" * 50)
else:
    print("No samples found that satisfy all conditions.")


print('-------------------------')
simtime=str(round((time.time()-simt)/60,3))
print(f'Computation time: {simtime} min')
print('-------------------------')


def match_reference_set(t, l, v, xFe_ref, xSi_ref, xB_ref):
    """
    Match each reference composition to the closest model composition
    and return the median predicted value.
    """
    r = 1 - t - l

    matched_vals = []

    for fe, si, b in zip(xFe_ref, xSi_ref, xB_ref):
        dist = np.sqrt(
            (t - fe)**2 +
            (l - si)**2 +
            (r - b)**2
        )
        idx = np.argmin(dist)
        matched_vals.append(v[idx])

    return np.median(matched_vals)

###### plot compositions
fig1, ax1 = plt.subplots(1,3)
ax1[0].hist(Comps[:,0],30)
ax1[0].set_title("Fe")
ax1[1].hist(Comps[:,1],30)
ax1[1].set_title("Si")
ax1[2].hist(Comps[:,2],30)
ax1[2].set_title("B")

FigpathOut=''.join([DirPathIn,f"Comps_{Htype}.png"])
fig1.savefig(FigpathOut, bbox_inches = "tight")

#### Plot valid compositions agains heating rate
if len(filtered_X[:, -1]) > 3:
    import matplotlib.colors as mcolors
    
    fig2 = plt.figure(figsize=(7, 4))
    
    v = filtered_X[:, -1]
    t = filtered_X[:, 0]
    l = filtered_X[:, 1]
    r = 1-t-l
    
    # Convert heating rate back to linear scale
    if Htype == "Quenching":
        v = np.exp(abs(v))  # positive values
    else:
        v = np.exp(v)
    
    norm = mcolors.LogNorm(vmin=min(v), vmax=max(v)) 
    
    ax = fig2.add_subplot(1, 1, 1, projection='ternary')
    ax.set_title(f"{target_total_fraction}% +-{ftot_tol}%")
    
    cs = ax.tripcolor(t, l, r, v, shading='flat', norm=norm, cmap='plasma')
    ax.set_tlabel('Fe')
    ax.set_llabel('Si')
    ax.set_rlabel('B')
    
    ########## Add refs if quenching
    # if Htype == "Quenching" and Tag == "Single":
    #     sc = ax.scatter(
    #         xFe_Ref_1_Q, xSi_Ref_1_Q, xB_Ref_1_Q,
    #         s=70, c=Rc_Ref_1_Q, cmap='plasma', norm=norm,
    #         edgecolors="lime", linewidths=1.5
    #     )
        
    #     sc = ax.scatter(
    #         xFe_Ref_2_Q, xSi_Ref_2_Q, xB_Ref_2_Q,
    #         s=70, c=Rc_Ref_2_Q, cmap='plasma', norm=norm,
    #         edgecolors="cyan", linewidths=1.5
    #     )
    
    if Htype == "Quenching" and Tag == "Single":
 
        sc = ax.scatter(
            xFe_Ref_1_Q,
            xSi_Ref_1_Q,
            xB_Ref_1_Q,
            s=70,
            marker='o',
            facecolors='none',        # no fill
            edgecolors='white',       # outline color
            linewidths=1.5
        )

        
        sc = ax.scatter(
            xFe_Ref_2_Q, xSi_Ref_2_Q, xB_Ref_2_Q,
            s=70, c = "white", marker = "^"
        )
        
        ########### Get contours of quenching for heating data
        
        # Target quenching rates and tolerance
        q_targets = [1e5, 1e6]
        q_tol = 0.07                # relative tolerance (5%)
        
        # Create ternary figure 
        fig22 = plt.figure()
        ax22 = fig22.add_subplot(1, 1, 1, projection='ternary')
        ax22.set_title("Figure 22 – Quenching‑rate‑filtered ternary data")
        ax22.set_tlabel("Fe")
        ax22.set_llabel("Si")
        ax22.set_rlabel("B")
        
        # Container for extracted ternary data
        ternary_data_q = {}
        
        for q0 in q_targets:
            mask = np.isclose(v, q0, rtol=q_tol)
        
            ternary_data_q[q0] = {
                "Fe": t[mask],
                "Si": l[mask],
                "B":  r[mask],
                "value": v[mask],
                "v": v[mask]
            }
        
            # Optional: attach points to ternary axes (light scatter, no emphasis)
            ax22.scatter(
                t[mask], l[mask], r[mask],
                s=15,
                marker='o',
                facecolors='none',
                edgecolors='white',
                linewidths=0.7
            )
            
        ax22.set_tlim(bounds[0][0], bounds[0][1])
        ax22.set_llim(bounds[1][0], bounds[1][1])
        ax22.set_rlim(bounds[2][0], bounds[2][1])  
        FigpathOut = ''.join([DirPathIn, "ContourForHeating.svg"])
        fig22.savefig(FigpathOut, bbox_inches="tight")

        ######### new bar plot
        
        def comp_key(fe, si, b, tol=1e-6):
            """Hashable key for a composition with tolerance."""
            return (
                round(fe / tol) * tol,
                round(si / tol) * tol,
                round(b  / tol) * tol
            )
        

        def comp_label(fe, si, b):
            return rf"$\mathrm{{Fe}}_{{{int(round(100*fe))}}}" \
                   rf"\mathrm{{Si}}_{{{int(round(100*si))}}}" \
                   rf"\mathrm{{B}}_{{{int(round(100*b))}}}$"

        
        
        refs = {}
        
        # Reference 1
        for fe, si, b, rc in zip(
                xFe_Ref_1_Q, xSi_Ref_1_Q, xB_Ref_1_Q, Rc_Ref_1_Q):
            key = comp_key(fe, si, b)
            if key not in refs:
                refs[key] = {
                    "Fe": fe, "Si": si, "B": b,
                    "ref1": None, "ref2": None
                }
            refs[key]["ref1"] = rc
        
        # Reference 2
        for fe, si, b, rc in zip(
                xFe_Ref_2_Q, xSi_Ref_2_Q, xB_Ref_2_Q, Rc_Ref_2_Q):
            key = comp_key(fe, si, b)
            if key not in refs:
                refs[key] = {
                    "Fe": fe, "Si": si, "B": b,
                    "ref1": None, "ref2": None
                }
            refs[key]["ref2"] = rc
            
            
        def nearest_model_value(t, l, v, fe, si, b):
            r = 1 - t - l
            dist = np.sqrt((t-fe)**2 + (l-si)**2 + (r-b)**2)
            return v[np.argmin(dist)]
        
        
        labels = []
        model_vals = []
        ref1_vals = []
        ref2_vals = []
        
        for data in refs.values():
            fe, si, b = data["Fe"], data["Si"], data["B"]
            
            if si == 1e-2:
                si = 0
                
            labels.append(comp_label(fe, si, b))
            model_vals.append(nearest_model_value(t, l, v, fe, si, b))
        
            ref1_vals.append(data["ref1"])
            ref2_vals.append(data["ref2"])
            
        
        
        x = np.arange(len(labels))
        bar_width = 0.3
        
        fig9, ax9 = plt.subplots(figsize=(0.3*len(labels), 4))
        
        for i, (ym, yr1, yr2) in enumerate(zip(model_vals, ref1_vals, ref2_vals)):
        
            # Determine which bars exist
            heights = [ym]
            colors  = ["tab:blue"]
        
            if yr1 is not None:
                heights.append(yr1)
                colors.append("tab:orange")
        
            if yr2 is not None:
                heights.append(yr2)
                colors.append("tab:green")
        
            n_bars = len(heights)
        
            # Choose offsets dynamically
            if n_bars == 2:
                offsets = [-bar_width/2, +bar_width/2]
            elif n_bars == 3:
                offsets = [-bar_width, 0.0, +bar_width]
            else:
                raise RuntimeError("Unexpected number of bars")
        
            # Plot bars
            for h, off, c in zip(heights, offsets, colors):
                ax9.bar(x[i] + off, h, bar_width, color=c)
        
        # Legend (manual to avoid duplicates)
        ax9.legend(handles=[
            plt.Line2D([0],[0], color="tab:blue", lw=6, label="This work"),
            plt.Line2D([0],[0], color="tab:orange", lw=6, label="Reference 1"),
            plt.Line2D([0],[0], color="tab:green", lw=6, label="Reference 2"),
        ])
        
        ax9.set_yscale("log")
        ax9.set_ylabel(r"Quenching rate (K/s)")
        
        ax9.yaxis.tick_right()                 # move ticks to the right
        ax9.yaxis.set_label_position("right")  # move y-label to the right
        ax9.spines["right"].set_visible(True)  # ensure right spine is visible
        ax9.spines["left"].set_visible(False)  # optional: hide left spine

        ax9.set_xticks(x)
        ax9.set_xticklabels(labels, rotation=45, ha="right")
        ax9.grid(True, which="both", linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        
        FigpathOut = ''.join([DirPathIn, "ComparisionRefQuench.svg"])
        fig9.savefig(FigpathOut, bbox_inches="tight")

        
        ###################


    ##########
    
    ax.set_tlim(bounds[0][0], bounds[0][1])
    ax.set_llim(bounds[1][0], bounds[1][1])
    ax.set_rlim(bounds[2][0], bounds[2][1])      
    
    # Colorbar with log ticks
    cax = ax.inset_axes([1.25, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig2.colorbar(cs, cax=cax)
    if Htype == "Quenching":
        colorbar.set_label('Quenching rate (K/s)', rotation=270, va='baseline')
    else:
        colorbar.set_label('Heating rate (K/s)', rotation=270, va='baseline')
    
    FigpathOut=''.join([DirPathIn,f"FilteredComps_{Htype}_{target_total_fraction}_{Tag}.png"])
    fig2.savefig(FigpathOut, bbox_inches = "tight",dpi=300)
    FigpathOut=''.join([DirPathIn,f"FilteredComps_{Htype}_{target_total_fraction}_{Tag}.svg"])
    fig2.savefig(FigpathOut, bbox_inches = "tight")

plt.show()

