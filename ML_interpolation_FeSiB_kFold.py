#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:36:13 2025

@author: ag0406
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import json
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from scipy.special import logit
# import mpltern
# import matplotlib.colors as mcolors

from ML_utilities import load_data_Kfold, Network, train, plot_ternary_rows, ReverseTransform

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

TargetPhase = "bcc"

Fe_Lim = np.array([0.7, 1.0]) 
Si_Lim = np.array([0.0, 0.3])
B_Lim = np.array([0.0, 0.3])


#%%
print(f"--------- {Htype} ---------")
data = load_data_Kfold(Htype,Type)

################ Load optuna parameters
DirPathOut = data["DirPathOut"]

# with open("/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/Heating/best_params.json", "r") as f:
#     config = json.load(f)
    
with open(f"{DirPathOut}best_params.json", "r") as f:
    config = json.load(f)

best_params = config["best_params"]
print("Loaded best params:", best_params)

################

PD = data["PD"]
plotPhaseIdx = np.where(PD[0,:]== TargetPhase)[0][0]
# create the network
input_dimension = data["X_all"].shape[1] 
    
output_dimension = data["y_all"].shape[1]

######################
model = Network(
    input_dimension,
    [best_params[f"n_units_layer{i}"] for i in range(best_params["n_layers"])],
    output_dimension,
    # activation=best_params["activation"],
    activation="leaky_relu",
    # activation="tanh",
    dropout=best_params["dropout"]
)


X_opt = data["X_opt"]
y_opt = data["y_opt"]

# Internal split for early stopping
n = len(X_opt)
n_val_final = int(0.1 * n)   # 10% for final validation of early stopping

rng = torch.Generator().manual_seed(123)
perm = torch.randperm(n, generator=rng)

val_idx_final = perm[:n_val_final]
train_idx_final = perm[n_val_final:]

X_train_final = X_opt[train_idx_final]
y_train_final = y_opt[train_idx_final]

X_val_final = X_opt[val_idx_final]
y_val_final = y_opt[val_idx_final]

train_val_data_final = {
    "train": {"X": X_train_final, "y": y_train_final},
    "val":   {"X": X_val_final,   "y": y_val_final}
}


Cost_train_final, Cost_val_final = train(
    model,
    best_params["epochs_adam"],
    # 20000,
    "Adam",
    train_val_data_final,
    lr=best_params["lr_adam"],
    lam=best_params["lambda"],
    batch_size=best_params["batch_size"],
    patience=best_params["patience"]
    # patience=None
)

Cost_train_2 = np.zeros(1) 
Cost_val_2 = np.zeros(1) 

torch.save(model.state_dict(), ''.join([data["DirPathOut"], "TrainedModel.pth"])) 
##########################  Extract trained data, scale back, and prepare plotting

R_scaler = data["R_scaler"]
X4_scaler = data["X4_scaler"]
Hrs_raw = data["Hrs_raw"]
f_scaler = data["f_scaler"]
ftot_scaler = data["ftot_scaler"]
t_scaler = data["t_scaler"]  # list of MinMaxScaler for each heating rate
Hrs = data["Hrs"]              # unique heating rates
X4 = data["X_all"][:, -1].detach().cpu().numpy()  # heating rate column

TrueHrs = np.unique(X4_scaler.inverse_transform(X4.reshape(-1, 1)))

X = data["X_all"]
y = data["y_all"].detach().numpy()      # or data["y_all"] if ground truth

bundle = {
    "model_state_dict": model.state_dict(),  # torch model weights
    "R_scaler": R_scaler,
    "X4_scaler": X4_scaler,
    "f_scaler": f_scaler,
    "ftot_scaler": ftot_scaler,
    "Hrs": list(Hrs),
    "t_scaler": t_scaler,
}
joblib.dump(bundle, f"deploy_bundle_{Htype}.joblib")

##########################
##########################

# # Higer resolution of interpolated data
def bounded_simplex_grid(Fe_min, Fe_max, Si_min, Si_max, B_min, B_max,
                         n_si=50, n_b_base=50, tol=1e-12):
    """
    Generate compositions (xFe,xSi,xB) with xFe+xSi+xB = 1,
    respecting element-wise bounds:
      Fe_min <= xFe <= Fe_max
      Si_min <= xSi <= Si_max
      B_min  <= xB  <= B_max

    Strategy: loop over xSi values and for each compute feasible xB interval:
      b_low  = max(B_min,   1 - Fe_max - xSi)
      b_high = min(B_max,   1 - Fe_min - xSi)
    Then sample xB in [b_low, b_high] and compute xFe = 1 - xSi - xB.

    Returns an (N,3) NumPy array of [xFe,xSi,xB].
    """
    # Sanity
    if Fe_min + Si_min + B_min > 1 + tol:
        raise ValueError("sum of lower bounds > 1 -> infeasible")
    if Fe_max + Si_max + B_max < 1 - tol:
        raise ValueError("sum of upper bounds < 1 -> infeasible")

    xSi_vals = np.linspace(Si_min, Si_max, n_si)
    rows = []

    # Use a base number of xB samples and scale it by the available interval length
    B_span = B_max - B_min if B_max > B_min else 1.0

    for s in xSi_vals:
        b_low = max(B_min, 1.0 - Fe_max - s)
        b_high = min(B_max, 1.0 - Fe_min - s)

        if b_low > b_high + tol:
            # no feasible xB for this xSi
            continue

        # choose number of xB points — at least 2
        frac = (b_high - b_low) / (B_span if B_span>0 else 1.0)
        n_b = max(2, int(np.ceil(n_b_base * max(frac, 0.05))))  # scale by interval length (min density)
        xb = np.linspace(b_low, b_high, n_b)

        xfe = 1.0 - s - xb

        # numerical safety clip
        xfe = np.clip(xfe, Fe_min, Fe_max)
        # keep only truly feasible (redundant but safe)
        mask = (xfe >= Fe_min - tol) & (xfe <= Fe_max + tol)
        if not np.any(mask):
            continue

        rows.append(np.column_stack([xfe[mask], np.full(mask.sum(), s), xb[mask]]))

    if len(rows) == 0:
        return np.empty((0, 3), dtype=float)

    return np.vstack(rows)



hRes = 100
n_b_base = 100

map_list = []
for hr in Hrs:
    mask = X[:, -1] == hr
    if mask.sum() == 0:
        continue
    xFe_Hrs = X[mask, 0]
    xSi_Hrs = X[mask, 1]
    xB_Hrs = 1 - xFe_Hrs - xSi_Hrs

    Fe_min, Fe_max = float(xFe_Hrs.min()), float(xFe_Hrs.max())
    Si_min, Si_max = float(xSi_Hrs.min()), float(xSi_Hrs.max())
    B_min,  B_max  = float(xB_Hrs.min()),  float(xB_Hrs.max())

    grid3 = bounded_simplex_grid(Fe_min, Fe_max, Si_min, Si_max, B_min, B_max,
                                 n_si=hRes, n_b_base=n_b_base)
    if grid3.size == 0:
        continue

    hr_col = np.full((grid3.shape[0], 1), hr)
    full = np.hstack([grid3, hr_col])   # columns: xFe,xSi,xB,Hr
    map_list.append(full)

if len(map_list) == 0:
    Xh = torch.empty((0, 4), dtype=torch.float32)
else:
    Map = np.vstack(map_list)
    Xh = torch.tensor(Map, dtype=torch.float32)

Xh = torch.hstack((Xh[:, :2], Xh[:, 3:4])) 

########################## Scale back predicted data 

xFe_pred = Xh[:,0]
xSi_pred = Xh[:,1]
xB_pred = 1 - xFe_pred - xSi_pred
HR_pred = Xh[:,2]

# HR_pred = np.log(HR_pred)
                                                   
y_pred = model(Xh).detach().numpy()

Scale = data["scaler"]

y_pred = ReverseTransform(y_pred,Scale,Type,data)
y_pred[:,-1] /= abs(X4_scaler.inverse_transform(Xh[:,-1].reshape(-1, 1)).ravel())

########################## Scale back true data
xFe_true = X[:,0]
xSi_true = X[:,1]
xB_true = 1 - xFe_true - xSi_true
HR_true = X[:,2]
# HR_true = np.log(HR_true)
y_true = y.copy()

y_true = ReverseTransform(y_true,Scale,Type,data)
print(y_true[:,-1])
print(abs(X4_scaler.inverse_transform(HR_true.reshape(-1, 1)).ravel()))
y_true[:, -1] /= abs(X4_scaler.inverse_transform(HR_true.reshape(-1, 1)).ravel())

######################## Plot data
######################## 
######################## 


######################## vol target phase
fig1, ax1 = plot_ternary_rows(
    y_true, y_pred,
    xFe_true, xSi_true, xB_true, HR_true,
    xFe_pred, xSi_pred, xB_pred, HR_pred,
    Hrs, TrueHrs, Htype,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=1,
    vmin=None,
    vmax=None,
    isLog=False,
    suptitle=f"{PD[0,plotPhaseIdx]}, Pred 20%",
    top_cbar_label="True vol (%)",
    bottom_cbar_label="Pred vol (%)",
    per_panel_colorbars=False
)

######################## rm(target phase)

# if vmin == 0:
#     vmin = 1e-8
# vmax = max(y_true[:, 2])
if Htype == "Quenching":
    # vmax = 1e2
    vmax = max(y_true[:, 2])
    # vmin = min(y_true[y_true[:,2]>0, 2])
    vmin = 1e-1
else:
    vmax = max(y_true[:, 2])
    # vmin = min(y_true[:, 2])
    vmin = 1e-1
print(f"rmin: {vmin:4.2f} nm, rmax: {vmax:4.2f} nm")

fig3, ax3 = plot_ternary_rows(
    y_true, y_pred,
    xFe_true, xSi_true, xB_true, HR_true,
    xFe_pred, xSi_pred, xB_pred, HR_pred,
    Hrs, TrueHrs, Htype,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=2,
    vmin=vmin,
    vmax=vmax,
    isLog=True,
    suptitle="Peak radius, Pred 20%",
    top_cbar_label="True r (nm)",
    bottom_cbar_label="Pred r (nm)",
    per_panel_colorbars=False
)

######################## times
fig4, ax4 = plot_ternary_rows(
    y_true, y_pred,
    xFe_true, xSi_true, xB_true, HR_true,
    xFe_pred, xSi_pred, xB_pred, HR_pred,
    Hrs, TrueHrs, Htype,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=-1,
    vmin=None,
    vmax=None,
    isLog=False,
    suptitle="Times vol, Pred 20%",
    top_cbar_label="True t (s)",
    bottom_cbar_label="Pred t (s)",
    per_panel_colorbars=True
)


######################## total volume
fig7, ax7 = plot_ternary_rows(
    y_true, y_pred,
    xFe_true, xSi_true, xB_true, HR_true,
    xFe_pred, xSi_pred, xB_pred, HR_pred,
    Hrs, TrueHrs, Htype,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=0,
    vmin=None,
    vmax=None,
    isLog=False,
    suptitle="Total vol, Pred 20%",
    top_cbar_label="Total vol (%)",
    bottom_cbar_label="Pred vol (%)",
    per_panel_colorbars=False   # every subplot has one
)

######################## 
######################## 
######################## 

##### Plot Cost(epoch) Cost_train_1

fig2 = plt.figure()

plt.semilogy(Cost_train_final,'k-',label = 'train_tot')

plt.semilogy(Cost_val_final,'r-',label = 'val_tot')

plt.ylabel(r"$C(\Phi)$")
plt.xlabel("$epochs$")
plt.legend()

######################### pred versus true
#########################
#########################

# training and validation

y_pred_raw =  model(X).detach().numpy()

r2 = r2_score(y.flatten(), y_pred_raw.flatten())
MSE = mean_squared_error(y.flatten(), y_pred_raw.flatten())
print(f"R²: {r2:.4f}")
print(f"MSE: {MSE:.4f}")

print("phases left:\n",PD[0,:])

fig5, ax5 = plt.subplots(1) 
# ax5.scatter(y.flatten(),y_pred_raw.flatten())
#ftot
ax5.scatter(y[:,0].flatten(),y_pred_raw[:,0].flatten(),color = "orange")
#f
# ax5.scatter(y[:,len(PD[0,:])].flatten(),y_pred_raw[:,len(PD[0,:])].flatten(),color = "b")
ax5.scatter(y[:,1].flatten(),y_pred_raw[:,1].flatten(),color = "b")
# Rm
# ax5.scatter(y[:,plotPhaseIdx + len(PD[0,:])].flatten(),y_pred_raw[:,plotPhaseIdx + len(PD[0,:])].flatten(),color = "m")
ax5.scatter(y[:,2].flatten(),y_pred_raw[:,2].flatten(),color = "m")
# Time
ax5.scatter(y[:,-1].flatten(),y_pred_raw[:,-1].flatten(), color = "c")

ax5.plot(y.flatten(),y.flatten(),'k')
ax5.set_xlabel("True data")
ax5.set_ylabel("Predicted data")
ax5.set_title(f"R$^2$: {r2:.4f}, MSE: {MSE:.4f} ,{Htype}")

# same for test
fig55, ax55 = plt.subplots(1)   
   
X_test = data["X_test"]
y_test = data["y_test"].detach().numpy() 

y_pred_raw_test =  model(X_test).detach().numpy()

r2 = r2_score(y_test.flatten(), y_pred_raw_test.flatten())
MSE = mean_squared_error(y_test.flatten(), y_pred_raw_test.flatten())
 
ax55.scatter(y_test[:,0].flatten(),y_pred_raw_test[:,0].flatten(),color = "orange")
#f
# ax5.scatter(y[:,len(PD[0,:])].flatten(),y_pred_raw[:,len(PD[0,:])].flatten(),color = "b")
ax55.scatter(y_test[:,1].flatten(),y_pred_raw_test[:,1].flatten(),color = "b")
# Rm
# ax5.scatter(y[:,plotPhaseIdx + len(PD[0,:])].flatten(),y_pred_raw[:,plotPhaseIdx + len(PD[0,:])].flatten(),color = "m")
ax55.scatter(y_test[:,2].flatten(),y_pred_raw_test[:,2].flatten(),color = "m")
# Time
ax55.scatter(y_test[:,-1].flatten(),y_pred_raw_test[:,-1].flatten(), color = "c")

ax55.plot(y_test.flatten(),y_test.flatten(),'k')
ax55.set_xlabel("True data")
ax55.set_ylabel("Predicted data")
ax55.set_title(f"R$^2$: {r2:.4f}, MSE: {MSE:.4f} ,{Htype}")

# And plot histograms of data points:
    
fig8, ax8 = plt.subplots(2) 

# ax8.scatter(y[:,0].flatten(),y_pred_raw[:,0].flatten(),color = "orange")
ax8[0].hist(y.flatten(),50)
ax8[0].set_ylabel("CNGT data")

ax8[1].hist(y_pred_raw.flatten(),50)
ax8[1].set_ylabel("ML data")
    
######################### Check "bad values"

import matplotlib.pyplot as plt
import mpltern
import matplotlib as mpl
import matplotlib.colors as mcolors

def plot_zeros_across_outputs_ternary(
    y,                        # (N, D) true targets
    xFe, xSi, xB,             # (N,) compositions, aligned with y
    HR,            # (N,) heating rate labels, aligned with y
    Htype,                    # "Quenching", "Heating", "AM", or "Iso"
    k_list=(0, 1, 2, 3),      # output columns to analyze
    eps=1e-6,                 # threshold for "near zero"
    suptitle="Near-zero targets in composition–HR space",
    cmap_hr="plasma",
):
    """
    For each k in k_list (length 4):
      - Scatter all samples faint (gray)
      - Highlight near-zero (y[:,k] <= eps) colored by HR_per_sample
      - Compute mean of near-zero and mean of non-zero (rest), with guards
      - Add legend line with counts + means

    Returns (fig, axs, stats), where stats is a list of dict per k with summary numbers.
    """
    # Labels per output index (edit for your semantics)
    k_labels = ["ftot", "fbcc", "rbcc", "time"]

    k_list = list(k_list)
    if len(k_list) != 4:
        raise ValueError("Provide exactly 4 outputs (k_list length 4) for a 2x2 layout.")

    # HR = np.exp(abs(X4_scaler.inverse_transform(HR.reshape(-1, 1))))
    HR = abs(X4_scaler.inverse_transform(HR.reshape(-1, 1)))
    hr_min, hr_max = HR.min(), HR.max()
    norm = mcolors.LogNorm(vmin=hr_min, vmax=hr_max)
    cmap = mpl.colormaps.get_cmap(cmap_hr)

    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), subplot_kw={'projection': 'ternary'})
    axs = axs.ravel()

    # Background style
    bg_color = "lightgray"
    bg_alpha = 0.25
    edge_color = "k"

    # Collect per-k stats to inspect/print if desired
    stats = []

    for ax, k in zip(axs, k_list):
        yk = y[:, k]
        near_zero = (yk <= eps)
        rest = ~near_zero

        # Means with guards
        if near_zero.any():
            mean_zero = float(np.mean(yk[near_zero]))
        else:
            mean_zero = np.nan

        if rest.any():
            mean_rest = float(np.mean(yk[rest]))
        else:
            mean_rest = np.nan

        # Keep stats
        stats.append({
            "k": k,
            "label": (k_labels[k] if k < len(k_labels) else f"k={k}"),
            "N_total": int(len(yk)),
            "N_zero": int(near_zero.sum()),
            "N_rest": int(rest.sum()),
            "mean_zero": mean_zero,
            "mean_rest": mean_rest,
        })

        # Background: all points faint
        ax.scatter(xFe, xSi, xB, s=8, color=bg_color, alpha=bg_alpha, label="all")

        # Highlight near-zero in color by HR
        sc = None
        if near_zero.any():
            sc = ax.scatter(
                xFe[near_zero], xSi[near_zero], xB[near_zero],
                s=26, c=HR[near_zero], cmap=cmap, norm=norm,
                edgecolors=edge_color, linewidths=0.2,
                label=(
                    f"{k_labels[k] if k < len(k_labels) else f'k={k}'}≤{eps:g}, (N={int(near_zero.sum())}\n" 
                    f"μ0={mean_zero:.2e}, μrest={mean_rest:.2e})"
                    if rest.any() else
                    f"{k_labels[k] if k < len(k_labels) else f'k={k}'}≤{eps:g}, (N={int(near_zero.sum())}\n" 
                    f"μ0={mean_zero:.2e})"
                )
            )
            # Per-axes colorbar
            cax = ax.inset_axes([1.16, 0.10, 0.05, 0.80], transform=ax.transAxes)
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("HR (K/s)" , rotation=270, va='baseline', labelpad=10)

        # Ternary labels
        ax.set_tlabel('Fe'); ax.set_llabel('Si'); ax.set_rlabel('B')
        # If you want fixed ternary limits, uncomment and define Fe_Lim/Si_Lim/B_Lim in your scope:
        ax.set_tlim(Fe_Lim[0], Fe_Lim[1])
        ax.set_llim(Si_Lim[0], Si_Lim[1])
        ax.set_rlim(B_Lim[0], B_Lim[1])

        ax.set_title(
            f"{k_labels[k] if k < len(k_labels) else f'k={k}'} ~ 0: "
            f"{int(near_zero.sum())} / {len(yk)}"
        )

        # Legend only if we have highlighted points
        if sc is not None:
            ax.legend(frameon=True, loc='upper left', bbox_to_anchor=(0.95, 1.55), fontsize=9)

    fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    return fig, axs, stats

figZ, axsZ, statsZ = plot_zeros_across_outputs_ternary(
    y=y,                 # (N, D)
    xFe=xFe_true, xSi=xSi_true, xB=xB_true,
    HR=HR_true,    # <-- per-sample HR labels, aligned with y
    Htype=Htype,
    k_list=(0, 1, 2, 3),
    eps=1e-2,                 # your threshold
    suptitle=f"Near-zero samples: composition & heating rate, {Htype}",
    cmap_hr="plasma",
)

# Optional: print a compact summary
for s in statsZ:
    print(f"k={s['k']} ({s['label']}): "
          f"N_zero={s['N_zero']}, N_rest={s['N_rest']}, "
          f"mean_zero={s['mean_zero']:.3e}, mean_rest={s['mean_rest']:.3e}")



######################### correlation matrix
#########################
#########################
fig6, ax6 = plt.subplots(1) 
# cm = spearmanr(y_pred_raw,axis=0)[0]
cm = spearmanr(np.hstack((X.numpy(),y_pred_raw)),axis=0)[0]
cm_img = ax6.imshow(cm, cmap='viridis')
fig6.colorbar(cm_img,ax=ax6)
# ax5.set_xticks(range(nnl), Par_names, size='small')
# ax5.set_yticks(range(nnl), Par_names, size='small')
ax6.set_title(f"Correlation matrix {Htype}")
    
#########################
plt.show()

FigpathOut=''.join([DirPathOut,f"Comparision_f_{PD[0,plotPhaseIdx]}_{Htype}.png"])
fig1.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Cost_{Htype}.png"])
fig2.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Comparision_rm_{PD[0,plotPhaseIdx]}_{Htype}.png"])
fig3.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Comparision_Times_{Htype}.png"])
fig4.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Comparision_ftot_{Htype}.png"])
fig7.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Precision_{Htype}.png"])
fig5.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Precision_test_{Htype}.png"])
fig55.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Correlation_{Htype}.png"])
fig6.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Data_hist_{Htype}.png"])
fig8.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"nearZero_{Htype}.png"])
figZ.savefig(FigpathOut, bbox_inches = "tight")

# FigpathOut=''.join([DirPathOut,f"Comparision_f_{PD[0,plotPhaseIdx]}_{Htype}.svg"])
# fig1.savefig(FigpathOut)
# FigpathOut=''.join([DirPathOut,f"Cost_{Htype}.svg"])
# fig2.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Comparision_rm_{PD[0,plotPhaseIdx]}_{Htype}.svg"])
# fig3.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Comparision_Times_{Htype}.svg"])
# fig4.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Precision_{Htype}.svg"])
# fig5.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Precision_test_{Htype}.svg"])
# fig55.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Comparision_ftot_{Htype}.svg"])
# fig7.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Data_hist_{Htype}.svg"])
# fig8.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"nearZero_{Htype}.svg"])
figZ.savefig(FigpathOut, bbox_inches = "tight")


