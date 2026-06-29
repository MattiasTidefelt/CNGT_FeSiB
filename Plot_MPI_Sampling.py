#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:42:17 2025

@author: ag0406
"""

import numpy as np
from os import makedirs
from os.path import exists
import h5py
import matplotlib.pyplot as plt
import mpltern
# import os
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator

from paramFeSiB import sysParam

# from LiqFeSiB import TransE
# from GetTm import GetTm
from ReadTempData import initialize_Tspline_AML

from ML_utilities import load_or_generate_Tg_Tm_grid_simplex, load_or_generate_tend_grid_AM_TgTm, plot_ternary_data

# Htype = "Heating"
# Htype = "Quenching"
Htype = "AM"
# Htype = "Iso" 

# TargetPhase = "bcc"
# TargetPhase = "fcc"
# TargetPhase = "FeB"
# TargetPhase = "Fe2B"
# TargetPhase = "Fe3B"
# TargetPhase = "Fe5SiB2"
# TargetPhase = "Fe5Si2B"
TargetPhase = "Fe10Si4B3"

fmax = 0.4

if TargetPhase == "bcc":
    cmap = "Oranges"
elif TargetPhase == "fcc":
    cmap = "Blues"
elif TargetPhase == "FeB":
    cmap = "Reds"
elif TargetPhase == "Fe2B":
    cmap = "Greens"
elif TargetPhase == "Fe3B":
    cmap = "Reds"
elif TargetPhase == "Fe5SiB2": 
    cmap = "Purples"
elif TargetPhase == "Fe5Si2B": 
    cmap = "Purples"
elif TargetPhase == "Fe10Si4B3": 
    cmap = "Purples"

x0Blank = np.array([0.8,0.1,0.1])
Parameters=sysParam(x0Blank)
PD=Parameters[0]  

plotPhaseIdx = np.where(PD[0,:] == TargetPhase)[0][0]

# --------------------  Get data sets ------------------------------  
DirPathIn = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/"

with h5py.File(f'{DirPathIn}/CompData_{Htype}.hdf5', 'r') as fh:
    CompsAndHR = np.array([fh["CompData"]])[0]
    
DirPathOut = f"{DirPathIn}{Htype}/"
if not exists(DirPathOut):
    makedirs(DirPathOut)   
        

Nsamples = CompsAndHR.shape[0]

Time = np.zeros([Nsamples])
T = Time.copy()
ftot = Time.copy()
nit = Time.copy()

f = np.zeros([Nsamples,PD.shape[1]])
Rmean = f.copy()

with h5py.File(f'{DirPathOut}/CrystallizationData.hdf5', 'r') as fh:
    for idx in range(Nsamples):
        
        Time[idx] = np.array(fh[f'{idx}/t'])
        T[idx] = np.array(fh[f'{idx}/T'])
        ftot[idx] = np.array(fh[f'{idx}/ftot'])
        nit[idx] = np.array(fh[f'{idx}/nit'])
        
        f[idx,:] = np.array(fh[f'{idx}/f'])
        Rmean[idx,:] = np.array(fh[f'{idx}/rm'])


xFe = CompsAndHR[:,0] 
xSi = CompsAndHR[:,1]
xB = CompsAndHR[:,2]

HR = CompsAndHR[:,3]

if Htype == "Quenching":
    Hrs = np.unique(HR)[::-1]
else:
    Hrs = np.unique(HR)

Fe_Lim = np.array([0.7, 1.0]) 
Si_Lim = np.array([0.0, 0.3])
B_Lim = np.array([0.0, 0.3])

# ########################
# ########################
# ############################## Get end time, as from apply ML model, based on grids and interpolation.

# lb = [b[0] for b in bounds]  # Fe_min, Si_min, B_min
# ub = [b[1] for b in bounds] # Fe_max, Si_max, B_max
############# Tg Tm
bounds = [
    (0.70, 0.98),  # xFe bounds
    (0.01, 0.29),  # xSi
    (0.01, 0.29)   # xB
]


X_int, Tg_int, Tm_int = load_or_generate_Tg_Tm_grid_simplex(
    filename="FeSiB_TgTm_grid_simplex_50.npz",
    bounds=bounds,
    PD=PD,
    n_grid=50,                 # 50×50 grid (filtered by bounds)
    independent=("Fe", "Si"),  # treat Fe & Si as independent; B computed
)


X2 = X_int[:, :2]  # (xFe, xSi) since independent=("Fe","Si")

Tg_interp = LinearNDInterpolator(X2, Tg_int)
Tm_interp = LinearNDInterpolator(X2, Tm_int)

Tgs = np.empty(len(xFe))
Tms = Tgs.copy()

############# t_end, AM


Q_tend, tend_vals = load_or_generate_tend_grid_AM_TgTm(
    filename="FeSiB_tend_AM_TgTmLogHr.npz",
    Tg_vals=Tg_int,     # from Tg/Tm grid generation (true)
    Tm_vals=Tm_int,     # from Tg/Tm grid generation (true)
    logHr_bounds=(min(Hrs), max(Hrs)),
    n_hr=7,
    as_grid_hr=True
)

tend_interp = LinearNDInterpolator(Q_tend, tend_vals)

print("--- Data loaded ---")
#############
nanTic = 0
time_end = np.empty(len(xFe))
for i in range(len(xFe)):
    x_i = np.array([xFe[i],xSi[i],xB[i]])
    Hr_i = HR[i]
        
    Tg = Tg_interp(x_i[:2])[0]
    Tm = Tm_interp(x_i[:2])[0]
        
    Tgs[i] = Tg
    Tms[i] = Tm
    if np.isnan(Tg):
        nanTic += 1
        Tgs[i] = Tgs[i-1] 
        Tms[i] = Tms[i-1] 
       
    if Htype == "AM":
        q = np.array([Tgs[i], Tms[i], np.log(Hr_i)], dtype=float)
        t_end = tend_interp(q)[0]

        # Fallback if out of convex hull:
        if np.isnan(t_end):
            t_end = initialize_Tspline_AML(Tgs[i], Tms[i], Hr_i)
        
        time_end[i] = t_end

    else:
        if Htype == "Heating":
            time_end[i] = (Tm-Tg)/Hr_i
        elif Htype == "Quenching":
            time_end[i] = (Tm-Tg)/np.abs(Hr_i)
        elif Htype == "Iso":
            time_end[i] = 0.0

print(f"number of Tg nans: {nanTic}")
###### Time plot

mask = (
    (ftot < fmax) &
    (nit != 0) &
    (ftot > 0) &
    (np.all(f >= 0, axis=1)) &
    (~np.isnan(ftot))
)

CleanIdx = np.where(mask)[0]

### Print out corrupt data before cutting it
for i in range(len(Hrs)):
    v_i =  ftot[HR == Hrs[i]]
    print(f"-------- Hr: {Hrs[i]:1.1e} ---------")
    print(f"Samples in: {len(v_i)}")
    print(f"Smaples > 0.21%:  {len(ftot[HR == Hrs[i]][ftot[HR == Hrs[i]] > 0.21])}")
    print(f"Failed runs:  {len(ftot[HR == Hrs[i]][ftot[HR == Hrs[i]] == 0])}")
    print(f"Negative values:  {len(v_i[v_i < 0])}")
    print(f"Returning nan:  {len(ftot[HR == Hrs[i]][np.isnan(ftot[HR == Hrs[i]])])}")
    
    mask = (
        (ftot[HR == Hrs[i]] < fmax) &
        (nit[HR == Hrs[i]] != 0) &
        (v_i > 0) &
        (~np.isnan(ftot[HR == Hrs[i]]))
    )
    
    CleanIdx_i = np.where(mask)[0]
    print(f"Clean samples: {len(CleanIdx_i)}")

### Cut out corrupted data
ftot = ftot[CleanIdx]
f = f[CleanIdx,:]
Rmean = Rmean[CleanIdx,:]
Time = Time[CleanIdx]
xFe = xFe[CleanIdx]
xSi = xSi[CleanIdx]
xB = xB[CleanIdx]
HR = HR[CleanIdx]
time_end = time_end[CleanIdx]
T = T[CleanIdx]

data = np.hstack((ftot.reshape(-1,1),f[:, plotPhaseIdx].reshape(-1,1),
                  (Rmean[:, plotPhaseIdx]*1e9).reshape(-1,1), T.reshape(-1,1), Time.reshape(-1,1)))

######################## Plot delta times
    
fig1, ax1 = plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=-1,
    vmin=0,
    vmax=1,
    isLog=False,
    suptitle="|($t_{end}$-$t$)/$t$|, Target 20%",
    cbar_label="t (s)",
    per_panel_colorbars=False,
    cmap = 'viridis'
)
    
######################## f target

# vmin = min(ftot[ftot>0])
# # vmax = max(ftot)
# vmax = 0.2

fig2, ax2 = plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=1,
    vmin=0,
    vmax=20,
    isLog=False,
    suptitle=f"{PD[0,plotPhaseIdx]}, Target 20%",
    cbar_label="Vol (%)",
    per_panel_colorbars=False,
    cmap = cmap
)

######################## Rmedian
fig3, ax3 = plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=2,
    vmin=None,
    vmax=None,
    isLog=False,
    suptitle="Median radius, Target 20%",
    cbar_label="$r_{median}$ (nm)",
    per_panel_colorbars=True,
    cmap = cmap
)


######################## total volume
fig4, ax4 = plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=0,
    vmin=0,
    vmax=20,
    isLog=False,
    suptitle="Total vol, Target 20%",
    cbar_label="Total vol (%)",
    per_panel_colorbars=False,   # every subplot has one
    cmap = 'viridis'
)

######################## Temp
fig5, ax5 = plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=3,
    vmin=None,
    vmax=None,
    isLog=False,
    suptitle="T end, Target 20%",
    cbar_label="T (K)",
    per_panel_colorbars=False,   # every subplot has one
    cmap = 'plasma'
)

plt.show()

# FigpathOut=''.join([DirPathOut,f"Times_n{CompsAndHR.shape[0]}.png"])
# fig1.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Vol_{PD[0,plotPhaseIdx]}_n{CompsAndHR.shape[0]}.png"])
# fig2.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"Rmean_{PD[0,plotPhaseIdx]}_n{CompsAndHR.shape[0]}.png"])
# fig3.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"ftot_n{CompsAndHR.shape[0]}.png"])
# fig4.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([DirPathOut,f"T_n{CompsAndHR.shape[0]}.png"])
# fig5.savefig(FigpathOut, bbox_inches = "tight")

FigpathOut=''.join([DirPathOut,f"Times_n{CompsAndHR.shape[0]}.svg"])
fig1.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Vol_{PD[0,plotPhaseIdx]}_n{CompsAndHR.shape[0]}.svg"])
fig2.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"Rmean_{PD[0,plotPhaseIdx]}_n{CompsAndHR.shape[0]}.svg"])
fig3.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"ftot_n{CompsAndHR.shape[0]}.svg"])
fig4.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([DirPathOut,f"T_n{CompsAndHR.shape[0]}.svg"])
fig5.savefig(FigpathOut, bbox_inches = "tight")

