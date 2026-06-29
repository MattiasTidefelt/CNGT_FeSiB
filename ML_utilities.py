#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 22:49:29 2026

@author: ag0406
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, PowerTransformer
from os import makedirs
from os.path import exists
import h5py
import os
import matplotlib.colors as mcolors
import mpltern
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from paramFeSiB import sysParam
from ReadTempData import initialize_Tspline_AML
from LiqFeSiB import TransE
from GetTm import GetTm

#%%################ Loading data
################ 
################ 

def write_colIdx(Htype, colIdx, filename="colIdx_data.txt"):
    import os

    # Convert list to string
    colIdx_str = ",".join(map(str, colIdx))

    # Read existing lines
    lines = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    lines[key] = val

    # Update or add the line for the current Htype
    lines[Htype] = colIdx_str

    # Write back all lines
    with open(filename, "w") as f:
        for key, val in lines.items():
            f.write(f"{key}:{val}\n")

#%% reg data
def load_data(Htype):
    
    fmax = 0.4
    x0Blank = np.array([0.8,0.1,0.1])
    Parameters=sysParam(x0Blank)
    PD=Parameters[0]  
        
    # --------------------  Get data sets ------------------------------  
    DirPathIn = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/"

    with h5py.File(f'{DirPathIn}/CompData_{Htype}.hdf5', 'r') as fh:
        CompsAndHR = torch.from_numpy(np.array([fh["CompData"]])[0]).float()
        
    DirPathOut = f"{DirPathIn}{Htype}/"
    if not exists(DirPathOut):
        makedirs(DirPathOut)                  

    Nsamples = CompsAndHR.shape[0]

    Time = np.zeros([Nsamples])
    Temp = Time.copy()
    ftot = Time.copy()
    nit = Time.copy()

    f = np.zeros([Nsamples,PD.shape[1]])
    Rmean = f.copy()

    with h5py.File(f'{DirPathOut}/CrystallizationData.hdf5', 'r') as fh:
        for idx in range(Nsamples):
            
            Time[idx] = torch.from_numpy(np.array(fh[f'{idx}/t'])).float()
            Temp[idx] = torch.from_numpy(np.array(fh[f'{idx}/T'])).float()
            ftot[idx] = torch.from_numpy(np.array(fh[f'{idx}/ftot'])).float()
            nit[idx] = torch.from_numpy(np.array(fh[f'{idx}/nit'])).float()
            
            f[idx,:] = torch.from_numpy(np.array(fh[f'{idx}/f'])).float()
            Rmean[idx,:] = torch.from_numpy(np.array(fh[f'{idx}/rm'])).float()

    mask = (
        (ftot < fmax) &
        (nit != 0) &
        (ftot > 0) &
        (np.all(f >= 0, axis=1)) &
        (~np.isnan(ftot))
    )
    
    CleanIdx = np.where(mask)[0]

    Time = Time[CleanIdx]
    Temp = Temp[CleanIdx]
    ftot = ftot[CleanIdx]
    nit = nit[CleanIdx]
    
       
    X1 = CompsAndHR[CleanIdx,0]   # xFe
    X2 = CompsAndHR[CleanIdx,1]   # xSi
    X3 = CompsAndHR[CleanIdx,2]    # xB
    X4 = CompsAndHR[CleanIdx,3]    # HR
    
    f = f[CleanIdx,:]
    
    Rmean = Rmean[CleanIdx,:]*1e9 # to nm
    
    X4_log = torch.log(abs(X4)) 
    
    # take away data from phase that do not contribute to volume fraction in investigated Hrs to improve precision
    colIdx = []
    for i in range(f.shape[1]):
        if max(f[:,i]) < 5e-2:
            colIdx.append(i)
            
    PD_red = np.delete(PD,colIdx, axis = 1)
    f_red = np.delete(f, colIdx, axis=1)
    Rmean_red = np.delete(Rmean, colIdx, axis=1)
    print(f"Removed phases: {PD[0,colIdx]}, does not contribute to transformation")
    
    #     print("Number of times < 1e-7: ", len(Time[Time<1e-7]))
    #     print("Number of r < 1e-6: ", len(Rmean_red[Rmean_red<1e-6].ravel()))
    #     print(f"min Rmean: {Rmean_red.min()}, n Rmean min: {len(Rmean_red[np.where(Rmean_red == 0)])}")
    #     print("Number of f< 1e-6: ", len(f_red[f_red<1e-6].ravel()))
    #     print(f"min f: {f_red.min()}, n fmin: {len(f_red[np.where(f_red == 0)])}")
    
    # f_red[f_red<1e-3] = 1e-3
    # Rmean_red[Rmean_red<1e-3] = 1e-3
    # ftot[ftot<1e-3] = 1e-3
    
    f_red = np.log(1 + f_red)
    Rmean_red = np.log(1 + Rmean_red)
    Time = np.log(1 + Time)
    # Time = np.log(1 + Time*torch.from_numpy(X4))
    ftot = np.log(1 + ftot)

    f_red = f_red[:,PD_red[0,:]=="bcc"]
    Rmean_red = Rmean_red[:,PD_red[0,:]=="bcc"]    
        
    ###########################  Scale data
    ########################### 
    ########################### 
    
    # X4_scaler = MinMaxScaler(feature_range=(0, 1))
    # X4_scaler = StandardScaler()
    # X4_scaler = RobustScaler()
    X4_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='uniform')
    X4_scaled = X4_scaler.fit_transform(X4_log.reshape(-1, 1))
    
    # Hrs = np.unique(X4)
    Hrs = np.unique(X4_scaled)
    
    Scale = 1.
    # Scale = .5
    
    # R_scaler = MinMaxScaler((0, 1))
    # R_scaler = StandardScaler()
    # R_scaler = RobustScaler()
    # R_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='uniform')
    R_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    R_scaled = R_scaler.fit_transform(Rmean_red)
    R_scaled = 1/(1+np.exp(-R_scaled/Scale))
    
    # f_scaler = MinMaxScaler((0, 1))
    # f_scaler = StandardScaler()
    # f_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='uniform')
    f_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    f_scaled = f_scaler.fit_transform(f_red)
    f_scaled = 1/(1+np.exp(-f_scaled/Scale))
    
    # ftot_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='uniform')
    ftot_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
    ftot_scaled = 1/(1+np.exp(-ftot_scaled/Scale))
       
    # t_scaler =  MinMaxScaler((0, 1))
    # t_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='uniform')
    # t_scaler = RobustScaler()
    # t_scaler = StandardScaler()
    t_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
    t_scaled = 1/(1+np.exp(-t_scaled/Scale))
    
    #     # Plot Time to check validity
    #     fig, ax10 = plt.subplots(2, 2, figsize=(10, 8))
    #     ax10[0,0].set_title("Before (log-space)")
    #     ax10[0,0].hist(Time, bins=30)
        
    #     # Step 1: Gaussianize with QuantileTransformer
    #     t_scaler_ORG = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    #     t_z = t_scaler_ORG.fit_transform(Time.reshape(-1, 1))  # shape (N,1), approx standard normal
        
    #     # Step 2: Map normal -> (0,1) with scaled sigmoid
    #     t_sig = 1.0 / (1.0 + np.exp(-t_z / Scale))  # shape (N,1)
        
    #     ax10[0,1].set_title("After (bounded 0..1)")
    #     ax10[0,1].hist(t_sig.ravel(), bins=30)
        
    #     # --------------- INVERSE -----------------
        
    #     t_z_hat = Scale * logit(t_sig.ravel())              # shape (N,)
    #     t_z_hat = t_z_hat.reshape(-1, 1)
        
    #     # Step 2 (inverse): inverse of QuantileTransformer (back to log-space)
    #     Time_recon_log = t_scaler_ORG.inverse_transform(t_z_hat).ravel()  # log-space
        
    #     ax10[1,0].set_title("Scaled back (log-space)")
    #     ax10[1,0].hist(Time_recon_log, bins=30)
        
    #     # Diff in log-space (relative)
    #     den = np.where(Time_recon_log == 0, 1e-12, Time_recon_log)
    #     ax10[1,1].set_title("Diff (relative, log-space)")
    #     ax10[1,1].hist((Time - Time_recon_log) / den, bins=30)
    
    # y = np.hstack((f_red, R_scaled, t_scaled))
    # y = np.hstack((f_scaled, R_scaled, t_scaled))
    # y = np.hstack((f_red, Rmean_red, t_scaled))
    y = np.hstack((ftot_scaled, f_scaled.reshape(-1, 1), R_scaled.reshape(-1, 1), t_scaled))

    write_colIdx(Htype, colIdx)     # Store colIdx not used for Apply ML model
    
    ###########################     
    # More complicated data now since we need two sets of randomized boundary and interior points 
    train_frac=0.6
    val_frac=1- train_frac
    seed=42
    
    # Shuffle for reproducibility
    N = len(X1)
    all_idx = torch.arange(N)
    rng = torch.Generator().manual_seed(seed)
    all_idx_shuf = all_idx[torch.randperm(N, generator=rng)]

    # Split into train/validate data
    n_train = int(train_frac * len(all_idx_shuf))   
    train_idx = all_idx_shuf[:n_train]
    
    n_val = int(val_frac * len(all_idx_shuf))
    val_idx = all_idx_shuf[n_train:n_train + n_val]

    # Extract datasets from analytical data
    X = torch.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1), torch.from_numpy(X4_scaled.reshape(-1, 1)).float()))
    y = torch.tensor(y, dtype=torch.float32) 
    
    X_train = X[train_idx]
    y_train = y[train_idx] 

    X_val = X[val_idx]
    y_val = y[val_idx]  
    
    data = {
        "scaler": Scale,
        "X_all": X,
        "y_all": y,
        "R_scaler": R_scaler,
        "t_scaler": t_scaler,
        "X4_scaler": X4_scaler,
        "f_scaler": f_scaler,
        "ftot_scaler": ftot_scaler,
        "Hrs_raw": X4,
        "Hrs": Hrs,
        "Time": Time,
        "T": Temp,
        "DirPathOut": DirPathOut,
        "ftot": ftot,
        "nit": nit,
        "PD": PD_red,
        "train": {
            "X": X_train,
            "y": y_train
        },
        "val": {
            "X": X_val,
            "y": y_val
        }
    }

    return data

#%% k-fold data
def load_data_Kfold(Htype, Type, nfolds=5, test_frac=0.10, seed=42):
    
    fmax = 0.4
    x0Blank = np.array([0.8,0.1,0.1])
    Parameters=sysParam(x0Blank)
    PD=Parameters[0]  
        
    # --------------------  Get data sets ------------------------------  
    DirPathIn = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/"

    with h5py.File(f'{DirPathIn}/CompData_{Htype}.hdf5', 'r') as fh:
        CompsAndHR = torch.from_numpy(np.array([fh["CompData"]])[0]).float()
        
    DirPathOut = f"{DirPathIn}{Htype}/"
    if not exists(DirPathOut):
        makedirs(DirPathOut)                  

    Nsamples = CompsAndHR.shape[0]

    Time = np.zeros([Nsamples])
    Temp = Time.copy()
    ftot = Time.copy()
    nit = Time.copy()

    f = np.zeros([Nsamples,PD.shape[1]])
    Rmean = f.copy()

    with h5py.File(f'{DirPathOut}/CrystallizationData.hdf5', 'r') as fh:
        for idx in range(Nsamples):
            
            Time[idx] = torch.from_numpy(np.array(fh[f'{idx}/t'])).float()
            Temp[idx] = torch.from_numpy(np.array(fh[f'{idx}/T'])).float()
            ftot[idx] = torch.from_numpy(np.array(fh[f'{idx}/ftot'])).float()
            nit[idx] = torch.from_numpy(np.array(fh[f'{idx}/nit'])).float()
            
            f[idx,:] = torch.from_numpy(np.array(fh[f'{idx}/f'])).float()
            Rmean[idx,:] = torch.from_numpy(np.array(fh[f'{idx}/rm'])).float()

    mask = (
        (ftot < fmax) &
        (nit != 0) &
        (ftot > 0) &
        (np.all(f >= 0, axis=1)) &
        (~np.isnan(ftot))
    )
    
    CleanIdx = np.where(mask)[0]

    Time = Time[CleanIdx]
    Temp = Temp[CleanIdx]
    ftot = ftot[CleanIdx]
    nit = nit[CleanIdx]
    
       
    X1 = CompsAndHR[CleanIdx,0]   # xFe
    X2 = CompsAndHR[CleanIdx,1]   # xSi
    X3 = CompsAndHR[CleanIdx,2]    # xB
    X4 = CompsAndHR[CleanIdx,3]    # HR
    
    f = f[CleanIdx,:]
    
    Rmean = Rmean[CleanIdx,:]*1e9 # to nm
    
    # X4 = torch.log(abs(X4)) #log X4
    X4 = abs(X4) #log X4
    
    Time *= X4.numpy()
    
    # take away data from phase that do not contribute to volume fraction in investigated Hrs to improve precision
    colIdx = []
    for i in range(f.shape[1]):
        if max(f[:,i]) < 5e-2:
            colIdx.append(i)
            
    PD_red = np.delete(PD,colIdx, axis = 1)
    f_red = np.delete(f, colIdx, axis=1)
    Rmean_red = np.delete(Rmean, colIdx, axis=1)
    

    
    f_red = f_red[:,PD_red[0,:]=="bcc"]
    Rmean_red = Rmean_red[:,PD_red[0,:]=="bcc"] 
    print(f"Removed phases: {PD[0,colIdx]}, does not contribute to transformation")
    
    f_red[f_red<1e-3] = 1e-3
    Rmean_red[Rmean_red<1e-2] = 1e-2
    
    f_red[f_red>3e-1] = 3e-1
    Rmean_red[Rmean_red>1e2] = 1e2
    
    f_red = np.clip(f_red, 1e-3,None)
    ftot = np.clip(ftot,1e-3,None)
    Rmean_red = np.clip(Rmean_red,1e-1,None)
    
        
    ############ Test over sampling

        
    # # Ensure ftot is 2D
    # ftot = ftot.reshape(-1, 1)
    
    # Time = Time.reshape(-1, 1)
    
    # # Transition boundaries
    # ftot_vec = ftot[:, 0]      # flatten
    # f_bcc    = f_red[:, 0]
    
    # transition_mask = ((ftot_vec > 0.01) & (ftot_vec < 0.20)) | \
    #                   ((f_bcc    > 0.01) & (f_bcc    < 0.20))
    
    # print(f"Number of points that are used for over sampling {len(transition_mask[transition_mask==True])}")
    
    # # Extract transition rows
    # X1_tr  = X1[transition_mask]
    # X2_tr  = X2[transition_mask]
    # X3_tr  = X3[transition_mask]
    # X4_tr  = X4[transition_mask]
    # f_tr   = f_red[transition_mask]
    # R_tr   = Rmean_red[transition_mask]
    # ftot_tr = ftot[transition_mask]     # <-- ALWAYS (n,1)
    # Time_tr = Time[transition_mask]     # <-- ALWAYS (n,1)
    
    # # Oversample factor
    # k = 5
    
    # # Repeat rows k times (keeps column shape)
    # X1_rep = np.repeat(X1_tr, k)
    # X2_rep = np.repeat(X2_tr, k)
    # X3_rep = np.repeat(X3_tr, k)
    # X4_rep = np.repeat(X4_tr, k)
    
    # f_rep    = np.repeat(f_tr,    k, axis=0)
    # R_rep    = np.repeat(R_tr,    k, axis=0)
    # ftot_rep = np.repeat(ftot_tr, k, axis=0)  # <-- stays (n*k, 1)
    # Time_rep = np.repeat(Time_tr, k, axis=0)  # <-- stays (n*k, 1)
    
    # # Combine
    # X1 = torch.from_numpy(np.concatenate([X1, X1_rep]))
    # X2 = torch.from_numpy(np.concatenate([X2, X2_rep]))
    # X3 = torch.from_numpy(np.concatenate([X3, X3_rep]))
    # X4 = torch.from_numpy(np.concatenate([X4, X4_rep]))
    
    # f_red     = np.vstack([f_red,    f_rep])
    # Rmean_red = np.vstack([Rmean_red, R_rep])
    # ftot      = np.vstack([ftot,     ftot_rep])
    # Time      = np.vstack([Time,     Time_rep])
    
    #############################
    # Ensure ftot is 2D
    ftot = ftot.reshape(-1, 1)
    Time = Time.reshape(-1, 1)
    
    # Transition boundaries
    ftot_vec = ftot[:, 0]
    f_bcc = f_red[:, 0]
    
    transition_mask = ((ftot_vec > 0.001) & (ftot_vec < 0.20)) | \
                      ((f_bcc    > 0.001) & (f_bcc    < 0.20))
    
    print(f"Number of points that are used for over sampling {np.sum(transition_mask)}")
    
    # Extract transition rows
    X1_tr  = X1[transition_mask]
    X2_tr  = X2[transition_mask]
    X3_tr  = X3[transition_mask]
    X4_tr  = X4[transition_mask]
    
    f_tr    = f_red[transition_mask]
    R_tr    = Rmean_red[transition_mask]
    ftot_tr = ftot[transition_mask]
    Time_tr = Time[transition_mask]
    
    # Oversample factor
    k = 5
    
    # --- Noise scale (tune this!) ---
    eps_x   = 1e-3   # inputs
    # eps_f   = 1e-4   # fractions
    # eps_r   = 1e-2   # radii (nm scale dependent!)
    # eps_t   = 1e-3   # time
    
    # --- Repeat base ---
    X1_rep = np.repeat(X1_tr, k)
    X2_rep = np.repeat(X2_tr, k)
    X3_rep = np.repeat(X3_tr, k)
    X4_rep = np.repeat(X4_tr, k)
    
    f_rep    = np.repeat(f_tr,    k, axis=0)
    R_rep    = np.repeat(R_tr,    k, axis=0)
    ftot_rep = np.repeat(ftot_tr, k, axis=0)
    Time_rep = np.repeat(Time_tr, k, axis=0)
    
    # --- Add small perturbations ---
    rng = np.random.default_rng()
    
    # Inputs (composition etc.)
    X1_rep = X1_rep + eps_x * rng.normal(size=X1_rep.shape)
    X2_rep = X2_rep + eps_x * rng.normal(size=X2_rep.shape)
    X3_rep = X3_rep + eps_x * rng.normal(size=X3_rep.shape)
    # X4_rep = X4_rep + eps_x * rng.normal(size=X4_rep.shape)
    
    
  
    # --- Ensure 1D arrays before stacking ---
    X1_rep = X1_rep.reshape(-1)
    X2_rep = X2_rep.reshape(-1)
    X3_rep = X3_rep.reshape(-1)
    
    # --- Stack correctly ---
    X_stack = np.column_stack([X1_rep, X2_rep, X3_rep])  # safer than vstack.T
    
    # --- Clip (avoid negatives)
    X_stack = np.clip(X_stack, 1e-12, None)
    
    # --- Normalize rows
    X_stack /= np.sum(X_stack, axis=1, keepdims=True)
    
    # --- Unpack
    X1_rep = X_stack[:, 0]
    X2_rep = X_stack[:, 1]
    X3_rep = X_stack[:, 2]


    
    # Outputs
    # f_rep    = f_rep    + eps_f * rng.normal(size=f_rep.shape)
    # ftot_rep = ftot_rep + eps_f * rng.normal(size=ftot_rep.shape)
    # R_rep    = R_rep    + eps_r * rng.normal(size=R_rep.shape)
    # Time_rep = Time_rep + eps_t * rng.normal(size=Time_rep.shape)
    
    # --- Optional: enforce physical bounds ---
    
    # Fractions (if 0–1)
    # f_rep    = np.clip(f_rep, 0.0, 1.0)
    # ftot_rep = np.clip(ftot_rep, 0.0, 1.0)
    
    # # If using 0–100 instead, use:
    # # f_rep    = np.clip(f_rep, 0.0, 100.0)
    # # ftot_rep = np.clip(ftot_rep, 0.0, 100.0)
    
    # # Radii must be positive
    # R_rep = np.clip(R_rep, 0.0, None)
    
    # # Time must be positive
    # Time_rep = np.clip(Time_rep, 0.0, None)
    
    # --- Combine ---

    X1 = torch.from_numpy(np.concatenate([X1, X1_rep])).float()
    X2 = torch.from_numpy(np.concatenate([X2, X2_rep])).float()
    X3 = torch.from_numpy(np.concatenate([X3, X3_rep])).float()
    X4 = torch.from_numpy(np.concatenate([X4, X4_rep])).float()

    
    f_red     = np.vstack([f_red,    f_rep])
    Rmean_red = np.vstack([Rmean_red, R_rep])
    ftot      = np.vstack([ftot,     ftot_rep])
    Time      = np.vstack([Time,     Time_rep])
    ######################

    
    print("Number of times < 1e-5: ", len(Time[Time<1e-5]))
    print("Number of times > 1e-4: ", len(Time[Time>1e-4]))
    print(f"min time: {Time.min()}, n times_min: {len(Time[np.where(Time == Time.min())])}")
    print("Number of r < 1e-6: ", len(Rmean_red[Rmean_red<1e-6].ravel()))
    print("Number of r > 1e1: ", len(Rmean_red[Rmean_red>1e1].ravel()))
    print(f"min Rmean: {Rmean_red.min()}, n Rmean min: {len(Rmean_red[np.where(Rmean_red == Rmean_red.min())])}, max Rmean {Rmean_red.max()}")
    print("Number of f< 1e-6: ", len(f_red[f_red<1e-6].ravel()))
    print("Number of f > 2.2e-1: ", len(f_red[f_red>2.2e-1].ravel()))
    print(f"min f: {f_red.min()}, n fmin: {len(f_red[np.where(f_red == f_red.min())])}")
    print("Number of ftot< 1e-6: ", len(ftot[ftot<1e-6].ravel()))
    print("Number of ftot > 2.2e-1: ", len(ftot[ftot>2.2e-1].ravel()))
    print(f"min ftot: {ftot.min()}, n ftot_min: {len(ftot[np.where(ftot == ftot.min())])}")
    
    print(f"Total number of examples after over sampling: {len(Time)}")
    # Continue
    
    # f_red[f_red<1e-3] = 0
    # Rmean_red[Rmean_red<1e-3] = 0
    # ftot[ftot<1e-3] = 0
    
    # f_red[f_red>2e-1] = 2e-1
    # # Rmean_red[Rmean_red>1e1] = 1e1
    # ftot[ftot>2e-1] = 2e-1
    
    f_red *= 100
    ftot *= 100
           
    ###########################  Scale data
    ########################### 
    ########################### 
    
    # X4_scaler = MinMaxScaler(feature_range=(0, 1))
    # X4_scaler = StandardScaler()
    # X4_scaler = RobustScaler()
    X4_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
    X4_scaled = X4_scaler.fit_transform(X4.reshape(-1, 1))
    
    # Hrs = np.unique(X4)
    Hrs = np.unique(X4_scaled)
    
    Scale = 1.
    # Scale = 1.5
    # R_scaler = MinMaxScaler((0, 1))
    # R_scaler = RobustScaler()
    
    if Type == "Uniform":# Uniform [0,1]
        R_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        R_scaled = R_scaler.fit_transform(Rmean_red)
        
        f_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        f_scaled = f_scaler.fit_transform(f_red)
        
        ftot_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
           
        t_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))

    elif Type == "Normal":# Normal bounded by [0,1]

        R_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        R_scaled = R_scaler.fit_transform(Rmean_red)
        
        f_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        f_scaled = f_scaler.fit_transform(f_red)
        
        ftot_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
           
        t_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
    
    elif Type == "Custom":# Custom mix of scalers
        R_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        R_scaled = R_scaler.fit_transform(Rmean_red)
        
        f_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        f_scaled = f_scaler.fit_transform(f_red)
        
        ftot_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
           
        t_scaler = QuantileTransformer(n_quantiles=len(X4), random_state=1, output_distribution='uniform')
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
    
    elif Type == "NormalBounded":# Normal bounded by [0,1]

        R_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        R_scaled = R_scaler.fit_transform(Rmean_red)
        R_scaled = np.clip(R_scaled, -3, 3) # clip to +-3 std from 0, including 99.7% of values
        R_scaled = (R_scaled + 3) / 6
        
        f_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        f_scaled = f_scaler.fit_transform(f_red)
        f_scaled = np.clip(f_scaled, -3, 3)
        f_scaled = (f_scaled + 3) / 6
        
        ftot_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
        ftot_scaled = np.clip(ftot_scaled, -3, 3) 
        ftot_scaled = (ftot_scaled + 3) / 6
           
        t_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
        t_scaled = np.clip(t_scaled, -3, 3) 
        t_scaled = (t_scaled + 3) / 6
    
    elif Type == "NormalSigmoid":# Normal scaled to sigmoid bounded by [0,1]

        R_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        R_scaled = R_scaler.fit_transform(Rmean_red)
        R_scaled = 1/(1+np.exp(-R_scaled/Scale))
        
        f_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        f_scaled = f_scaler.fit_transform(f_red)
        f_scaled = 1/(1+np.exp(-f_scaled/Scale))
        
        ftot_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
        ftot_scaled = 1/(1+np.exp(-ftot_scaled/Scale))
           
        t_scaler = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
        t_scaled = 1/(1+np.exp(-t_scaled/Scale))
        
    elif Type == "StandardScaler":

        R_scaler = StandardScaler()
        R_scaled = R_scaler.fit_transform(Rmean_red)
        
        f_scaler = StandardScaler()
        f_scaled = f_scaler.fit_transform(f_red)
        
        ftot_scaler = StandardScaler()
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
           
        t_scaler = StandardScaler()
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
        
    elif Type == "Power":
        
        R_scaler = PowerTransformer()
        R_scaled = R_scaler.fit_transform(Rmean_red)
        
        f_scaler = PowerTransformer()
        f_scaled = f_scaler.fit_transform(f_red)
        
        ftot_scaler = PowerTransformer()
        ftot_scaled = ftot_scaler.fit_transform(ftot.reshape(-1, 1))
           
        t_scaler = PowerTransformer()
        t_scaled = t_scaler.fit_transform(Time.reshape(-1, 1))
    
    #     # Plot Time to check validity
    #     fig, ax10 = plt.subplots(2, 2, figsize=(10, 8))
    #     ax10[0,0].set_title("Before (log-space)")
    #     ax10[0,0].hist(Time, bins=30)
        
    #     # Step 1: Gaussianize with QuantileTransformer
    #     t_scaler_ORG = QuantileTransformer(n_quantiles=len(X4), output_distribution='normal')
    #     t_z = t_scaler_ORG.fit_transform(Time.reshape(-1, 1))  # shape (N,1), approx standard normal
        
    #     # Step 2: Map normal -> (0,1) with scaled sigmoid
    #     t_sig = 1.0 / (1.0 + np.exp(-t_z / Scale))  # shape (N,1)
        
    #     ax10[0,1].set_title("After (bounded 0..1)")
    #     ax10[0,1].hist(t_sig.ravel(), bins=30)
        
    #     # --------------- INVERSE -----------------
        
    #     t_z_hat = Scale * logit(t_sig.ravel())              # shape (N,)
    #     t_z_hat = t_z_hat.reshape(-1, 1)
        
    #     # Step 2 (inverse): inverse of QuantileTransformer (back to log-space)
    #     Time_recon_log = t_scaler_ORG.inverse_transform(t_z_hat).ravel()  # log-space
        
    #     ax10[1,0].set_title("Scaled back (log-space)")
    #     ax10[1,0].hist(Time_recon_log, bins=30)
        
    #     # Diff in log-space (relative)
    #     den = np.where(Time_recon_log == 0, 1e-12, Time_recon_log)
    #     ax10[1,1].set_title("Diff (relative, log-space)")
    #     ax10[1,1].hist((Time - Time_recon_log) / den, bins=30)
    
    # y = np.hstack((f_red, R_scaled, t_scaled))
    # y = np.hstack((f_scaled, R_scaled, t_scaled))
    # y = np.hstack((f_red, Rmean_red, t_scaled))
    # y = np.hstack((ftot_scaled, f_scaled.reshape(-1, 1), R_scaled.reshape(-1, 1), t_scaled))
    y = np.hstack((ftot_scaled, f_scaled.reshape(-1, 1), R_scaled.reshape(-1, 1), t_scaled))

    write_colIdx(Htype, colIdx)     # Store colIdx not used for Apply ML model
    
    ###########################

    # Extract datasets from analytical data
    X = torch.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1), torch.from_numpy(X4_scaled.reshape(-1, 1)).float()))
    y = torch.tensor(y, dtype=torch.float32) 
    
   
    # Total number of samples
    N = len(X)
    all_idx = torch.arange(N)

    # Shuffle
    rng = torch.Generator().manual_seed(seed)
    shuf_idx = all_idx[torch.randperm(N, generator=rng)]

    # ----------------------------
    # 2. Test split (10%)
    # ----------------------------
    n_test = int(test_frac * N)
    test_idx = shuf_idx[:n_test]
    opt_idx  = shuf_idx[n_test:]     # optimization data (for CV)

    X_test = X[test_idx]
    y_test = y[test_idx]

    X_opt  = X[opt_idx]
    y_opt  = y[opt_idx]

    # ----------------------------
    # 3. Package everything
    # ----------------------------
    data = {
        "scaler": Scale,
        "X_all": X,
        "y_all": y,
        "X_opt": X_opt,
        "y_opt": y_opt,
        "X_test": X_test,
        "y_test": y_test,
        "R_scaler": R_scaler,
        "t_scaler": t_scaler,
        "X4_scaler": X4_scaler,
        "f_scaler": f_scaler,
        "ftot_scaler": ftot_scaler,
        "Hrs_raw": X4,
        "Hrs": Hrs,
        "Time": Time,
        "T": Temp,
        "DirPathOut": DirPathOut,
        "ftot": ftot,
        "nit": nit,
        "PD": PD_red,
        "nfolds": nfolds
    }

    return data

def ReverseTransform(Y,Scale,Type,data):
    from scipy.special import logit
    
    R_scaler = data["R_scaler"]
    f_scaler = data["f_scaler"]
    ftot_scaler = data["ftot_scaler"]
    t_scaler = data["t_scaler"]  # list of MinMaxScaler for each heating rate
    
    if Type == "Uniform":
        Y[:,0] = ftot_scaler.inverse_transform(Y[:,0].reshape(-1, 1)).ravel() # ftot
        
        Y[:,1] = f_scaler.inverse_transform(Y[:,1].reshape(-1, 1)).ravel() # ftarget

        Y[:,2] = R_scaler.inverse_transform(Y[:,2].reshape(-1, 1)).ravel() # Scale back r

        Y[:,-1] = t_scaler.inverse_transform(Y[:,-1].reshape(-1, 1)).ravel() # time_end
        
    elif Type == "NormalBounded":
        Y[:,0] = Y[:,0] * 6 - 3
        Y[:,0] = ftot_scaler.inverse_transform(Y[:,0].reshape(-1, 1)).ravel()

        Y[:,1] = Y[:,1] * 6 - 3
        Y[:,1] = f_scaler.inverse_transform(Y[:,1].reshape(-1, 1)).ravel()

        Y[:,2] = Y[:,2] * 6 - 3
        Y[:,2] = R_scaler.inverse_transform(Y[:,2].reshape(-1, 1)).ravel() # Scale back r

        Y[:,-1] = Y[:,-1] * 6 - 3
        Y[:,-1] = t_scaler.inverse_transform(Y[:,-1].reshape(-1, 1)).ravel()
        
    elif Type == "NormalSigmoid":
        Y[:,0] = Scale*logit(Y[:,0])
        Y[:,0] = ftot_scaler.inverse_transform(Y[:,0].reshape(-1, 1)).ravel()

        Y[:,1] = Scale*logit(Y[:,1])
        Y[:,1] = f_scaler.inverse_transform(Y[:,1].reshape(-1, 1)).ravel()

        Y[:,2] = Scale*logit(Y[:,2])
        Y[:,2] = R_scaler.inverse_transform(Y[:,2].reshape(-1, 1)).ravel() # Scale back r

        Y[:,-1] = Scale*logit(Y[:,-1]) # scale back time
        Y[:,-1] = t_scaler.inverse_transform(Y[:,-1].reshape(-1, 1)).ravel()
        
    elif Type == "StandardScaler" or Type == "Power" or Type == "Normal":

        Y[:,0] = ftot_scaler.inverse_transform(Y[:,0].reshape(-1, 1)).ravel()

        Y[:,1] = f_scaler.inverse_transform(Y[:,1].reshape(-1, 1)).ravel()

        Y[:,2] = R_scaler.inverse_transform(Y[:,2].reshape(-1, 1)).ravel() # Scale back r

        Y[:,-1] = t_scaler.inverse_transform(Y[:,-1].reshape(-1, 1)).ravel()
        
    return Y
#%%################ Generate grids
################ 
################ 

def load_or_generate_Tg_Tm_grid_simplex(
    filename,
    bounds,
    PD=None,
    n_grid=50,
    seed=0,
    independent=("Fe", "Si"),
    save_fmt="npz",
    ):
    """
    Load X, Tg, Tm if file exists; otherwise build a structured ternary grid
    that satisfies sum(X)=1 and respects bounds, compute Tg/Tm, save, return.

    Parameters
    ----------
    filename : str
        File to read/write. Extension determines format (.npz or .csv).
    bounds : list of tuples
        [(xFe_min, xFe_max), (xSi_min, xSi_max), (xB_min, xB_max)]
        Bounds are enforced for the returned compositions.
    n_grid : int
        Grid density (points per axis for the two independent variables).
        Total points will be <= n_grid^2 after filtering by bounds.
    seed : int
        Included for API similarity; not used for structured grid (but kept).
    independent : tuple of str
        Which two components are treated as independent grid axes:
        ("Fe","Si"), ("Fe","B"), or ("Si","B").
        The third component is computed as 1 - sum(independent).
    save_fmt : str
        "npz" or "csv" (also inferred from filename extension if provided).

    Returns
    -------
    X : ndarray, shape (n,3)
        Compositions [xFe, xSi, xB] with sum=1 for each row.
    Tg : ndarray, shape (n,)
    Tm : ndarray, shape (n,)
    """

    # ------------------ Load if exists ------------------
    if os.path.exists(filename):
        print(f"[load_or_generate_simplex] Loading: {filename}")
        if filename.endswith(".npz"):
            data = np.load(filename)
            return data["X"], data["Tg"], data["Tm"]
        elif filename.endswith(".csv"):
            raw = np.loadtxt(filename, delimiter=",")
            X  = raw[:, :3]
            Tg = raw[:, 3]
            Tm = raw[:, 4]
            return X, Tg, Tm
        else:
            raise ValueError("Unsupported file extension. Use .npz or .csv.")

    print(f"[load_or_generate_simplex] File not found. Generating: {filename}")

    # Unpack bounds
    (fe_lo, fe_hi), (si_lo, si_hi), (b_lo, b_hi) = bounds

    fe = np.linspace(fe_lo, fe_hi, n_grid)
    si = np.linspace(si_lo, si_hi, n_grid)
    FE, SI = np.meshgrid(fe, si, indexing="ij")
    B = 1.0 - FE - SI

    mask = (B >= b_lo) & (B <= b_hi)
    X = np.column_stack([FE[mask], SI[mask], B[mask]])
    
    # Final safety: enforce bounds (numerical tolerance) and sum=1
    eps = 1e-6
    fe_ok = (X[:,0] >= fe_lo - eps) & (X[:,0] <= fe_hi + eps)
    si_ok = (X[:,1] >= si_lo - eps) & (X[:,1] <= si_hi + eps)
    b_ok  = (X[:,2] >= b_lo  - eps) & (X[:,2] <= b_hi  + eps)
    sum_ok = np.abs(X.sum(axis=1) - 1.0) <= 1e-10
    X = X[fe_ok & si_ok & b_ok & sum_ok]

    if X.shape[0] == 0:
        raise RuntimeError(
            "No valid compositions after applying simplex constraint and bounds.\n"
            "Try relaxing bounds, changing independent axes, or reducing n_grid."
        )

    # Compute Tg, Tm on this grid
    n = X.shape[0]
    Tg = np.empty(n, dtype=float)
    Tm = np.empty(n, dtype=float)

    print(f"[load_or_generate_simplex] Grid points kept: {n:,} (from <= {n_grid*n_grid:,})")
    for i in range(n):
        [_, Tg[i]] = TransE(1000, X[i,0], X[i,1], X[i,2])
        Tm[i] = GetTm(PD, X[i,:])

    # Save results
    if filename.endswith(".npz") or save_fmt.lower() == "npz":
        np.savez(filename, X=X, Tg=Tg, Tm=Tm)
    elif filename.endswith(".csv") or save_fmt.lower() == "csv":
        out = np.column_stack([X, Tg, Tm])
        np.savetxt(filename, out, delimiter=",",
                   header="xFe,xSi,xB,Tg,Tm", comments="")
    else:
        raise ValueError("Unsupported save format. Use .npz or .csv.")

    print("[load_or_generate_simplex] Saved:", filename)
    return X, Tg, Tm

def load_or_generate_tend_grid_AM_TgTm(
    filename,
    Tg_vals,         # "true" Tg values from Tg/Tm grid generation
    Tm_vals,         # "true" Tm values from Tg/Tm grid generation
    logHr_bounds,    # (logHr_min, logHr_max)
    n_hr=80,
    seed=0,
    as_grid_hr=False,
    min_delta_T=1e-9,
):
    """
    Builds/loads a dataset for AM:
        inputs:  [Tg, Tm, logHr]
        output:  t_end = initialize_Tspline_AML(Tg, Tm, exp(logHr))

    Saves as .npz or .csv like your other function.
    """

    if os.path.exists(filename):
        print(f"[load_or_generate_tend] Loading: {filename}")
        if filename.endswith(".npz"):
            d = np.load(filename)
            return d["Q"], d["t_end"]
        elif filename.endswith(".csv"):
            raw = np.loadtxt(filename, delimiter=",")
            return raw[:, :3], raw[:, 3]
        else:
            raise ValueError("Unsupported extension (use .npz or .csv).")

    print(f"[load_or_generate_tend] File not found. Generating: {filename}")

    rng = np.random.default_rng(seed)

    if as_grid_hr:
        logHr = np.linspace(logHr_bounds[0], logHr_bounds[1], n_hr)
    else:
        logHr = rng.uniform(logHr_bounds[0], logHr_bounds[1], n_hr)

    Tg_vals = np.asarray(Tg_vals, dtype=float).reshape(-1)
    Tm_vals = np.asarray(Tm_vals, dtype=float).reshape(-1)

    # Keep only valid Tg/Tm pairs
    valid = np.isfinite(Tg_vals) & np.isfinite(Tm_vals) & (Tm_vals > Tg_vals + min_delta_T)
    Tg_ok = Tg_vals[valid]
    Tm_ok = Tm_vals[valid]

    if Tg_ok.size == 0:
        raise RuntimeError("No valid Tg/Tm points to build AM t_end dataset.")

    Q_list = []
    t_list = []

    for lh in logHr:
        Hr = float(np.exp(lh))
        for Tg, Tm in zip(Tg_ok, Tm_ok):
            try:
                te = initialize_Tspline_AML(float(Tg), float(Tm), Hr)
                if np.isfinite(te):
                    Q_list.append([Tg, Tm, lh])
                    t_list.append(te)
            except Exception:
                # Skip invalid combos
                print("skipping")
                pass

    Q = np.asarray(Q_list, dtype=float)
    t_end = np.asarray(t_list, dtype=float)

    if Q.shape[0] < 10:
        raise RuntimeError("Too few valid AM tend points generated. Adjust bounds or n_hr.")

    if filename.endswith(".npz"):
        np.savez(filename, Q=Q, t_end=t_end)
    elif filename.endswith(".csv"):
        out = np.column_stack([Q, t_end])
        np.savetxt(filename, out, delimiter=",", header="Tg,Tm,logHr,t_end", comments="")
    else:
        raise ValueError("Unsupported extension (use .npz or .csv).")

    print("[load_or_generate_tend] Saved:", filename, "points:", Q.shape[0])
    return Q, t_end

# from scipy.interpolate import LinearNDInterpolator

#%%################ Network functions
################ 
################ 

def costFunction(Input_data, model, lam=0.0):
    X_true = Input_data["X"]
    y_true = Input_data["y"]
    # costFunc = torch.nn.MSELoss()
    y_pred = model(X_true)
    # mse_loss = costFunc(y_pred, y_true)
    # reg_loss = lam * sum(torch.norm(p) for p in model.parameters()) #Forb
    # reg_loss = lam * sum(p.abs().sum() for p in model.parameters()) #L1
    reg_loss = lam * sum((p**2).sum() for p in model.parameters()) #L2
    
    # Compute squared error
    sq_error = (y_pred - y_true)**2       # shape: (batch, n_outputs)
    
    # Extract ftot and f columns  (assuming they are outputs 0 and 1)
    # Change indices to match your output layout!
    ftot_true = y_true[:, 0]
    f_true    = y_true[:, 1]
    
    # Build weights (per-sample)
    # Weight only for ftot and f — others keep weight 1
    weight_ftot = torch.where(
        (ftot_true > 0.01) & (ftot_true < 0.2),
        torch.tensor(1.0, device=y_true.device),   # boosted weight
        torch.tensor(1.0, device=y_true.device)
    )
    
    weight_f = torch.where(
        (f_true > 0.01) & (f_true < 0.2),
        torch.tensor(1.0, device=y_true.device),
        torch.tensor(1.0, device=y_true.device)
    )
    
    # Combine weights — take max so either ftot or f triggers a higher weight
    sample_weights = torch.maximum(weight_ftot, weight_f)
    
    # Reshape to broadcast
    sample_weights = sample_weights.unsqueeze(1)    # (batch, 1)
    
    # Weighted MSE
    weighted_mse = (sample_weights * sq_error).mean()


    # return mse_loss + reg_loss
    return weighted_mse + reg_loss

def Network(input_dimension, hidden_layers, output_dimension, activation="tanh", dropout=0.0):
    act_map = {
        # "relu": torch.nn.ReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "silu": torch.nn.SiLU(),
        "gelu": torch.nn.GELU()
    }

    modules = []
    # First layer
    modules.append(torch.nn.Linear(input_dimension, hidden_layers[0]))
    modules.append(act_map[activation])
    if dropout > 0:
        modules.append(torch.nn.Dropout(dropout))

    # Hidden layers    
    for i in range(len(hidden_layers) - 1):
        modules.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        modules.append(act_map[activation])
        # modules.append(torch.nn.Sigmoid())
        if dropout > 0:
            modules.append(torch.nn.Dropout(dropout))

    # Output layer
    modules.append(torch.nn.Linear(hidden_layers[-1], output_dimension))
    modules.append(torch.nn.Sigmoid())
    # modules.append(torch.nn.Softplus())
    # modules.append(torch.nn.ReLU())
    # modules.append(torch.nn.LeakyReLU())

    return torch.nn.Sequential(*modules)

def train(model,
          epochs,
          opt,
          train_val_data,
          lr=1e-3,
          lam=0.0,
          batch_size=None,
          patience=None,
          print_every=100,
          device=None):
    """
    model: torch.nn.Module
    epochs: int
    opt: 'Adam' or 'LBFGS'
    train_val_data: {"train": {"X": ..., "y": ...}, "val": {"X": ..., "y": ...}}
    lr, lam: floats
    batch_size: int or None (used only for Adam). LBFGS is full-batch.
    patience: int or None for early stopping on val loss
    print_every: logging frequency
    device: torch.device or None (if None, inferred from model)
    """

    # --- Setup device ---
    if device is None:
        device = next(model.parameters()).device

    # Move data to device
    X_tr = train_val_data["train"]["X"].to(device)
    y_tr = train_val_data["train"]["y"].to(device)
    X_va = train_val_data["val"]["X"].to(device)
    y_va = train_val_data["val"]["y"].to(device)
        
    # Optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'LBFGS':
        # Note: LBFGS ignores batch_size; must use full batch with closure
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


    Cost_train = []
    Cost_val = []


    best_val = float("inf")
    best_state = None
    no_improve = 0

    # Mini-batching setup (Adam only)
    if opt == 'Adam' and batch_size is not None and batch_size > 0:
        n = X_tr.shape[0]
        # integer number of batches per epoch
        def batch_indices():
            idx = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                yield idx[start:end]

    for epoch in range(1, epochs + 1):
        model.train()

        if opt == 'Adam' and batch_size is not None and batch_size > 0:
            # --- Adam mini-batch ---
            epoch_train_losses = []
            for bidx in batch_indices():
                optimizer.zero_grad()
                train_batch = {"X": X_tr[bidx], "y": y_tr[bidx]}
                loss = costFunction(train_batch, model, lam=lam)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.detach().item())
            train_loss = float(torch.tensor(epoch_train_losses).mean().item())
        else:
            # --- Full-batch (Adam or LBFGS) ---
            if opt == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    loss = costFunction({"X": X_tr, "y": y_tr}, model, lam=lam)
                    loss.backward()
                    return loss
                train_loss = optimizer.step(closure).detach().item()
            else:
                optimizer.zero_grad()
                loss = costFunction({"X": X_tr, "y": y_tr}, model, lam=lam)
                loss.backward()
                optimizer.step()
                # train_loss = loss.detach().item()
                loss = float(closure().detach())

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_loss = costFunction({"X": X_va, "y": y_va}, model, lam=0.0).item()  # no reg on val by default

        Cost_train.append(train_loss)
        Cost_val.append(val_loss)

        # Logging
        if print_every is not None:
            if (opt == 'Adam' and epoch % print_every == 0) or (opt == 'LBFGS' and epoch % max(1, print_every // 10) == 0):
                print(f"[{opt}] Epoch {epoch:4d} | Train: {train_loss:.6e} | Val: {val_loss:.6e}")

        # Early stopping
        if patience is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    # restore best weights
                    if best_state is not None:
                        model.load_state_dict(best_state)
                        print(f"Final epoch of training: {epoch:4d}")
                    break

    return Cost_train, Cost_val

#%%############### Plod panels


def sci_fmt(val, unit="K/s", sign=True):
    # val is the float heating rate (physical)
    # sign=True keeps sign for quenching (negative)
    if not sign:
        val = abs(val)

    coeff, exp = f"{val:1.1e}".split("e")
    coeff = float(coeff)                       # e.g. '5.0'
    exp = int(exp)                             # e.g. '+05' -> 5

    return rf"$\phi: {coeff:.1f}\cdot 10^{{{exp}}}$ {unit}"

################
################
################ Comapre true vs predicted
def plot_ternary_rows(
    y_true, y_pred,
    xFe_true, xSi_true, xB_true, HR_true,
    xFe_pred, xSi_pred, xB_pred, HR_pred,
    Hrs, TrueHrs, Htype,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=0,
    vmin=None,             # scalar or None
    vmax=None,             # scalar or None
    isLog=False,
    suptitle="Figure",
    top_cbar_label="True",
    bottom_cbar_label="Pred",
    per_panel_colorbars=True,
    figsize=None,
    cmap='viridis'
):
    """
    Exactly mirrors your original loop, parameterized:
      - Plots y_true[:, v_index] (top row) and y_pred[:, v_index] (bottom row).
      - If vmin/vmax is None: compute per-panel from y_true[mask_t, -1].
      - If vmin/vmax is scalar: use same for all panels.
      - Colorbars: per_panel_colorbars=True -> every panel; False -> only last column in each row.
    """
        
    if figsize is None:
        figsize = (4 * len(Hrs), 5)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle)
    axes = []

    for i in range(len(Hrs)):
        # --- Extract TRUE data for this HR ---
        mask_t = (HR_true == Hrs[i])
        v_t = y_true[mask_t, v_index]
        # if v_index == -1:
        #     v_t *= np.exp(TrueHrs[i])
        t_t = xFe_true[mask_t]
        l_t = xSi_true[mask_t]
        r_t = xB_true[mask_t]

        # Per-panel scale: default from y_true[mask_t, -1], unless user passed vmin/vmax
        if (vmin is None) or (vmax is None):
            # Guard for empty mask to avoid min/max on empty slice
            if per_panel_colorbars:
                vmin_i = y_true[mask_t, v_index].min()
                vmax_i = y_true[mask_t, v_index].max()
                # if v_index == -1:
                #     vmin_i *= np.exp(TrueHrs[i])
                #     vmax_i *= np.exp(TrueHrs[i])
            else:
                vmin_i = y_true[:,v_index].min()
                vmax_i = y_true[:,v_index].max()
        else:
            vmin_i = vmin
            vmax_i = vmax
        
        # if v_index == -1 and i == 0:
        #     vmin_i = 1e10
        #     vmax_i = 0
        #     for j in range(len(Hrs)):
        #         vmin_j = (y_true[HR_true == Hrs[j],v_index]*np.exp(TrueHrs[j])).min()
        #         vmax_j = (y_true[HR_true == Hrs[j],v_index]*np.exp(TrueHrs[j])).max()
        #         if vmin_j < vmin_i:
        #             vmin_i = vmin_j
        #         if vmax_j >vmax_i:
        #             vmax_i = vmax_j
        
            
            
        # ===== First row (True) =====
        ax_top = fig.add_subplot(2, len(Hrs), i + 1, projection='ternary')
        axes.append(ax_top)
        


        if v_t.size:
            if not isLog:
                cs_top = ax_top.tripcolor(t_t, l_t, r_t, v_t,
                                          shading='flat', vmin=vmin_i, vmax=vmax_i, cmap=cmap)
            else:
                # norm = mcolors.LogNorm(vmin=vmin_i, vmax=vmax_i)
                norm = mcolors.SymLogNorm(linthresh=1e-6, vmin=vmin_i, vmax=vmax_i)
                cs_top = ax_top.tripcolor(t_t, l_t, r_t, v_t,
                                          shading='flat', norm=norm, cmap=cmap)
        else:
            cs_top = None

        ax_top.set_tlabel('Fe')
        ax_top.set_llabel('Si')
        ax_top.set_rlabel('B')

        ax_top.taxis.set_ticks(np.linspace(Fe_Lim[0], Fe_Lim[1], 3))
        ax_top.laxis.set_ticks(np.linspace(Si_Lim[0], Si_Lim[1], 3))
        ax_top.raxis.set_ticks(np.linspace(B_Lim[0], B_Lim[1], 3))

        if Htype == "Quenching":
            # hr_val = -np.exp(abs(TrueHrs[i]))
            hr_val = -abs(TrueHrs[i])
            ax_top.set_title(sci_fmt(hr_val, unit="K/s"))
        elif Htype in ("Heating","AM"):
            # hr_val = np.exp(TrueHrs[i])
            hr_val = TrueHrs[i]
            ax_top.set_title(sci_fmt(hr_val, unit="K/s"))
        elif Htype == "Iso":
            hr_val = np.exp(TrueHrs[i])
            hr_val = TrueHrs[i]
            ax_top.set_title(sci_fmt(hr_val, unit="K"))

        # Limits
        ax_top.set_tlim(Fe_Lim[0], Fe_Lim[1])
        ax_top.set_llim(Si_Lim[0], Si_Lim[1])
        ax_top.set_rlim(B_Lim[0], B_Lim[1])

        # Colorbar for TRUE row
        if (cs_top is not None) and (per_panel_colorbars or (i == len(Hrs) - 1)):
            cax = ax_top.inset_axes([1.3, 0.1, 0.05, 0.9], transform=ax_top.transAxes)
            cb = fig.colorbar(cs_top, cax=cax)
            # cb.set_ticks(np.linspace(cb.vmin, cb.vmax, 3))
            # cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
            cb.set_label(top_cbar_label, rotation=270, va='baseline')

        # --- Extract PRED data for this HR ---
        mask_p = (HR_pred == Hrs[i])
        v_p = y_pred[mask_p, v_index]
        # if v_index == -1:
        #     v_p *= np.exp(TrueHrs[i])
        t_p = xFe_pred[mask_p]
        l_p = xSi_pred[mask_p]
        r_p = xB_pred[mask_p]

        # ===== Second row (Pred) =====
        ax_bot = fig.add_subplot(2, len(Hrs), len(Hrs) + i + 1, projection='ternary')
        axes.append(ax_bot)

        if v_p.size:
            if not isLog:
                cs_bot = ax_bot.tripcolor(t_p, l_p, r_p, v_p,
                                          shading='flat', vmin=vmin_i, vmax=vmax_i, cmap=cmap)
            else:
                # norm = mcolors.LogNorm(vmin=vmin_i, vmax=vmax_i)
                norm = mcolors.SymLogNorm(linthresh=1e-6, vmin=vmin_i, vmax=vmax_i)
                cs_bot = ax_bot.tripcolor(t_p, l_p, r_p, v_p,
                                          shading='flat', norm=norm, cmap=cmap)
        else:
            cs_bot = None

        ax_bot.set_tlabel('Fe')
        ax_bot.set_llabel('Si')
        ax_bot.set_rlabel('B')
        
        
        ax_bot.taxis.set_ticks(np.linspace(Fe_Lim[0], Fe_Lim[1], 3))
        ax_bot.laxis.set_ticks(np.linspace(Si_Lim[0], Si_Lim[1], 3))
        ax_bot.raxis.set_ticks(np.linspace(B_Lim[0], B_Lim[1], 3))


        # Limits
        ax_bot.set_tlim(Fe_Lim[0], Fe_Lim[1])
        ax_bot.set_llim(Si_Lim[0], Si_Lim[1])
        ax_bot.set_rlim(B_Lim[0], B_Lim[1])

        # Colorbar for PRED row
        if (cs_bot is not None) and (per_panel_colorbars or (i == len(Hrs) - 1)):
            cax = ax_bot.inset_axes([1.3, 0.1, 0.05, 0.9], transform=ax_bot.transAxes)
            cb = fig.colorbar(cs_bot, cax=cax)
            # cb.set_ticks(np.linspace(cb.vmin, cb.vmax, 3))
            # cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
            cb.set_label(bottom_cbar_label, rotation=270, va='baseline')

    # Adjust spacing to avoid overlap
    plt.subplots_adjust(wspace=1, hspace=0.5, top=0.85, bottom=0.08)
    return fig, axes

################
################
################ Plot CNGT data
def plot_ternary_data(
    data,
    xFe, xSi, xB, HR,
    Hrs, Htype, time_end,
    Fe_Lim, Si_Lim, B_Lim,
    v_index=0,
    vmin=None,             # scalar or None
    vmax=None,             # scalar or None
    isLog=False,
    suptitle="Figure",
    cbar_label="True",
    per_panel_colorbars=True,
    figsize=None,
    cmap='viridis'
):
    """
    Exactly mirrors your original loop, parameterized:
      - Plots y_true[:, v_index] (top row) and y_pred[:, v_index] (bottom row).
      - If vmin/vmax is None: compute per-panel from y_true[mask_t, -1].
      - If vmin/vmax is scalar: use same for all panels.
      - Colorbars: per_panel_colorbars=True -> every panel; False -> only last column in each row.
    """
        
    if figsize is None:
        figsize = (4*len(Hrs),3.7)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle)
    axes = []

    for i in range(len(Hrs)):
        # --- Extract data for this HR ---
        mask_t = (HR== Hrs[i])
        v_t = data[mask_t,v_index]
        if v_index == 0 or v_index == 1:
            v_t *= 100
        time_end_i = time_end[HR == Hrs[i]]
        if v_index == -1:
            v_t = np.abs((time_end_i-v_t)/v_t)
            v_t[v_t>1] = 1
            v_t[v_t<0.2] = 0
        t_t = xFe[mask_t]
        l_t = xSi[mask_t]
        r_t = xB[mask_t]

        # Per-panel scale: default from y_true[mask_t, -1], unless user passed vmin/vmax
        if (vmin is None) or (vmax is None):
            # Guard for empty mask to avoid min/max on empty slice
            if per_panel_colorbars:
                vmin_i = data[mask_t,v_index].min()
                vmax_i = data[mask_t,v_index].max()
            else:
                vmin_i = data[:,v_index].min()
                vmax_i = data[:,v_index].max()
        else:
            vmin_i = vmin
            vmax_i = vmax
            
        # ===== First row (True) =====
        ax_top = fig.add_subplot(1, len(Hrs), i + 1, projection='ternary')
        axes.append(ax_top)
        
        if v_t.size:
            if not isLog:
                cs_top = ax_top.tripcolor(t_t, l_t, r_t, v_t,
                                          shading='flat', vmin=vmin_i, vmax=vmax_i, cmap=cmap)
            else:
                # norm = mcolors.LogNorm(vmin=vmin_i, vmax=vmax_i)
                norm = mcolors.SymLogNorm(linthresh=1e-6, vmin=vmin_i, vmax=vmax_i)
                cs_top = ax_top.tripcolor(t_t, l_t, r_t, v_t,
                                          shading='flat', norm=norm, cmap=cmap)
        else:
            cs_top = None

        ax_top.set_tlabel('Fe')
        ax_top.set_llabel('Si')
        ax_top.set_rlabel('B')

        ax_top.taxis.set_ticks(np.linspace(Fe_Lim[0], Fe_Lim[1], 3))
        ax_top.laxis.set_ticks(np.linspace(Si_Lim[0], Si_Lim[1], 3))
        ax_top.raxis.set_ticks(np.linspace(B_Lim[0], B_Lim[1], 3))

        if Htype == "Quenching":
            ax_top.set_title(sci_fmt(Hrs[i], unit="K/s"))
        elif Htype in ("Heating","AM"):
            ax_top.set_title(sci_fmt(Hrs[i], unit="K/s"))
        elif Htype == "Iso":
            ax_top.set_title(sci_fmt(Hrs[i], unit="K"))

        # Limits
        ax_top.set_tlim(Fe_Lim[0], Fe_Lim[1])
        ax_top.set_llim(Si_Lim[0], Si_Lim[1])
        ax_top.set_rlim(B_Lim[0], B_Lim[1])

        # Colorbar for TRUE row
        if (cs_top is not None) and (per_panel_colorbars or (i == len(Hrs) - 1)):
            cax = ax_top.inset_axes([1.07, 0.1, 0.05, 0.9], transform=ax_top.transAxes)
            cb = fig.colorbar(cs_top, cax=cax)
            # cb.set_ticks(np.linspace(cb.vmin, cb.vmax, 3))
            # cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
            cb.set_label(cbar_label, rotation=270, va='baseline')

    # Adjust spacing to avoid overlap
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    return fig, axes

#%% More stuff


