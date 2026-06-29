#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:31:58 2026

@author: ag0406
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import time


from SolidSolFeSiB import SolidSolFeSiB

R=8.31446261815324  

T = 1000
PD = "bcc"

SolidSolFeSiB(T,PD)
from SolidSolFeSiB import Gibbs_SolidSol
from SolidSolFeSiB import TransGMO 

dx = 1e-6

xB = dx
xFe = np.linspace(0.7,1-2*dx,50)
xSi = 1 -xB - xFe

# xB = 0.01
# xFe = np.linspace(0.7,0.98,50)
# xSi = 1 - xFe

[G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_SolidSol[0]
[G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_SolidSol[1]
[G_Liq_FeSi_0,G_Liq_FeSi_1,G_Liq_FeSi_2,G_Liq_FeSi_3]=Gibbs_SolidSol[2]
[G_Liq_SiB_0,G_Liq_SiB_1,G_Liq_SiB_2]=Gibbs_SolidSol[3]
[G_Liq_FeSiB_0,G_Liq_FeSiB_1,G_Liq_FeSiB_2]=Gibbs_SolidSol[4]

L1 = -2*1260*R
   
T=float(T)


#--------------Gibbs Matrix Zr60Cu30Al10-------------

def FindGm_A2(T, PD, xB, xSi, xFe):
    
    Gref=xB*G_Liq_B_0 + xSi*G_Liq_Si_0 + xFe*G_Liq_Fe_0   
    
    Gide=R*T*(xB*np.log(xB) + xSi*np.log(xSi) + xFe*np.log(xFe))
    
    GexBin=(xFe*xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB-xFe)**1 + G_Liq_FeB_2*(xB-xFe)**2 )
            + xFe*xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe-xSi)**1 + G_Liq_FeSi_2*(xFe-xSi)**2 + G_Liq_FeSi_3*(xFe-xSi)**3)
          + xB*xSi*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB-xSi)**1 + G_Liq_SiB_2*(xB-xSi)**2))
    
    Gex=GexBin + xB*xFe*xSi*((xB+0)*G_Liq_FeSiB_0 + (xFe+0)*G_Liq_FeSiB_1 + (xSi+0)*G_Liq_FeSiB_2)
    
    Gmo = TransGMO(T,xFe,xSi,xB,PD)
    
    Gm = Gref + Gide + Gex + Gmo
    
    return Gm

def Gm_B2_ordered_from_eta(eta, xFe, xSi, T, L1):

    # Sublattice site fractions from eta
    y1_Si = xSi + eta
    y2_Si = xSi - eta

    y1_Fe = 1 - y1_Si
    y2_Fe = 1 - y2_Si

    # --- Reference term ---
    Gref = 0.5*(y1_Fe*G_Liq_Fe_0 + y1_Si*G_Liq_Si_0 +
                y2_Fe*G_Liq_Fe_0 + y2_Si*G_Liq_Si_0)

    # --- Ideal entropy ---
    Gide = 0.5*R*T*(y1_Fe*np.log(y1_Fe) + y1_Si*np.log(y1_Si) +
                    y2_Fe*np.log(y2_Fe) + y2_Si*np.log(y2_Si))

    # --- Ordering enthalpy ---
    G_ord = L1*(y1_Fe*y2_Si + y1_Si*y2_Fe)

    return Gref + Gide + G_ord

def Gm_B2_ordered_ref(xFe, xSi, T, L1):
    return Gm_B2_ordered_from_eta(0.0, xFe, xSi, T, L1)


def G_bcc_from_eta(eta, G_A2_const, xFe, xSi, T, L1):
    G_ord = Gm_B2_ordered_from_eta(eta, xFe, xSi, T, L1)
    G_ref = Gm_B2_ordered_ref(xFe, xSi, T, L1)
    return G_A2_const + (G_ord - G_ref)


def minimize_eta_and_get_sites(G_A2_const, xFe, xSi, xB, T, L1):
    # Allowed eta interval ensuring 0 ≤ y ≤ 1
    eta_min = max(-xSi, xSi - 1)
    eta_max = min( xSi, 1 - xSi )


    # If ordering impossible, eta = 0
    if eta_max - eta_min < 1e-14:
        eta_eq = 0.0
        G_bcc = G_bcc_from_eta(eta_eq, G_A2_const, xFe, xSi, T, L1)
    else:
        # Minimize Gibbs wrt eta
        res = minimize_scalar(
            lambda eta: G_bcc_from_eta(eta, G_A2_const, xFe, xSi, T, L1),
            bounds=(eta_min, eta_max),
            method="bounded"
        )
        eta_eq = res.x
        G_bcc = res.fun
        if eta_eq < 0:
            eta_eq = -eta_eq

    # Recover site fractions
    y1_Si = xSi + eta_eq
    y2_Si = xSi - eta_eq
    y1_Fe = 1 - y1_Si
    y2_Fe = 1 - y2_Si

    return G_bcc, eta_eq, y1_Si, y2_Si, y1_Fe, y2_Fe


Gm_B2 = np.empty(len(xFe))
Gm_A2 = Gm_B2.copy()
y1_Si = Gm_B2.copy()
y2_Si = Gm_B2.copy()

fig = plt.figure(666)
fig.suptitle("bcc A2-B2 transition: (Fe,Si,B)(Fe,Si,B)")
ax = fig.subplots(2)

simt = time.time()

for i in range(len(xFe)):    
    Gm_A2[i] = FindGm_A2(T, PD, xB, xSi[i], xFe[i]) 
    Gm_B2[i], _, y1_Si[i], y2_Si[i], _, _ = minimize_eta_and_get_sites(Gm_A2[i] ,xFe[i], xSi[i], xB, T, L1)

print('-------------------------')
simtime=str(round((time.time()-simt),3))
print(f'Computation time: {simtime} s')

ax[0].scatter(xFe, Gm_B2*1e-3,color = "k")
ax[0].scatter(xFe, Gm_A2*1e-3,color = "m")
ax[0].legend(["bcc_B2","bcc_A2"])
ax[0].set_ylabel("Gm (kJ/mol)")

ax[1].scatter(xFe, y1_Si, marker="1",color="r")
ax[1].scatter(xFe, y2_Si, marker="2",color="b")
ax[1].legend(["y1_Si","y2_Si"])
ax[1].set_xlabel("xFe")
ax[1].set_ylabel("Si site fraction on\n sublattice 1 and 2")
    
plt.show()


  
    
    