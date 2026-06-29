#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:01:49 2026

@author: ag0406
"""
import numpy as np
from SolidSolFeSiB import Gibbs_SolidSol 

[G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_SolidSol[0]

R = 8.31446261815324
L1 = -2*1260*R

def dG_B2_dxSi(eta, xFe, xSi, T):   
    # site fractions
    y1_Si = xSi + eta
    y2_Si = xSi - eta

    y1_Fe = 1 - y1_Si
    y2_Fe = 1 - y2_Si

    # dy/dxSi
    dy1Si_dx = 1.0
    dy2Si_dx = 1.0
    dy1Fe_dx = -1.0
    dy2Fe_dx = -1.0

    # --- reference derivative ---
    dGref_dx = (
        0.5*(dy1Fe_dx*G_Liq_Fe_0 + dy1Si_dx*G_Liq_Si_0 +
             dy2Fe_dx*G_Liq_Fe_0 + dy2Si_dx*G_Liq_Si_0)
    )

    # --- ideal entropy derivative ---
    dGid_dx = 0.5*R*T*(
        dy1Fe_dx*(1 + np.log(y1_Fe)) +
        dy1Si_dx*(1 + np.log(y1_Si)) +
        dy2Fe_dx*(1 + np.log(y2_Fe)) +
        dy2Si_dx*(1 + np.log(y2_Si))
    )

    # --- ordering derivative ---
    dGord_dx = L1*(
        dy1Fe_dx*y2_Si + y1_Fe*dy2Si_dx +
        dy1Si_dx*y2_Fe + y1_Si*dy2Fe_dx
    )

    return dGref_dx + dGid_dx + dGord_dx


def dG_B2_dxFe(eta, xFe, xSi, T):
    # site fractions
    y1_Si = xSi + eta
    y2_Si = xSi - eta
    y1_Fe = 1 - y1_Si
    y2_Fe = 1 - y2_Si

    # dy/dxFe
    dy1Si_dx = -1.0
    dy2Si_dx = -1.0
    dy1Fe_dx = +1.0
    dy2Fe_dx = +1.0

    # --- reference derivative ---
    dGref_dx = (
        0.5*(dy1Fe_dx*G_Liq_Fe_0 + dy1Si_dx*G_Liq_Si_0 +
             dy2Fe_dx*G_Liq_Fe_0 + dy2Si_dx*G_Liq_Si_0)
    )

    # --- ideal entropy derivative ---
    dGid_dx = 0.5*R*T*(
        dy1Fe_dx*(1 + np.log(y1_Fe)) +
        dy1Si_dx*(1 + np.log(y1_Si)) +
        dy2Fe_dx*(1 + np.log(y2_Fe)) +
        dy2Si_dx*(1 + np.log(y2_Si))
    )

    # --- ordering derivative ---
    dGord_dx = L1*(
        dy1Fe_dx*y2_Si + y1_Fe*dy2Si_dx +
        dy1Si_dx*y2_Fe + y1_Si*dy2Fe_dx
    )

    return dGref_dx + dGid_dx + dGord_dx

