#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:47:39 2022

@author: ag0406

transform population N2(r2)>0 to shape N1(r1)
Rquiers that r2(1)<r1(1)
"""

from scipy.interpolate import interp1d
import numpy as np

def N_Trans(r_new, r_old, N_new, N_old):
    r_min = max(r_new[0], r_old[0])  
    r_max = min(r_new[-1], r_old[-1]) 
    
    if r_min >= r_max:
        raise ValueError("No overlap between r1 and r2")  

    Idx_lb = np.searchsorted(r_new, r_min, side='left')
    Idx_ub = np.searchsorted(r_new, r_max, side='right') - 1

    if Idx_lb >= Idx_ub:
        raise ValueError("Invalid index range for overlap")

    FF = interp1d(r_old, N_old, kind='linear', fill_value='extrapolate')
    # NInter = N_new.copy()
    NInter = np.zeros_like(N_new)
    NInter[Idx_lb:Idx_ub+1] = FF(r_new[Idx_lb:Idx_ub+1])

    return NInter
