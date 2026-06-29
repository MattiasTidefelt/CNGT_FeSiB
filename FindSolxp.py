#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:47:34 2025

@author: ag0406
"""

# from Gibbs_Tangent import Gibbs_Tangent
import numpy as np
from Gibbs_Func import Gibbs_Func

def find_xp(vars,T,Mu_L,PD):
    xFe, xSi = vars
    
    xp = np.array([xFe,xSi,1-xFe-xSi])
    T_liq_p1=np.dot(xp,Mu_L)
    
    Gp = Gibbs_Func(T,xp[0],xp[1],xp[2],PD[-1])

    res = Gp-T_liq_p1
    return -res