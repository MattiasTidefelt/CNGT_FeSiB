#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:47:38 2024

@author: ag0406
"""

def StoicCompounds_dT(T,PD):   
    from SolidFeSiB_dT import Gibbs_Solid_dT 
    
    m=PD[2]
    n=PD[3]
    o=PD[4]

    G_Bin_FeB = Gibbs_Solid_dT[0]
    dGmo = Gibbs_Solid_dT[-1]

    DeltaG_dT = G_Bin_FeB[0]/(m+n+o) + dGmo
        
    return DeltaG_dT 