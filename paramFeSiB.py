#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:34:12 2021

@author: ag0406

Thermodynamic description: Yoshitomi et al. ISIJ international, Vol. 48 (2008)
"""

import numpy as np

from rho_vm import rho_vm
from NomCompStr import MatrixStr

# Diamond Si: 2.31 g/cm3
# Boron (105) : 2.32 g/cm3
# Boron (12) : 2.48 g/cm3
def sysParam(Comp):
    PD=np.zeros([8,12], dtype=object)                           #Particle Data 
    PD[0,:] = ['FeB','Fe2B','Fe3B','FeSi','Fe2Si','Fe5Si3','Fe5Si2B','Fe5SiB2','Fe10Si4B3','bcc','fcc','hcp']                   # Stochiometric compositions 
    # PD[1,:] = [6.88, 7.42, 7.54, 6.53, 7.32, 6.63, 6.88, 7.85, 6.63, 7.87, 7.59, 9.41]    # g/cm3    (I guess this is from materials project)
    PD[1,:] = [7, 7.3, 7.401, 6.1, 6.42, 6.51, 7.88, 6.85, 6.63, 7.87, 7.59, 9.41]
    #V = [48.55,0] Å^3
    #Curie Fe3B= 780 K, Fe2B = 1017 K           
    PD[2,:] = [1, 2, 3, 0.5, 0.666667, 0.625, 4.7, 5, 2, 1, 1, 1]           # o, Fe_m Si_n B_o                                  # m, Fe_m Si_n B_o
    PD[3,:] = [0, 0, 0, 0.5, 0.333334, 0.375, 2, 1, 0.4, 0, 0, 0]                                             # n, Fe_m Si_n B_o
    PD[4,:] = [1, 1, 1, 0, 0, 0, 1, 2, 0.6, 0, 0, 0]                                                        
    PD[5,:] = 0.5  # Scale r* to min_r seems nice on 0.6!
    PD[6,:] = 300   # Scale r* to max_r
    PD[7,:] = ['Stoic', 'Stoic', 'Stoic','Stoic', 'Stoic', 'Stoic', 'Stoic', 'Stoic', 'Stoic', 'Sol', 'Sol','Sol',]                   #stoichiometric description or not
    
    # r1=2.55E-10                             #nearest neighbour distance Fe-Fe (Kaban)
    r1 = 2.48e-10                           # nearest neighbour Fe77.5Si12.5B10
    d0=0.3*r1                               #characteristic length scale
    
    # [sigma(r),sigma0] #0: Tollman,1: mod Tollman,2: Kozerchnick
    # 0: Turbnull, 1: Granasy and Tegze, 2: Spaepen
    Stype = [2,0]    

    PD_strip = PD
        
    #### Get Vm for all Phases (Solid,Matrix)
    MatrixPhase = MatrixStr(Comp)
    [GlassSpecs,PhaseSpecs] = rho_vm(MatrixPhase,PD_strip)
    Vml = GlassSpecs[1]                                     # Vm for matrix [m3/mol]
    PD_strip[1,:] = PhaseSpecs                              # Vm for phases [m3/mol], change elements of rho to Vm
    
    PD_strip = PD_strip[:,:-1]    # Ignore hcp
    
    return [PD_strip,d0,Vml,Stype]

    
