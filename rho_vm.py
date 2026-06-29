#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:37:18 2024

@author: ag0406
"""

import periodictable as pd
import re
import numpy as np

NA = 6.02214179e23  #atom/mol

def rho_vm(glass,Parameters=None):
    
    #######################################     Glass
    #######################################
    #######################################
    
    matches = re.findall(r'([A-Za-z]+)(\d+\.\d+)', glass)
    
    Elements = [element for element, _ in matches]                      # atom species
    mf = np.array([float(fraction) for _, fraction in matches])         # mol fraction
    
    ##### APF B tetragonal (Structural stability and electronic properties of Beta-tetragonal boron: a first- principles study)
    N = 190
    a=10.61
    b=10.31
    c=14.15
    V_Ang = a*b*c   #Ångström^3
    V_cm = V_Ang*1e-24
    
    # materials project
    # V_Ang = 385.32   #Ångström^3
    # V_cm = V_Ang*1e-24
    # N = 50
    Vp = 4/3*np.pi*(0.84e-8)**3
    APF_B_tetra = N*Vp/V_cm
    
    Pf= {'cubic': 0.5236,'bcc': 0.68,'fcc': 0.74,'hcp': 0.74,'tetragonal': APF_B_tetra,'diamond': 0.34,'random': 0.64}
    
    am = np.zeros(len(Elements))        # g/mol
    rho = am.copy()                      # g/cm3
    s = []                              #elemtn symmetry
    n = am.copy()                       # atom/cm3
    V_e = am.copy()                     #cm3/atom
    Pf_e =am.copy()                     # Packing factor
    
    
    # V_tetra =a**2*c
    
    for i in range(len(Elements)):
        Elem = eval(f'pd.{Elements[i]}')
        # r_c = Elem.covalent_radius
        # print(f"{Elem}: radii {r_c:1.2e}")
        
        am[i] = Elem.mass 
        rho[i] = Elem.density
        # print(f"{Elements[i]}, rho: {rho[i]}, am {am[i]}")
        s.append(Elem.crystal_structure['symmetry'])
        n[i] = Elem.number_density
        V_e[i] = 1/n[i]
        try: 
            Pf_e[i] = Pf[s[i].lower()]
        except:
            Pf_e[i] = Pf['random']
        
    V_glass = sum(V_e*Pf_e/Pf['random']*mf)     #cm3/glass_atom
    M_glass = sum(am*mf)                        #g/mol
    rho_glass = M_glass/(V_glass*NA)            # g/cm3
    mVol_glass = (M_glass/rho_glass)*1e-6        #m3/mol
    
    GlassSpecs = [rho_glass, mVol_glass]
    print(f'{glass}; Vml: {mVol_glass:1.2e} m3/mol')
    
    if isinstance(Parameters, (list, np.ndarray)):
        #######################################     Phases
        #######################################
        #######################################
        
        phases = Parameters[0]   
        rho = Parameters[1]
        m = Parameters[2]
        n = Parameters[3]
        o = Parameters[4]

        Mcell = am[0]*m + am[1]*n + am[2]*o
        Vm = Mcell/((m+n+o)*rho)*1e-6 #cm3/mol, molar volume               
            
        for j in range(len(phases)):
            print(f'{phases[j]}; Vm: {Vm[j]:1.2e} m3/mol')
            
        PhaseSpecs = Vm
        #######################################
            
        return [GlassSpecs, PhaseSpecs]
    elif Parameters is None:
        return GlassSpecs

    

