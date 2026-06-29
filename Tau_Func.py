#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:56:31 2022

@author: ag0406
"""

import numpy as np
def Tau_Func(T,r_star,D,sigma,xm,xp,Vm,Vml):

    k_B=1.38064852E-23
    N_A=6.02214086E23     
    
    l=2*(3*Vml/(4*np.pi*N_A))**(1/3)
    Z=Vm/(2*np.pi*r_star**2*N_A)*np.sqrt(sigma/(k_B*T))
    # k_star=4*np.pi*r_star**2/l**4*min(D*xm)                             #OBS
    multFact = np.sum((xp-xm)**2/(xm*D))
    k_star=(4*np.pi*r_star**2/l**4)/multFact
    #-----------st nuc rate
    # Tau=1/(4*np.pi*Z**2*k_star)
    Tau=1/(2*Z**2*k_star)
    return(Tau)