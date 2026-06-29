#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:23:12 2022

@author: ag0406

uygiygkgh
"""

import numpy as np
# def Jst_mult(T,G_star,r_star,D,sigma,xm,xp,N0,Vm,Vml):
def Jst_mult(T,G_vol,G_surf,G_star,r_star,D,sigma,xm,xp,N0,Vm,Vml):
    k_B=1.38064852E-23
    N_A=6.02214086E23

    l=2*(3*Vml/(4*np.pi*N_A))**(1/3)        

    # Z=Vm/(2*np.pi*r_star**2*N_A)*np.sqrt(sigma/(k_B*T))
    Z = np.sqrt(l**6/(64*np.pi**2*k_B*T)*G_vol**4/G_surf**3) #Kozerchnik
    
    multFact = np.sum((xp-xm)**2/(xm*D))
    k_star=(4*np.pi*r_star**2/(l**4))/multFact
    #-----------st nuc rate
    J_st=N0*Z*k_star*np.exp(-G_star/(k_B*T))
    
    Tau = 1/(4*np.pi*k_star*Z**2)
    return J_st,Tau