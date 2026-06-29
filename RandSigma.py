#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:23:59 2024

@author: ag0406
"""

import numpy as np
from GetSigma0 import GetSigma0
# from scipy.optimize import root
from scipy.optimize import root_scalar

k_B=1.38064852E-23
# def RandSigma(dc,Vm,Hf_Sf,T,d0,gSize,PD,Stype):
def RandSigma(dc,xp,Vm,Hf_Sf,T,d0,gSize,PD,Stype):
    
    # Sigma0 = GetSigma0(T, Hf_Sf[0], Hf_Sf[1], Vm, PD[0], Stype[1])
    Sigma0 = GetSigma0(T, xp, Hf_Sf[0], Hf_Sf[1], Vm, PD[0], Stype[1])
    
    if Stype[0] == 0:
        Gr=lambda xr: -4*np.pi*xr**3*dc/(3*Vm) + 4*np.pi*Sigma0*xr**3/(xr+2*d0)
        dGr = lambda xr: -4*np.pi*Sigma0*xr**3/(2*d0 + xr)**2 + 12*np.pi*Sigma0*xr**2/(2*d0 + xr) - 4*np.pi*dc*xr**2/Vm
        
    elif Stype[0] == 1:
        Gr=lambda xr: -4*np.pi*xr**3*dc/(3*Vm) + 4*np.pi*xr**2*Sigma0/((1+d0/(xr))**2)
        dGr = lambda xr: 8*np.pi*Sigma0*d0/(d0/xr + 1)**3 + 8*np.pi*Sigma0*xr/(d0/xr + 1)**2 - 4*np.pi*dc*xr**2/Vm
        
    elif Stype[0] == 2:
        r1 = d0/0.3
        Alpha = lambda xr: 1-6/11*r1/xr + 0.0921*(r1/xr)**2 + 0.045*np.log(10/3*xr/r1)*(r1/xr)**2

        Gr=lambda xr: -4*np.pi*xr**3*dc/(3*Vm) + 4*np.pi*Sigma0*xr**2*(1-6/11*r1/xr + 0.0921*(r1/xr)**2 + 0.045*np.log(10/3*xr/r1)*(r1/xr)**2)
        dGr = lambda xr: 4*np.pi*Sigma0*xr**2*(-0.09*r1**2*np.log(3.33333333333333*xr/r1)/xr**3 - 0.1392*r1**2/xr**3 + 0.545454545454545*r1/xr**2) + 8*np.pi*Sigma0*xr*(0.045*r1**2*np.log(3.33333333333333*xr/r1)/xr**2 + 0.0921*r1**2/xr**2 - 0.545454545454545*r1/xr + 1) - 4*np.pi*dc*xr**2/Vm
        
    sol = root_scalar(lambda x: dGr(x), bracket=(1e-12, 1e-6), method='brentq')
    # try:
    #     sol = root_scalar(lambda x: dGr(x), bracket=(1e-12, 1e-6), method='brentq')
    # except:
    #     r=np.logspace(np.log10(PD[5]*1e-12),np.log10(PD[6]*1e-6),gSize) 
    #     print(Gr(r))
        
    r_star = sol.root

    
    # print(PD[0],Sigma0,dc,Vm,T,r_star0,sigma_star0)

    
    # r=np.linspace(PD[5]*r_star,PD[6]*r_star,gSize)
    r=np.logspace(np.log10(PD[5]*r_star),np.log10(PD[6]*r_star),gSize) 
    # r=np.linspace(1e-10,1e-7,gSize)
    
    dr=np.gradient(r)
    
    G_star = Gr(r_star)
        
    #-------------interfacial energy  
    if Stype[0] == 0:
 
        sigma=Sigma0/(1+2*d0/(r+dr)) 
        sigma_star=Sigma0/(1+2*d0/r_star) 
        
    elif Stype[0] == 1:

        sigma=Sigma0/((1+d0/(r+dr))**2) 
        sigma_star=Sigma0/((1+d0/r_star)**2) 
        
    elif Stype[0] == 2:
        sigma = Alpha(r)*Sigma0
        sigma_star = Alpha(r_star)*Sigma0   
        
    # sigma[sigma<0] = 1e-2
      
    return [G_star,r,dr,r_star,sigma,sigma_star]

