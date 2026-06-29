#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:31:28 2025

@author: ag0406

BCCandFCC_dc: attempts to activate the solution phase, BCC, FCC (and HCP)
which is more stable.
Then transform one phase to the other, i.e., the second order transition.
"""
import numpy as np

from RandSigma import RandSigma
from DcXp import DcXp
from N_Trans import N_Trans
from Jst_mult import Jst_mult
from H import H
from SolidSolFeSiB import SolidSolFeSiB

k_B=1.38064852E-23
R=8.31446261815324 
        
def BCCandFCC_dc(r_BF,N_BF,r_old_BF,N_old_BF,delta_f,Active_BF,  dc_BF,SolIdx,T,xp_force,xm,TangCoeff,Mu,isStoich,gSize,PD,BF_flag,isHeating,isQuenching): 
    dc = np.zeros([len(SolIdx),1])
    xp = np.zeros([len(SolIdx),3])  

    for idx, p in enumerate((SolIdx)):                
        SolidSolFeSiB(T,PD[0,p])      #Update Liquid Gibbs(T) Polynomials
        
        #1, Compute dc, mu and particle composition              
        [dc[idx],xp[idx,:]] = DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])

    SolActiveIdx = SolIdx[np.where(dc == np.max(dc))[0]][0]
    SolInActiveIdx = SolIdx[SolIdx != SolActiveIdx][0]
        
    if PD[0,SolActiveIdx] == 'fcc' and isHeating:
        if BF_flag == 3:
            ##################
            ##################
            r_BF = np.zeros(gSize)
            N_BF = r_BF.copy()
               
            r_old_BF = N_BF.copy()
            N_old_BF = N_BF.copy()
            delta_f = 0
            Active_BF = False
            ##################
            ##################
            
        BF_flag = 2
        if dc[SolIdx != SolActiveIdx] > 1e1: 
            dc_BF = dc[SolIdx == SolActiveIdx][0] - dc[SolIdx != SolActiveIdx][0]
        else:
            dc_BF = dc[SolIdx == SolActiveIdx][0]
        
    if PD[0,SolActiveIdx] == 'bcc' and isQuenching:
        if BF_flag == 2:
            ##################
            ##################
            r_BF = np.zeros(gSize)
            N_BF = r_BF.copy()
               
            r_old_BF = N_BF.copy()
            N_old_BF = N_BF.copy()
            delta_f = 0
            Active_BF = False
            ##################
            ##################
            
        BF_flag = 3
        if dc[SolIdx != SolActiveIdx] > 1e1: 
            dc_BF = dc[SolIdx == SolActiveIdx][0] - dc[SolIdx != SolActiveIdx][0]
        else:
            dc_BF = dc[SolIdx == SolActiveIdx][0]
    
    return r_BF,N_BF,r_old_BF,N_old_BF,delta_f,Active_BF,  dc, dc_BF, xp, SolActiveIdx, SolInActiveIdx, BF_flag

def BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf,d0,gSize,N,N_old,r_old,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx):
    
    if BF_flag == 2:     #bcc is matrix
        # D = 4.6e-5*np.exp(-218e3/(R*T))     #Kozerschnick p.314
        D = 6.4e-5*np.exp(-291e3/(R*T))         # in review papeer
                    
    if BF_flag == 3:     #fcc is matrix
        # D = 7e-5*np.exp(-286e3/(R*T))       #Kozerschnick p.314
        D = 1.77e-5*np.exp(-236.5e3/(R*T))        # in review papeer
        
    Vm = Vm[SolActiveIdx]
    xp_BF = xp[SolActiveIdx,:]
    xm_BF = xp[SolInActiveIdx,:]
    PD = PD[:,SolActiveIdx]
    
    # 
    if (xp_BF == xm_BF).all():
        # print("------ Perturb xp, same as xm ------")
        # print(f"old xp_BF: {xp_BF}")
        perturb = np.random.normal(scale=1e-6, size=xp_BF.shape)
        xp_perturbed = xp_BF + perturb
    
        # Ensure positivity and renormalize to sum to 1
        xp_perturbed = np.clip(xp_perturbed, 1e-8, None)
        xp_perturbed /= np.sum(xp_perturbed)
        
        xp_BF= xp_perturbed
        # print(f"new xp_BF: {xp_BF}")
    
    [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc_BF,xp_BF,Vm,Hf_Sf,T,d0,gSize,PD,Stype)
    # [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc_BF,xp_BF,Vm,Hf_Sf[:,SolActiveIdx],T,d0,gSize,PD,Stype) 

    if sigma_star <= 0:
        MinIdx = np.where(Sigma>0)[0][-1]
        sigma_star = Sigma[MinIdx]
        r_star = r[MinIdx]      
    
    if Active_BF and np.linalg.norm(r-r_old) > 0:
        N = N_Trans(r,r_old,N,N_old)
        
    Active_BF = True
                        
    GT = (2*Sigma/(r+dr))*Vm
    preFact = (dc_BF - GT)/(R*T*r)                    
    compFact = np.sum((xp_BF-xm_BF)**2/(xm_BF*D))

    # if compFact == 0:
    #     # Add a small perturbation to xp_BF
    #     perturb = np.random.normal(scale=1e-6, size=xp_BF.shape)
    #     xp_BF_perturbed = xp_BF + perturb
    
    #     # Ensure positivity and renormalize to sum to 1
    #     xp_BF_perturbed = np.clip(xp_BF_perturbed, 1e-8, None)
    #     xp_BF_perturbed /= np.sum(xp_BF_perturbed)
    
    #     # Recalculate compFact
    #     compFact = np.sum((xp_BF_perturbed - xm_BF)**2 / (xm_BF * D))
    #     print("Adjust compfact")
        
    v = preFact/compFact 
    
    # if np.isinf(v).any():
    #     print(f"D : \n {D}")
    #     print(f"xm_BF : \n {xm_BF}")
    #     print(f"xp_BF : \n {xp_BF}")
        
    # if (v > 1e1).any():
    #     print(f"compFact: {compFact}")
    #     print(f"preFact: {preFact}")
        

    rlim = r > 0.8*r_star
    dt = 0.5*min(dr[rlim]/abs(v[rlim]))
    # print(v.max())
    
    return v, dt, Active_BF, N, r, dr, r_star, sigma_star, G_star

def BCCandFCC_JN(dc_BF,Vm,xp,T,f,delta_f,dt,v,r,dr,r_star,G_star,sigma_star,N,Nn,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx,PD):
    
    if BF_flag == 2 or BF_flag == 3:
        if BF_flag == 2:
            # D = 4.6e-5*np.exp(-218e3/(R*T))     #Kozerschnick p.314
            D = 6.4e-5*np.exp(-291e3/(R*T))         # in old review papeer
                                   
        if BF_flag == 3:
            # D = 7e-5*np.exp(-286e3/(R*T))       #Kozerschnick p.314
            D = 1.77e-5*np.exp(-236.5e3/(R*T))        # in old review papeer

        Vml = Vm[SolInActiveIdx]  
        Vm = Vm[SolActiveIdx]
        xp_BF = xp[SolActiveIdx,:]
        xm_BF = xp[SolInActiveIdx,:]
        tot = f[SolInActiveIdx]
        PD = PD[:,SolActiveIdx]
        
        # 
        if (xp_BF == xm_BF).all():
            # print("------ Perturb xp, same as xm ------")
            # print(f"old xp_BF: {xp_BF}")
            perturb = np.random.normal(scale=1e-6, size=xp_BF.shape)
            xp_perturbed = xp_BF + perturb
        
            # Ensure positivity and renormalize to sum to 1
            xp_perturbed = np.clip(xp_perturbed, 1e-8, None)
            xp_perturbed /= np.sum(xp_perturbed)
            
            xp_BF= xp_perturbed
            # print(f"new xp_BF: {xp_BF}")
        
        if dc_BF > 1e2: 
            rkBT = r_star+0.5*np.sqrt(k_B*T/(np.pi*sigma_star))
            
            # G_star = 16*np.pi/3*Vm**2*sigma_star**3/dc_BF**2
            # r_star = 2*Vm*sigma_star/dc_BF    
            # J_st,_ = Jst_mult(T,G_star,r_star,D,sigma_star,xm_BF,xp_BF,Nn,Vm,Vml) 
            
            G_vol = dc_BF/Vml
            G_surf = sigma_star
            # G_star = 16*np.pi/3*G_surf**3/G_vol**2
            # r_star = 2*G_surf/G_vol
            J_st,tau  = Jst_mult(T,G_vol,G_surf,G_star,r_star,D,sigma_star,xm_BF,xp_BF,Nn,Vm,Vml) 
                   
            dN = (tot-delta_f)*J_st*dt
                    
            if dN < 1e0 or np.isnan(dN):
                dN = 0
            else:
                N[(np.abs(r - rkBT)).argmin()] += dN[0]
            
            #4,  Grow size classes--------------------------------------------------------
            N[0] = N[0] - dt/dr[0]*v[0]*(H(v[0])*N[0] + H(-v[0])*N[1])
            N[1:-1] = N[1:-1] + dt/dr[1:-1]*(v[0:-2]*(H(v[0:-2])*N[0:-2] + H(-v[0:-2])*N[1:-1])
                      -v[1:-1]*(H(v[1:-1])*N[1:-1] + H(-v[1:-1])*N[2:]))
            N[-1] = N[-1] + dt/dr[-1]*(v[-2]*(H(v[-2])*N[-2] + H(-v[-2])*N[-1]) - v[-1]*(H(v[-1])*N[-1]))
            
            #5, Calculate volume fraction of crystallized phase and update matrix composition---- 
            N[N<0] = 0
            delta_f = 4*np.pi/3*sum(N*r**3)
            
            if np.isnan(N).any():
                print(f"damaged dt: \n {dt}")
                print(f"damaged dr: \n {dr}")
                print(f"damaged v: \n {v}")
        else:
            delta_f = 0
    else:
       delta_f = 0

    return delta_f, N
