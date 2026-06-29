#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 15:36:04 2025

@author: ag0406

Find Jss(T) in order to start quenching at appropriate t
This is to avoid extremely small dt at high T 
(nothing will happen anyway at high T if the material was melted)
"""

import numpy as np

#functions
from DeffMult import DMult
from Jst_mult import Jst_mult
from DcXp import DcXp
from LiqFeSiB import LiqFeSiB, TransE
from SolidSolFeSiB import SolidSolFeSiB
from SolidFeSiB import SolidFeSiB
# from paramFeSiB import sysParam
from Get_HandS import Get_HandS
from RandSigma import RandSigma
from GetTm import GetTm

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

from CheckForDcAtComp import CheckDcAtMatrixComp

#--------------Simulation parameters ---------------
N_A=6.02214086E23
k_B=1.38064852E-23  
R=8.31446261815324             

# xm = np.array([0.759,0.070,0.171])
# xm = np.array([0.9-0.05,0.1,0.05])
# xm = np.array([0.9-0.05,0.05,0.1])
# xm = np.array([0.9-5e-2,5e-2,0.1])
# xm = np.array([0.899,0.001,0.1])
# xm = np.array([0.8111596 , 0.12589529, 0.06294511])
# xm = np.array([0.8,0.1,0.1])
# xm = np.array([0.74691075, 0.14755855, 0.1055307])

def Findt0T0quench(HR,xm,gsize,isStoich,Vm,d0,Vml,Stype,PD):

    xp_force = np.array([1-2*1e-6,1e-6,1e-6])               # Does only apply for sol phases!
    
    # Parameters=sysParam(xm)
    # PD=Parameters[0]                                        # PhaseData
    # d0=Parameters[1]                                        # Characteristic length scale
    # Vml=Parameters[2]                                       # Molar volume liquid
    # # Vm = PD[1,:]
    # Stype = Parameters[3]                              
    
    Hf_Sf, PD = Get_HandS(PD,xm)                             # Get reduced PD if phase does not exhibit dc in temp itnerval
    
    Hf_Sf, PD, Vm_blank = CheckDcAtMatrixComp(Vm,PD,Hf_Sf,xm)            # Get reduced PD if phase does not exhibit dc in temp itnerval, but at matrix composition  
        
    phases=PD.shape[1]
    
    [_,Tg] = TransE(1000,xm[0],xm[1],xm[2])  
    Tm = GetTm(PD,xm)                              
    Temp=np.arange(Tg,Tm,50)  #600-1250     
     
    gSize= 50   #50 for log      
    dcMin = 1e2
    
    #------------initiate arrays-----------------
    J_T=np.zeros([phases,len(Temp)])
    dt_p = np.zeros(phases)
    dt = np.zeros([1,len(Temp)])
    
    N0=N_A/Vml                               #Nucsites per unit volume
            
    for j in range(len(Temp)):
        T=float(Temp[j])  
        LiqFeSiB(T,'Liquid')      #Update Liquid Gibbs(T) Polynomials
        TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
        fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
        Mu = fp + TangCoeff - np.dot(xm,TangCoeff)
         
        for p in range(phases):
            if PD[-1,p] == 'Stoic':
                SolidFeSiB(T,PD[0,p]) #Update phase Gibbs(T) Polynomials
            elif PD[-1,p] == 'Sol':
                SolidSolFeSiB(T,PD[0,p])      #Update Liquid Gibbs(T) Polynomials
    
            Vm = PD[1,p]        
    
            # Get dc and xp
            [dc,xp] = DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])
            
            if dc > dcMin:
                    
                [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm,Hf_Sf,T,d0,gSize,PD[:,p],Stype)
                # [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm,Hf_Sf[:,p],T,d0,gSize,PD[:,p],Stype)
                v = np.empty_like(r) 
     
                #-------------Scaled diffusion in matrix
                [_,Tg] = TransE(T,xm[0],xm[1],xm[2]) 
                D = DMult(T,Tg,Tm,Vml)
                
                GT = (2*Sigma/(r))*Vm
                preFact = (dc - GT)/(R*T*r)
                compFact = np.sum((xp-xm)**2/(xm*D))
                v = preFact/compFact 
                
                if max(v[-int(gSize/10):]) > 5*np.median(v[-int(gSize/10):]):
                    v[-int(gSize/10):] = np.median(v[-int(gSize/10):])     # remove outliers at end of v cruve
                    
                if np.median(v[-int(gSize/5):]) < 0:
                    print(f"{PD[0,p]}, flipping v curve, median of end: {np.median(v[-20:]):1.1e}")
                    v *= -1
            
                dt_p[p] = 0.5*min(dr/abs(v))
                
                G_vol = dc/Vml
                G_surf = sigma_star
                G_star = 16*np.pi/3*G_surf**3/G_vol**2
                r_star = 2*G_surf/G_vol
                J_st,_  = Jst_mult(T,G_vol,G_surf,G_star,r_star,D,sigma_star,xm,xp,N0,Vm,Vml) 
     
                J_T[p,j]=J_st
        
        if dt_p.any()>0:
            dt[0,j] = min(dt_p[dt_p > 0])
            
    dN_T = J_T*dt
       
    dN_T[np.isnan(dN_T)] = 0
    Start_T = 0
    Start_dt = 0
    lim = 1e1
    for j in range(len(Temp)):
        for p in range(phases):
            if dN_T[p,j] > lim and Temp[j] > Start_T:
                Start_T = Temp[j]
                Start_dt = dt[0,j]
                print(f"{PD[0,p]}, T:{Temp[j]:1.1f}")
    
    print('----------------------------------')
    print(f'Starting Temp: {Start_T:1.1f} (K), dt :{Start_dt:1.1e} (s), gSize: {gSize}')
    print('----------------------------------')
    
    # Start_t = (Tm - Start_T)*Start_dt
    Start_t = (Tm - Start_T)/HR # should give the actual time according to the used HR....
    return [Start_t, Start_T]


from ReadTempData import Tempfunc
def Findt0T0_AM(Tin,Time,Tgtg_i,xm,isStoich,Vm,d0,Vml,Stype,PD):

    xp_force = np.array([1-2*1e-6,1e-6,1e-6])               # Does only apply for sol phases!
                 
    Hf_Sf, PD = Get_HandS(PD,xm)                             # Get reduced PD if phase does not exhibit dc in temp itnerval
    
    Hf_Sf, PD, Vm_blank = CheckDcAtMatrixComp(Vm,PD,Hf_Sf,xm)            # Get reduced PD if phase does not exhibit dc in temp itnerval, but at matrix composition  
        
    phases=PD.shape[1]
    
    Tg_t = Tgtg_i[0]
    TempRes = 50
    TimeRange = np.linspace(Time,Tg_t,TempRes)
    Temp = np.empty(TempRes)
    for i in range(len(TimeRange)):
        Temp[i] = Tempfunc(TimeRange[i])
     
    Tm = GetTm(PD,xm)                                  
     
    gSize= 70   #50 for log      
    dcMin = 1e2
    
    #------------initiate arrays-----------------
    J_T=np.zeros([phases,len(Temp)])
    dt_p = np.zeros(phases)
    dt = np.zeros([1,len(Temp)])
    
    N0=N_A/Vml                               #Nucsites per unit volume
            
    for j in range(len(Temp)):
        T=float(Temp[j])  
        LiqFeSiB(T,'Liquid')      #Update Liquid Gibbs(T) Polynomials
        TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
        fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
        Mu = fp + TangCoeff - np.dot(xm,TangCoeff)
         
        for p in range(phases):
            if PD[-1,p] == 'Stoic':
                SolidFeSiB(T,PD[0,p]) #Update phase Gibbs(T) Polynomials
            elif PD[-1,p] == 'Sol':
                SolidSolFeSiB(T,PD[0,p])      #Update Liquid Gibbs(T) Polynomials
    
            Vm = PD[1,p]        
    
            # Get dc and xp
            [dc,xp] = DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])
            # print(dc)
            
            if dc > dcMin:
                    
                [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm,Hf_Sf,T,d0,gSize,PD[:,p],Stype)
                # [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm,Hf_Sf[:,p],T,d0,gSize,PD[:,p],Stype)
                v = np.empty_like(r) 
     
                #-------------Scaled diffusion in matrix
                [_,Tg] = TransE(T,xm[0],xm[1],xm[2]) 
                D = DMult(T,Tg,Tm,Vml)
                
                GT = (2*Sigma/(r))*Vm
                preFact = (dc - GT)/(R*T*r)
                compFact = np.sum((xp-xm)**2/(xm*D))
                v = preFact/compFact 
                
                if max(v[-int(gSize/10):]) > 5*np.median(v[-int(gSize/10):]):
                    v[-int(gSize/10):] = np.median(v[-int(gSize/10):])     # remove outliers at end of v cruve
                    
                if np.median(v[-int(gSize/5):]) < 0:
                    print(f"{PD[0,p]}, flipping v curve, median of end: {np.median(v[-20:]):1.1e}")
                    v *= -1
            
                dt_p[p] = 0.5*min(dr/abs(v))
                
                G_vol = dc/Vml
                G_surf = sigma_star
                G_star = 16*np.pi/3*G_surf**3/G_vol**2
                r_star = 2*G_surf/G_vol
                J_st,_  = Jst_mult(T,G_vol,G_surf,G_star,r_star,D,sigma_star,xm,xp,N0,Vm,Vml) 
     
                J_T[p,j]=J_st
        
        if dt_p.any()>0:
            dt[0,j] = min(dt_p[dt_p > 0])
        
            
    dN_T = J_T*dt
       
    dN_T[np.isnan(dN_T)] = 0
    Start_T = 0
    lim = 1e1
    for j in range(len(Temp)):
        for p in range(phases):
            if dN_T[p,j] > lim and Temp[j] > Start_T:
                Start_T = Temp[j]
                Start_t = TimeRange[j]
                print(f"{PD[0,p]}, T:{Temp[j]:1.1f}")
    
    print('----------------------------------')
    print(f'Going from T: {Tin:1.2f} (K), t: {Time:1.2e} (s)')
    print(f'Starting Temp: {Start_T:1.1f} (K), Time {Start_t:1.2e} (s)')
    print('----------------------------------')
    
    return [Start_t, Start_T]
