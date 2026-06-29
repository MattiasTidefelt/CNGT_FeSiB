#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:26:54 2025

@author: ag0406

Compute and illustrate the particle size distribution (PSD), supersaturation,
growth velocity and radial composition in matrix in time.

Only minor adjustments should be needed to change phases or system

For the substitutional solution model

Transfering the model of second order transition and generally updated one. 20/10-25

Using Tg from therm assessment

Uses Solid solution for BCC and FCC and accounts for second order transition
"""

# Modules
import numpy as np
import time

# Functions
from DeffMult import DMult
from Jst_mult import Jst_mult
from Tau_Func import Tau_Func
from H import H
from DcXp import DcXp
from N_Trans import N_Trans
from LiqFeSiB import LiqFeSiB, TransE
from SolidSolFeSiB import SolidSolFeSiB
from SolidFeSiB import SolidFeSiB
from RandSigma import RandSigma
from paramFeSiB import sysParam
from Get_HandS import Get_HandS, Get_HandS_BF
from CheckForDcAtComp import CheckDcAtMatrixComp
from GetTm import GetTm

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

from BCCandFCC import BCCandFCC_dc, BCCandFCC_v, BCCandFCC_JN

from FindJssT import Findt0T0quench


#---------------Simulation parameters-------------------
N_A=6.02214086E23
k_B=1.38064852E-23
R=8.31446261815324 

def MPI_Nuc_FeSiB_HQ(x0,HR):
    
    if HR < 0:
        isHeating = False
        isQuenching = True
    elif HR > 0:
        isHeating = True
        isQuenching = False
        
    HR = abs(HR)

    isStoich = False

    R=8.31446261815324 

    x0Str = "_".join(f"{x:.3f}" for x in x0)

    xp_force = np.array([1-2*1e-6,1e-6,1e-6])               # Does only apply for sol phases!

    Parameters=sysParam(x0)
    PD=Parameters[0]                                        # PhaseData
    d0=Parameters[1]                                        # Characteristic length scale
    Vm = PD[1,:]
    Vml=Parameters[2]                                       # Molar volume liquid
    Stype = Parameters[3]   

    phaseBlank = PD[0,:]                          

    Hf_Sf, PD = Get_HandS(PD,x0)                             # Get reduced PD if phase does not exhibit dc in temp itnerval

    Hf_Sf, PD, Vm = CheckDcAtMatrixComp(Vm,PD,Hf_Sf,x0)            # Get reduced PD if phase does not exhibit dc in temp itnerval, but at matrix composition     

    SolIdx = np.where(PD[-1,:] == 'Sol')[0]
            
    phases=PD.shape[1]
                                         
    fmax=0.20                                               # Volume fraction limit
    dcMin = 1e2

    dt_force = 1.5/HR
    dt_eval = 10**-np.linspace(2,15,50)

    # -------------Load Temperature data----------------
    [_,Tg] = TransE(1000,x0[0],x0[1],x0[2]) 
    Tm = GetTm(PD,x0)
    
    if isHeating:
        gSize = 70#150 
        Time = 1e-20
        T = Tg
        T_old = Tg-5
    elif isQuenching:
        gSize = 70#150 
        [Time,T] = Findt0T0quench(HR,x0,gSize,isStoich,Vm,d0,Vml,Stype,PD)
        T_old = T+5
        
    N0 = N_A/Vml                                              # Initial number of nucleation sites
    Nn = N0.copy()                                          # Variable number of nucleation sites

    #------------ initiate arrays-----------------
    dc = np.zeros([phases,1])                               #Chemical driving force [J/mol]
    dt_p = dc.copy()                                        #Max time step allowed for particle
    dN = dc.copy()                                          #Clusters formed within time step for each phase
    G_star = dc.copy()
    r_star = dc.copy()
    sigma_star = dc.copy()
    Rmean_p_t = dc.copy()

    xp = np.zeros([phases,3])                               #molar fractional particle composition

    dr = np.zeros([phases,gSize])
    v = dr.copy()                                            #Growth velocity
    Sigma = dr.copy()                                        #Interfacial energy

    r = np.zeros([phases,gSize])
    N = r.copy()
       
    r_old = N.copy()
    N_old = N.copy()

    ############# For secondary transition of BCC-FCC
    r_BF = np.zeros(gSize)
    N_BF = r_BF.copy()
       
    r_old_BF = N_BF.copy()
    N_old_BF = N_BF.copy()

    BF_flag = 1
    delta_f = 0
    SolActiveIdx = 100
    SolInActiveIdx = 100
    dc_BF = 0
    ####################

    f = np.zeros([phases,1])                              #Volume fractions
    NBlank = np.zeros(gSize)

    #-----------Initiate simulation tics------------
    xm=x0.copy()
    xm_old = xm.copy()
    it = 0
    it_old=0
    ftot = 0
    simt = time.time()

    Active = False*np.empty(phases)
    Active_BF = False
            
    #%%
    while ftot<fmax:
        
        if np.abs(T-T_old) >= 2 or it-it_old>5e3:            # Only update things if something has changed, i.e., T
        
            LiqFeSiB(T,'Liquid')      #Update Liquid Gibbs(T) Polynomials
            TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
            fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
            Mu = fp + TangCoeff - np.dot(xm,TangCoeff)
            
            D = DMult(T,Tg,Tm,Vml)
            
            for p in range(phases):        
                
                if PD[-1,p] == 'Stoic':
                    SolidFeSiB(T,PD[0,p])    #Update phase Gibbs(T) Polynomials
                elif PD[-1,p] == 'Sol':
                    SolidSolFeSiB(T,PD[0,p])      #Update Liquid Gibbs(T) Polynomials
        
                #1, Compute dc, mu and particle composition 
                if p < SolIdx[0]:
                    [dc[p],xp[p,:]] = DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])
                
                ######################
                ######################
                elif p == SolIdx[0]:
                    r_BF,N_BF,r_old_BF,N_old_BF,delta_f,Active_BF,  dc[SolIdx], dc_BF, xp[SolIdx,:], SolActiveIdx, SolInActiveIdx, BF_flag = BCCandFCC_dc(r_BF,N_BF,r_old_BF,N_old_BF,delta_f,Active_BF,   dc_BF,SolIdx,T,xp_force,xm,TangCoeff,Mu,isStoich,gSize,PD,BF_flag,isHeating,isQuenching)             
                
                ######################
                ######################
                if p != SolInActiveIdx:
                    if dc[p] > dcMin:  # Avoid corruption by only allowing possitive driving force
                        [G_star[p],r[p],dr[p],r_star[p],Sigma[p],sigma_star[p]] = RandSigma(dc[p],xp[p,:],Vm[p],Hf_Sf,T,d0,gSize,PD[:,p],Stype)
                        # [G_star[p],r[p],dr[p],r_star[p],Sigma[p],sigma_star[p]] = RandSigma(dc[p],xp[p,:],Vm[p],Hf_Sf[:,p],T,d0,gSize,PD[:,p],Stype) 
                        
                        if Active[p]:
                            N[p,:] = N_Trans(r[p,:],r_old[p,:],N[p,:],N_old[p,:])
                            
                        Active[p] = True
                        
                        if it-it_old>5e3:
                            it_old = it
                                        
                        GT = (2*Sigma[p,:]/(r[p,:]+dr[p,:]))*Vm[p]
                        preFact = (dc[p] - GT)/(R*T*r[p,:])                    
                        compFact = np.sum((xp[p,:]-xm)**2/(xm*D))
                        v[p,:] = preFact/compFact 
            
                        if it ==0:
                            dt_p[p] = 0.5*min(dr[p,:]/abs(v[p,:]))
                        else:
                            rlim = r[p,:] > 0.8*r_star[p]
                            dt_p[p] = 0.5*min(dr[p,rlim]/abs(v[p,rlim]))
                            
                elif p == SolInActiveIdx and BF_flag == 2 and dc_BF > dcMin or p == SolInActiveIdx and BF_flag == 3 and dc_BF > dcMin:
                    # print("----------------------------")
                    v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF, G_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf,d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
                    
                    # Hf_Sf_BF = Get_HandS_BF(PD,xp[SolActiveIdx,:],SolActiveIdx,SolInActiveIdx)
                    # v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF, G_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf_BF,d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
                    
                    # v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf_BF[:,0],d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
                    # print("----------------------------")
                    if dt_p[SolInActiveIdx] < 1e-8:
                        # print("Liq diff",D)
                        # print(r_star_BF, sigma_star_BF)
                        print(f"T: {T}, Active: {PD[0,SolActiveIdx]}, InActive {PD[0,SolInActiveIdx]}")
                        print(dt_p.reshape(1,-1)[0])
                        print("max v",v_BF.max())
                    
            if dc[dc > dcMin].any():
                # dt_max = min(dt_p[dt_p > 0])
                exclude = np.zeros(dt_p.shape[0], dtype=bool)
                exclude[SolInActiveIdx] = True
                dt_max = dt_p[(dt_p.squeeze() > 0) & (~exclude)].min()
                if isHeating:
                    New_T = T + HR*dt_max
                elif isQuenching:
                    New_T = T - HR*dt_max
                    
                if np.abs(T - New_T) >= 0.1:
                    # print("here 1")
                    for dt_i in dt_eval:
                        if isHeating:
                            New_T = T + HR*dt_i
                        elif isQuenching:
                            New_T = T - HR*dt_i
                        T_step = np.abs(T - New_T)
                        # print(f"{T_step:1.1e}, {dt_i:1.1e}, {dt_max:1.1e}")
                        if T_step <= 0.1 and dt_i < dt_max:
                            dt = dt_i                       
                            break 
                        else:
                            dt = dt_max 
                else:
                    dt = dt_max 
            else:
                dt = dt_force
            
        # Grow phases on grid from calculated growth velocity
        for p in range(phases):
            if dc[p] > dcMin:           # Avoid corruption by only allowing possitive driving force         
        
                if p != SolInActiveIdx:
                    rkBT = r_star[p]+0.5*np.sqrt(k_B*T/(np.pi*sigma_star[p]))
                    
                    G_vol = dc[p]/Vml
                    G_surf = sigma_star[p]
                    # G_star[p] = 16*np.pi/3*G_surf**3/G_vol**2
                    # r_star[p] = 2*G_surf/G_vol
                    
                    # G_star[p] = 16*np.pi/3*Vm[p]**2*sigma_star[p]**3/dc[p]**2          
                    # r_star[p] = 2*Vm[p]*sigma_star[p]/dc[p]
                    
                    J_st,Tau  = Jst_mult(T,G_vol,G_surf,G_star[p],r_star[p],D,sigma_star[p],xm,xp[p,:],N0,Vm[p],Vml) 
                    # J_st,Tau = Jst_mult(T,G_star[p],r_star[p],D,sigma_star[p],xm,xp[p,:],N0,Vm[p],Vml) 
                    
                    
                    ##################       
                    if isQuenching or BF_flag == 2 or BF_flag == 3:
                        dN[p] = (1-f[p])*J_st*dt
                    elif isHeating and BF_flag == 1:
                    ##################
                        if Tau <0:
                            Tau = 0
                        if Time == 0:
                            Time == 1e-7
                        J = J_st * np.exp(-Tau/Time)
                        dN[p] = (1-f[p])*J*dt 
        
                    if dN[p]< 1e0:
                        dN[p] = 0
                    else:
                        N[p,(np.abs(r[p,:] - rkBT)).argmin()] += dN[p][0]
    
                    #4,  Grow size classes--------------------------------------------------------
                    N[p,0] = N[p,0] - dt/dr[p,0]*v[p,0]*(H(v[p,0])*N[p,0] + H(-v[p,0])*N[p,1])
                    N[p,1:-1] = N[p,1:-1] + dt/dr[p,1:-1]*(v[p,0:-2]*(H(v[p,0:-2])*N[p,0:-2] + H(-v[p,0:-2])*N[p,1:-1])
                              -v[p,1:-1]*(H(v[p,1:-1])*N[p,1:-1] + H(-v[p,1:-1])*N[p,2:]))
                    N[p,-1] = N[p,-1] + dt/dr[p,-1]*(v[p,-2]*(H(v[p,-2])*N[p,-2] + H(-v[p,-2])*N[p,-1]) - v[p,-1]*(H(v[p,-1])*N[p,-1]))
                 
                    #5, Calculate volume fraction of crystallized phase and update matrix composition----              
                    N[p,np.where(r[p,:]<0.8*r_star[p])] = 0
                    f[p] = 4*np.pi/3*sum(N[p,:]*r[p,:]**3)
                    
            else:
                dN[p] = 0

    
        if BF_flag == 2 and f[SolInActiveIdx] > 0  and dc_BF > dcMin or BF_flag == 3 and f[SolInActiveIdx] > 0 and dc_BF > dcMin:
                Nn_BF = sum(N[SolInActiveIdx,:])
                delta_f, N_BF = BCCandFCC_JN(dc_BF,Vm,xp,T,f,delta_f,dt,v_BF,r_BF,dr_BF,r_star_BF,G_star_BF,sigma_star_BF,N_BF,Nn_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx,PD)
                
                # N_BF[np.where(r_BF<0.8*r_star_BF)] = 0
                
                Ntrans_BD_IA = N_Trans(r[SolInActiveIdx,:],r_BF,NBlank,N_BF)
                N[SolInActiveIdx,:] -= Ntrans_BD_IA
                N[SolInActiveIdx,N[SolInActiveIdx,:]<0] = 0
                # f[SolInActiveIdx] = 4*np.pi/3*sum(N[SolInActiveIdx,:]*r[SolInActiveIdx,:]**3)
                f[SolInActiveIdx] -= delta_f
                
                if f[SolInActiveIdx] > 0:
                    Ntrans_BD_A = N_Trans(r[SolActiveIdx,:],r_BF,NBlank,N_BF)
                    N[SolActiveIdx,:] += Ntrans_BD_A
                    f[SolActiveIdx] = 4*np.pi/3*sum(N[SolActiveIdx,:]*r[SolActiveIdx,:]**3)
                else:
                    f[SolInActiveIdx] = 0
                    N[SolInActiveIdx,:] = 0
                    # BF_flag = 4
        
        # Nn = N0 - np.sum(N)
        # if Nn <= 0:
        #     Nn = 0
         
        ftot = np.sum(f)
        xm = (x0-np.sum(f*xp,axis=0))/(1-ftot)  
        if (xm<0).any():
            xm[xm<0] = 1e-6
            
        
        #Transfer all populations to a uniform
        for p in range(phases): #Interpolate N[p] to N[0] as N_tot and store modus of dist
            if max(N[p,:]) > 0:
                Rmean_p_t[p] = r[p,np.where(N[p,:] == max(N[p,:]))] #max N(r), modus 
                # Rmean_p_t[p] = r[p,(N[p,:]-np.median(N[p,:])).argmin()] #median of N[p,r]

        r_old = r.copy()
        N_old = N.copy()   
        N_old_BF = N_BF.copy()
        r_old_BF = r_BF.copy()       
        
        if np.linalg.norm(xm-xm_old)>1e-3:
            xm_old = xm.copy()
            [_,Tg] = TransE(T,xm[0],xm[1],xm[2]) 
            Tm = GetTm(PD,xm)   
        
        it += 1
        Time = Time + dt  
        if np.abs(T-T_old) >= 2:
            T_old = T
            
        if isHeating:
            T = T + HR*dt
            
            if T > Tm:
                print("----------------------------")
                print("Tm reached, breaking simulation")
                print("----------------------------")
                break
            
        elif isQuenching:
            T = T - HR*dt
            
            if T < Tg:
                print("----------------------------")
                print("Tg reached, breaking simulation")
                print("----------------------------")
                break   
    
        if it-it_old>5e3 or it == 0:  
            status = (
                f"\r {x0Str}: "
                f"dt:{dt:2.2e} "
                f"t:{Time:2.4f} "
                f"T:{T:4.1f} "
                f"it:{it} "
                f"ftot:{ftot:1.2e} "
                f"HR:{HR:1.1e}"
            )
        
            print(status, end='\n')
            # it_old = it
            
    print('-------------------------')
    simtime=str(round((time.time()-simt)/60,1))
    Final_status = (
        f"\r {x0Str}: "
        f"t:{Time:2.4f} "
        f"T:{T:4.1f} "
        f"it:{it} "
        f"ftot:{ftot:1.2e} "
        f"wall-time:{simtime} min\n"
        f"f:[{', '.join(f'{f[pp][0]:1.2e}' for pp in range(phases))}] "
        f"dc:[{', '.join(f'{dc[pp,0]:1.2e}' for pp in range(phases))}] "
        f"maxV:[{', '.join(f'{max(v[pp,:]):4.2e}' for pp in range(phases))}] "
        f"dN:[{', '.join(f'{dN[pp][0]:4.2e}' for pp in range(phases))}]"
    )
    
    print(Final_status, end='\n')
    print('-------------------------')
    
    mapped_f = np.zeros(len(phaseBlank), dtype=float)
    mapped_Rmean = np.zeros(len(phaseBlank), dtype=float)
    
    # Fill in the positions where sub_list elements appear
    for s, val in zip(PD[0,:], f):
        idx = np.where(phaseBlank == s)[0][0]  # Get position of the element in the full list
        mapped_f[idx] = val  
    for s, val in zip(PD[0,:], Rmean_p_t):
        idx = np.where(phaseBlank == s)[0][0]  # Get position of the element in the full list
        mapped_Rmean[idx] = val  
        
    
    return [Time,T,ftot,mapped_f,mapped_Rmean,it]

    
    