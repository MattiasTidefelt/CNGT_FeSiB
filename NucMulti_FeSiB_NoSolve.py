#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:28:57 2022

@author: ag0406

Compute and illustrate the particle size distribution (PSD), supersaturation,
growth velocity and radial composition in matrix in time.

Only minor adjustments should be needed to change phases or system

For the substitutional solution model

Starting the attempt of implementing Fe-Si-B multiphase model 18/08-25

TEST TO EXTRACT FROM TDB TO GET COMPOSITION!!!!!!!!!!!!!!!!!!!!!!!
However, to low at the moment (400-650 K), would expect it to be a bit higher
"""
# Modules
import matplotlib.pyplot as plt
import numpy as np
import time
from os import makedirs
from os.path import exists

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
from plotLabel import plotLabel
from RandSigma import RandSigma
from HEXCOL import HEXCOL
from paramFeSiB import sysParam
from Get_HandS import Get_HandS, Get_HandS_BF
from CheckForDcAtComp import CheckDcAtMatrixComp
from GetTm import GetTm

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

from BCCandFCC import BCCandFCC_dc, BCCandFCC_v, BCCandFCC_JN

from FindJssT import Findt0T0quench

# isHeating = False
# isQuenching = True

isHeating = True
isQuenching = False

isStoich = False

basedir = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/Figures/"

plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.linewidth'] = 1.5
plt.close('all')

#---------------Simulation parameters-------------------
N_A=6.02214086E23
k_B=1.38064852E-23
R=8.31446261815324 

# HR = 3.2e+08
# HR = 1e7
HR = 6e7

# [xFe, xSi, XB]
# x0 = np.array([0.758,0.111,0.131])
# x0 = np.array([0.760,0.133,0.107])
# x0 = np.array([0.755, 0.143, 0.102])
# x0 = np.array([0.932,  0.013, 0.056])
# x0 = np.array([0.9-0.05,0.1,0.05])
# x0 = np.array([0.9-0.05,0.05,0.1])
# x0 = np.array([0.9-5e-2,5e-2,0.1])
# x0 = np.array([0.899,0.001,0.1])
# x0 = np.array([0.8111596 , 0.12589529, 0.06294511])
# x0 = np.array([0.8,0.1,0.1])                      #Rc=4e6, Hc= 5e6
# x0 = np.array([0.829 , 0.048, 0.122])
# x0 = np.array([0.832,0.031,0.137])
# x0 = np.array([0.85,0.06,0.09])
# x0 = np.array([0.8,0.01,0.19])                    
# x0 = np.array([0.9,0.01,0.09])                        #Rc=4e6, Hc= 6e7
# x0 = np.array([0.74691075, 0.14755855, 0.1055307])
# x0 = np.array([8.5008329e-01, 8.1207208e-02, 6.8709537e-02]) #from optimization
# x0 = np.array([0.779 , 0.125, 0.096])
x0 = np.array([0.800 , 0.087, 0.113])
# x0 = np.array([0.762 , 0.211, 0.027])
# x0 = np.array([0.7 , 0.01, 0.29])
# x0 = np.array([0.77411887, 0.21552039, 0.01036074])
# x0 = np.array([0.7 , 0.15, 0.15])
# x0 = np.array([0.76669942, 0.05922546, 0.17407512])
# x0 = np.array([0.75407612, 0.19001434, 0.05590954])
# x0 = np.array([0.85389158, 0.09837894, 0.04772948])

# x0 = np.array([0.7, 0.29, 0.01])
# x0 = np.array([0.83, 0.16, 0.01])
# x0 = np.array([0.81, 0.18, 0.01])
# x0 = np.array([0.79, 0.20, 0.01])
# x0 = np.array([0.9, 0.09, 0.01])

# x0 = np.array([0.7, 0.01, 0.29])
# x0 = np.array([0.81, 0.01, 0.18])
# x0 = np.array([0.86, 0.01, 0.13])
# x0 = np.array([0.9, 0.01, 0.09])
x0 = np.array([0.95, 0.01, 0.04])


x0Str = "_".join(f"{x:.3f}" for x in x0)

xp_force = np.array([1-2*1e-6,1e-6,1e-6])               # Does only apply for sol phases!

Parameters=sysParam(x0)
PD=Parameters[0]                                        # PhaseData
d0=Parameters[1]                                        # Characteristic length scale
Vm = PD[1,:]
Vml=Parameters[2]                                       # Molar volume liquid
Stype = Parameters[3]                              

Hf_Sf, PD = Get_HandS(PD,x0)                             # Get reduced PD if phase does not exhibit dc in temp itnerval

Hf_Sf, PD, Vm = CheckDcAtMatrixComp(Vm,PD,Hf_Sf,x0)            # Get reduced PD if phase does not exhibit dc in temp itnerval, but at matrix composition     

SolIdx = np.where(PD[-1,:] == 'Sol')[0]
        
phases=PD.shape[1]

COLORS=HEXCOL(phases,1)
                                     
fmax=0.20                                               # Volume fraction limit
dcMin = 1e2

dt_force = 1.5/HR
# dt_eval = np.linspace(1e-4,dt_force,50)
dt_eval = 10**-np.linspace(2,15,50)
NDD_vols=np.linspace(1e-4,fmax,8)

OutDir =f'{basedir}/Comp_{x0Str}/{HR}_Ks/'
if not exists(OutDir):
    makedirs(OutDir)
# -------------Load Temperature data----------------
[_,Tg] = TransE(1000,x0[0],x0[1],x0[2]) 
Tm = GetTm(PD,x0)   
print(f"Tg: {Tg:1.1f} K")
if isHeating:
    gSize = 70 #70
    Time = 1e-20
    T = Tg
    T_old = Tg-5
elif isQuenching:
    gSize = 70 #70
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
Rmean_p = dc.copy()
Rmean_p_t = dc.copy()
dissolutionflag = dc.copy() 

xp = np.zeros([phases,3])                               #molar fractional particle composition

dr = np.zeros([phases,gSize])
v = dr.copy()                                            #Growth velocity
Sigma = dr.copy()                                        #Interfacial energy

r = np.zeros([phases +1,gSize])
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

f = np.zeros([phases+1,1])                              #Volume fractions
V_tot = f.copy()
N_tot = f.copy()
N_tot_t = f.copy()

NBlank = np.zeros(gSize)

TimePar= 0
TempPar= 0

# ------------- Get Hex colors to use -------------
COLORS=HEXCOL(len(NDD_vols),1)
COLOR2 = HEXCOL(phases,2)

#-----------Initiate simulation tics------------
xm=x0.copy()
xm_old = xm.copy()
NDD_tic = 0
PlotVol = fmax/5
it = 0
it_old=0
tt = Time
PlotTime = fmax/5
simt = time.time()

print_flag = True
Active = False*np.empty(phases)
Active_BF = False
        
#%%
if phases <= 4:
    fig2,ax2 = plt.subplots(phases+1, sharex=True)          # Initiate PSD fig
    big_flag = False
elif phases > 4:
    if int(phases/int(np.ceil(phases/2))) == 2:
        fig2,ax2 = plt.subplots(int(np.ceil(phases/2))+1 ,2, sharex='col')     
    else:
        fig2,ax2 = plt.subplots(int(np.ceil(phases/2)) ,2, sharex='col')
    big_flag = True

while f[-1]<fmax:
    
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
                                
                    # # #2, Get velocity--------- 
                    if print_flag and it-it_old>5e3:
                        print_flag = False
                        
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
                v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF, G_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf,d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
                
                # Hf_Sf_BF = Get_HandS_BF(PD,xp[SolActiveIdx,:],SolActiveIdx,SolInActiveIdx)
                # v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF, G_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf_BF,d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
                
                # v_BF, dt_p[SolInActiveIdx], Active_BF, N_BF, r_BF, dr_BF, r_star_BF, sigma_star_BF = BCCandFCC_v(dc_BF,Vm,xp,T,Hf_Sf_BF[:,0],d0,gSize,N_BF,N_old_BF,r_old_BF,PD,Stype,Active_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx)
    
        if dc[dc > dcMin].any():
            r[-1,:] = np.logspace(np.log10((0.8 * r_star[(dt_p > 0) & (r_star > 0)]).min()),np.log10((50 * r_star[(dt_p > 0) & (r_star > 0)]).max()),gSize)
            
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
                if dc[p] > dcMin: 
                    rkBT = r_star[p]+0.5*np.sqrt(k_B*T/(np.pi*sigma_star[p]))
                    
                    G_vol = dc[p]/Vml
                    G_surf = sigma_star[p]
                    # G_star[p] = 16*np.pi/3*G_surf**3/G_vol**2
                    # r_star[p] = 2*G_surf/G_vol
                    J_st,Tau  = Jst_mult(T,G_vol,G_surf,G_star[p],r_star[p],D,sigma_star[p],xm,xp[p,:],N0,Vm[p],Vml) 
 
                                 
                    # G_star[p] = 16*np.pi/3*Vm[p]**2*sigma_star[p]**3/dc[p]**2          
                    # r_star[p] = 2*Vm[p]*sigma_star[p]/dc[p]
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

    ## The reason for the observed flipping comes from the minimizer finding different minimum at threshold temperature!
    if it % 5 == 0:
        if BF_flag == 1:
            print(f"{PD[0,SolActiveIdx]}: {f[SolActiveIdx][0]:1.2e}, T: {T:4.2f}, {r_star[SolActiveIdx][0]:1.2e}, BF_flag: {BF_flag} IA: {SolInActiveIdx}, A: {SolActiveIdx}")
        elif BF_flag == 2 or BF_flag ==3:
            print(f"------------ depleating {PD[0,SolInActiveIdx]} -------------- BF_flag: {BF_flag} IA: {SolInActiveIdx}, A: {SolActiveIdx}")
            print(f"{PD[0,SolInActiveIdx]}: {f[SolInActiveIdx][0]:1.1e} dc {dc[SolInActiveIdx][0]:1.1e}, {PD[0,SolActiveIdx]}: {f[SolActiveIdx][0]:1.1e} dc {dc[SolActiveIdx][0]:1.1e},T: {T:4.2f} df {delta_f:1.1e} r* {r_star_BF:1.1e}")
           
    if BF_flag == 2 and f[SolInActiveIdx] > 0  and dc_BF > dcMin or BF_flag == 3 and f[SolInActiveIdx] > 0 and dc_BF > dcMin:
            Nn_BF = sum(N[SolInActiveIdx,:])
            delta_f, N_BF = BCCandFCC_JN(dc_BF,Vm,xp,T,f,delta_f,dt,v_BF,r_BF,dr_BF,r_star_BF,G_star_BF,sigma_star_BF,N_BF,Nn_BF,SolIdx,BF_flag,SolActiveIdx,SolInActiveIdx,PD)
            
            # N_BF[np.where(r_BF<0.8*r_star_BF)] = 0
            
            Ntrans_BD_IA = N_Trans(r[SolInActiveIdx,:],r_BF,NBlank,N_BF)
            Ntrans_BD_IA[Ntrans_BD_IA<0] = 0
            
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
    
    # if it % 10 == 0:
    #     if BF_flag == 1:
    #         print(f"{PD[0,SolActiveIdx]}: {f[SolActiveIdx][0]:1.2e}, T: {T:4.2f}, {r_star[SolActiveIdx][0]:1.2e}")
    #     elif BF_flag == 2 or BF_flag ==3:
    #         print(f"------------ depleating {PD[0,SolInActiveIdx]} --------------")
    #         print(f"{PD[0,SolInActiveIdx]}: {f[SolInActiveIdx][0]:1.2e}, {PD[0,SolActiveIdx]}: {f[SolActiveIdx][0]:1.2e},T: {T:4.2f} {delta_f:1.2e}")
    
    # Nn = N0 - np.sum(N[-1,:])
    # if Nn <= 0:
    #     Nn = 0
        
    if it == 0:
        ini = [dc.T,np.max(v, axis=1),dN.T] 
     
    f[-1] = sum(f[:-1])
    xm = (x0-np.sum(f[:-1]*xp,axis=0))/(1-f[-1])  
    if (xm<0).any():
        xm[xm<0] = 1e-6
    
    #Transfer all populations to a uniform
    N[-1,:]=0
    for p in range(phases): #Interpolate N[p] to N[0] as N_tot and store modus of dist
        if max(N[p,:]) > 0:
            Rmean_p_t[p] = r[p,np.where(N[p,:] == max(N[p,:]))] #max N(r), modus 
            # Rmean_p_t[p] = r[p,(N[p,:]-np.median(N[:,p])).argmin()] # median (should be larger than modus)
            N[-1,:] += N_Trans(r[-1,:],r[p,:],NBlank,N[p,:])     

    r_old = r.copy()
    N_old = N.copy()   
    N_old_BF = N_BF.copy()
    r_old_BF = r_BF.copy()       
    
    ######## Sample data to plot at certain volumes or times.
    ############################
    ############################
    if f[-1] >= PlotVol or Time >= tt or f[-1] >= NDD_vols[NDD_tic]:    
        
        if f[-1] >= PlotVol or Time >= tt: 
            
            N_tot_t[:,0] = np.sum(N,axis=1)
            N_tot = np.append(N_tot,N_tot_t,axis=1)
            V_tot = np.append(V_tot,f,axis=1)
            Rmean_p = np.append(Rmean_p,Rmean_p_t,axis=1)
            TimePar = np.append(TimePar,Time)
            TempPar = np.append(TempPar,T)
                
            if f[-1]>=PlotVol:
                PlotVol += fmax/5
            if Time>=tt:
                tt += 0.05

        if f[-1] >= NDD_vols[NDD_tic]:
            coltic = 0
            rowtic = 0   
            limIdx = 0
            limidx_r = 0
            Nmax = 1e5
            for p in range(phases+1):
                if p < phases-1:
                    if np.max(N[:-1,:],axis = 1)[p] > Nmax:
                        Nmax = np.max(N[:-1,:],axis = 1)[p]
                        limIdx = p
                        limidx_r = np.where(N[p,:] >= Nmax)[-1]
                    
                if big_flag == True:
                    ax2[p-rowtic,coltic].plot(r[p,:]*1e9,N[p,:],COLORS[NDD_tic],linewidth=2)
                    if p < phases:
                        ax2[p-rowtic,coltic].set_ylabel(plotLabel(PD[0,p]),fontsize=15)
                    
                    if p == int(np.ceil(phases/2))-1:
                        ax2[p-rowtic,coltic].set_xlabel('$r$ (nm)')
                        # ax2[p-rowtic,coltic].set_xlim(left=5e-10*1e9, right=5*Rmean_p_t[limIdx]*1e9)
                        ax2[p-rowtic,coltic].set_xlim(left=0.1, right=3*r[limIdx,limidx_r]*1e9)
                        coltic += 1
                        if int(phases/int(np.ceil(phases/2))) == 2:
                            rowtic = int(np.ceil(phases/2)) + 1 
                        else:
                            rowtic = int(np.ceil(phases/2))
                        
                    if p == phases:
                        ax2[p-rowtic,coltic].plot(r[-1,:]*1e9,N[-1,:],COLORS[NDD_tic],linewidth=2,label=f"t: {Time:1.2e}")
                        ax2[p-rowtic,coltic].set_ylabel('Sum')
                        ax2[p-rowtic,coltic].set_xlabel('$r$ (nm)')
                        # ax2[p-rowtic,coltic].set_xlim(left=5e-10*1e9, right=5*Rmean_p_t[limIdx]*1e9)
                        ax2[p-rowtic,coltic].set_xlim(left=0.1, right=3*r[limIdx,limidx_r]*1e9)
                        for row in range(rowtic):
                            ax2[row, coltic].yaxis.set_label_position("right")
                            ax2[row, coltic].yaxis.tick_right()
                        
                else:                   
                    ax2[p].plot(r[p,:]*1e9,N[p,:],COLORS[NDD_tic],linewidth=2)
                    if p < phases:
                        ax2[p].set_ylabel(plotLabel(PD[0,p]),fontsize=15)
                    if p == phases:
                        ax2[p].plot(r[-1,:]*1e9,N[-1,:],COLORS[NDD_tic],linewidth=2,label=f"t: {Time:1.2e}")
                        ax2[p].set_ylabel('Sum')
                        ax2[p].set_xlabel('$r$ (nm)')
                        # ax2[p].set_xlim(left=5e-10*1e9, right=5*max(Rmean_p_t)*1e9)
                        ax2[p].set_xlim(left=0.1, right=3*r[limIdx,limidx_r]*1e9)
                
            fig2.legend(bbox_to_anchor=(0.08,0.92,0.85,0.2),loc="lower left",ncol=4,handlelength=0.9, handletextpad=0.3, columnspacing=0.7, mode="expand", borderaxespad=0.)
             
            NDD_tic += 1
    
    ############################
    ############################

    if not print_flag:  
        status = (
            f"\r t:{Time:2.4f} "
            f"T:{T:4.1f} "
            f"it:{it} \n"
            f"f:[{', '.join(f'{f[pp][0]:1.2e}' for pp in range(phases+1))}] "
            f"dc:[{', '.join(f'{dc[pp,0]:1.2e}' for pp in range(phases))}] "
            f"maxV:[{', '.join(f'{max(v[pp,:]):4.2e}' for pp in range(phases))}] "
            f"dN:[{', '.join(f'{dN[pp][0]:4.2e}' for pp in range(phases))}]"
        )
    
        print(status, end='\n')
        
        print_flag = True
        it_old = it
            
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

Final_status = (
    f"\r t:{Time:2.4f}, "
    f"T:{T:4.1f}, "
    f"it:{it}, "
    f"dt:{dt:1.2e} \n"
    f"f:[{', '.join(f'{f[pp][0]:1.2e}' for pp in range(phases+1))}] "
    f"dc:[{', '.join(f'{dc[pp,0]:1.2e}' for pp in range(phases))}] "
    f"maxV:[{', '.join(f'{max(v[pp,:]):4.2e}' for pp in range(phases))}] "
    f"dN:[{', '.join(f'{dN[pp][0]:4.2e}' for pp in range(phases))}]"
)

print(Final_status, end='\n')

print('-------------------------')
print('Initial:')
Initial_status = (
    f"dc:[{', '.join(f'{ini[0][0,pp]:1.2e}' for pp in range(phases))}] "
    f"maxV:[{', '.join(f'{ini[1][pp]:4.2e}' for pp in range(phases))}] "
    f"dN:[{', '.join(f'{ini[2][0,pp]:4.2e}' for pp in range(phases))}]"
)

print(Initial_status, end='\n')

if f[-1] >= 1e-2:
    print("-------------------------")
    print("Crystallization occured (1%), no critical rate")

print('-------------------------')
simtime=str(round((time.time()-simt)/60,3))
print(f'Computation time: {simtime} min')
print(f'Iterations: {it}')
#%%

if phases <= 4:
    fig1,ax1 = plt.subplots(phases+1)          # Initiate PSD fig
    big_flag = False
elif phases > 4:
    fig1,ax1 = plt.subplots(int(np.ceil(phases/2)) ,2)     
    big_flag = True

fig1.suptitle(f'Velocity (m/s), {T:4.0f} (K), ftot: {f[-1][0]:1.1e}')   

coltic = 0
rowtic = 0   
for p in range(phases):        
    if big_flag == True:
        ax1[p-rowtic,coltic].plot(r[p,:], v[p,:].reshape([len(v[0,:]),1]),'k',linewidth=2, label = f"{plotLabel(PD[0,p])}")
        ax1[p-rowtic,coltic].set_xscale('log')
        ax1[p-rowtic,coltic].legend(loc = "upper right")
        ax1[p-rowtic,coltic].set_xlim(0.8*r_star[p],r[p,-1])
        ax1[p-rowtic,coltic].set_ylim(1.2*np.min(v[p,r[p,:]>0.8*r_star[p]]),1.2*np.max(v[p,r[p,:]>0.8*r_star[p]]))
        
        if p == int(np.ceil(phases/2))-1:
            ax1[p-rowtic,coltic].set_xlabel('$r$ (nm)')
            coltic += 1
            rowtic = int(np.ceil(phases/2))
        if p == phases -1:
            ax1[-1,-1].set_xlabel('$r$ (nm)')
    else:                   
        ax1[p].plot(r[p,:], v[p,:].reshape([len(v[0,:]),1]),'k',linewidth=2, label = f"{plotLabel(PD[0,p])}")        
        ax1[p].legend(loc = "upper right")
        if p == phases -1:
            ax1[p].set_xlabel('$r$ (nm)')

if isHeating:                 
    FigpathOut=f'{OutDir}Velocities_Heating.png'
    fig1.savefig(FigpathOut, bbox_inches = "tight")
elif isQuenching:
    FigpathOut=f'{OutDir}Velocities_Quenching.png'
    fig1.savefig(FigpathOut, bbox_inches = "tight")

#############################

#%%        
fig3,ax=plt.subplots(4, sharex=True,figsize=(9,12))
FracStr=[]

for p in range(phases):
    ax[0].plot(TimePar[1:], Rmean_p[p,1:],'-*',color=COLOR2[p],linewidth=2,label=plotLabel(PD[0,p]))
ax[0].set_ylabel('$r_m$  (m)',fontsize=15)
ax[0].set_yscale('log')
ax[0].set_ylim(ymin=Rmean_p[:,1:].min(), ymax=2*Rmean_p.max())
ax[0].yaxis.set_minor_locator(plt.MaxNLocator(5))
ax[0].legend(bbox_to_anchor=(0.08,0.8,0.85,0.2),loc="lower left",ncol=4,handlelength=0.9, handletextpad=0.3, columnspacing=0.7, mode="expand", borderaxespad=0.)

for p in range(phases):
    ax[1].plot(TimePar[1:], N_tot[p,1:],'-*',color=COLOR2[p],linewidth=2)
ax[1].set_ylabel('$N_{tot}$  (m$^{-3}$)',fontsize=15)
ax[1].set_yscale('log')
ax[1].set_ylim(ymin=N_tot[:,1:].min(), ymax=10**(np.ceil(np.log10(N_tot.max()))+1))
ax[1].yaxis.set_minor_locator(plt.MaxNLocator(5))

for p in range(phases):
    ax[2].plot(TimePar[1:], V_tot[p,1:],'-*',color=COLOR2[p],linewidth=2,label='_nolegend_')
    if f[p] == 0:
        FracStr=np.append(FracStr,0)
    else:
        FracStr=np.append(FracStr,str(round((f[p]/f[-1])[0]*100,1)))
ax[2].set_ylabel('$V$  (%)',fontsize=15)
ax[2].set_yscale('log')
ax[2].set_ylim(ymin=V_tot[:,1:].min(), ymax=1)
ax[2].yaxis.set_minor_locator(plt.MaxNLocator(5))

ax[3].plot(TimePar[1:], TempPar[1:],'-*',linewidth=2,label='_nolegend_')
ax[3].set_xlabel('$t$  (s)',fontsize=15)
ax[3].set_ylabel('$T$  (K)',fontsize=15)
ax[3].yaxis.set_minor_locator(plt.MaxNLocator(5))

FracStr_print =[]
for p in range(phases):
    if eval(FracStr[p]) > 1e-3:
        FracStr_print.append(f"{PD[0,p]}: {eval(FracStr[p]):1.1f}%, ") 
        
fig3.suptitle([" ".join(FracStr_print).rstrip(", "),f"tot: {f[-1][0]:1.1e}"])

plt.draw()

if isHeating:
    FigpathOut=f'{OutDir}PSD_Heating.png'
    fig2.savefig(FigpathOut, bbox_inches = "tight")
    FigpathOut=f'{OutDir}GrowthPar_Heating.png'
    fig3.savefig(FigpathOut, bbox_inches = "tight")
else:
    FigpathOut=f'{OutDir}PSD_Quenching.png'
    fig2.savefig(FigpathOut, bbox_inches = "tight")
    FigpathOut=f'{OutDir}GrowthPar_Quenching.png'
    fig3.savefig(FigpathOut, bbox_inches = "tight")    
    
    