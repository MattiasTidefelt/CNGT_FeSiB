#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 6 2025

@author: ag0406

18/8-25, this is good, all phases that exhibit a driving force in the range 400-2500 K are included
in the simulations, otherwise they're excluded. The driving forces that i get are compatible with those obtained
from Fe-B system if only this binary is considered. As in the binary case, the solution phases does not exhibit
no real solubility (no common tangent can be found), 
hence, they're treated as stoichiometric phases in the sens of beeing end members of ~pure Fe.

Next step should be to test and sample some compositions and then write it to C code and prepare for parallellization with
MPI4py.

Mobility data would be interesting for the diffusion coefficients. 
The flat interfe energy as modelled from Kapty would also be of interest. 

TEST TO EXTRACT FROM TDB TO GET COMPOSITION!!!!!!!!!!!!!!!!!!!!!!!
However, to low at the moment (400-650 K), would expect it to be a bit higher
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
from matplotlib.legend_handler import HandlerTuple
from os import makedirs
from os.path import exists

#functions
from DeffMult import DMult
from Jst_mult import Jst_mult
from DcXp import DcXp
from LiqFeSiB import LiqFeSiB, TransE
from SolidSolFeSiB import SolidSolFeSiB
from SolidFeSiB import SolidFeSiB
from plotLabel import plotLabel
from Tau_Func import Tau_Func
from paramFeSiB import sysParam
from Get_HandS import Get_HandS
from RandSigma import RandSigma
from HEXCOL import HEXCOL  
from CheckForDcAtComp import CheckDcAtMatrixComp
# from find_xGuess import find_xGuess_avg, find_xGuess
from GetTm import GetTm

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

# from Kfunc import initialize_spline
# initialize_spline() 

isStoich = False

basedir = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/Figures/"

#--------------Simulation parameters ---------------
N_A=6.02214086E23
k_B=1.38064852E-23  
R=8.31446261815324             

# xm = np.array([0.80906114, 0.17027636, 0.01916881])
# xm = np.array([0.762, 0.114, 0.123])
# xm = np.array([0.9-0.05,0.1,0.05])
# xm = np.array([0.9-0.05,0.05,0.1])
# xm = np.array([0.849,0.001,0.15])
# xm = np.array([0.8111596 , 0.12589529, 0.06294511])
# xm = np.array([0.8,0.1,0.1])
# xm = np.array([0.67448258, 0.1089052 , 0.21661222])
# xm = np.array([0.9,0.09,0.01])
# xm = np.array([0.8,0.01,0.19])  
# xm = np.array([0.9,0.01,0.09]) 
# xm = np.array([0.89,0.01,0.1]) 
# xm = np.array([0.74691075, 0.14755855, 0.1055307])
# xm = np.array([8.3518493e-01, 6.0763441e-02, 1.0405160e-01]) 
# xm = np.array([0.8320919,  0.05669853, 0.11120956])
# xm = np.array([0.82750536, 0.03799919, 0.13449545])
# xm = np.array([0.767 , 0.178, 0.055])
# xm = np.array([0.806 , 0.083, 0.111])
# xm = np.array([0.800 , 0.087, 0.113])
# xm = np.array([0.88633399, 0.07551005, 0.03815596])
# xm = np.array([0.80, 0.05, 0.15])
# xm = np.array([0.70, 0.05, 0.25])
# xm = np.array([0.70, 0.15, 0.15])
# xm = np.array([0.762 , 0.211, 0.027])
# xm = np.array([0.829 , 0.048, 0.122])
# xm = np.array([0.665, 0.289, 0.045])
# xm = np.array([0.752, 0.042, 0.206])
# xm = np.array([0.81, 0.12, 0.07])
# xm = np.array([0.76, 0.18, 0.06])
# xm = np.array([0.750,0.240,0.010])
# xm = np.array([0.750,0.010,0.240])
# xm = np.array([0.781,0.208,0.011])
# xm = np.array([0.7,0.01,0.29])
# xm = np.array([0.7,0.29,0.01])
# xm = np.array([0.82875918, 0.08442636, 0.08681446])
# xm = np.array([0.7,0.15,0.15])

# xm = np.array([0.7, 0.29, 0.01])
# xm = np.array([0.79, 0.20, 0.01])
# xm = np.array([0.81, 0.18, 0.01])
# xm = np.array([0.83, 0.16, 0.01])
# xm = np.array([0.9, 0.09, 0.01])

# xm = np.array([0.7, 0.01, 0.29])
# xm = np.array([0.81, 0.01, 0.18])
# xm = np.array([0.86, 0.01, 0.13])
xm = np.array([0.9, 0.01, 0.09])
# xm = np.array([0.95, 0.01, 0.04])

# xm = np.array([0.86, 0.03, 0.11])
# xm = np.array([0.6, 0.39, 0.01]) # Here the scilicides start to dominate!

# xm =  np.array([0.791,0.035,0.174])

################ g/cm3
xFe = 0.85
xSi = 0.15
mSi = 2.3296
mFe = 7.874
mFe_ref = 7.87/mFe*xFe +mSi*xSi
###############

x0Str = "_".join(f"{x:.3f}" for x in xm)
OutDir =f'{basedir}/Comp_{x0Str}/'
if not exists(OutDir):
    makedirs(OutDir)

xp_force = np.array([1-2*1e-6,1e-6,1e-6])               # Does only apply for sol phases!

Parameters=sysParam(xm)
PD=Parameters[0]                                        # PhaseData
d0=Parameters[1]                                        # Characteristic length scale
Vm = PD[1,:] 
Vml=Parameters[2]                                       # Molar volume liquid
Stype = Parameters[3]                              

Hf_Sf, PD = Get_HandS(PD,xm)                             # Get reduced PD if phase does not exhibit dc in temp itnerval

Hf_Sf, PD, Vm = CheckDcAtMatrixComp(Vm,PD,Hf_Sf,xm)            # Get reduced PD if phase does not exhibit dc in temp itnerval, but at matrix composition     

phases=PD.shape[1]

COLORS=HEXCOL(phases,1)

[_,Tg] = TransE(1000,xm[0],xm[1],xm[2])  
Tm = GetTm(PD,xm)                              
Temp=np.arange(Tg,Tm,50)  #600-1250     

# xGuess = find_xGuess_avg(Tg, PD, xm, xp_force)    

gSize=50   #150 for log      
dcMin = 1e1

#------------initiate arrays-----------------
dc_plot=np.zeros([phases,len(Temp)])
Nuc_plot=dc_plot.copy()
v_plot=dc_plot.copy()
r_plot=dc_plot.copy()
Sig_plot=dc_plot.copy()
Tau_plot=dc_plot.copy()

D_plot=np.zeros([3,len(Temp)]) 
                      
#------------ Initiate V/J_plot fig
fig1, ax1 = plt.subplots()
fig1.subplots_adjust(right=0.85)
twin1 = ax1.twinx()
#------------ Initiate dc/r*/D fig
fig2,ax2=plt.subplots(4, sharex=True, figsize =(8,6))
    
#------------ Initiate Sigma(r,T) fig
fig3,ax3=plt.subplots(1)

#------------ Initiate diffusion fig
fig4, ax4=plt.subplots(1)

simt=time.time()

N0=N_A/Vml                               #Nucsites per unit volume
        
for j in range(len(Temp)):
    T=float(Temp[j])  
    LiqFeSiB(T,'Liquid')      #Update Liquid Gibbs(T) Polynomials
    TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
    fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
    Mu = fp + TangCoeff - np.dot(xm,TangCoeff)
    
    print(f'\r------------------ Temperature: {T:4.0f} ------------------------')    
    
    for p in range(phases):
        if PD[-1,p] == 'Stoic':
            SolidFeSiB(T,PD[0,p])           #Update phase Gibbs(T) Polynomials
        elif PD[-1,p] == 'Sol':
            SolidSolFeSiB(T,PD[0,p])        #Update Liquid Gibbs(T) Polynomials
        
        # Get dc and xp
        [dc,xp] = DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])
        if PD[0,p] == 'bcc':
            print(xp)
        # dc *= 0.8
        
        [_,Tg] = TransE(T,xm[0],xm[1],xm[2]) 
        D = DMult(T,Tg,Tm,Vml)
        D_plot[:,j]=D

        if dc>= dcMin:
            dc_plot[p,j]=dc   
            
            [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm[p],Hf_Sf,T,d0,gSize,PD[:,p],Stype)
            # [G_star,r,dr,r_star,Sigma,sigma_star] = RandSigma(dc,xp,Vm[p],Hf_Sf[:,p],T,d0,gSize,PD[:,p],Stype)
            # print("-----------------"*2)
            # print(r_star)
            
            G_vol = dc/Vml      # J/m3
            G_surf = sigma_star #J/m2
            # G_star = 16*np.pi/3*G_surf**3/G_vol**2
            # r_star = 2*G_surf/G_vol
            
            G_star = 16*np.pi/3*Vm[p]**2*sigma_star**3/dc**2          
            r_star = 2*Vm[p]*sigma_star/dc

            r_plot[p,j]=r_star
            Sig_plot[p,j]=sigma_star
            # print(r_star)
            
            # if j==len(Temp)-1:
            if j==0:
                ax3.plot(r*1e9, Sigma,label=plotLabel(PD[0,p]))
                ax3.set_xlabel('$r$  [nm]',fontsize=15)
                ax3.set_xscale('log')
                ax3.set_ylabel('$\sigma(r,T)$  [J/m$^2$]',fontsize=15)
                ax3.set_title(f'Sigma(r,{T:1.0f})')
                ax3.legend(loc='lower right',fontsize=10)
                        
            #-------------Scaled diffusion in matrix
            #aims to find the composition guess for phase equilibrium 
            # based on particle composition and matrix phase
            # xGuesses = find_xGuess(T,PD[:,p],xm,xp)
            # xGuesses = xp
            # xGuess_plot[:,p,j] = xGuesses
            
            # xGuess_plot[:,p,j] = xGuess[p,:]
            
            # GT = (2*Sigma/(r+dr))
            # preFact = (G_vol - GT)/(R*T*r)
            
            GT = (2*Sigma/(r))*Vm[p]
            preFact = (dc - GT)/(R*T*r)
            compFact = np.sum((xp-xm)**2/(xm*D))
            
            v = preFact/compFact 
            
            v_plot[p,j]=max(v) 
                       
            rkBT=r_star+0.5*np.sqrt(k_B*T/(np.pi*sigma_star)) 

            J_st,Tau  = Jst_mult(T,G_vol,G_surf,G_star,r_star,D,sigma_star,xm,xp,N0,Vm[p],Vml) 
            # J_st,Tau = Jst_mult(T,G_star,r_star,D,sigma_star,xm,xp,N0,Vm[p],Vml) 
            
            Tau_plot[p,j] = Tau
            Nuc_plot[p,j] = J_st
            
            #%% Plots
            # ------------------------------S plot----------------------------
    #         ax[p].plot(r, v,'k',linewidth=2)
    #         ax[p].set_xscale('log')
    #         ax[p].ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    #         ax[p].set_title(PD[0,p],fontsize=15)
    #         if p==0:
    #             ax[p].vlines(x=rkBT, ymin=min(v), ymax=v[(np.abs(r - rkBT)).argmin()], ls='--', lw=2, label='r$_kBT$')
    #             ax[p].plot(r_star,v[(np.abs(r - r_star)).argmin()],'ro',label='r *')
    #             ax[p].set_ylabel('$v$  (m/s)',fontsize=15)
    #             ax[p].legend(loc='upper right',fontsize=10)
    #         else:
    #             ax[p].vlines(x=rkBT, ymin=min(v), ymax=v[(np.abs(r - rkBT)).argmin()], ls='--', lw=2)
    #             ax[p].plot(r_star,v[(np.abs(r - r_star)).argmin()],'ro')
        
    # if isSave==True:
    #     FigpathOut=''.join([OutDir,f'ParamCheck_{T:1.0f}.png'])
    #     fig.savefig(FigpathOut, bbox_inches = "tight")

simtime=str(round((time.time()-simt),3))
print(f'Computation time: {simtime} s')
#%%  
# ax4.semilogy(Temp,D_plot[:],label = "Deff") 
for i in range(3):
    ax4.semilogy(Temp,D_plot[i,:])
    
ax4.set_xlabel("T (K)")
ax4.set_ylabel("D")
ax4.set_title(f'Fe:{xm[0]:1.3f} Si:{xm[1]:1.3f} B:{xm[2]:1.3f}, Tg: {Tg:1.1f}')
ax4.legend(['Fe','Si','B'])
  
######### Guess plot
# print("------------------------------------------")
# coltic = 0
# rowtic = 0  
# for p in range(phases):  
#     nonZeroIdx = np.where(xGuess_plot[0,p,:] != 0)[0]
#     if len(nonZeroIdx) == 0:
#         Guess_mean = np.array([0,0,0])
#     else:    
#         Guess_mean = np.median(xGuess_plot[:,p,nonZeroIdx],axis=1)
#     print(f"{PD[0,p]} median Guess: {Guess_mean}") 
#     for i in range(3):  
#         if big_flag == True:
#             ax5[p-rowtic,coltic].plot(Temp[nonZeroIdx],xGuess_plot[i,p,nonZeroIdx],linewidth=2)
#             if len(nonZeroIdx) != 0:
#                 ax5[p-rowtic,coltic].hlines(y=Guess_mean[i],xmin = Temp[nonZeroIdx[0]], xmax = Temp[nonZeroIdx[-1]], ls='--', lw=1)
#             ax5[p-rowtic,coltic].set_ylabel(plotLabel(PD[0,p]),fontsize=15)
            
#             if i == 2:
#                 if p == int(np.ceil(phases/2))-1:
#                     ax5[p-rowtic,coltic].set_xlabel("T (K)")
#                     coltic += 1
#                     if int(phases/int(np.ceil(phases/2))) == 2:
#                         rowtic = int(np.ceil(phases/2)) 
#                     else:
#                         rowtic = int(np.ceil(phases/2))
                    
#                 if p == phases-1:
#                     for row in range(rowtic):
#                         ax5[row, coltic].yaxis.set_label_position("right")
#                         ax5[row, coltic].yaxis.tick_right()
                
#         else:                   
#             ax5[p].plot(Temp[nonZeroIdx],xGuess_plot[i,p,nonZeroIdx],linewidth=2)
#             ax5[p].hlines(y=Guess_mean[i],xmin = Temp[nonZeroIdx[0]], xmax = Temp[nonZeroIdx[-1]], ls='--', lw=2)
#             ax5[p].set_ylabel(plotLabel(PD[0,p]),fontsize=15)
#             if p == phases:
#                 ax5[p].set_ylabel('Sum')
#                 ax5[p].set_xlabel("T (K)")
    
# fig5.suptitle("xGuess") 
#%% ##########  
p1l=list()
p2l=list() 
for p in range(phases):
    plotLim=len(dc_plot[p,np.where(dc_plot[p,:]>1e2)][0])   
    # ------------------------------V/J plot---------------------------- 
    
    markStyle=['o','x','D','s','<','>','^','.','v','+',',','1','2','3']
    
    ax1.set_ylim([1E-3, np.max(Nuc_plot[~np.isnan(Nuc_plot)])*1e1])
    ax1.set_yscale('log')
    twin1.set_ylim([1E-12, np.max(v_plot[~np.isnan(v_plot)])*1e1])
    twin1.set_yscale('log')
        
    p1, = ax1.plot(Temp[:plotLim], Nuc_plot[p,:plotLim], 'r-',marker=markStyle[p],label=PD[0,p])
    p2, = twin1.plot(Temp[:plotLim], v_plot[p,:plotLim], 'b-',marker=markStyle[p])
 
    p1l.append(p1)
    p2l.append(p2)
       
    ax1.set_xlabel("Temp")
    ax1.set_ylabel("J$_{ss}$  (m$^{-3}$s$^{-1}$)",fontsize=15)
    twin1.set_ylabel("$v$  (m/s)",fontsize=15)
    
    ax1.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    
    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax1.tick_params(axis='x', **tkw)
    fig1.suptitle(f'Fe:{xm[0]:1.3f} Si:{xm[1]:1.3f} B:{xm[2]:1.3f}, Tg: {Tg:1.1f}') 
    
    #--------------------- dc/r*/D plot---------------------------------
    ax2[0].plot(Temp[:plotLim],dc_plot[p,:plotLim]/1e3,color=COLORS[p],label=PD[0,p])
    ax2[0].set_ylabel('dc (kJ/mol)',fontsize=15)
    ax2[0].legend(loc='upper right',fontsize=10)
    ax2[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:2.1f}"))
    
    ax2[1].plot(Temp[:plotLim],r_plot[p,:plotLim]*1e9,color=COLORS[p])
    ax2[1].set_yscale('log')
    ax2[1].set_ylabel('r* (nm)',fontsize=15)
    ax2[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:2.1f}"))
    
    ax2[3].plot(Temp[:plotLim],Tau_plot[p,:plotLim],color=COLORS[p])
    ax2[3].set_yscale('log')
    ax2[3].set_xlabel('Temp (K)',fontsize=15)
    ax2[3].set_ylabel('Tau (s)',fontsize=15)
    
    ax2[2].plot(Temp[:plotLim],Sig_plot[p,:plotLim],color=COLORS[p])
    ax2[2].set_ylabel('$\sigma^*$ [J/m$^2$]',fontsize=15)
    ax2[2].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:2.1f}"))

ax2[0].legend(ncol = int(np.ceil(phases/2)), bbox_to_anchor=(0.09, 1.05))   
handles = []
labels = []

# Loop to accumulate handles and labels for each phase
for i in range(phases):
    handles.append((p1l[i], p2l[i]))  # Add handles for each phase
    labels.append(PD[0, i])  # Add labels for each phase

# Create the legend using the accumulated handles and labels
ax1.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower right')
plt.show()

FigpathOut=''.join([OutDir,'Jv.png'])
fig1.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([OutDir,'dcPlot.png'])
fig2.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([OutDir,'Sigma_rT.png'])
fig3.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([OutDir,'xGuesses.png'])
# fig5.savefig(FigpathOut, bbox_inches = "tight")

