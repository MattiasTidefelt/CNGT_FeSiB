#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:53:55 2021
Thermodynamic description: Yoshitomi et al. ISIJ international, Vol. 48 (2008)
@author: ag0406

Compute and illustrate liquid Gibbs ternary curve with line compounds.
The maximum driving force approach is calculated as the maximum between the 
tangents of the liquid phase and the solid phases.

For the substitutional solution model


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import makedirs
from os.path import exists
from matplotlib import rcParams
import matplotlib.ticker as mtick

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func_surf import Gibbs_Func_surf
from Gibbs_Func import Gibbs_Func
from LiqFeSiB import LiqFeSiB, TransE
from SolidSolFeSiB import SolidSolFeSiB
from SolidFeSiB import SolidFeSiB
from DcXp import DcXp
from plotLabel import plotLabel
from StoicCompounds import StoicCompounds
from paramFeSiB import sysParam
from GetTm import GetTm

rcParams.update({'font.size': 13})
rcParams['axes.linewidth'] = 2

basedir = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/Figures/"
isStoich = False
#-----------------
# xm = np.array([0.759,0.070,0.171])
# xm = np.array([0.9-0.05,0.1,0.05])
# xm = np.array([0.8699968 , 0.03000012, 0.11000339])
# xm = np.array([0.9-0.05,0.05,0.1])
# xm = np.array([0.8111596 , 0.12589529, 0.06294511])
# xm = np.array([0.8,0.1,0.1])
# xm = np.array([0.67448258, 0.1089052 , 0.21661222])
# xm = np.array([0.74691075, 0.14755855, 0.1055307])
# xm = np.array([0.9,0.01,0.09]) 
# xm = np.array([0.83517896, 0.06073153, 0.10408948])
# xm = np.array([0.767 , 0.178, 0.055])

# xm = np.array([0.7, 0.29, 0.01])
# xm = np.array([0.81, 0.18, 0.01])
# xm = np.array([0.9, 0.09, 0.01])

# xm = np.array([0.7, 0.01, 0.29])
# xm = np.array([0.81, 0.01, 0.18])
xm = np.array([0.9, 0.01, 0.09])
# xm = np.array([0.95, 0.01, 0.04])

x0Str = "_".join(f"{x:.3f}" for x in xm)
OutDir =f'{basedir}/Comp_{x0Str}/'
if not exists(OutDir):
    makedirs(OutDir)

xp_force =np.array([1-2*1e-6,1e-6,1e-6])

T=float(1000)
print(f"T: {T:1.0f} K")

[_,Tg] = TransE(1000,xm[0],xm[1],xm[2]) 
print(f"Tg: {Tg:1.0f} K")

Parameters=sysParam(xm)
PD=Parameters[0]                                        # PhaseData
d0=Parameters[1]                                        # Characteristic length scale
Vml=Parameters[2]                                       # Molar volume liquid
Stype = Parameters[3]

phases=PD.shape[1]
reso=20

#-----------------Initiate arrays
dc=np.zeros([phases])
#check non stoichiometric phases
S_p = PD[-1,np.where(PD[-1,:] == 'Stoic')[0]]
Sol_p = PD[-1,np.where(PD[-1,:] == 'Sol')[0]]

yB=dc.copy()
MuAC_liq=dc.copy()
MuBC_liq=dc.copy()
MuAC_p=dc.copy()
MuBC_p=dc.copy()

xp=np.zeros([phases,3])

Gp=np.zeros([phases,reso])
Tp=Gp.copy()
T_Liq_p=Gp.copy()

mx=np.zeros([reso,3,phases])

#-----------------
# dx=1e-3
dx= 1e-6
X=np.linspace(0+dx,1-2*dx,reso)   #A=xFe,B=xSi,C=xA

LiqFeSiB(T,'Liquid')
TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
Mu = fp + TangCoeff - np.dot(xm,TangCoeff)

#Liquid Gibbs and tangent plane
xB=X                                 #A=xFe,B=xSi,C=xB
Map_liq=pd.DataFrame()
Map_T=pd.DataFrame()
Map_Tg=pd.DataFrame()
Map_Tm=pd.DataFrame()
   
for k in range(len(xB)):    #slice surface and stack each line in a dataframe with corresponding function value
    xC=np.linspace(dx,1-xB[k],len(xB)-k)
    xA=(1-xB[k])-xC
    
    Gm=Gibbs_Func_surf(T,xA,np.array([xB[k]]),xC,'Liquid')   
 
    Map2=pd.DataFrame({'0':xB[k]*np.ones(len(xC)),'1':xC,'2':Gm})
    Map_liq=pd.concat([Map_liq,Map2],ignore_index=True) 

    Tplane=Mu[0]*xA + Mu[1]*xB[k] + Mu[2]*xC
    
    Map2=pd.DataFrame({'0':xB[k]*np.ones(len(xC)),'1':xC,'2':Tplane})
    Map_T=pd.concat([Map_T,Map2],ignore_index=True)   
     
print('     MuA       MuB         MuC')  
print([round(Mu[0],2),round(Mu[1],2),round(Mu[2],2)])

for k in range(len(xB)):    #slice surface and stack each line in a dataframe with corresponding function value
    xC=np.linspace(dx,1-xB[k],len(xB)-k)
    # xC=np.linspace(0.01,0.99-xB[k],len(xB)-k)
    xA=(1-xB[k])-xC

    [_,Tg] = TransE(1000,xA,np.array([xB[k]]),xC) 
    Map3=pd.DataFrame({'xA':xA,'xB':xB[k]*np.ones(len(xC)),'xC':xC,'Tg':Tg})
    Map_Tg=pd.concat([Map_Tg,Map3],ignore_index=True)  
    
    Tm = np.zeros(len(xA))
    for i in range(len(xA)):
        xTm = np.array([xA[i],xB[k],xC[i]])
        Tm[i] = GetTm(PD,xTm)
    Map4=pd.DataFrame({'xA':xA,'xB':xB[k]*np.ones(len(xC)),'xC':xC,'Tm':Tm})
    Map_Tm=pd.concat([Map_Tm,Map4],ignore_index=True)  
    
LiqFeSiB(T,'Liquid')
#%%


#---------------Specified slice of liq, B rich and Si rich-----------------
fig2 ,ax2 = plt.subplots(1,2, figsize=(12,5))
# B rich
xA_s = X
xB_s = X[0]*np.ones(reso)
xC_s = ((1-dx)-X)

Gm=Gibbs_Func_surf(T,xA_s,xB_s,xC_s,'Liquid')   
Map_liq_s=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Gm}) 
#Tangent slice
Tplane=Mu[0]*xA_s + Mu[1]*xB_s + Mu[2]*xC_s
Map_T_s=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Tplane})

Lig_phases_p=np.zeros([Map_T_s.shape[0],Map_T_s.shape[1],len(Sol_p)])
for i in range(len(Sol_p)):
    SolidSolFeSiB(T,PD[0,-len(Sol_p)+i])
    Gp=Gibbs_Func_surf(T,xA_s,xB_s,xC_s,PD[0,-len(Sol_p)+i])   
    Lig_phases_p[:,:,i]=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Gp}) 
    
#----------------Slice of 3D Gibbs B ------------------
ax2[0].plot(Map_liq_s['0'],Map_liq_s['2']*1e-3,linewidth=2,label='Liquid')
ax2[0].plot(Map_T_s['0'],Map_T_s['2']*1e-3,linewidth=2,label='Tangent')

for i in range(len(Sol_p)): 
    ax2[0].plot(Lig_phases_p[:,0,i],Lig_phases_p[:,2,i]*1e-3,linewidth=2,label=PD[0,-len(Sol_p)+i])
    
ax2[0].ticklabel_format(style='sci',scilimits=(0,0), axis='y')
ax2[0].set_xlabel('Fe fraction')
ax2[0].set_ylabel('G$_m$  [kJ/mol]')
ax2[0].set_title(f'Gm Fe-B, T:{T}')
ax2[0].set_xlim(0,1)
ax2[0].set_ylim(1.3*Map_liq_s['2'].min()*1e-3,1.1*Lig_phases_p[:,2,:].max()*1e-3)
ax2[0].legend()

# Si rich
xA_s = X
xB_s = ((1-dx)-X)
xC_s = X[0]*np.ones(reso)

Gm=Gibbs_Func_surf(T,xA_s,xB_s,xC_s,'Liquid')   
Map_liq_s=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Gm}) 
#Tangent slice
Tplane=Mu[0]*xA_s + Mu[1]*xB_s + Mu[2]*xC_s
Map_T_s=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Tplane})

Lig_phases_p=np.zeros([Map_T_s.shape[0],Map_T_s.shape[1],len(Sol_p)])
for i in range(len(Sol_p)):
    SolidSolFeSiB(T,PD[0,-len(Sol_p)+i])
    Gp=Gibbs_Func_surf(T,xA_s,xB_s,xC_s,PD[0,-len(Sol_p)+i])   
    Lig_phases_p[:,:,i]=pd.DataFrame({'0':xA_s,'1':xC_s,'2':Gp}) 
    
#----------------Slice of 3D Gibbs Si ------------------

ax2[1].plot(Map_liq_s['0'],Map_liq_s['2']*1e-3,linewidth=2,label='Liquid')
ax2[1].plot(Map_T_s['0'],Map_T_s['2']*1e-3,linewidth=2,label='Tangent')

for i in range(len(Sol_p)): 
    ax2[1].plot(Lig_phases_p[:,0,i],Lig_phases_p[:,2,i]*1e-3,linewidth=2,label=PD[0,-len(Sol_p)+i])
    
ax2[1].ticklabel_format(style='sci',scilimits=(0,0), axis='y')
ax2[1].set_xlabel('Fe fraction')
ax2[1].set_ylabel('G$_m$  [kJ/mol]')
ax2[1].set_title(f'Gm Fe-Si, T:{T}')
ax2[1].yaxis.set_label_position("right")
ax2[1].yaxis.tick_right()
ax2[1].set_xlim(0,1)
ax2[1].set_ylim(1.2*Map_liq_s['2'].min()*1e-3,Lig_phases_p[:,2,:].max()*1e-3+1)
ax2[1].invert_xaxis()
ax2[1].legend()

# Common exponent

# Gather ALL y values from both subplots before scaling by 1e-3
y_all = []

# Liquid + tangent + solids (B-rich)
y_all.append(Map_liq_s['2'].values * 1e-3)
y_all.append(Map_T_s['2'].values * 1e-3)
for i in range(len(Sol_p)):
    y_all.append(Lig_phases_p[:,2,i] * 1e-3)

# Liquid + tangent + solids (Si-rich)
# (Note: You recreate Map_liq_s and Map_T_s later, so repeat)
y_all.append(Map_liq_s['2'].values * 1e-3)
y_all.append(Map_T_s['2'].values * 1e-3)
for i in range(len(Sol_p)):
    y_all.append(Lig_phases_p[:,2,i] * 1e-3)

# Flatten to 1 array
y_all = np.concatenate(y_all)

# Compute shared exponent
exp = int(np.floor(np.log10(np.max(np.abs(y_all))))) -1


def shared_fmt(y, pos):
    # Divide by 10**exp so both axes show same scale
    return f"{y / 10**exp:.0f}"

formatter = mtick.FuncFormatter(shared_fmt)


for ax in ax2:      # ax2 is your array of 2 axes
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel(rf"G$_m$ (10$^{exp}$ kJ/mol)")


##############################

Lig_phases=np.zeros([Map_T.shape[0],Map_T.shape[1],len(Sol_p)])
Lig_T_phases=np.zeros([Map_T.shape[0],Map_T.shape[1],len(Sol_p)])
Sol_idx=0

# plt.figure()
#------------------Line compunds (A,B)_m C_n-----------------
for p in range(phases):
    m = PD[2,p]
    n = PD[3,p]
    o = PD[4,p]
    
    if PD[-1,p] == 'Stoic':     #Stoichiometric or not
        SolidFeSiB(T,PD[0,p])                              #Update phase Gibbs(T) Polynomials
    
        xp[p,:]=[m/(m+n+o), n/(m+n+o), o/(m+n+o)]   #particle composition Fe, B             
        T_Liq_p[p,:]=Mu[0]*((1-X)*(1-xp[p,1])) + Mu[1]*xp[p,1] + Mu[2]*(X*(1-xp[p,1]))
        
        Tp[p,0]=StoicCompounds(T,PD[:,p])

        T_Liq=np.dot(Mu,xp[p,:])
        dc[p] = T_Liq - Tp[p,0]
        
        yB[p] = None
        MuAC_p[p]= None
        MuBC_p[p]= None

        MuAC_liq[p]=Mu[0]*(1-xp[p,1]) + Mu[1]*xp[p,1] + Mu[2]*0
        MuBC_liq[p]=Mu[0]*0 + Mu[1]*xp[p,1] + Mu[2]*(1-xp[p,1])
        
        mx[:,0,p]=xp[p,1]
        mx[:,1,p]=X*(1-xp[p,1])

    ########################################################################
    ########################################################################
    ########################################################################
    elif PD[-1,p]== 'Sol':
        SolidSolFeSiB(T,PD[0,p])
        
        Map_liq_p=pd.DataFrame()
            
        for k in range(len(xB)):    #slice surface and stack each line in a dataframe with corresponding function value
            xC=np.linspace(0,1-xB[k],len(xB)-k)
            xA=(1-xB[k])-xC
            
            Gm=Gibbs_Func_surf(T,xA,np.array([xB[k]]),xC,PD[0,p])   
            Map2=pd.DataFrame({'0':xB[k]*np.ones(len(xC)),'1':xC,'2':Gm})
            Map_liq_p=pd.concat([Map_liq_p,Map2],ignore_index=True) 
            
        
        Lig_phases[:,:,Sol_idx] = Map_liq_p
        Sol_idx+=1
    
    [dc[p],xp[p,:]]=DcXp(T,xp_force,xm,TangCoeff,Mu,isStoich,PD[:,p])
        
print('------------------ dc ----------------')
for p in range(phases):
    print(f" {PD[0,p]} {dc[p]:1.1e} ")   
#%%   
#---------------------Plot Energies------------------
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.ticklabel_format(style='sci',scilimits=(0,0), axis='z')
# Full 3DGibbs
ax.plot_trisurf(Map_liq['0'], Map_liq['1'], Map_liq['2']*1e-3,color="silver") 
# Tangent plane
ax.plot_trisurf(Map_T['0'], Map_T['1'], Map_T['2']*1e-3,color="grey") 

ax.scatter(xm[1],xm[2],Gibbs_Func_surf(T,np.array([xm[0]]),np.array([xm[1]]),np.array([xm[2]]),'Liquid')*1e-3,color='black',label='Comp')
#%%
for p in range(phases):
    m = PD[2,p]
    n = PD[3,p]
    o = PD[4,p]
    
    if PD[-1,p] == 'Stoic':     #Stoichiometric or not
        ax.scatter(xp[p,1],xp[p,2],Tp[p,0]*1e-3,"*",label=plotLabel(PD[0,p]))

ax.set_xlabel('xSi [mol]')
ax.set_ylabel('xB [mol]')
ax.set_zlabel('G$_m$ [kJ/mol]')
ax.set_title(f'Fe({xm[0]:1.3f})Si({xm[1]})B({xm[2]}), T={T} [K]')
ax.legend(bbox_to_anchor=(0.05, 1.02))
ax.view_init(15, 160)
# ax.view_init(15, 160)
plt.show()

#---------------------Plot Energies------------------
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.ticklabel_format(style='sci',scilimits=(0,0), axis='z')
# Full 3DGibbs
ax.plot_trisurf(Map_liq['0'], Map_liq['1'], Map_liq['2']*1e-3,color="silver",label="Liq") 
# Tangent plane
ax.plot_trisurf(Map_T['0'], Map_T['1'], Map_T['2']*1e-3,color="grey",label="Liq_T") 
# chemical potentials
ax.scatter([0,1,0],[0,0,1],[Mu[0]*1e-3,Mu[1]*1e-3,Mu[2]*1e-3],color='k')


PP = -2
SolidSolFeSiB(T,PD[0,PP])
dc_Gibbs_p = Gibbs_Func_surf(T,np.array([xp[PP,0]]),np.array([xp[PP,1]]),np.array([xp[PP,2]]),PD[0,PP])
dc_Gibbs_l = np.dot(xp[PP,:],Mu)

ax.scatter(xp[PP,1],xp[PP,2], dc_Gibbs_p * 1e-3,color='r')
ax.scatter(xp[PP,1],xp[PP,2], dc_Gibbs_l * 1e-3,color='c')

# SolPhases
CC = ["m","c","b"]
for pp in range(len(Sol_p)):
    # Full 3DGibbs
    ax.plot_trisurf(Lig_phases[:,0,pp], Lig_phases[:,1,pp], Lig_phases[:,2,pp]*1e-3,color = CC[pp], label=PD[0,PD[-1,:]=='Sol'][pp]) 
    # Tangent plane
    # ax.plot_trisurf(Lig_T_phases[:,0,p], Lig_T_phases[:,1,p], Lig_T_phases[:,2,p]*1e-3)

ax.set_xlabel('xSi [mol]')
ax.set_ylabel('xB [mol]')
ax.set_zlabel('G$_m$ [kJ/mol]')
ax.set_title(''.join(['Fe-Si-B, T=',str(T),' [K]']))
ax.view_init(15, 160)
# ax.view_init(15, 270)
ax.legend(bbox_to_anchor=(1.35, 1.02))

#----------------Ternplot of Tg------------------

import matplotlib.tri as tri

def ternary_to_cartesian(xA, xB, xC):
    # Equilateral triangle mapping
    x = 0.5 * (2*xB + xC) / (xA + xB + xC)
    y = (np.sqrt(3)/2) * xC / (xA + xB + xC)
    return x, y

x, y = ternary_to_cartesian(Map_Tg['xA'], Map_Tg['xB'], Map_Tg['xC'])

triang = tri.Triangulation(x, y)

fig5, ax5 = plt.subplots(1,2,figsize=(12, 5))

tpc = ax5[0].tripcolor(
    triang,
    Map_Tg['Tg'],       # Color values at each point
    shading='flat',
    vmin=Map_Tg['Tg'].min(),
    vmax=Map_Tg['Tg'].max(),
    cmap='inferno'
)

fig5.colorbar(tpc, ax=ax5[0], label='Tg')
ax5[0].set_aspect('equal')

tpc = ax5[1].tripcolor(
    triang,
    Map_Tm['Tm'],       # Color values at each point
    shading='flat',
    vmin=Map_Tm['Tm'].min(),
    vmax=Map_Tm['Tm'].max(),
    cmap='inferno'
)

fig5.colorbar(tpc, ax=ax5[1], label='Tm')
ax5[1].set_aspect('equal')


####################### Tg and Tm with mpltern

# import mpltern

# fig6, axes = plt.subplots(1, 2, figsize=(12, 5),
#                           subplot_kw={'projection': 'ternary'})


# # Define mask for the region limits
# mask = (
#     (Map_Tg['xA'] >= 0.7) & (Map_Tg['xA'] <= 1) &  # Fe range
#     (Map_Tg['xB'] >= 0.01) & (Map_Tg['xB'] <= 0.3) & # Si range
#     (Map_Tg['xC'] >= 0.01) & (Map_Tg['xC'] <= 0.3)   # B range
# )

# # # Apply mask to Tg values
# vmin = Map_Tg.loc[mask, 'Tg'].min()
# vmax = Map_Tg.loc[mask, 'Tg'].max()

# # Then use these in tripcolor
# ax1 = axes[0]
# # First plot: Tg
# cs1 = ax1.tripcolor(
#     Map_Tg['xA'], Map_Tg['xB'], Map_Tg['xC'],
#     Map_Tg['Tg'],
#     shading='flat',
#     vmin=vmin,
#     vmax=vmax,
#     cmap='plasma'
# )

# ax1.set_tlabel('Fe')
# ax1.set_llabel('Si')
# ax1.set_rlabel('B')
# ax1.set_tlim(0.7, 1)
# ax1.set_llim(0.0, 0.3)
# ax1.set_rlim(0.0, 0.3)

# # Inset colorbar for ax1
# cax1 = ax1.inset_axes([1.07, 0.1, 0.05, 0.9], transform=ax1.transAxes)
# colorbar1 = fig6.colorbar(cs1, cax=cax1)
# colorbar1.set_label('$T_g$ (K)', rotation=270, va='baseline')


# mask_tm = (
#     (Map_Tm['xA'] >= 0.7) & (Map_Tm['xA'] <= 1) &
#     (Map_Tm['xB'] >= 0.01) & (Map_Tm['xB'] <= 0.3) &
#     (Map_Tm['xC'] >= 0.01) & (Map_Tm['xC'] <= 0.3)
# )

# vmin_tm = Map_Tm.loc[mask_tm, 'Tm'].min()
# vmax_tm = Map_Tm.loc[mask_tm, 'Tm'].max()

# # Second plot: Tm
# ax2 = axes[1]

# cs2 = ax2.tripcolor(
#     Map_Tm['xA'], Map_Tm['xB'], Map_Tm['xC'],
#     Map_Tm['Tm'],
#     shading='flat',
#     vmin=vmin_tm,
#     vmax=vmax_tm,
#     cmap='plasma'
# )
# ax2.set_tlabel('Fe')
# ax2.set_llabel('Si')
# ax2.set_rlabel('B')
# ax2.set_tlim(0.7, 1)
# ax2.set_llim(0.0, 0.3)
# ax2.set_rlim(0.0, 0.3)

# # Inset colorbar for ax2
# cax2 = ax2.inset_axes([1.07, 0.1, 0.05, 0.9], transform=ax2.transAxes)
# colorbar2 = fig6.colorbar(cs2, cax=cax2)
# colorbar2.set_label('$T_m$ (K)', rotation=270, va='baseline')

# plt.subplots_adjust(wspace=0.5, hspace=0.5)

############# Gibbs liquid with mpltern

# import mpltern
# import matplotlib.colors as mcolors

# fig22 = plt.figure(figsize=(7, 4))

# v = Map_liq['2']#*1e-3
# t = Map_Tg['xA']
# l = Map_Tg['xB']
# r = Map_Tg['xC']

# ax = fig22.add_subplot(1, 1, 1, projection='ternary')
# # norm = mcolors.LogNorm(vmin=v.min(), vmax=v.max())
# # cs = ax.tripcolor(t, l, r, v, shading='flat', norm=norm, cmap='plasma')
# cs = ax.tripcolor(t, l, r, v, shading='flat', cmap='plasma')

# ax.set_tlabel('Fe')
# ax.set_llabel('Si')
# ax.set_rlabel('B')

#######################
plt.show()

FigpathOut=''.join([OutDir,"Esurf_stoich.png"])
fig1.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([OutDir,"Esurf_sol.png"])
fig4.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([OutDir,"Slice.png"])
fig2.savefig(FigpathOut, bbox_inches = "tight")
FigpathOut=''.join([OutDir,"Slice.svg"])
fig2.savefig(FigpathOut, bbox_inches = "tight")

# FigpathOut=''.join([OutDir,"Tg.png"])
# fig5.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([OutDir,"TgTm_mpltern.svg"])
# fig5.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([OutDir,"TgTm_mpltern.png"])
# fig6.savefig(FigpathOut, bbox_inches = "tight")
# FigpathOut=''.join([OutDir,"Liq.svg"])
# fig22.savefig(FigpathOut, bbox_inches = "tight")
