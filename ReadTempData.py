#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:00:59 2025

@author: ag0406

t-T data from central node in AM simulation of slab with FEM
"""

from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import root
        
cs_T = None

def initialize_Tspline(Tg,Tm, maxHR):
    
    fname = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/GEPRE_Testplate3_fine_readOut.txt"    
        
    delimiters = [' ', ',', '\t']  

    for delimiter in delimiters:
        try:
            file = np.loadtxt(open(fname,"r"),delimiter=delimiter, comments='#')
            break
        
        except ValueError:
            continue
        
    t = file[:,0]          # Initially in seconds
    T = file[:,1]+273.15   # C to K
    
    redIdx = np.where(T>400)
    t_red = t[redIdx]
    t_red -= t_red[0]
    T_red = T[redIdx]
     
    sol = root(
        lambda t_scale: tscaleFunc(t_scale, t_red, T_red, maxHR),
        x0=[1e-5],                 
    )
    t_Scale = sol.x
    print("-------------------------------")
    print(f"Scaling t by {t_Scale[0]:1.2e}")
    print("-------------------------------")  
    
    ### get times and temperatures at Tg and Tm
    global cs_T
    cs_T = interp1d(t_red*t_Scale,T_red, kind='linear', fill_value='extrapolate')     
    
    t_red_fine = np.linspace(t_red[0]*t_Scale, t_red[-1]*t_Scale ,1000000)
    T_red_fine = cs_T(t_red_fine)
    
    Tmt = []
    TmT = []
    Tgt = []
    TgT = []
    Tm_flag = False
    Tg_flag = True
    for i in range(len(t_red_fine)):
        Tsearch = T_red_fine[i]
        if Tsearch > Tm and not Tm_flag:
            TmT.append(Tsearch)
            Tmt.append(t_red_fine[i])
            Tm_flag = True
        elif Tsearch < Tm and Tm_flag:
            TmT.append(Tsearch)
            Tmt.append(t_red_fine[i])
            Tm_flag = False
            
        if Tsearch > Tg and Tg_flag:
            TgT.append(Tsearch)
            Tgt.append(t_red_fine[i])
            Tg_flag = False
        elif Tsearch < Tg and not Tg_flag:
            TgT.append(Tsearch)
            Tgt.append(t_red_fine[i])
            Tg_flag = True
            
    Tmtm = [np.array(Tmt),np.array(TmT)]   
    Tgtg = [np.array(Tgt),np.array(TgT)]
    
    # Tmtm = 0
    # Tgtg = 0
    ################
    
    redIdx = np.where(T>Tg-10)[0]
    t_red = t[redIdx[0]:]
    t_red -= t_red[0]
    T_red = T[redIdx[0]:]  
    
    t_red *= t_Scale        #modifying t according to the required limiting heating rate 
            
    # global cs_T
    # cs_T = interp1d(t_red,T_red, kind='linear', fill_value='extrapolate')  
    cs_T = interp1d(t_red,T_red, kind='linear', fill_value='extrapolate') 
    
    return t_red[-1]*0.55, Tmtm, Tgtg

def initialize_Tspline_AML(Tg,Tm, maxHR):
    
    fname = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/GEPRE_Testplate3_fine_readOut.txt"    
        
    delimiters = [' ', ',', '\t']  

    for delimiter in delimiters:
        try:
            file = np.loadtxt(open(fname,"r"),delimiter=delimiter, comments='#')
            break
        
        except ValueError:
            continue
        
    t = file[:,0]          # Initially in seconds
    T = file[:,1]+273.15   # C to K
    
    redIdx = np.where(T>400)
    t_red = t[redIdx]
    t_red -= t_red[0]
    T_red = T[redIdx]
     
    sol = root(
        lambda t_scale: tscaleFunc(t_scale, t_red, T_red, maxHR),
        x0=[1e-5],                 
    )
    t_Scale = sol.x
    # print("-------------------------------")
    # print(f"Scaling t by {t_Scale[0]:1.2e}")
    # print("-------------------------------")  
    
    redIdx = np.where(T>Tg-10)[0]
    t_red = t[redIdx[0]:]
    t_red -= t_red[0]
    T_red = T[redIdx[0]:]  
    
    t_red *= t_Scale        #modifying t according to the required limiting heating rate 
            
    return t_red[-1]*0.55
    
def tscaleFunc(t_scale,t_red,T_red,maxHR):
    maxIdx = np.where(T_red == max(T_red))[0][0]
    # dTdt_max = np.max(T_red[1:]-T_red[0:-1])/(t_red[1:]*t_scale - t_red[0:-1]*t_scale)[0]
    dTdt_max = (T_red[maxIdx]-T_red[0])/(t_red[maxIdx]*t_scale - t_red[0]*t_scale)
    # print(f"dTdt_max {dTdt_max}, t_scale {t_scale}")
    return maxHR - dTdt_max   

def Tempfunc(t):
    global cs_T
    if cs_T is None:
        raise ValueError("Spline not initialized. Call initialize_spline first.")
    return cs_T(t)


# import matplotlib.pyplot as plt
# from os import makedirs
# from os.path import exists

# fname = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/GEPRE_Testplate3_fine_readOut.txt"    
    
# delimiters = [' ', ',', '\t']  

# for delimiter in delimiters:
#     try:
#         file = np.loadtxt(open(fname,"r"),delimiter=delimiter, comments='#')
#         break
    
#     except ValueError:
#         continue
    
# t = file[:,0]     # modifying time to milisec (does not make sense otherwise)
# T = file[:,1]+273.15   # C to K
          
# print(f'File loaded as:\n {fname}')  

# redIdx = np.where(T>400)
# t_red = t[redIdx]
# t_red -= t_red[0]
# T_red = T[redIdx]

# fig, ax = plt.subplots(1,3, figsize = (14,5))
# fig.suptitle('Node temp from AM by FEM')
# ax[0].plot(t_red,T_red)
# ax[0].set_xlabel('t (s)')
# ax[0].set_ylabel('T (K)')
# ax[0].set_title('Raw')

# Tg = 725
# Tm = 1200
# # maxHR = 1.6e8                                           # Scale T(t) to get ~heating rate (dTdt)
# # maxHR = 1.58489319e+08
# maxHR = 1.7e8 

# t_end, Tmtm, Tgtg = initialize_Tspline(Tg,Tm,maxHR) 

# Tgtg_idx = np.where(Tgtg[0] <= t_end/10)[0]
# Tmtm_idx = np.where(Tmtm[0] <= t_end/10)[0]
# t = np.linspace(0,t_end/10,5000)
# T = np.zeros(len(t))
# for i in range(len(t)):
#     T[i] = Tempfunc(t[i])
    
# ax[1].plot(t,T)
# ax[1].scatter(Tmtm[0][Tmtm_idx],Tmtm[1][Tmtm_idx])
# ax[1].scatter(Tgtg[0][Tgtg_idx],Tgtg[1][Tgtg_idx])
# ax[1].set_xlabel('t (s)')
# ax[1].set_title('Interpolated section')

# ax[2].plot(t[1:],(T[1:]-T[0:-1])/(t[1:]-t[0:-1]))
# ax[2].set_xlabel('t (s)')
# ax[2].set_title('Heating rate (dTdt)')
# ax[2].yaxis.set_label_position("right")
# ax[2].yaxis.tick_right()

# DirPathOut = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/AM_Simulation/"
# if not exists(DirPathOut):
#     makedirs(DirPathOut)
# FigpathOut=''.join([DirPathOut,"AM_TempsTimes.png"])
# fig.savefig(FigpathOut, bbox_inches = "tight")
