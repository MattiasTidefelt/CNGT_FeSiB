#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:12:39 2024

@author: ag0406
"""

def GetSigma0(T,xp,Hf,Sf,Vm,name,SigType):
    NA=6.02214086e23
    R=8.31446261815324
    
    #Calculated Vm for pure elem according to Rho_Vm script 
    # Fe = 7.09232918465837e-06     #bcc
    # Si = 1.2053862660944204e-05   #diamond
    # B = 4.620085470085471e-06 (boron 105) #rhomb/hex
    # B = 4.358870967741936e-06 (boron 12)
    
    #Enthalpy and entropy of fusion for pure elements according to thermal assessment
    # Hf: [[1.32465057e+04 5.02085012e+04 5.01994782e+04]
    # Sf: [7.47828636e+00 2.97685891e+01 2.13845280e+01]]
    
    #Sigma0 for pure elements Sigtype = 0, alpha = 0.49
    # Fe = 0.2082193969436184
    # Si = 0.5541647214126797
    # B = 1.050054014777807
    
    #Sigma0 for pure elements Sigtype = 1 (T = 800 K, ~Tg), alpha = 0.49
    # Fe = 0.15112962391677806
    # Si = 0.40850792802002833
    # B = 0.703952448733658

    # Vm_Fe = 7.09232918465837e-06
    # Vm_Si = 1.2053862660944204e-05
    # Vm_B = 4.620085470085471e-06
    
    Vm_Fe = 7.00790*1e-6*(1 + (3.42756*1e-5 + 1.62801*1e-8*T -0.291672/T**2))
    Vm_Si = (11.99189 + (1.485*1e-4*T + 7*1e-9*T**2 +7.84/T -380/T**2))*1e-6
    Vm_B = 4.620085470085471e-06*(1 + ((2*0.98*1e-6 + 1.82*1e-6)/3*T))
        
    if SigType == 0:
        # alpha = 0.49 +-0.08
        alpha_Fe = 0.46  # 0.46
        # alpha_Fe = 0.49  # 0.46
        alpha_Si = 0.32     # 0.32
        alpha_B = 0.32  #0.32
        
        # Funkar inte, e.g.; 0.751_0.012_0.237: t:0.6212 T:1361.0 it:85717140 ftot:1.19e-14 HR:1.0e+03
        # alpha_Fe = 0.49
        # alpha_Si = 0.32
        # alpha_B = 0.32
        
        sigma0_Fe = alpha_Fe*Hf[0]/(Vm_Fe**(2/3)*NA**(1/3))
        sigma0_Si = alpha_Si*Hf[1]/(Vm_Si**(2/3)*NA**(1/3))
        sigma0_B = alpha_B*Hf[2]/(Vm_B**(2/3)*NA**(1/3))
        
    elif SigType == 1:
        #alpha = 0.49 +-0.08
        alpha_Fe = 0.41     # 0.55
        alpha_Si = 0.546     # 0.49
        alpha_B = 0.433  #0.41
        
        # alpha_Fe = 0.46  # 0.49
        # alpha_Si = 0.32     # 0.32
        # alpha_B = 0.32  #0.32
        
        Gf_Fe = Hf[0] + T*Sf[0]
        Gf_Si = Hf[1] + T*Sf[1]
        Gf_B = Hf[2] + T*Sf[2]
        
        sigma0_Fe = alpha_Fe*Gf_Fe/(2*Vm_Fe**(2/3)*NA**(1/3))
        sigma0_Si = alpha_Si*Gf_Si/(2*Vm_Si**(2/3)*NA**(1/3))
        sigma0_B = alpha_B*Gf_B/(2*Vm_B**(2/3)*NA**(1/3))
        
        
    elif SigType == 2:
        alpha_Fe = 0.76
        alpha_Si = 0.86
        alpha_B = 0.86
        
        Gf_Fe = T*Sf[0]
        Gf_Si = T*Sf[1]
        Gf_B = T*Sf[2]
        
        sigma0_Fe = alpha_Fe*Gf_Fe/(Vm_Fe**(2/3)*NA**(1/3))
        sigma0_Si = alpha_Si*Gf_Si/(Vm_Si**(2/3)*NA**(1/3))
        sigma0_B = alpha_B*Gf_B/(Vm_B**(2/3)*NA**(1/3))
        
    
    Sigma0 = xp[0]*sigma0_Fe + xp[1]*sigma0_Si + xp[2]*sigma0_B#*0.6
    if Sigma0 < 0:
        Sigma0 = 1e-4
    # print(f"{name}, sig: {Sigma0}, xFe: {xp[0]}")
    
    
    #############################    
    # alpha = 0.49 +-0.08
    # Sols = ["bcc","fcc","hcp"]
    # if name in Sols:
    #     alpha = 0.53
    # else:
    #     alpha = 0.45
        
    # # alpha = 0.49

    # if SigType == 0:
    #     Sigma0 = alpha*Hf/(Vm**(2/3)*NA**(1/3))
    # elif SigType == 1:
    #     Gf = Hf + T*Sf
    #     Sigma0 = alpha*Gf/(2*Vm**(2/3)*NA**(1/3))
    # elif SigType == 2:
    #     a = 0
    #     b = 0.925
    #     Sigma0 = (a*Hf +b*R*T)/(Vm**(2/3)*NA**(1/3))
        
        
    return Sigma0
