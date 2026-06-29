#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:29:10 2025

@author: ag0406
"""

from DcXp import DcXp
from LiqFeSiB import LiqFeSiB
from SolidSolFeSiB import SolidSolFeSiB
from SolidFeSiB import SolidFeSiB

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

import numpy as np
from scipy.optimize import root_scalar

def CheckDcAtMatrixComp(Vm,PD,Hf_Sf,xm):
    print('-------------------------------------------------------------')
    print('Checking if phase exhibit driving force at matrix composition')
    print('-------------------------------------------------------------')
    colIdx = []
    for p in range(PD.shape[1]):
        m = PD[2,p]
        n = PD[3,p]
        o = PD[4,p]
        
        x0=np.array([m/(m+n+o),n/(m+n+o),o/(m+n+o)])                            #Same matrix composition as particle 
        dx = 1e-6
        if x0[0] == 0:
            x0=np.array([m/(m+n+o)+2*dx,n/(m+n+o)-dx,o/(m+n+o)-dx])                         
        elif x0[1] == 0:
            x0=np.array([m/(m+n+o)-dx,n/(m+n+o)+2*dx,o/(m+n+o)-dx]) 
        elif x0[2] == 0:
            x0=np.array([m/(m+n+o)-dx,n/(m+n+o)-dx,o/(m+n+o)+2*dx]) 
            
        if PD[-1,p] == 'Sol':
            x0=np.array([1-2*dx,0+dx,0+dx]) 
        
        try:
            T = root_scalar(lambda T: TmFuncNew(T, x0, xm, PD[:, p]), bracket=[800, 2400], method='brentq').root            
        except:
            print(f'No driving force for {PD[0,p]} at Matrix composition in temperature interval, excluded from calculations!')
            colIdx.append(p)
            
    PD_red = np.delete(PD, colIdx, axis=1)
    Vm_red = np.delete(Vm, colIdx)
    # Hf_Sf_red = np.delete(Hf_Sf,colIdx, axis =1)
    Hf_Sf_red = Hf_Sf
    
    return Hf_Sf_red, PD_red, Vm_red

def TmFuncNew(T,x0,xm,PD):
    LiqFeSiB(T,'Liquid')
    TangCoeff = Gibbs_Tangent(T,xm[0],xm[1],xm[2],'Liquid',0)
    fp,_ = Gibbs_Func(T,xm[0],xm[1],xm[2],'Liquid') #Gibbs(p)
    Mu = fp + TangCoeff - np.dot(xm,TangCoeff)
    
    if PD[-1] == 'Sol':
        SolidSolFeSiB(T,PD[0])
    else:
         SolidFeSiB(T,PD[0])
    
    isStoich = False
    dc,_ = DcXp(float(T),x0,xm,TangCoeff,Mu,isStoich,PD)
    # if PD[-1] == 'Sol':
    #     print(f"{PD[0]}, T {T:1.2f}, dc: {dc:1.2e}")

    return dc
