#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:35:45 2025

@author: ag0406
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:12:39 2024

@author: ag0406
"""

from DcXp import DcXp
from LiqFeSiB import LiqFeSiB
from LiqFeSiB_dT import LiqFeSiB_dT
from SolidSolFeSiB import SolidSolFeSiB
from SolidSolFeSiB_dT import SolidSolFeSiB_dT
from SolidFeSiB import SolidFeSiB
from SolidFeSiB_dT import SolidFeSiB_dT

from Gibbs_Tangent import Gibbs_Tangent
from Gibbs_Func import Gibbs_Func

from Gibbs_Func import Gibbs_Func
from LiqPhase_dT import LiqPhase_dT

import numpy as np
from scipy.optimize import root_scalar

################################# Only use Hf and Sf for pure elements, then approximate sigma0 as a normalized linear combination
# Reference structures are Si: diamond (GHSER-Si), B: rhomb (GHSHER-B), Fe: bcc-a2 (SolidSol)
def Get_HandS(PD,xm):         
    Hf_Sf_pure = np.zeros([2,3])
    
    Elem = ['Fe','Si','B']
    for p in range(3):
        print(f'--------------------   {Elem[p]}   --------------------')        
        x0=np.array([1,1,1])*1e-6                            #Same matrix composition as particle 
        x0[p] += 1-2e-6
        
        try:
            T = root_scalar(lambda T: TmFuncNew(T, x0,xm, Elem[p] ,PD[:,-2]), bracket=[400, 2500], method='brentq').root
            
            print(f'Tm: {T:1.1f}')

            [Hl,Hp,dGm,dGp]=EnthalpyEntropy(T,x0[0],x0[1],x0[2],Elem[p] ,PD[:,-2])
            # print(f'Hl {Hl:1.3f}, Hp {Hp:1.3f}, dGm {dGm:1.3f}, dGp {dGp:1.3f}')
            
            Hf_Sf_pure[0,p] = abs(Hl-Hp)
            Hf_Sf_pure[1,p] = abs(dGm-dGp)     #dG/dT = Delta S
            print(f'H_fus: {Hf_Sf_pure[0,p]:1.3e}, S_fus: {Hf_Sf_pure[1,p]:1.3e}')
            PD_red = PD
            
        except:
            print(f'No driving force for {PD[0,p]} in temperature interval, excluded from calculations!')
    
    return Hf_Sf_pure, PD_red

def TmFuncNew(T,x0,xm,Elem,PD):
    LiqFeSiB(T,'Liquid')
    TangCoeff = Gibbs_Tangent(T,x0[0],x0[1],x0[2],'Liquid',0)
    fp,_ = Gibbs_Func(T,x0[0],x0[1],x0[2],'Liquid') #Gibbs(p)
    Mu = fp + TangCoeff - np.dot(x0,TangCoeff)
    
    PolT=np.array([1, T, T*np.log(T), T**2, T**3, T**(-1), T**7, T**(-9)]).reshape(8,1)
    
    if Elem == 'Fe':
        SolidSolFeSiB(T,PD[0])
        isStoich = True
        dc,_= DcXp(float(T),x0,xm,TangCoeff,Mu,isStoich,PD)
    elif Elem == 'Si':
        if T <= 1687:
            GHSER_Si = np.array([-8162.609, 137.227259, -22.8317533, -0.001912904, -3.552E-09, 176667,0,0])
        elif T>1687:
            GHSER_Si = np.array([-9457.642, 167.271767, -27.196, 0, 0, 0,0,-4.20369E+30])
            
        dc = np.dot(x0,Mu) - np.dot(GHSER_Si,PolT)[0]
        
    elif Elem == 'B':
        #------------ GHSER_i
        if T <= 1100:
            GHSER_B = np.array([-7735.284, 107.111864, -15.6641, -0.006864515, 6.18878e-7, 370843,0,0])
        elif T>1100 and T<= 2348:
            GHSER_B = np.array([-16649.474, 184.801744, -26.6047, -0.79809e-3, -0.02556e-6, 1748270,0,0])
        elif T>2348 and T<= 3000:
            GHSER_B = np.array([-36667.582, 231.336244, -31.5957527, -.00159488, +1.34719E-07, 11205883,0,0])
            
        dc = np.dot(x0,Mu) - np.dot(GHSER_B,PolT)[0]
    
    return dc

def EnthalpyEntropy(T,xFe,xSi,xB,Elem,PD):        

    LiqFeSiB(T,'Liquid')                    #Update Liquid Gibbs(T) Polynomials
    Gm,_ = Gibbs_Func(T,xFe,xSi,xB,'Liquid') 
    
    LiqFeSiB_dT(T,'Liquid')                 #Update Liquid S Polynomials        
    dGm = LiqPhase_dT(T,xFe,xSi,xB,'Liquid') 
    
    Hl = Gm - T*dGm
    
    #%%     PARTICLE 
    PolT=np.array([1, T, T*np.log(T), T**2, T**3, T**(-1), T**7, T**(-9)]).reshape(8,1)
    PoldT=np.array([0, 1, np.log(T)+1, 2*T, 3*T**2, -T**(-2), 7*T**6, -9*T**(-10)]).reshape(8,1)
    if Elem == 'Fe':
        SolidSolFeSiB(T,PD[0])                    #Update Liquid Gibbs(T) Polynomials
        Gp,eta = Gibbs_Func(T,xFe,xSi,xB,PD[0]) 
        
        SolidSolFeSiB_dT(T,PD[0])                 #Update Liquid S Polynomials
        dGp = LiqPhase_dT(T,xFe,xSi,xB,PD[0]) 
        
    elif Elem == 'Si':
        if T <= 1687:
            GHSER_Si = np.array([-8162.609, 137.227259, -22.8317533, -0.001912904, -3.552E-09, 176667,0,0])
        elif T>1687:
            GHSER_Si = np.array([-9457.642, 167.271767, -27.196, 0, 0, 0,0,-4.20369E+30])

        Gp = np.dot(GHSER_Si,PolT)[0]
        dGp = np.dot(GHSER_Si,PoldT)[0]
        
    elif Elem == 'B':
        #------------ GHSER_i
        if T <= 1100:
            GHSER_B = np.array([-7735.284, 107.111864, -15.6641, -0.006864515, 6.18878e-7, 370843,0,0])
        elif T>1100 and T<= 2348:
            GHSER_B = np.array([-16649.474, 184.801744, -26.6047, -0.79809e-3, -0.02556e-6, 1748270,0,0])
        elif T>2348 and T<= 3000:
            GHSER_B = np.array([-36667.582, 231.336244, -31.5957527, -.00159488, +1.34719E-07, 11205883,0,0])
        
        Gp = np.dot(GHSER_B,PolT)[0]
        dGp = np.dot(GHSER_B,PoldT)[0]

    Hp = Gp - T*dGp
    
    return [Hl,Hp,dGm,dGp]

#################################
#################################
################################# For BCC to FCC transformation

def Get_HandS_BF(PD,xp,SolActiveIdx,SolInActiveIdx):         
    Hf_Sf_pure = np.zeros([2,3])
        
    #only iron exhibit a temperature where fcc becomes more stable than bcc      
    # x0=np.array([1-2e-6,1e-6,1e-6])                          #Same matrix composition as particle 
    x0 = xp
    
    try:
        # T = root_scalar(lambda T: TmFuncNew_BF(T,x0,SolActiveIdx,SolInActiveIdx ,PD), bracket=[800, 1700], method='brentq').root
        
        TC_Fe= 1043
        ETCFESI = 63
            
        T = TC_Fe*x0[0] + x0[0]*x0[1]*8*ETCFESI*(x0[0]-x0[1])  
        # print(f'Tm: {T:1.1f}')
    
        [Hl,Hp,dGm,dGp]=EnthalpyEntropy_BF(T,x0[0],x0[1],x0[2],SolActiveIdx,SolInActiveIdx,PD)
        
        Hf_Sf_pure[0,:] = abs(Hl-Hp)
        Hf_Sf_pure[1,:] = abs(dGm-dGp)     #dG/dT = Delta S
        
    except:
        print('Failed BCC-FCC enthalpy/entropy')

    return Hf_Sf_pure

# def TmFuncNew_BF(T, x0,SolActiveIdx,SolInActiveIdx,PD):
        
#     SolidSolFeSiB(T,PD[0,SolInActiveIdx])                    #Update Liquid Gibbs(T) Polynomials
#     Gm = Gibbs_Func(T,x0[0],x0[1],x0[2],PD[0,SolInActiveIdx]) 

#     #%%     PARTICLE       
#     SolidSolFeSiB(T,PD[0,SolActiveIdx])                    #Update Liquid Gibbs(T) Polynomials
#     Gp = Gibbs_Func(T,x0[0],x0[1],x0[2],PD[0,SolActiveIdx]) 
    
#     dc = Gp - Gm

#     return dc

#################################
#################################
################################# 

def EnthalpyEntropy_BF(T,xFe,xSi,xB,SolActiveIdx,SolInActiveIdx,PD):    
        
    SolidSolFeSiB(T,PD[0,SolInActiveIdx])                    #Update Liquid Gibbs(T) Polynomials
    Gm,eta = Gibbs_Func(T,xFe,xSi,xB,PD[0,SolInActiveIdx]) 

    SolidSolFeSiB_dT(T,PD[0,SolInActiveIdx])                 #Update Liquid S Polynomials
    dGm = LiqPhase_dT(T,xFe,xSi,xB,PD[0,SolInActiveIdx]) 

    Hl = Gm - T*dGm

    #%%     PARTICLE       
    SolidSolFeSiB(T,PD[0,SolActiveIdx])                    #Update Liquid Gibbs(T) Polynomials
    Gp,eta = Gibbs_Func(T,xFe,xSi,xB,PD[0,SolActiveIdx]) 

    SolidSolFeSiB_dT(T,PD[0,SolActiveIdx])                 #Update Liquid S Polynomials
    dGp = LiqPhase_dT(T,xFe,xSi,xB,PD[0,SolActiveIdx]) 

    Hp = Gp - T*dGp

    return [Hl,Hp,dGm,dGp]

#################################
#################################
################################# treating all particles as unary metals

# def Get_HandS(PD,xm):     
#     phases=PD.shape[1]
    
#     Hf_Sf = np.zeros([2,phases])
    
#     for p in range(phases):
#         print(f'--------------------   {PD[0,p]}   --------------------')
#         m = PD[2,p]
#         n = PD[3,p]
#         o = PD[4,p]
        
#         x0=np.array([m/(m+n+o),n/(m+n+o),o/(m+n+o)])                            #Same matrix composition as particle 
#         dx = 1e-6
#         if x0[0] == 0:
#             x0=np.array([m/(m+n+o)+dx,n/(m+n+o)-dx/2,o/(m+n+o)-dx/2])                         
#         elif x0[1] == 0:
#             x0=np.array([m/(m+n+o)-dx/2,n/(m+n+o)+dx,o/(m+n+o)-dx/2]) 
#         elif x0[2] == 0:
#             x0=np.array([m/(m+n+o)-dx/2,n/(m+n+o)-dx/2,o/(m+n+o)+dx]) 
            
#         if PD[-1,p] == 'Sol':
#             x0=np.array([1-2*dx,0+dx,0+dx]) 
        
#         try:
#             T = root_scalar(lambda T: TmFuncNew(T, x0,xm, PD[:, p]), bracket=[400, 2500], method='brentq').root
            
#             print(f'Tm: {T:1.1f}')

#             [Hl,Hp,dGm,dGp]=EnthalpyEntropy(T,x0[0],x0[1],x0[2],PD[:,p])
#             # print(f'Hl {Hl:1.3f}, Hp {Hp:1.3f}, dGm {dGm:1.3f}, dGp {dGp:1.3f}')
            
#             Hf_Sf[0,p] = abs(Hl-Hp)
#             Hf_Sf[1,p] = abs(dGm-dGp)     #dG/dT = Delta S
#             print(f'H_fus: {Hf_Sf[0,p]:1.3e}, S_fus: {Hf_Sf[1,p]:1.3e}')
#             PD_red = PD
            
#         except:
#             print(f'No driving force for {PD[0,p]} in temperature interval, excluded from calculations!')
            
#         PD_red = np.delete(PD, np.where(Hf_Sf[0,:]==0)[0], axis=1)
#         Hf_Sf_red = np.delete(Hf_Sf,np.where(Hf_Sf[0,:]==0)[0], axis =1)
#     return Hf_Sf_red, PD_red

# def TmFuncNew(T,x0,xm,PD):
#     LiqFeSiB(T,'Liquid')
#     TangCoeff = Gibbs_Tangent(T,x0[0],x0[1],x0[2],'Liquid',0)
#     fp,_ = Gibbs_Func(T,x0[0],x0[1],x0[2],'Liquid') #Gibbs(p)
#     Mu = fp + TangCoeff - np.dot(x0,TangCoeff)
    
#     if PD[-1] == 'Sol':
#         SolidSolFeSiB(T,PD[0])
#     else:
#           SolidFeSiB(T,PD[0])
    
#     isStoich = True
#     dc,_= DcXp(float(T),x0,xm,TangCoeff,Mu,isStoich,PD)

#     return dc

# def EnthalpyEntropy(T,xFe,xSi,xB,PD):    
#     from StoicCompounds import StoicCompounds
#     from StoicCompounds_dT import StoicCompounds_dT
    
#     from Gibbs_Func import Gibbs_Func
#     from LiqPhase_dT import LiqPhase_dT

#     LiqFeSiB(T,'Liquid')                    #Update Liquid Gibbs(T) Polynomials
#     Gm,_ = Gibbs_Func(T,xFe,xSi,xB,'Liquid') 
    
#     LiqFeSiB_dT(T,'Liquid')                 #Update Liquid S Polynomials        
#     dGm = LiqPhase_dT(T,xFe,xSi,xB,'Liquid') 
    
#     Hl = Gm - T*dGm
    
#     #%%     PARTICLE    
#     if PD[-1] == 'Stoic':
#         SolidFeSiB(T,PD[0])          #Update phase Gibbs(T)dT Polynomials
#         Gp = StoicCompounds(T,PD)
        
#         SolidFeSiB_dT(T,PD[0])       #Update phase S Polynomials
#         dGp = StoicCompounds_dT(T,PD)   #entropy
        
#     elif PD[-1] == 'Sol':   
#         SolidSolFeSiB(T,PD[0])                    #Update Liquid Gibbs(T) Polynomials
#         Gp,eta = Gibbs_Func(T,xFe,xSi,xB,PD[0]) 
        
#         SolidSolFeSiB_dT(T,PD[0])                 #Update Liquid S Polynomials
#         dGp = LiqPhase_dT(T,xFe,xSi,xB,PD[0]) 

#     Hp = Gp - T*dGp
    
#     return [Hl,Hp,dGm,dGp]