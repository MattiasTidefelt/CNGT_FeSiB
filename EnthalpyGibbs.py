#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:24:13 2022

@author: ag0406
"""

def EnthalpyGibbs(T,xFe,xNb,xB,yFe,yNb,PD):
    import numpy as np
    from LiqFeNbB import Gibbs_Liq
    from LiqFeNbB_dT import Gibbs_Liq_dT    
    from SolidFeNbB import Gibbs_Solid    
    from SolidFeNbB_dT import Gibbs_Solid_dT    
    
    [G_Liq_B_0,G_Liq_Fe_0,G_Liq_Nb_0]=Gibbs_Liq[0]
    [G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_Liq[1]
    [G_Liq_NbB_0,G_Liq_NbB_1,G_Liq_NbB_2]=Gibbs_Liq[2]
    [G_Liq_FeNb_0,G_Liq_FeNb_1,G_Liq_FeNb_2]=Gibbs_Liq[3]
    [G_Liq_FeNbB_0]=Gibbs_Liq[4]
    
    [dG_Liq_B_0,dG_Liq_Fe_0,dG_Liq_Nb_0]=Gibbs_Liq_dT[0]
    [dG_Liq_FeB_0,dG_Liq_FeB_1,dG_Liq_FeB_2]=Gibbs_Liq_dT[1]
    [dG_Liq_NbB_0,dG_Liq_NbB_1,dG_Liq_NbB_2]=Gibbs_Liq_dT[2]
    [dG_Liq_FeNb_0,dG_Liq_FeNb_1,dG_Liq_FeNb_2]=Gibbs_Liq_dT[3]
    [dG_Liq_FeNbB_0]=Gibbs_Liq_dT[4]
    
 
    [G_Bin1,G_Bin2] = Gibbs_Solid[0]
    [G_Tern1,G_Tern2] = Gibbs_Solid[1] 
 
    [dG_Bin1, dG_Bin2]=Gibbs_Solid_dT[0]
    [dG_Tern1, dG_Tern2]=Gibbs_Solid_dT[1]    
    
    R=8.31446261815324
    
    #--------------Gibbs Matrix FeNbB-------------
    # Gref=xB*GHSER_Cu + xNb*GHSER_Zr + xFe*GHSER_Al
    Gref=xB*G_Liq_B_0 + xNb*G_Liq_Nb_0 + xFe*G_Liq_Fe_0
    
    ## ONLY SURFACE PLOT
    if len(xB)==1:
        xB=xB*np.ones(max([len(xNb),len(xFe)]))
    elif len(xNb)==1:
        xNb=xNb*np.ones(max([len(xB),len(xFe)]))
    elif len(xFe)==1:
        xFe=xFe*np.ones(max([len(xNb),len(xB)]))
        
    Gide=np.zeros(max([len(xB),len(xNb),len(xFe)]))   
    for i in range(len(Gide)):
        if xB[i]==0 and xNb[i]>0 and xFe[i]>0: 
            Gide[i]=R*T*(xNb[i]*np.log(xNb[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xNb[i]==0 and xFe[i]>0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xNb[i]>0 and xFe[i]==0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xNb[i]*np.log(xNb[i]))
        elif xB[i]==0 and xNb[i]==0 and xFe[i]>0: 
            Gide[i]=R*T*(xFe[i]*np.log(xFe[i]))
        elif xB[i]==0 and xNb[i]>0 and xFe[i]==0: 
            Gide[i]=R*T*(xNb[i]*np.log(xNb[i]))
        elif xB[i]>0 and xNb[i]==0 and xFe[i]==0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]))
        else:
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xNb[i]*np.log(xNb[i]) + xFe[i]*np.log(xFe[i]))
    
    GexBin=(xFe*xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xFe-xB)**1 + G_Liq_FeB_2*(xFe-xB)**2 )
         + xFe*xNb*(G_Liq_FeNb_0 + G_Liq_FeNb_1*(xFe-xNb)**1 + G_Liq_FeNb_2*(xFe-xNb)**2)
         + xB*xNb*(G_Liq_NbB_0 + G_Liq_NbB_1*(xNb-xB)**1 + G_Liq_NbB_2*(xNb-xB)**2))
    Gex=GexBin + xFe*xB*xNb*((xFe+0)*G_Liq_FeNbB_0)
    Gm=Gref+Gide+Gex
    
    #--------------Gibbs Matrix FeNbB-------------dT
    dGref=xB*dG_Liq_B_0 + xNb*dG_Liq_Nb_0 + xFe*dG_Liq_Fe_0
    
    ## ONLY SURFACE PLOT
    if len(xB)==1:
        xB=xB*np.ones(max([len(xNb),len(xFe)]))
    elif len(xNb)==1:
        xNb=xNb*np.ones(max([len(xB),len(xFe)]))
    elif len(xFe)==1:
        xFe=xFe*np.ones(max([len(xNb),len(xB)]))
        
    dGide=np.zeros(max([len(xB),len(xNb),len(xFe)]))   
    for i in range(len(Gide)):
        if xB[i]==0 and xNb[i]>0 and xFe[i]>0: 
            dGide[i]=R*T*(xNb[i]*np.log(xNb[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xNb[i]==0 and xFe[i]>0: 
            dGide[i]=R*T*(xB[i]*np.log(xB[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xNb[i]>0 and xFe[i]==0: 
            dGide[i]=R*T*(xB[i]*np.log(xB[i]) + xNb[i]*np.log(xNb[i]))
        elif xB[i]==0 and xNb[i]==0 and xFe[i]>0: 
            dGide[i]=R*T*(xFe[i]*np.log(xFe[i]))
        elif xB[i]==0 and xNb[i]>0 and xFe[i]==0: 
            dGide[i]=R*T*(xNb[i]*np.log(xNb[i]))
        elif xB[i]>0 and xNb[i]==0 and xFe[i]==0: 
            dGide[i]=R*T*(xB[i]*np.log(xB[i]))
        else:
            # dGide[i]=R*T*(xB[i]*np.log(xB[i]) + xNb[i]*np.log(xNb[i]) + xFe[i]*np.log(xFe[i]))
            dGide[i]=R*(xB[i]*np.log(xB[i]) + xNb[i]*np.log(xNb[i]) + xFe[i]*np.log(xFe[i]))
    
    dGexBin=(xFe*xB*(dG_Liq_FeB_0 + dG_Liq_FeB_1*(xFe-xB)**1 + dG_Liq_FeB_2*(xFe-xB)**2 )
         + xFe*xNb*(dG_Liq_FeNb_0 + dG_Liq_FeNb_1*(xFe-xNb)**1 + dG_Liq_FeNb_2*(xFe-xNb)**2)
         + xB*xNb*(dG_Liq_NbB_0 + dG_Liq_NbB_1*(xNb-xB)**1 + dG_Liq_NbB_2*(xNb-xB)**2))
    dGex=dGexBin + xFe*xB*xNb*((xFe+0)*G_Liq_FeNbB_0)
    dGm=dGref+dGide+dGex
    
    #----------------
    Gm_mix = Gex + Gide;
    Hl_mix = Gex - T*dGex;
    Hl=Gm - T*dGm
    # #####################     PARTICLE     ##################### 
    ##################### ##################### ##################### 
    ##################### ##################### ##################### 
    

    
    # if len(yNb)==1:
    #     yNb=yNb*np.ones(len(yFe))
    # elif len(yFe)==1:
    #     yFe=yFe*np.ones(len(yNb))
          
    # Gide=np.zeros(max([len(yNb),len(yFe)])) 
    # dGide=np.zeros(max([len(yNb),len(yFe)])) 
    #----------Gibbs particle ------------          
        
    m = PD[3]
    n = PD[4]
    o = PD[5]
    
    # for i in range(len(Gide)):
    #     if yNb[i]==0 and yFe[i]>0: 
    #         Gide[i]=n*R*T*(yFe[i]*np.log(yFe[i]))
    #     elif yNb[i]>0 and yFe[i]==0: 
    #         Gide[i]=n*R*T*(yNb[i]*np.log(yNb[i]))
    #     else:
    #         Gide[i]=n*R*T*(yFe[i]*np.log(yFe[i]) + yNb[i]*np.log(yNb[i]))
    
    Gide=n*R*T*(yFe*np.log(yFe) + yNb*np.log(yNb))
    
    Gref=yFe*G_Bin1 + yNb*G_Bin2 
    Gex=yFe*yNb*(G_Tern1*(yFe-yNb)**0 + G_Tern2*(yFe-yNb)**1)
    Gp=(Gref + Gide + Gex)#/(m+n+o)         
    #----------------------------------------------dT  
    # for i in range(len(Gide)):
    #     if yNb[i]==0 and yFe[i]>0: 
    #         # dGide[i]=m*R*T*(yFe[i]*np.log(yFe[i]))
    #         dGide[i]=n*R*(yFe[i]*np.log(yFe[i]))
    #     elif yNb[i]>0 and yFe[i]==0: 
    #         # dGide[i]=m*R*T*(yNb[i]*np.log(yNb[i]))
    #         dGide[i]=n*R*(yNb[i]*np.log(yNb[i]))
    #     else:
    #         # dGide[i]=n*R*T*(yFe[i]*np.log(yFe[i]) + yNb[i]*np.log(yNb[i]))
    #         dGide[i]=n*R*(yFe[i]*np.log(yFe[i]) + yNb[i]*np.log(yNb[i]))
    dGide=n*R*(yFe*np.log(yFe) + yNb*np.log(yNb))
            
    dGref=yFe*dG_Bin1 + yNb*dG_Bin2 
    dGex=yFe*yNb*(dG_Tern1*(yFe-yNb)**0 + dG_Tern2*(yFe-yNb)**1)
    dGp=(dGref+dGide+dGex)#/(m+n+o)  
       
    #----------------    
    Gp_mix = (Gex + Gide)#/(m+n+o) 
    Hp_mix = (Gex - T*dGex)#/(m+n+o) #Delta S=-dG/dT
    # Hp = Gp - T*dGp
    Hp = (Gp - T*dGp)/(m+n+o)
    
    return [Hl,Gm_mix,Hl_mix,Hp,Gp_mix,Hp_mix]
 
