#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:15:07 2022

@author: ag0406

Thermodynamic description: Yoshitomi et al. ISIJ international, Vol. 48 (2008)
G_0 and GHSER from SGTE_REF.tdb

Described as B-Fe-Si

Unclear in how to treat magnetic contribution, or transition contribution..
Lowest driving force is obtained by positive transformation energy
But if not negative magnetic contribution is used, fcc is stable at low temperatures instead of bcc
"""
import numpy as np

R=8.31446261815324

def GMO(TC,T,BMAGN,p):
    # Magnetic ordering contribution to Gibbs energy
    try:
        Tau = T/TC
        
        # A = 518/1125 + 11692/159575*(1/p-1)
        # if Tau <=1:
        #     f_Tau = 1-1/A*((79*Tau**-1/(140*p) + 474/497*(1/p-1))*(Tau**3/6 + Tau**9/135 +Tau**15/600))
        # elif Tau>1:
        #     f_Tau = -1/A*((Tau**-5)/10 + (Tau**-15)/315 + (Tau**-25)/1500)
        
        if Tau <=1:
            f_Tau = 1-0.905299383*Tau**-1 -0.153008346*Tau**3 - 0.00680037095*Tau**9 - 0.0015300846*Tau**15
        elif Tau>1:
            f_Tau = -0.0641731208*Tau**-5 -0.00203724193*Tau**-15 - (4.278208005e-4)*Tau**-25
        
        # A = 0.33471979 + 0.49649686*(1/p-1)
        # if Tau <=1:
        #     f_Tau = 1-1/A*((0.38438376*Tau**-1/p + 0.63570895*(1/p-1))*(Tau**3/6 + Tau**9/135 +Tau**15/600 +Tau**21/1617))
        # elif Tau>1:
        #     f_Tau = -1/A*((Tau**-7)/21 + (Tau**-21)/630 + (Tau**-35)/2975 + (Tau**-49)/8232)
    except:
        f_Tau = 0

    Gmo = R*T*f_Tau*np.log(BMAGN+1)
    
    return Gmo

def TransGMO(T,xFe,xSi,xB,PD):   
    
    if PD=='bcc': # 2 1   3 !
        TC_Fe= 1043
        ETCFESI = 63
            
        TC = TC_Fe*xFe + xFe*xSi*8*ETCFESI*(xFe-xSi)   # This removes the stability of Fe(Si)..
        BMAGN= 2.22*xFe
        
        # TC = TC_Fe                     
        # BMAGN= 2.22
        p = 0.4
        # n = 4.7041
        Gmo = GMO(TC,T,BMAGN,p)
        # Gmo = 0
        
    elif PD=='fcc':       # 2 1   1 !
        TC= 201/3 # -201 TN (antiferromagnetic Neel Temperature) AMF factor is -3
        BMAGN= 2.1/3*xFe#-2.1
        # BMAGN= 2.1/3#-2.1
        p = 0.28
        # n = 5.1665
        # Gmo = GMO(TC,T,BMAGN,p)
        Gmo = 0
        
    elif PD == 'hcp':
        Gmo =0
    
    # Gmo = 0
    return Gmo

def SolidSolFeSiB(T,PD):

    #GHSER data
    global Gibbs_SolidSol
    #------------ 
    

    PolT=np.array([1, T, T*np.log(T), T**2, T**3, T**(-1), T**7, T**(-9)]).reshape(8,1)
    
    #------------ GHSER_i
    if T <= 1100:
        GHSER_B = np.array([-7735.284, 107.111864, -15.6641, -0.006864515, 6.18878e-7, 370843,0,0])
    elif T>1100 and T<= 2348:
        GHSER_B = np.array([-16649.474, 184.801744, -26.6047, -0.79809e-3, -0.02556e-6, 1748270,0,0])
    elif T>2348 and T<= 3000:
        GHSER_B = np.array([-36667.582, 231.336244, -31.5957527, -.00159488, +1.34719E-07, 11205883,0,0])
        
    if T <= 1811:
        GHSER_Fe = np.array([1225.7, 124.134, -23.5143, -.00439752, -5.89269E-08, 77358,0,0])
    elif T > 1811 and T<= 6000:
        GHSER_Fe = np.array([-25383.581, 299.31255, -46, 0, 0, 0 ,0,2.2960305E+31])
        
    if T <= 1687:
        GHSER_Si = np.array([-8162.609, 137.227259, -22.8317533, -0.001912904, -3.552E-09, 176667,0,0])
    elif T>1687:
        GHSER_Si = np.array([-9457.642, 167.271767, -27.196, 0, 0, 0,0,-4.20369E+30])
    ###########################################################################
    ###########################################################################
    ###########################################################################

    if PD=='bcc': # 2 1   3 !
    # ONSTITUENT BCC_A2  :B,FE%,SI : VA% :  !        
        FESIW1 = 1260*R
        L1 = - 11544
        L2 = 3890
        L0BCC = np.dot(np.array([-27809, 11.62, 0, 0, 0, 0, 0, 0]),PolT)[0]
        #------------ B
        if T<=6000:
            G_B_0 = np.array([43514, -12.217, 0, 0, 0, 0,0,0]) + GHSER_B 
            
        #------------ Fe    
        if T<=6000:
            G_Fe_0 = GHSER_Fe 
            
        #------------ Si
        if T<=3600:
            G_Si_0 = np.array([47000, -22.5, 0, 0, 0, 0,0,0]) + GHSER_Si # ISIJ = SGTE
        
        #FeB
        G_FeB_0= np.array([-3.30923011E+04, 1.56047893E+01, 0, 0, 0, 0, 0, 0])
        G_FeB_1= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_FeB_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        #FeSi
        # G_FeSi_0= 4*L0BCC -4*FESIW1 # L0BCC
        G_FeSi_0= np.array([4*L0BCC -4*FESIW1, 0, 0, 0, 0, 0, 0, 0]) # L0BCC
        G_FeSi_1= np.array([8*L1, 0, 0, 0, 0, 0, 0, 0]) # L1BCC
        G_FeSi_2= np.array([16*L2, 0, 0, 0, 0, 0, 0, 0]) # L2BCC
        G_FeSi_3= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        #FeSi
        G_BSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        #BFeSi    (L0)
        G_BFeSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])

    ###########################################################################
    ###########################################################################
    ###########################################################################
        
    elif PD=='fcc':       # 2 1   1 !
    #CONSTITUENT FCC_A1  :B,FE%,SI : VA% :  !
        #------------ B 
        if T<=6000:
            G_B_0 = np.array([50208, -13.472, 0, 0, 0, 0,0,0]) + GHSER_B
            
        #------------ Fe    
        if T<=1811:
            G_Fe_0 = np.array([-1462.4, 8.282, -1.15, 6.4e-4, 0, 0, 0, 0]) + GHSER_Fe
        elif T> 1811 and T <= 6000:
            G_Fe_0 = np.array([-27098.266, 300.25256, -46, 0, 0, 0, 0, 2.78854E+31]) 
            
        #------------ Si
        if T<=3600:
            G_Si_0 = np.array([51000, -21.8, 0, 0, 0,  0,  0, 0]) + GHSER_Si
            
        #FeB
        G_FeB_0= np.array([-5.7793964E+04, 4.0432377E+01, 0, 0, 0, 0, 0, 0]) 
        G_FeB_1= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeB_2= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        
        #FeSi
        G_FeSi_0= np.array([-125248, 41.116, 0, 0, 0, 0, 0, 0])
        G_FeSi_1= np.array([-142708, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeSi_2= np.array([89907, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeSi_3= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        #BSi
        G_BSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_BSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_BSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        
        #BFeSi     (L0)
        G_BFeSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
    elif PD=='hcp':       # *  2 1   .5 !
    # CONSTITUENT HCP_A3  :FE,SI : VA% :  !
        #------------ B 
        if T<=6000:
            G_B_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
            
        #------------ Fe    
        if T<=1811:
            G_Fe_0 = np.array([-3705.78, 12.591, -1.15, 6.4e-4, 0, 0, 0, 0]) + GHSER_Fe
        elif T> 1811 and T <= 6000:
            G_Fe_0 = np.array([-3957.199, 5.24951, 0, 0, 0, 0, 0, 4.9251E+30]) + GHSER_Fe
            
        #------------ Si
        if T<=3600:
            G_Si_0 = np.array([49200, -20.8, 0, 0, 0,  0,  0, 0]) + GHSER_Si
            
        #FeB
        G_FeB_0= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeB_1= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeB_2= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        
        #FeSi
        G_FeSi_0= np.array([-123468, 41.116, 0, 0, 0, 0, 0, 0])
        G_FeSi_1= np.array([-142708, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeSi_2= np.array([89907, 0, 0, 0, 0, 0, 0, 0]) 
        G_FeSi_3= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        #BSi
        G_BSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_BSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        G_BSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])         
        
        #BFeSi     (L0)
        G_BFeSi_0= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_1= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_BFeSi_2= np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    #-------------------------  
    G_B_0= np.dot(G_B_0,PolT)[0] 
    G_Fe_0= np.dot(G_Fe_0,PolT)[0]      #+ Gmo
    G_Si_0= np.dot(G_Si_0,PolT)[0] 
    
    #FeB
    G_FeB_0= np.dot(G_FeB_0,PolT)[0] 
    G_FeB_1= np.dot(G_FeB_1,PolT)[0] 
    G_FeB_2= np.dot(G_FeB_2,PolT)[0] 
    
    #FeSi
    G_FeSi_0= np.dot(G_FeSi_0,PolT)[0]
    G_FeSi_1= np.dot(G_FeSi_1,PolT)[0]
    G_FeSi_2= np.dot(G_FeSi_2,PolT)[0]
    G_FeSi_3= np.dot(G_FeSi_3,PolT)[0]
    
    #BSi
    G_BSi_0= np.dot(G_BSi_0,PolT)[0]
    G_BSi_1= np.dot(G_BSi_1,PolT)[0]
    G_BSi_2= np.dot(G_BSi_2,PolT)[0]
    
    #BFeSi     (L0-L2)
    G_BFeSi_0= np.dot(G_BFeSi_0,PolT)[0] 
    G_BFeSi_1= np.dot(G_BFeSi_1,PolT)[0] 
    G_BFeSi_2= np.dot(G_BFeSi_2,PolT)[0] 
    
    
    Ref=[G_B_0, G_Fe_0, G_Si_0]
    FeB=[G_FeB_0, G_FeB_1, G_FeB_2]
    FeSi=[G_FeSi_0, G_FeSi_1, G_FeSi_2, G_FeSi_3]
    BSi=[G_BSi_0, G_BSi_1, G_BSi_2]
    Ternary=[G_BFeSi_0,G_BFeSi_1,G_BFeSi_2]
    
    Gibbs_SolidSol=[Ref,FeB,FeSi,BSi,Ternary]
    