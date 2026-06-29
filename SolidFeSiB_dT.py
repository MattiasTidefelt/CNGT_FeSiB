#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:18:22 2022

@author: ag0406

Thermodynamic description: Yoshitomi et al. ISIJ international, Vol. 48 (2008)
"""

import numpy as np
R=8.31446261815324

def GMO(TC,T,BMAGN,p):
    # Magnetic ordering contribution to Gibbs energy
    try:
        Tau = T/TC
        # A = 518/1125 +11692/159575*(1/p-1)
        # if Tau <=1:
        #     f_Tau = 1-1/A*((79*Tau**-1/(140*p) + 474/497*(1/p-1))*(Tau**3/6 + Tau**9/135 +Tau**15/600))
        # elif Tau>1:
        #     f_Tau = -1/A*((Tau**-5)/10 + (Tau**-15)/315 + (Tau**-25)/1500)
        if Tau <=1:
            f_Tau = 1-0.9052999383*Tau**-1 -0.153008346*Tau**3 - 0.00680037095*Tau**9 - 0.0015300846*Tau**15
        elif Tau>1:
            f_Tau = -0.0641731208*Tau**-5 -0.00203724193*Tau**-15 - (4.278208005e-4)*Tau**-25
    except:
        f_Tau = 0

    Gmo = R*f_Tau*np.log(BMAGN+1)
    
    return Gmo
        

def SolidFeSiB_dT(T,PD):
    import numpy as np

    #GHSER data
    global Gibbs_Solid_dT
    
    # m=PD[3]
    # n=PD[4]

    # R=8.31446261815324
    #BhorMagneton = 9.2740100783e-24
    #-------------------Standard element reference (SER)
    PoldT=np.array([0, 1, np.log(T)+1, 2*T, 3*T**2, -T**(-2), 7*T**6, -9*T**(-10)]).reshape(8,1)
    
    #------------ GHSER_i
    if T <= 1100:
        GHSER_B = np.array([-7735.284, 107.111864, -15.6641, -0.006864515, 6.18878e-7, 370843,0,0])
    elif T>1100 and T<= 2348:
        GHSER_B = np.array([-16649.474, 184.801744, -26.6047, -0.79809e-3, -0.02556e-6, 1748270,0,0])
    elif T>2348 and T<= 3000:
        GHSER_B = np.array([-36667.582, 231.336244, -31.5957527, -.00159488, +1.34719E-07, 11205883,0,0])
        
    if T <= 1811:
        GHSER_Fe = np.array([1225.7, 124.134, -23.5143, -4.39752e-3, -0.058927e-6, 77359,0,0])
    elif T > 1811 and T<= 6000:
        GHSER_Fe = np.array([-25383.581, 299.31255, -46, 0, 0, 0 ,0,2.2960305E+31])
        
    if T <= 1687:
        GHSER_Si = np.array([-8162.609, 137.227259, -22.8317533, -0.001912904, -3.552E-09, 176667,0,0])
    elif T>1687:
        GHSER_Si = np.array([-9457.642, 167.271767, -27.196, 0, 0, 0,0,-4.20369E+30])
    ###########################################################################
    ###########################################################################
    ###########################################################################

        
    #-------------------LINE COMPOUNDS POLYNOMIALS
    if PD =='FeB':    # 2 1   1 !      
        G_Bin1= np.array([-7.39333794E+04, 6.83396500E+00, 0, 0, 0, 0, 0, 0]) +GHSER_Fe + GHSER_B
        G_Bin2= np.array([0, 0, 0, 0, 0, 0, 0, 0,]) 
   
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        TC = 599                   
        # BMAGN= 1.26
        BMAGN= 0.58
        p = 0.4
        dGmo = GMO(TC,T,BMAGN,p)
         
    elif PD =='Fe2B': # 2 2   1 !
         G_Bin1= np.array([-8.1226188E+04, 3.1072761E+00, 0, 0, 0, 0, 0, 0]) +2*GHSER_Fe + GHSER_B 
         G_Bin2= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
    
         G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
         G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

         TC = 1016                   
         # BMAGN= 1.95
         BMAGN= 1.223
         p = 0.4
         dGmo = GMO(TC,T,BMAGN,p)
     
    elif PD =='Fe3B': # 2 3   1 !       
        G_Bin1= np.array([-7.7749550e4, 2.5959351e0, 0, 0, 0, 0, 0, 0]) +3*GHSER_Fe + GHSER_B 
        G_Bin2= np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
   
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        TC = 790                     
        # BMAGN= 1.94
        BMAGN= 1.3825
        p = 0.4
        dGmo = GMO(TC,T,BMAGN,p)
         
    elif PD=='FeSi': # 2 .5   .5 !
        G_Bin1 = np.array([-36381, 2.22, 0, 0, 0, 0, 0, 0]) + 0.5*GHSER_Fe + 0.5*GHSER_Si
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        dGmo = 0
    
    elif PD=='Fe2Si': # 2 .666667   .333333 !
        G_Bin1 = np.array([-23752, -3.54, 0, 0, 0, 0, 0, 0]) + .6666667*GHSER_Fe + .3333333*GHSER_Si
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        TC = 780                   
        BMAGN= 0.66
        p = 0.4
        dGmo = GMO(TC,T,BMAGN,p)
        
    elif PD=='Fe5Si2B':  # 3 4.7   2   1 !     
        G_Bin1 = np.array([-2.48130111E+05, 5, 0, 0, 0, 0, 0, 0]) + 4.7*GHSER_Fe + 2*GHSER_Si + GHSER_B
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        dGmo = 0
    
    elif PD=='Fe5SiB2':   # 3 5   1   2 !        
        G_Bin1 = np.array([-2.69801881E+05, 2.35921193E+01, 0, 0, 0, 0, 0, 0]) + 5*GHSER_Fe + GHSER_Si + 2*GHSER_B
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        TC = 784 
        BMAGN = 1.12625
        p = 0.4
        dGmo = GMO(TC,T,BMAGN,p)
        
    elif PD=='Fe10Si4B3': # 3 2   .4   .6 !
        G_Bin1 = np.array([-9.26653078E+04, 6.67608425E+00, 0, 0, 0, 0, 0, 0]) + 2*GHSER_Fe + 0.4*GHSER_Si + 0.6*GHSER_B
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])  
        
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        dGmo = 0
        
    elif PD=='Fe5Si3': # 2 .625   .375 !
        G_Bin1 = np.array([-30143, .27, 0, 0, 0, 0, 0, 0]) + .625*GHSER_Fe + 0.375*GHSER_Si
        G_Bin2 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        
        #(L0,L1)
        G_Tern1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        G_Tern2 = np.array([0 , 0, 0, 0, 0, 0, 0, 0])
        
        TC = 100 + 273.15                   
        BMAGN= 1.2
        p = 0.4
        dGmo = GMO(TC,T,BMAGN,p)
        
    #---------------------------------------------
    G_Bin1= np.dot(G_Bin1,PoldT)[0]
    G_Bin2= np.dot(G_Bin2,PoldT)[0]
   
    G_Tern1 = np.dot(G_Tern1,PoldT)[0]
    G_Tern2 = np.dot(G_Tern2,PoldT)[0]
    
    Bin=[G_Bin1,G_Bin2]
    Ternary=[G_Tern1,G_Tern2]
    
    Gibbs_Solid_dT=[Bin,Ternary,dGmo]
    