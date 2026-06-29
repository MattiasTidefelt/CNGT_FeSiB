#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:12:05 2022

@author: ag0406

Thermodynamic description: Yoshitomi et al. ISIJ international, Vol. 48 (2008)

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
        # A = 518/1125 +11692/159575*(1/p-1)
        # if Tau <1:
        #     f_Tau = 1-1/A*((79*Tau**-1/(140*p) + 474/497*(1/p-1))*(Tau**3/6 + Tau**9/135 +Tau**15/600))
        # elif Tau>=1:
        #     f_Tau = -1/A*((Tau**-5)/10 + (Tau**-15)/315 + (Tau**-25)/1500)
        if Tau <=1:
            f_Tau = 1-0.9052999383*Tau**-1 -0.153008346*Tau**3 - 0.00680037095*Tau**9 - 0.0015300846*Tau**15
        elif Tau>1:
            f_Tau = -0.0641731208*Tau**-5 -0.00203724193*Tau**-15 - (4.278208005e-4)*Tau**-25
    except:
        f_Tau = 0

    Gmo = - R*f_Tau*np.log(BMAGN+1)
    # Gmo = 0
    
    return Gmo

def dTransE(T,xFe,xSi,xB):
    p = 0.99
    # p = 0.2
    
    Tg_Fe= 452.75 
    Tg_Si= 422 
    Tg_B= 587.25  
    
    ag_Fe = 1.511526
    ag_Si = 3.49000000E+01
    ag_B = 12.069916
    
    A0_FeB = 3.94497952E+02
    A1_FeB = 1.15226105E+03
    A2_FeB = 8.70819606E+02
    
    A0_SiFeB = -2.03949395E+03
    A1_SiFeB = 6.62325398E+02
    A2_SiFeB = -3.48292295E+03
    
    Om0_FeB = 4.7158236e+02
    
    Om0_FeSiB = -7.70491622E+04
    Om1_FeSiB = 4.04365578E+04
    Om2_FeSiB = -5.99086462E+04 
    
    #Kauzmann temperature
    # The good thing is that Tk= 0.8*Tg; composition dependent Tg
    Tk = xFe*Tg_Fe + xSi*Tg_Si + xB*Tg_B + xFe*xB*Om0_FeB + xFe*xSi*xB*(xB*Om0_FeSiB  + xFe*Om1_FeSiB + xSi*Om2_FeSiB) 
    if Tk < 0:
        # It is written somewhere, I have to find it again
        # that for antiferromagnetic material, if T < 0, then it should be divided by -1
        # see fcc phase for example
        # Tk = 1e-6 
        Tk *= -1 
        
    alpha = xFe*ag_Fe + xSi*ag_Si + xB*ag_B + xB*xFe*(A0_FeB + A1_FeB*(xB-xFe) + A2_FeB*(xB-xFe)**2) + xB*xFe*xSi*(xB*A0_SiFeB  + xFe*A1_SiFeB + xSi*A2_SiFeB)
    
    if xFe > 0 and xSi > 0 and xB > 0:
        Gtrans = GMO(Tk/3,T,abs(alpha)/3,p)
    else:
        Gtrans = 0
    # Gtrans = 0
    
    return Gtrans
    
def LiqFeSiB_dT(T,PD):

    #GHSER data
    global Gibbs_Liq_dT
    #------------ 
    
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
    if PD=='Liquid': # 1  1.0  !
    # CONSTITUENT LIQUID:L :B,FE,SI :  !
        
        #------------ B 
        if T <= 500:
            G_B_0 = np.array([40723.275, 86.843839, -15.6641, -.006864515, 6.18878E-07, 370843, 0, 0])
        elif T>500 and T<= 2348:
            G_B_0 = np.array([41119.703, 82.101722, -14.9827763, -.007095669, 5.07347E-07, 335484, 0, 0])
        elif T>2348 and T<= 6000:
            G_B_0 = np.array([28842.012, 200.94731, -31.4, 0, 0, 0, 0, 0])
    
        #------------ Fe   
        if T<=1811:
            G_Fe_0 = np.array([12040.17, -6.55843, 0, 0, 0, 0, -3.6751551E-21, 0]) + GHSER_Fe
        elif T>1811 and T<= 6000:
            G_Fe_0 = np.array([14544.751, -8.01055, 0, 0, 0, 0, 0, -2.2960305E+31]) + GHSER_Fe
            
        #------------ Si
        if T<=1678:
            G_Si_0 = np.array([50696.36, -30.099439, 0, 0, 0, 0, 2.09307E-21, 0]) + GHSER_Si 
        elif T > 1678 and T<= 3600:
            G_Si_0 = np.array([49828.165, -29.559069, 0, 0, 0, 0, 0, 4.20369E+30]) + GHSER_Si 
            
        #FeB            
        G_FeB_0= np.array([-1.22861908E+05, 1.45892955E+01, 0, 0, 0, 0, 0, 0])        
        G_FeB_1= np.array([1.95234499E+04, 0, 0, 0, 0, 0, 0, 0])
        G_FeB_2= np.array([5.10703439E+04, 0, 0, 0, 0, 0, 0, 0])
    
        #FeSi
        G_FeSi_0= np.array([-164435, 41.977, 0, 0, 0, 0, 0, 0])
        G_FeSi_1= np.array([0, -21.523, 0, 0, 0, 0, 0, 0])
        G_FeSi_2= np.array([-18821, 22.07, 0, 0, 0, 0, 0, 0])
        G_FeSi_3= np.array([9696, 0, 0, 0, 0, 0, 0, 0])
        
        #BSi
        G_BSi_0= np.array([-68220.33, 41.76042, 0, 0, 0, 0, 0, 0])
        G_BSi_1= np.array([10902.63, -11.10014, 0, 0, 0, 0, 0, 0])
        G_BSi_2= np.array([39692.79, -17.31724, 0, 0, 0, 0, 0, 0])
        
        #BFeSi     (L0)
        G_FeBSi_0= np.array([-5.23427598E-06, 0, 0, 0, 0, 0, 0, 0])
        G_FeBSi_1= np.array([-55686, 0, 0, 0, 0, 0, 0, 0])
        G_FeBSi_2= np.array([93217, 0, 0, 0, 0, 0, 0, 0])
    
    #-------------------------  
    G_B_0= np.dot(G_B_0,PoldT)[0] 
    G_Fe_0= np.dot(G_Fe_0,PoldT)[0]      # + dGmo
    G_Si_0= np.dot(G_Si_0,PoldT)[0] 
    
    #FeB
    G_FeB_0= np.dot(G_FeB_0,PoldT)[0] 
    G_FeB_1= np.dot(G_FeB_1,PoldT)[0] 
    G_FeB_2= np.dot(G_FeB_2,PoldT)[0] 
    
    #FeSi
    G_FeSi_0= np.dot(G_FeSi_0,PoldT)[0]
    G_FeSi_1= np.dot(G_FeSi_1,PoldT)[0]
    G_FeSi_2= np.dot(G_FeSi_2,PoldT)[0]
    G_FeSi_3= np.dot(G_FeSi_3,PoldT)[0]
    
    #BSi
    G_BSi_0= np.dot(G_BSi_0,PoldT)[0]
    G_BSi_1= np.dot(G_BSi_1,PoldT)[0]
    G_BSi_2= np.dot(G_BSi_2,PoldT)[0]
    
    #FeBSi     (L0-L2)
    G_FeBSi_0= np.dot(G_FeBSi_0,PoldT)[0] 
    G_FeBSi_1= np.dot(G_FeBSi_1,PoldT)[0] 
    G_FeBSi_2= np.dot(G_FeBSi_2,PoldT)[0] 
    
    
    Ref=[G_B_0, G_Fe_0, G_Si_0]
    FeB=[G_FeB_0, G_FeB_1, G_FeB_2]
    FeSi=[G_FeSi_0, G_FeSi_1, G_FeSi_2, G_FeSi_3]
    BSi=[G_BSi_0, G_BSi_1, G_BSi_2]
    Ternary=[G_FeBSi_0,G_FeBSi_1,G_FeBSi_2]
    
    Gibbs_Liq_dT=[Ref,FeB,FeSi,BSi,Ternary]
    