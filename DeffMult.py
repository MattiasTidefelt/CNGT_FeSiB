#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:54:18 2022

@author: ag0406
"""

from Phi import Phi
import numpy as np
def DMult(T,Tg,Tm,Vml):
    k_B=1.38064852E-23                      # Boltzman constant
    N_A=6.02214086E23                       # Avocragos constant
    h=6.62607015e-34                        # Planks constant
    #------viscocity
    q=4.536
    z=2.889
    # Tg += 100
    # Ta=2.02*Tg
    # Ta = (1.075 - 0.188/4)*Tm     # +- 0.188
    Ta = (1.075)*Tm     # +- 0.188
    # Ta = (1.075 + 0.188/4)*Tm     # +- 0.188
    # Ta = (1.075 + 0.188/2)*Tm     # +- 0.188
    Tr = (Ta-T)/Ta
    if Tr < 0:
        Tr = 0
    Einf = 6.466*Ta
    
    eta0 = h*N_A/Vml      #Approx from article
    E = Einf + Ta*(q*Tr)**z*Phi(Ta-T)
    eta = eta0*np.exp(E/T)
    
    l = 2*(3*Vml/(4*np.pi*N_A))**(1/3)
    Deff = k_B*T/(3*np.pi*l*eta)  #m2/s
    
    #Fe, Si, B
    # A_W=np.array([55.845, 28.0855, 10.811]) # a.u  
       
    # D=np.array([A_W[2]/A_W[0], A_W[2]/A_W[1] , 1])*Deff
    
    A_r=np.array([1.32, 1.11, 0.84]) # Å  
       
    D=np.array([A_r[2]/A_r[0], A_r[2]/A_r[1] , 1])*Deff

    # D = Deff*np.ones(3)
    
    # D=np.array([1, 0.01 , 50])*Deff
    # D=np.array([1/5, 1/50 , 5e3])*Deff
    # D=np.array([1, 1/500 , 50])*Deff
    # D=np.array([1/5, 1/50 , 1])*Deff
    
    return(D)