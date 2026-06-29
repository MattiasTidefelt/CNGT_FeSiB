#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:32:19 2022

@author: ag0406
"""

def HEXCOL(N,Type):

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import to_hex
    # import matplotlib.pyplot as plt
    import numpy as np
    
    if Type ==1:
        ColLim=["#762a83","#e7d4e8","#1b7837"] #Green-violet-purple
    elif Type == 2:
        ColLim=["#ff335e","#bb33ff","#33fffc"] #red-purple-cyan
    elif Type == 3:
        ColLim=["#33ff9f","#beff33","#ffb533"] #green-lemon-orange
    elif Type == 4:
        # ColLim=["#FF6929","#f1224B","#7E2F8E"] #orange-lemon-purple
        ColLim=["#FF6929","#eb3480","#7E2F8E"] #orange-lemon-purple
    
    cmap=LinearSegmentedColormap.from_list('', ColLim , N=20)
    
    
    
    COLOR=[]
    Ncol=np.linspace(0,1,N)
    
    for i in Ncol:
        COLOR.append(to_hex(cmap(i)))
    
    return COLOR
# x=np.linspace(0,1)

# plt.figure(1)
# for i in range(Size):
#     plt.plot(x, x*(1+i), COLOR[i])
    