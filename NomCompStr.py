#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:53:31 2024

@author: ag0406
"""

def MatrixStr(Comp):
    xFe = Comp[0]    
    xSi = Comp[1]
    xB = Comp[2]
    
    composition_string = f"Fe{xFe:.2f}Si{xSi:.2f}B{xB:.2f}"
    return composition_string
