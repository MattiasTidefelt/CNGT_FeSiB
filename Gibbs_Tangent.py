def Gibbs_Tangent(T,xFe,xSi,xB,PD,eta):
    import numpy as np
    if PD == 'Liquid':
        from LiqFeSiB import Gibbs_Liq 
        
        [G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_Liq[0]
        [G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_Liq[1]
        [G_Liq_FeSi_0,G_Liq_FeSi_1,G_Liq_FeSi_2,G_Liq_FeSi_3]=Gibbs_Liq[2]
        [G_Liq_SiB_0,G_Liq_SiB_1,G_Liq_SiB_2]=Gibbs_Liq[3]
        [G_Liq_FeSiB_0,G_Liq_FeSiB_1,G_Liq_FeSiB_2]=Gibbs_Liq[4]
    else:
        from SolidSolFeSiB import Gibbs_SolidSol 
        
        [G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_SolidSol[0]
        [G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_SolidSol[1]
        [G_Liq_FeSi_0,G_Liq_FeSi_1,G_Liq_FeSi_2,G_Liq_FeSi_3]=Gibbs_SolidSol[2]
        [G_Liq_SiB_0,G_Liq_SiB_1,G_Liq_SiB_2]=Gibbs_SolidSol[3]
        [G_Liq_FeSiB_0,G_Liq_FeSiB_1,G_Liq_FeSiB_2]=Gibbs_SolidSol[4]
        
        if PD == "bcc":
            from B2_derivatives import dG_B2_dxSi, dG_B2_dxFe
            dG_B2_Si = dG_B2_dxSi(eta, xFe, xSi, T)
            dG_B2_Fe = dG_B2_dxFe(eta, xFe, xSi, T)


    
    R=8.31446261815324
    
    if xFe <= 0:
        xFe = 1e-6
    if xSi <= 0:
        xSi = 1e-6
    if xB <= 0:
        xB = 1e-6
    

    ###################
    # import sympy as sp

    # # Declare variables
    # xFe, xSi, xB, T = sp.symbols('xFe xSi xB T', real=True, positive=True)
    # R = sp.Symbol('R', real=True, positive=True)
    
    # # Gibbs energy constants
    # G_Liq_Fe_0, G_Liq_Si_0, G_Liq_B_0 = sp.symbols('G_Liq_Fe_0 G_Liq_Si_0 G_Liq_B_0')
    # G_Liq_FeB_0, G_Liq_FeB_1, G_Liq_FeB_2 = sp.symbols('G_Liq_FeB_0 G_Liq_FeB_1 G_Liq_FeB_2')
    # G_Liq_FeSi_0, G_Liq_FeSi_1, G_Liq_FeSi_2, G_Liq_FeSi_3 = sp.symbols('G_Liq_FeSi_0 G_Liq_FeSi_1 G_Liq_FeSi_2 G_Liq_FeSi_3')
    # G_Liq_SiB_0, G_Liq_SiB_1, G_Liq_SiB_2 = sp.symbols('G_Liq_SiB_0 G_Liq_SiB_1 G_Liq_SiB_2')
    # G_Liq_FeSiB_0, G_Liq_FeSiB_1, G_Liq_FeSiB_2 = sp.symbols('G_Liq_FeSiB_0 G_Liq_FeSiB_1 G_Liq_FeSiB_2')
    
    # Gtrans = sp.Symbol('Gtrans')
    
    # # Reference part
    # Gref = xB*G_Liq_B_0 + xSi*G_Liq_Si_0 + xFe*G_Liq_Fe_0
    
    # # Ideal mixing
    # Gide = R*T*(xB*sp.log(xB) + xSi*sp.log(xSi) + xFe*sp.log(xFe))
    
    # # Binary excess terms
    # Gex=((xFe*xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB-xFe)**1 + G_Liq_FeB_2*(xB-xFe)**2 )
    #       + xFe*xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe-xSi)**1 + G_Liq_FeSi_2*(xFe-xSi)**2 + G_Liq_FeSi_3*(xFe-xSi)**3)
    #       + xB*xSi*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB-xSi)**1 + G_Liq_SiB_2*(xB-xSi)**2))
    #       + xB*xFe*xSi*((xB+0)*G_Liq_FeSiB_0 + (xFe+0)*G_Liq_FeSiB_1 + (xSi+0)*G_Liq_FeSiB_2))
      
    # # Total Gibbs energy
    # Gm = Gref + Gide + Gex + Gtrans
    
    # # Compute symbolic partial derivatives
    # mu_Fe = sp.simplify(sp.diff(Gm, xFe))
    # mu_Si = sp.simplify(sp.diff(Gm, xSi))
    # mu_B  = sp.simplify(sp.diff(Gm, xB))
    
    # # Display results
    # print("μ_Fe = ")
    # sp.pprint(mu_Fe, use_unicode=True)
    # print("\nμ_Si = ")
    # sp.pprint(mu_Si, use_unicode=True)
    # print("\nμ_B = ")
    # sp.pprint(mu_B, use_unicode=True)
    # from sympy.printing.pycode import pycode

    # # Compute symbolic partial derivatives
    # mu_Fe = sp.simplify(sp.diff(Gm, xFe))
    # mu_Si = sp.simplify(sp.diff(Gm, xSi))
    # mu_B  = sp.simplify(sp.diff(Gm, xB))

    # # Display results as Python expressions
    # print("mu_Fe = ")
    # print(pycode(mu_Fe))
    # print("\nmu_Si = ")
    # print(pycode(mu_Si))
    # print("\nmu_B = ")
    # print(pycode(mu_B))
    ###################
    # Gref=xB*G_Liq_B_0 + xSi*G_Liq_Si_0 + xFe*G_Liq_Fe_0
    
    # Gide=R*T*(xB*np.log(xB) + xSi*np.log(xSi) + xFe*np.log(xFe))
    
    # GexBin=(xFe*xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB-xFe)**1 + G_Liq_FeB_2*(xB-xFe)**2 )
    #       + xFe*xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe-xSi)**1 + G_Liq_FeSi_2*(xFe-xSi)**2 + G_Liq_FeSi_3*(xFe-xSi)**3)
    #       + xB*xSi*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB-xSi)**1 + G_Liq_SiB_2*(xB-xSi)**2))
    # Gex=GexBin + xFe*xB*xSi*((xB+0)*G_Liq_FeSiB_0 + (xFe+0)*G_Liq_FeSiB_1 + (xSi+0)*G_Liq_FeSiB_2)
    
    # Gm=Gref+Gide+Gex + Gtrans
    # Gm=Gref+Gide+Gex + Gtrans

    ################## dGdxFe
    # dGref_Fe=G_Liq_Fe_0
    
    # dGide_Fe=R*T*(np.log(xFe)+1) 
    
    # dGexBin_Fe = (xB*(G_Liq_FeB_0 - G_Liq_FeB_1*(xFe - xB) + G_Liq_FeB_2*(xFe - xB)**2 + xFe*(G_Liq_FeB_1 - 2*G_Liq_FeB_2*(xFe - xB)))
    #               +(xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe - xSi) + G_Liq_FeSi_2*(xFe - xSi)**2 + G_Liq_FeSi_3*(xFe - xSi)**3 + xFe*xSi*(G_Liq_FeSi_1 + 2*G_Liq_FeSi_2*(xFe - xSi) + 3*G_Liq_FeSi_3*(xFe - xSi)**2))))
    
    # dGex_Fe=dGexBin_Fe + xSi*xB**2*G_Liq_FeSiB_0 + 2*xFe*xSi*xB*G_Liq_FeSiB_1 + xSi**2*xB*G_Liq_FeSiB_2
    
    # dGm_Fe=dGref_Fe + dGide_Fe + dGex_Fe
    
    dGm_Fe = G_Liq_FeSiB_1*xB*xFe*xSi + G_Liq_Fe_0 + R*T*(np.log(xFe) + 1) - xB*xFe*(G_Liq_FeB_1 + 2*G_Liq_FeB_2*(xB - xFe)) + xB*xSi*(G_Liq_FeSiB_0*xB + G_Liq_FeSiB_1*xFe + G_Liq_FeSiB_2*xSi) + xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB - xFe) + G_Liq_FeB_2*(xB - xFe)**2) + xFe*xSi*(G_Liq_FeSi_1 + 2*G_Liq_FeSi_2*(xFe - xSi) + 3*G_Liq_FeSi_3*(xFe - xSi)**2) + xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe - xSi) + G_Liq_FeSi_2*(xFe - xSi)**2 + G_Liq_FeSi_3*(xFe - xSi)**3)
    
    ################## dGdxSi
    # dGref_Si=G_Liq_Si_0
    
    # dGide_Si=R*T*(np.log(xSi)+1) 
    
    # dGexBin_Si = (xFe*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe - xSi) + G_Liq_FeSi_2*(xFe - xSi)**2  + G_Liq_FeSi_3*(xFe - xSi)**3 - xFe*xSi*(G_Liq_FeSi_1 + 2*G_Liq_FeSi_2*(xFe - xSi) + 3*G_Liq_FeSi_3*(xFe - xSi)**2))
    #               +(xB*(G_Liq_SiB_0 - G_Liq_SiB_1*(xB - xSi) + G_Liq_SiB_2*(xB - xSi)**2 + xSi*(G_Liq_SiB_1 - 2*G_Liq_SiB_2*(xB - xSi)))))
    
    # dGex_Si=dGexBin_Si + xFe*xB**2*G_Liq_FeSiB_0 + xFe**2*xB*G_Liq_FeSiB_1 + 2*xFe*xSi*xB*G_Liq_FeSiB_2
    
    # dGm_Si = dGref_Si + dGide_Si + dGex_Si
    dGm_Si = G_Liq_FeSiB_2*xB*xFe*xSi + G_Liq_Si_0 + R*T*(np.log(xSi) + 1) + xB*xFe*(G_Liq_FeSiB_0*xB + G_Liq_FeSiB_1*xFe + G_Liq_FeSiB_2*xSi) - xB*xSi*(G_Liq_SiB_1 + 2*G_Liq_SiB_2*(xB - xSi)) + xB*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB - xSi) + G_Liq_SiB_2*(xB - xSi)**2) - xFe*xSi*(G_Liq_FeSi_1 + 2*G_Liq_FeSi_2*(xFe - xSi) + 3*G_Liq_FeSi_3*(xFe - xSi)**2) + xFe*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe - xSi) + G_Liq_FeSi_2*(xFe - xSi)**2 + G_Liq_FeSi_3*(xFe - xSi)**3)
    
    ################## dGdxB
    # dGref_B=G_Liq_B_0
    
    # dGide_B=R*T*(np.log(xB) + 1) 
    
    # dGexBin_B = (xFe*(G_Liq_FeB_0 - G_Liq_FeB_1*(xFe - xB) + G_Liq_FeB_2*(xFe - xB)**2 - xB*(G_Liq_FeB_1 - 2*G_Liq_FeB_2*(xFe - xB)))
    #               +(xSi*(G_Liq_SiB_0 - G_Liq_SiB_1*(xB - xSi) + G_Liq_SiB_2*(xB - xSi)**2 - xB*(G_Liq_SiB_1 - 2*G_Liq_SiB_2*(xB - xSi)))))
    
    # dGex_B=dGexBin_B + 2*xFe*xSi*xB*G_Liq_FeSiB_0 + xFe**2*xSi*G_Liq_FeSiB_1 + xSi**2*xFe*G_Liq_FeSiB_2
    
    # dGm_B = dGref_B +dGide_B +dGex_B
    dGm_B = G_Liq_B_0 + G_Liq_FeSiB_0*xB*xFe*xSi + R*T*(np.log(xB) + 1) + xB*xFe*(G_Liq_FeB_1 + 2*G_Liq_FeB_2*(xB - xFe)) + xB*xSi*(G_Liq_SiB_1 + 2*G_Liq_SiB_2*(xB - xSi)) + xFe*xSi*(G_Liq_FeSiB_0*xB + G_Liq_FeSiB_1*xFe + G_Liq_FeSiB_2*xSi) + xFe*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB - xFe) + G_Liq_FeB_2*(xB - xFe)**2) + xSi*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB - xSi) + G_Liq_SiB_2*(xB - xSi)**2)
    
    if PD == "bcc":
        return [dGm_Fe + dG_B2_Fe, dGm_Si + dG_B2_Si, dGm_B] 
    else:
        return [dGm_Fe, dGm_Si, dGm_B]  
