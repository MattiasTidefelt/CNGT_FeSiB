def Gibbs_Func_surf(T,xFe,xSi,xB,PD):
    import numpy as np
    from scipy.optimize import minimize_scalar
    
    if PD == 'Liquid':
        from LiqFeSiB import Gibbs_Liq 
        from LiqFeSiB import TransE
        
        [G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_Liq[0]
        [G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_Liq[1]
        [G_Liq_FeSi_0,G_Liq_FeSi_1,G_Liq_FeSi_2,G_Liq_FeSi_3]=Gibbs_Liq[2]
        [G_Liq_SiB_0,G_Liq_SiB_1,G_Liq_SiB_2]=Gibbs_Liq[3]
        [G_Liq_FeSiB_0,G_Liq_FeSiB_1,G_Liq_FeSiB_2]=Gibbs_Liq[4]
    else:
        from SolidSolFeSiB import Gibbs_SolidSol 
        from SolidSolFeSiB import TransGMO 
        
        [G_Liq_B_0,G_Liq_Fe_0,G_Liq_Si_0]=Gibbs_SolidSol[0]
        [G_Liq_FeB_0,G_Liq_FeB_1,G_Liq_FeB_2]=Gibbs_SolidSol[1]
        [G_Liq_FeSi_0,G_Liq_FeSi_1,G_Liq_FeSi_2,G_Liq_FeSi_3]=Gibbs_SolidSol[2]
        [G_Liq_SiB_0,G_Liq_SiB_1,G_Liq_SiB_2]=Gibbs_SolidSol[3]
        [G_Liq_FeSiB_0,G_Liq_FeSiB_1,G_Liq_FeSiB_2]=Gibbs_SolidSol[4]

    
    R=8.31446261815324  
    T=float(T)
    
    
    #--------------Gibbs Matrix Zr60Cu30Al10-------------
    Gref=xB*G_Liq_B_0 + xSi*G_Liq_Si_0 + xFe*G_Liq_Fe_0
    
    ## ONLY SURFACE PLOT
    if len(xB)==1:
        xB=xB*np.ones(max([len(xSi),len(xFe)]))
    elif len(xSi)==1:
        xSi=xSi*np.ones(max([len(xB),len(xFe)]))
    elif len(xFe)==1:
        xFe=xFe*np.ones(max([len(xSi),len(xB)]))
        
    Gide=np.zeros(max([len(xB),len(xSi),len(xFe)]))   
    for i in range(len(Gide)):            
        if xB[i]==0 and xSi[i]>0 and xFe[i]>0: 
            Gide[i]=R*T*(xSi[i]*np.log(xSi[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xSi[i]==0 and xFe[i]>0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xFe[i]*np.log(xFe[i]))
        elif xB[i]>0 and xSi[i]>0 and xFe[i]==0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xSi[i]*np.log(xSi[i]))
        elif xB[i]==0 and xSi[i]==0 and xFe[i]>0: 
            Gide[i]=R*T*(xFe[i]*np.log(xFe[i]))
        elif xB[i]==0 and xSi[i]>0 and xFe[i]==0: 
            Gide[i]=R*T*(xSi[i]*np.log(xSi[i]))
        elif xB[i]>0 and xSi[i]==0 and xFe[i]==0: 
            Gide[i]=R*T*(xB[i]*np.log(xB[i]))
        else:
            Gide[i]=R*T*(xB[i]*np.log(xB[i]) + xSi[i]*np.log(xSi[i]) + xFe[i]*np.log(xFe[i]))

    GexBin=(xFe*xB*(G_Liq_FeB_0 + G_Liq_FeB_1*(xB-xFe)**1 + G_Liq_FeB_2*(xB-xFe)**2 )
           + xFe*xSi*(G_Liq_FeSi_0 + G_Liq_FeSi_1*(xFe-xSi)**1 + G_Liq_FeSi_2*(xFe-xSi)**2 + G_Liq_FeSi_3*(xFe-xSi)**3)
          + xB*xSi*(G_Liq_SiB_0 + G_Liq_SiB_1*(xB-xSi)**1 + G_Liq_SiB_2*(xB-xSi)**2))

    Gex=GexBin + xB*xFe*xSi*((xB+0)*G_Liq_FeSiB_0 + (xFe+0)*G_Liq_FeSiB_1 + (xSi+0)*G_Liq_FeSiB_2)
    
    ######################
    def Gm_B2_ordered_from_eta(eta, xFe, xSi, T, L1):

        # Sublattice site fractions from eta
        y1_Si = xSi + eta
        y2_Si = xSi - eta

        y1_Fe = 1 - y1_Si
        y2_Fe = 1 - y2_Si

        # --- Reference term ---
        Gref = 0.5*(y1_Fe*G_Liq_Fe_0 + y1_Si*G_Liq_Si_0 +
                    y2_Fe*G_Liq_Fe_0 + y2_Si*G_Liq_Si_0)

        # --- Ideal entropy ---
        Gide = 0.5*R*T*(y1_Fe*np.log(y1_Fe) + y1_Si*np.log(y1_Si) +
                        y2_Fe*np.log(y2_Fe) + y2_Si*np.log(y2_Si))

        # --- Ordering enthalpy ---
        G_ord = L1*(y1_Fe*y2_Si + y1_Si*y2_Fe)

        return Gref + Gide + G_ord

    def Gm_B2_ordered_ref(xFe, xSi, T, L1):
        return Gm_B2_ordered_from_eta(0.0, xFe, xSi, T, L1)


    def G_bcc_from_eta(eta, G_A2_const, xFe, xSi, T, L1):
        G_ord = Gm_B2_ordered_from_eta(eta, xFe, xSi, T, L1)
        G_ref = Gm_B2_ordered_ref(xFe, xSi, T, L1)
        return G_A2_const + (G_ord - G_ref)

    def minimize_eta_and_get_sites(G_A2_const, xFe, xSi, xB, T, L1):
        # Allowed eta interval ensuring 0 ≤ y ≤ 1
        eta_min = max(-xSi, xSi - 1)
        eta_max = min( xSi, 1 - xSi )

        # If ordering impossible, eta = 0
        if eta_max - eta_min < 1e-14:
            eta_eq = 0.0
            G_bcc = G_bcc_from_eta(eta_eq, G_A2_const, xFe, xSi, T, L1)
        else:
            # Minimize Gibbs wrt eta
            res = minimize_scalar(
                lambda eta: G_bcc_from_eta(eta, G_A2_const, xFe, xSi, T, L1),
                bounds=(eta_min, eta_max),
                method="bounded"
            )
            eta_eq = res.x
            G_bcc = res.fun
            if eta_eq < 0:
                eta_eq = -eta_eq

        # Recover site fractions
        y1_Si = xSi + eta_eq
        y2_Si = xSi - eta_eq
        y1_Fe = 1 - y1_Si
        y2_Fe = 1 - y2_Si

        return G_bcc, eta_eq, y1_Si, y2_Si, y1_Fe, y2_Fe
    
    ######################
    if PD == 'Liquid':
        Gtrans = np.empty_like(Gide)
        for i in range(len(Gtrans)):
            [Gtrans[i],_] = TransE(T,xFe[i],xSi[i],xB[i])
        # print(Gtrans)
        Gm=Gref+Gide+Gex + Gtrans
    else: 
        Gtrans = np.empty_like(Gide)
        for i in range(len(Gtrans)):
            Gtrans[i] = TransGMO(T,xFe[i],xSi[i],xB[i],PD)
        Gm=(Gref+Gide+Gex + Gtrans)
        if PD == "bcc":
            L1 = -2*1260*R
            Gm_temp = np.empty_like(Gm)
            for i in range(len(Gm)):
                Gm_temp[i], _, _, _, _, _ = minimize_eta_and_get_sites(Gm[i] ,xFe[i], xSi[i], xB[i], T, L1)
            Gm = Gm_temp 
      
    return Gm
