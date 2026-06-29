def StoicCompounds(T,PD):
    from SolidFeSiB import Gibbs_Solid 
    
    m=PD[2]
    n=PD[3]
    o=PD[4]

    G_Bin_FeB = Gibbs_Solid[0]
    Gmo = Gibbs_Solid[-1]

    Gp = G_Bin_FeB[0]/(m+n+o) + Gmo
        
    return Gp 
