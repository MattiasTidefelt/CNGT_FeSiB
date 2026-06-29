#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:38:45 2021

@author: ag0406
"""

from scipy.optimize import minimize
# from FindSolxp import find_xp
from Gibbs_Tangent import Gibbs_Tangent

from scipy.optimize import least_squares, minimize

def DcXp(T,xp,xm,TangCoeff_Liq,Mu_L,isStoich,PD):
    import numpy as np
    from Gibbs_Func import Gibbs_Func
    from StoicCompounds import StoicCompounds
    
    #particle at equal slope
    if PD[-1] == 'Stoic':
        m=PD[2]
        n=PD[3]
        o=PD[4]
        
        xp=[m/(m+n+o), n/(m+n+o), o/(m+n+o)]   #particle composition Fe, Si, B
        
        # dx = 1e-3
        # if np.any(xp == 0):  # perturb zeros slightly
        #     xp = np.clip(xp + dx, 0, 1)
        #     xp /= xp.sum()
        
        T_liq_p1=np.dot(xp,Mu_L)
        
        T_p1=StoicCompounds(T,PD)
        
        dc=T_liq_p1-T_p1 
        
    elif PD[-1] == 'Sol':
        # # Force composition of solution phase, edge cases produce problems
        # # and the solubility is ~0 in these phases for the investigated compositions
        # isStoich = True
        ################################
        ################################
        
        if isStoich:
        
            T_liq_p1=np.dot(xp,Mu_L)
            Gp,eta = Gibbs_Func(T,xp[0],xp[1],xp[2],PD[0])
            dc = T_liq_p1-Gp
            
        elif not isStoich:
                          
            eps = 1e-10
            
            def chemical_potentials(T, x, phase):
                """
                x: length-3 array [xFe, xSi, xB]
                phase: string used by your Gibbs functions
                Returns: mu (len-3) chemical potentials for that phase at T,x
                """
                fp,eta = Gibbs_Func(T, x[0], x[1], x[2], phase)         # scalar G
                TangCoeff = Gibbs_Tangent(T, x[0], x[1], x[2], phase, eta)  # dG/dx_i (vector len-3)
                mu = fp + TangCoeff - np.dot(x, TangCoeff)         # mu_i = G + dG/dx_i - sum_j x_j dG/dx_j
                return np.asarray(mu, dtype=float)
            
            def clamp_initial(vars2):
                """Make an initial guess feasible and not on the boundary."""
                a = max(eps, min(1 - 2*eps, vars2[0]))
                b = max(eps, min(1 - 2*eps, vars2[1]))
                if a + b >= 1 - eps:
                    s = (1 - 2*eps) / (a + b)
                    a *= s
                    b *= s
                return np.array([a, b], dtype=float)
            
            def compose(vars2):
                xFe, xSi = vars2
                xB = 1.0 - xFe - xSi
                return np.array([xFe, xSi, xB], dtype=float)
            
            def find_precipitate_composition(T, xm, Mu_L, PD, eps_local=1e-8, 
                                              tol_mu=1e-12, try_least_squares=False):
                """
                T : temperature
                xm_matrix : matrix composition array-like [xFe,xSi,xB]
                PD : phase descriptor (we assume PD[-1] is the solid-phase name)
                Returns: (xp (len-3 array), DeltaG = T_liq(xp) - G_solid(xp))
                """
            
                solid_phase = PD[0]

                def best_initial_guess(T, xSi_fixed, Mu_L, PD, fe_bounds=(0.75, 0.98), n_trials=20):
                    """
                    Randomly sample xFe with fixed xSi, compute xp and DeltaG,
                    and return the composition with the highest driving force.
                    """
                    solid_phase = PD[0]
                    best_x0 = None
                    best_DeltaG = -np.inf
                    xFe_n = np.linspace(fe_bounds[0],fe_bounds[1],n_trials)
                    # xSi_n = np.linspace(0.24,0.01,n_trials)
                    xSi_n = np.linspace(0.24,1e-6,n_trials)
                    
                    for i in range(n_trials):
                        xFe = xFe_n[i]
                        # x0 = clamp_initial([xFe, xSi_fixed])
                        xSi = xSi_n[i]
                        x0 = clamp_initial([xFe, xSi])
                        xp = compose(x0)
                        Gp,eta = Gibbs_Func(T, xp[0], xp[1], xp[2], solid_phase)
                        DeltaG = np.dot(Mu_L, xp) - Gp
                
                        if DeltaG > best_DeltaG:
                            best_DeltaG = DeltaG
                            best_x0 = x0
                    # print(f"{PD[0]}, {best_x0}")
                    return best_x0, best_DeltaG
                
                # x0_vars, best_DeltaG = best_initial_guess(T, 0.01 , Mu_L, PD, fe_bounds=(0.75, 0.98), n_trials=20)
                x0_vars, best_DeltaG = best_initial_guess(T, 1e-6 , Mu_L, PD, fe_bounds=(0.75, 1-1e-6), n_trials=20)
                
                if best_DeltaG > 0:
            
                    lower = np.array([eps_local, eps_local])
                    upper = np.array([1 - eps_local, 1 - eps_local])
                
                    # Objective to minimize: f = Gp - dot(Mu_L, x)  (because we want to maximize DeltaG = dot - Gp)
                    def obj_and_grad(vars2):
                        xFe, xSi = vars2
                        xB = 1.0 - xFe - xSi
                        if xFe <= eps_local or xSi <= eps_local or xB <= eps_local:
                            # penalty for invalid composition
                            return 1e6, np.array([0.0, 0.0])
                        xs = np.array([xFe, xSi, xB])
                        Gp,eta = Gibbs_Func(T, xs[0], xs[1], xs[2], solid_phase)
                        # objective value
                        f = Gp - np.dot(Mu_L, xs)
                        # gradient: d f / d x_i = mu_solid_i - mu_liq_i  (i=Fe,Si)
                        mu_s = chemical_potentials(T, xs, solid_phase)
                        grad = mu_s[:2] - Mu_L[:2]
                        return float(f), grad
                
                    def obj_wrapper(v):
                        f, _ = obj_and_grad(v)
                        return f
                
                    def jac_wrapper(v):
                        _, g = obj_and_grad(v)
                        return g
                
                    # Optionally add inequality constraint that T_liq(x) - Gp >= 0 (i.e., dot - Gp >= 0)
                    use_penalty = True
                    penalty_factor = 1e6
                
                    def penalized_obj(v):
                        f, grad = obj_and_grad(v)
                        xFe, xSi = v
                        xs = compose(v)
                        dot = np.dot(Mu_L, xs)
                        violation = max(0.0, (Gp := Gibbs_Func(T, xs[0], xs[1], xs[2], solid_phase)[0]) - dot)
                        if violation > 0:
                            f += penalty_factor * (violation**2)
                        return f
                
                    x0 = x0_vars
                
                    # minimize (use jac for speed)
                    res = minimize(penalized_obj if use_penalty else obj_wrapper,
                                    x0,
                                    method='SLSQP',
                                    jac=(jac_wrapper if not use_penalty else None),  # if using penalty we don't provide Jacobian
                                    bounds=list(zip(lower, upper)), 
                                    options={'ftol':1e-12, 'maxiter':500, 'disp': False})
                
                    if not res.success:
                        # Fallback to ~Stoichometric Fe
                        xp = compose(x0)
                        Gp,eta = Gibbs_Func(T,xp[0],xp[1],xp[2],PD[0])
                        DeltaG = np.dot(xp,Mu_L) - Gp
                        # print(f"Failed dc {PD[0]}, T: {T}, dc {DeltaG}, used x: {x0}")
                        return xp, DeltaG
                
                    # success
                    xFe, xSi = res.x
                    xp = compose([xFe, xSi])
                    if xp[xp<0].any():
                        xp = compose(x0)
                    Gp,eta = Gibbs_Func(T, xp[0], xp[1], xp[2], solid_phase)
                    DeltaG =  np.dot(Mu_L, xp) - Gp
                    return xp, DeltaG
                
                else:
                    xp = compose(x0_vars)
                    return xp, best_DeltaG
            
            xp, dc = find_precipitate_composition(T, xm, Mu_L, PD, eps_local=1e-8, 
                                              tol_mu=1e-12, try_least_squares=False)
                       
    return [dc,np.array(xp)]
