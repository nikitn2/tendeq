#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import config as cfg
import numpy as np

import TnMachinery.MPSfunctions as mps

"""
Here are various functions for calculating p(x,t) solutions of Peyman's (3D spatial, KD reactant) Fokker-Planck equation.
"""

# Creates the functions:
# reac_1 = -r1*phi1,
# reac_2 = -r2*phi1,
#       .
#       .
# reac_K = -rK*phi1.

def initLinReacTermKD(NK, K, K_left, split, c, consts):
    
    # Initialise
    dphi = c/2.0**NK
    
    # Get linear-phis in mps form and kron them together
    prod_mps = mps.constDistrLinear(NK, K_left,   0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    uni_mps = mps.constDistrUniformFP(NK)
    for k in range(0,K-1): prod_mps = mps.mpsKron(prod_mps, uni_mps)
    
    # Interleave if necessary
    if split is False: prod_mps = mps.mpsInterleave(mps = prod_mps, K = K_left+K)
    
    # Scale reac_mps with the terms in consts to get reac1, reac2, ... reacK
    reac_mpses = []
    for const in consts: 
        if const == 0: const = cfg.zero
        reac_mpses+=[-const*prod_mps]
    
    return reac_mpses

# Creates the functions:
# reac_1 = -r1*phi1*phi2*...phiK,
# reac_2 = -r2*phi1*phi2*...phiK,
#       .
#       .
# reac_K = -rK*phi1*phi2*...phiK.

def initLinLinReacTermKD(NK, K, K_left, split, c, consts):
    
    # Initialise
    dphi = c/2.0**NK
    
    # Get linear-phis in mps form and kron them together
    prod_mps = mps.constDistrLinear(NK, K_left,   0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    phik_mps = mps.constDistrLinear(NK, 0,        0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    for k in range(0,K-1): prod_mps = mps.mpsKron(prod_mps, phik_mps)
    
    # Interleave if necessary
    if split is False: prod_mps = mps.mpsInterleave(mps = prod_mps, K = K_left+K)
    
    # Scale reac_mps with the terms in consts to get reac1, reac2, ... reacK
    reac_mpses = []
    for const in consts: 
        if const == 0: const = 1e-64
        reac_mpses+=[-const*prod_mps]
    
    return reac_mpses

# Outputs p_init and the following mpses used in “timestepFP_init_spat3DreacKD”:
# gamma, alpha, omega, S_j; u_i, phi_j.

def timestepFP_distributions_spat3DreacKD(NK, K_phi, l, c, split, nu, const_gamma, const_omega, reac_pars, velFieldFunc, velFieldPars, initialDistrFP, kwargs_initialDistrFP):
    
    # Initialise
    dx = l/2.0**NK
    spatialBoundryCond = "periodic"
    concntrBoundryCond = "normconserving"
    dphi = c/2.0**NK
    
    # First get the phis
    phi_mpses = [mps.constDistrLinear(NK=NK, K_left=3+k, K_right=K_phi-1-k, split = split, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) for k in range(K_phi)] #NikPhi
    
    # Now produce each of the MPSes for the FP equation in the case of 3 spatial dims and two reactants
    
    # Begin with initial flow field
    pInit_mps = initialDistrFP(NK=NK, split=split, l=l, c=c, **kwargs_initialDistrFP)
    
    # Reaction terms
    consts_reac = reac_pars[:-1]
    if    reac_pars[-1] == "LinLin": reac_mpses = initLinLinReacTermKD(NK, K_phi, 3, split, c, consts_reac)
    elif  reac_pars[-1] == "Lin":    reac_mpses = initLinReacTermKD(   NK, K_phi, 3, split, c, consts_reac)
    else: print("reac_pars's last element is invalid! Exiting."); sys.exit(1)
    
    # Artificial diffusion alpha:
    uni_mps = mps.constDistrUniformFP(NK*(3+K_phi))
    
    # Spatial diffusion nu
    nu_mps    = uni_mps.multiply(nu, inplace=False, spread_over="all" )
    
    # Create velocity field and modelled spatial diffusion gamma
    u1_mps, u2_mps, u3_mps = velFieldFunc(NK, K_padLeft=0, K=3, K_padRight=0, split=split, l=l, pars=velFieldPars)
    gamma_mps = getTotStrain3D(u1_mps,u2_mps,u3_mps, split, NK, 0, dx, preFactor = const_gamma)
    
    # Pad ui_mps and gamma
    padRight = mps.constDistrUniformFP(NK*K_phi)
    u1_mps = mps.mpsKron(u1_mps, padRight)
    u2_mps = mps.mpsKron(u2_mps, padRight)
    u3_mps = mps.mpsKron(u3_mps, padRight)
    gamma_mps = mps.mpsKron(gamma_mps, padRight)
    
    # Produce the differentiation MPOs
    
    # First spatial
    d_dx1    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 0,      NpadRight=NK*(2+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx2    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK,     NpadRight=NK*(1+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx3    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK*2,   NpadRight=NK*(0+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    
    # Now the phi ones
    d_dphis = []; dd_dphisSq = []
    for j in range(0,K_phi):
        d_dphis   += [mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK*(3+j), NpadRight=NK*(K_phi-1-j), h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)]
        dd_dphisSq+= [mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = NK*(3+j), NpadRight=NK*(K_phi-1-j), h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)]    
        
    return pInit_mps, nu_mps, gamma_mps, reac_mpses, phi_mpses, u1_mps, u2_mps, u3_mps, d_dx1, d_dx2, d_dx3, d_dphis, dd_dphisSq

# Outputs “p_1 = p_0 + dt*F(p_0p5_mps), with
# F = F(p)
#      = sum_i { 
#      - u_iHdp/dx_i 
#      + d/dx_i [(nu+gamma)Hdp/dx_i]
#      }  
#      + sum_j { 
#        dd/dphi_j^2 [alphaHp]                     <-- Artificial dissipation term
#      + d/dphi_j [omegaH({phi_j} - <Phi_j>) p]  
#      - d/dphi_j [S_jHp]
#      }
def timestepFP_init_spat3DreacKD(p_0_mps, p_0p5_mps, l, c, dt, split, chi, nu_mps, gamma_mps, const_omega, const_alpha, S_mpses, u1_mps, u2_mps, u3_mps, phi_mpses, d_dx1, d_dx2, d_dx3, d_dphis, dd_dphisSq, w, t, tol_zipup = 1e-16, tol_compression = 1e-16):    
    
    # Initialise
    chiHadInter = 2*chi
    chiOpsInter = 4*chi
    K_phi = len(phi_mpses)
    K = 3+ K_phi
    N = p_0p5_mps.L
    NK = N//K
    dphi = c/2.0**NK
    if K_phi<1: print("ERROR: need at least one species (so K_phi > 0)! Exiting.\n"); sys.exit(1)
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :tol_compression, "max_bond" :chiOpsInter, "method" : 'svd'}#'isvd', "renorm" : True }
    if const_omega == 0: const_omega = cfg.zero
    
    # Adjust u_i & gamma terms according to time
    coswt = np.cos(w*t) + cfg.zero
    u1_mps.multiply(coswt,    spread_over="all", inplace=True)
    u2_mps.multiply(coswt,    spread_over="all", inplace=True)
    u3_mps.multiply(coswt,    spread_over="all", inplace=True)
    gamma_mps.multiply(coswt, spread_over="all", inplace=True)
    
    # Get the reactant subgrid model omega
    nuPgamma_mps = nu_mps.add_MPS(gamma_mps, inplace=False, **COMP_OPTS)
    omega_mps = const_omega*nuPgamma_mps
    
    # Begin with the spatial terms, convection & (spatial) diffusion
    
    # First compute convection= -sum_k{u_kHdp_0p5/dx}
    dp_dx_1 = d_dx1.apply(p_0p5_mps, **COMP_OPTS)
    convection = mps.mpsHadamardZipupProd(u1_mps, dp_dx_1, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    dp_dx_2 = d_dx2.apply(p_0p5_mps, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(u2_mps, dp_dx_2, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    convection.add_MPS(temp, inplace=True, **COMP_OPTS)
    dp_dx_3 = d_dx3.apply(p_0p5_mps, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(u3_mps, dp_dx_3, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    convection.add_MPS(temp, inplace=True, **COMP_OPTS)
    convection.multiply(-1.0,inplace=True)    
    
    # Now compute spatial diffusion = sum_i{ d/dx_i [ (nu + gamma)Hdp/dx_i] }
    temp = mps.mpsHadamardZipupProd(nuPgamma_mps, dp_dx_1, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    diffusion = d_dx1.apply(temp, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(nuPgamma_mps, dp_dx_2, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    diffusion.add_MPS(d_dx2.apply(temp), inplace=True, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(nuPgamma_mps, dp_dx_3, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
    diffusion.add_MPS(d_dx3.apply(temp), inplace=True, **COMP_OPTS)
    
    # Sum spatial terms into F
    F = convection.add_MPS(diffusion, inplace = False, **COMP_OPTS)
    
    # Time for the phi terms
    
    # Begin with the artificial diffusion
    for j in range(0,K_phi):    
        temp = dd_dphisSq[j].apply(const_alpha*p_0p5_mps, **COMP_OPTS)
        F.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Now do the mixing term d/dphi_j [omegaH({phi_j} - <Phi_j>)]
    for j in range(0,K_phi):
        # First compute <phi_j>p_0p5
        phijHp  = mps.mpsHadamardZipupProd(p_0p5_mps, phi_mpses[j], max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
        avgphij = mps.averageMpsAccrossLastKDimensions(phijHp, NK, split, dvolToAvg= dphi**K_phi, KtoAvg=K_phi, padBack = True)
        avgphijHp= mps.mpsHadamardZipupProd(p_0p5_mps, avgphij, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
        
        # And then sum it with phijHp_0p5, multiply by omega, differentiate wrt to phi_j and add to F
        temp = phijHp.add_MPS(-1*avgphijHp, inplace=False, **COMP_OPTS)
        temp = mps.mpsHadamardZipupProd(temp, omega_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
        temp = d_dphis[j].apply(temp, **COMP_OPTS)
        F.add_MPS(temp, inplace = True, **COMP_OPTS)
    
    # Add the reaction term and finish
    for j in range(0,K_phi):
        temp = mps.mpsHadamardZipupProd(S_mpses[j], p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_zipup)
        temp = d_dphis[j].apply(temp, **COMP_OPTS)
        F.add_MPS(-1*temp, inplace = True, **COMP_OPTS)
    
    # Finish up by multiplying F with dt and add to p_0_mps
    F.multiply(dt, spread_over="all",inplace=True)
    p_1_mps = F.add_MPS(p_0_mps, inplace=False, **COMP_OPTS)
    
    # And finally, return.
    return p_1_mps, F

# Computes strain = sqrt(sum_ab {S_ij S_ij} ),
# with S_ab  = 0.5*(du_adx_b + du_bdx_a). 
def getTotStrain3D(u1,u2,u3, split, NK, K_phi, dx, spatialBoundryCond = "periodic", preFactor= 1, tol_zipup = 1e-16):
    
    # Initialise
    DU = [[None, None, None], [None, None, None], [None, None, None]]
    chi = max([u1.max_bond(), u2.max_bond(), u3.max_bond()])
    chiMax = 4*chi
    if preFactor == 0: preFactor = cfg.zero
    
    # First produce spatial differentiation MPOs
    d_dx1    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 0,      NpadRight=NK*(2+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx2    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK,     NpadRight=NK*(1+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx3    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK*2,   NpadRight=NK*(0+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    
    # Now get the du_i/dx_j tensor elements and fill into DU
    DU[0][0] = d_dx1.apply(u1, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[0][1] = d_dx1.apply(u2, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[0][2] = d_dx1.apply(u3, compress=True, cutoff_mode='rel', cutoff=1e-15)
    DU[1][0] = d_dx2.apply(u1, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[1][1] = d_dx2.apply(u2, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[1][2] = d_dx2.apply(u3, compress=True, cutoff_mode='rel', cutoff=1e-15);
    DU[2][0] = d_dx3.apply(u1, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[2][1] = d_dx3.apply(u2, compress=True, cutoff_mode='rel', cutoff=1e-15); DU[2][2] = d_dx3.apply(u3, compress=True, cutoff_mode='rel', cutoff=1e-15);
    
    # Then compute sqrt{ sum_ij (DU[i][j] + DU[j][i])^2 }
    temp = cfg.zero*DU[0][0]
    for i in range(3):
        for j in range(3):
            temp_ij = DU[i][j].add_MPS(DU[j][i])
            temp.add_MPS(mps.mpsHadamardZipupProd(temp_ij, temp_ij, max_intermediateBond = chiMax, max_finalBond = chi, tol_bond = tol_zipup), inplace=True, compress=True, cutoff_mode='rel', cutoff=1e-15)
    
    # Finally, take the square root, and return.
    #strain = preFactor*mps.mpsSqrt_dmrg(Ssum)
    strain = preFactor*mps.mpsSqrt_direct(temp, chi=chiMax)
        
    return strain
