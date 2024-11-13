#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import TnMachinery.MPSfunctions as mps
from config import phiStart

"""
Here are various functions for calculating p(x,t) solutions of Pope's test case  (0D spatial, 2D reactant) Fokker-Planck equation.
"""

def timestepFP_distributions_spat0Dreac2D(NK, c, split, const_beta, consts_gammas, concntrBoundryCond, diffAcc, initialDistrFP, kwargs_initialDistrFP, initReacTerm, kwargs_initReacTerm):
    
    # Initialise
    #concntrBoundryCond = "homogeneous" 
    #concntrBoundryCond = "zerograd"
    dphi = c/2.0**NK
    
    # Produce the differentiation MPOs
    
    if diffAcc == 2:   difmpoGenerator = mps.mpoCreateAcc2SpatialDiff
    elif diffAcc == 8: difmpoGenerator = mps.mpoCreateAcc8SpatialDiff
    
    d_dphi_1    = difmpoGenerator(N=NK, diffOrder=1, NpadLeft = 0*NK, NpadRight=NK,   h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dphi_2    = difmpoGenerator(N=NK, diffOrder=1, NpadLeft = 1*NK, NpadRight=0,    h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dphi_1Sq = difmpoGenerator(N=NK, diffOrder=2, NpadLeft = 0*NK, NpadRight=NK,   h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dphi_2Sq = difmpoGenerator(N=NK, diffOrder=2, NpadLeft = 1*NK, NpadRight=0,    h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    
    differentiators = [d_dphi_1,d_dphi_2,dd_dphi_1Sq,dd_dphi_2Sq ]
    
    # Produce each of the MPSs for the FP equation in the case of 2 spatial dims and two reactants
    
    # First initial flow field
    pInit_mps = initialDistrFP(NK=NK, split=split, c=c, **kwargs_initialDistrFP)
    
    # Then alpha, beta and gamma MPSs
    uni2K = mps.constDistrUniformFP(NK*2)
    beta_mps   = uni2K.multiply(const_beta,         inplace=False, spread_over="all" )
    gamma1_mps = uni2K.multiply(consts_gammas[0],   inplace=False, spread_over="all" )
    gamma2_mps = uni2K.multiply(consts_gammas[1],   inplace=False, spread_over="all" )
    
    # Delta MPSs are a bit special. First initialise phis
    #dphi = c/2**NK
    phi1_mps = mps.constDistrLinear(NK=NK, K_left=0, K_right=1, split = split, pars= [0.0+phiStart,c-dphi+phiStart]) #NikPhi
    phi2_mps = mps.constDistrLinear(NK=NK, K_left=1, K_right=0, split = split, pars= [0.0+phiStart,c-dphi+phiStart]) #NikPhi
    
    # Now construct the delta
    delta1_mps, delta2_mps = initReacTerm(NK = NK, K_left = 0, split= split, c=c, **kwargs_initReacTerm)
    
    return pInit_mps, beta_mps, gamma1_mps, gamma2_mps, delta1_mps, delta2_mps, phi1_mps, phi2_mps, differentiators

# Outputs “p_1 = p_0 + dt*F(p_0p5_mps)
# F(p) =
#      + sum_j { 
#        dd/dphi_j^2 [betaHp]                     <-- Artificial dissipation term
#      + d/dphi_j [gamma_jH( {phi_j} - <Phi_j>)Hp]  
#      - d/dphi_j [delta_jHp]
#      }
def timestepFP_init_spat0Dreac2D(p_0_mps, p_0p5_mps, l, c, dt, chi, split, beta_mps, gamma1_mps,  gamma2_mps, delta1_mps, delta2_mps, phi1_mps, phi2_mps, differentiators, t = None, tol_compression = 1e-16, tol_bond_zipup = 1e-16):    
    
    # Initialise
    K = 2 # phi_1, phi_2
    N = p_0p5_mps.L
    NK = N//K
    chiHadInter = 4*chi
    chiOpsInter = 8*chi
    dphi = c/2.0**NK
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :tol_compression, "max_bond" :chiOpsInter, "method" : 'svd'}#'isvd', "renorm" : True }
    
    # Extract the differentiators
    d_dphi_1    = differentiators[0]
    d_dphi_2    = differentiators[1]
    dd_dphi_1Sq = differentiators[2]
    dd_dphi_2Sq = differentiators[3]    
    
    # Now compute phi_1, phi_2 diffusion
    betaHp = mps.mpsHadamardZipupProd(beta_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    F = dd_dphi_1Sq.apply(betaHp, **COMP_OPTS)
    F.add_MPS(dd_dphi_2Sq.apply(betaHp, **COMP_OPTS), inplace=True, **COMP_OPTS)
    
    # Now need sum_i d/dphi_i ( gamma_iH(phi_i - <phi_i>)p_0p5 ). First initiate the integrator
    
    # Compute -<phi_1>p_0p5
    phi1p = mps.mpsHadamardZipupProd(p_0p5_mps, phi1_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    avgphi1 = mps.averageMpsAccrossLastKDimensions(phi1p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi1p= mps.mpsHadamardZipupProd(p_0p5_mps, avgphi1, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    
    # And -<phi_2>p_0p5
    phi2p = mps.mpsHadamardZipupProd(p_0p5_mps, phi2_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    avgphi2 = mps.averageMpsAccrossLastKDimensions(phi2p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi2p= mps.mpsHadamardZipupProd(p_0p5_mps, avgphi2, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    
    # Then sum phi1p_0p5 with -<phi1>p_0p5 and hadamard the result with gamma1, before differentiating with regards to phi1
    temp = phi1p.add_MPS(-1*avgphi1p, inplace=False, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(gamma1_mps, temp, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    F.add_MPS(d_dphi_1.apply(temp, **COMP_OPTS), inplace=True, **COMP_OPTS)
    
    # Ditto for phi_2 component
    temp = phi2p.add_MPS(-1*avgphi2p, inplace=False, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(gamma2_mps, temp, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = d_dphi_2.apply(temp, **COMP_OPTS)
    F.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Need to add the source term - sum_i d_dphi_i [ delta_i H p]
    temp = mps.mpsHadamardZipupProd(delta1_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    sourceTerm = d_dphi_1.apply(temp, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(delta2_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = d_dphi_2.apply(temp, **COMP_OPTS)
    sourceTerm.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Add source term into F
    F.add_MPS(-1*sourceTerm, inplace=True, **COMP_OPTS)
    
    # Finish up by multiplying F with dt and add to p_0_mps
    F.multiply(dt, spread_over="all",inplace=True)
    p_1_mps = F.add_MPS(p_0_mps, inplace=False, **COMP_OPTS)
        
    # And finally, return.    
    return p_1_mps, F