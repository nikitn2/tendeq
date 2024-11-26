
# -*- coding: utf-8 -*-
import numpy as np
import quimb as qu
import config as cfg

import TnMachinery.MPSfunctions as mps
import TnMachinery.genTNfunctions as fgtn
import PDEs.FokkerPlanck.FokkerPlanckDistrs as fpDis

"""
Here are various functions for calculating p(x,t) solutions of Peyman's case 4 (2D spatial, 2D reactant) Fokker-Planck equation.
It basically represents a simple fuel-oxidiser mixture. 
"""

# Creates the two function:
# reac_1 = - Const*phi1     and
# reac_2 = - Const*phi1

def initLin1ReacTerm2D(NK, K_left, split, c, Const):
    
    # Initialise
    dphi = c/2.0**NK
    
    # Get linear-phis in mps form and kron them together
    linL_mps = mps.constDistrLinear(NK, K_left, 0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    uniR_mps = mps.constDistrUniformFP(NK)
    prod_mps = mps.mpsKron(linL_mps,uniR_mps)
    
    # Interleave if necessary
    if split is False: prod_mps = mps.mpsInterleave(mps = prod_mps, K = K_left+2)
    
    # Scale reac_mps with -Const to get reac1, reac2
    reac1_mps = prod_mps.multiply(-Const, spread_over="all", inplace=False)
    reac2_mps = prod_mps.multiply(-Const, spread_over="all", inplace=False)
    
    return reac1_mps, reac2_mps

# Creates the two function:
# reac_1 = - Da*phi1*phi2,     and
# reac_2 = - r*Da*phi1*phi2.

def initLinLinReacTerm2D(NK, K_left, split, c, consts):
    
    # Initialise
    r = consts[0]
    Da= consts[1]
    dphi = c/2.0**NK
    
    # Get linear-phis in mps form and kron them together
    linL_mps = mps.constDistrLinear(NK, K_left, 0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    linR_mps = mps.constDistrLinear(NK, 0,      0, True, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    prod_mps = mps.mpsKron(linL_mps,linR_mps)
    
    # Interleave if necessary
    if split is False: prod_mps = mps.mpsInterleave(mps = prod_mps, K = K_left+2)
    
    # Scale reac_mps with -Da, -r*Da to get reac1, reac2
    reac1_mps = prod_mps.multiply(-Da, spread_over="all", inplace=False)
    reac2_mps = prod_mps.multiply(-r*Da, spread_over="all", inplace=False)
    
    return reac1_mps, reac2_mps

# Creates the two functions
# reac_1/r0 = - phi1*exp(-Ze/phi2),     and
# reac_2/r0 = - Q*reac_1.

# NOTE: This initialisation is exponentially expensive in NK,
# but it's OK because this can in theory be done in poly(NK) cost with TT-cross

def initLinExponReacTerm2D(NK, K_left, split, c, consts):
    
    # Initialise
    r0 = -consts[0]
    Q  =  consts[1]
    Ze =  consts[2]
    dphi = c/2.0**NK
    
    # Get linear-phi in mps form
    reac1_mps = mps.constDistrLinear(NK, K_left, 0, split, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    
    # Get the exponential function in mps form
    phi2        = np.linspace(cfg.phiStart,c+cfg.phiStart-dphi,2.0**NK) #NikPhi
    phi2[0]     = np.nan
    exp_phi2     = r0*np.exp(-Ze/phi2); exp_phi2[0] = 0
    exp_phi2_mps = mps.mpsDecompFlow1D_timestep(exp_phi2)
    
    # Now combine these two to get phi1*exp(-Ze/phi2)
    
    # First rename site indices and legs appropriately
    exp_phi2_mps = mps.shiftIndextagMps(exp_phi2_mps, Nextra = (K_left+1)*NK)
    
    # Now kron-in expPhi_mps
    reac1_mps &= exp_phi2_mps
    qu.tensor.tensor_core.new_bond( reac1_mps["s{}".format( (K_left+1)*NK -1 )], reac1_mps["s{}".format( (K_left+1)*NK )] )
    reac1_mps._L += NK
    
    # Interleave if necessary
    if split is False: reac1_mps = mps.mpsInterleave(mps = reac1_mps, K = K_left+2)
    
    # Multiply reac1_mps by Q to get reac2_mps
    reac2_mps = reac1_mps.multiply(Q, spread_over="all", inplace=False)
    
    return reac1_mps, reac2_mps

def timestepFP_distributions_spat2Dreac2D(NK,l, c, split, const_alpha, const_beta, consts_gammas, consts_deltas, TG_pars, concntrBoundryCond, initialDistrFP, kwargs_initialDistrFP):
    
    # Initialise
    spatialBoundryCond = "periodic"
    #concntrBoundryCond = "homogeneous" 
    #concntrBoundryCond = "normpreserving"
    dx = l/2.0**NK
    dphi = c/2.0**NK
    
    # Produce the differentiation MPOs
    d_dx1    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 0,      NpadRight=NK*3, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx2    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK,     NpadRight=NK*2, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx1Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = 0,      NpadRight=NK*3, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx2Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = NK,     NpadRight=NK*2, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dphi_1    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 2*NK, NpadRight=NK,   h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    d_dphi_2    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 3*NK, NpadRight=0,    h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dphi_1Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = 2*NK, NpadRight=NK,   h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dphi_2Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = 3*NK, NpadRight=0,    h=dphi, boundryCond=concntrBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    differentiators = [d_dx1, d_dx2, dd_dx1Sq, dd_dx2Sq,d_dphi_1,d_dphi_2,dd_dphi_1Sq,dd_dphi_2Sq ]
    
    # Produce each of the MPSs for the FP equation in the case of 2 spatial dims and two reactants
    
    # First initial flow field
    pInit_mps = initialDistrFP(NK=NK, split=split, l=l, c=c, **kwargs_initialDistrFP)
    
    # Then alpha, beta and gamma MPSs
    uni4K = mps.constDistrUniformFP(NK*4)
    alpha_mps  = uni4K.multiply(const_alpha,        inplace=False, spread_over="all" )
    beta_mps   = uni4K.multiply(const_beta,         inplace=False, spread_over="all" )
    gamma1_mps = uni4K.multiply(consts_gammas[0],   inplace=False, spread_over="all" )
    gamma2_mps = uni4K.multiply(consts_gammas[1],   inplace=False, spread_over="all" )
    
    # Delta MPSs are a bit special. First initialise phis
    #dphi = c/2.0**NK
    phi1_mps = mps.constDistrLinear(NK=NK, K_left=2, K_right=1, split = split, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    phi2_mps = mps.constDistrLinear(NK=NK, K_left=3, K_right=0, split = split, pars= [0.0+cfg.phiStart,c-dphi+cfg.phiStart]) #NikPhi
    
    # Now construct the delta
    delta1_mps, delta2_mps = initLinLinReacTerm2D(NK = NK, K_left = 2, split= split, c=c, consts=consts_deltas)
    
    # The Taylor-Green vortex flow fields
    u1_mps, u2_mps = fpDis.flowFieldTaylorGreenVortex(NK, K_padLeft=0, K=2, K_padRight=2, split=split, l=l, pars=TG_pars)
    
    return pInit_mps, alpha_mps, beta_mps, gamma1_mps, gamma2_mps, delta1_mps, delta2_mps, phi1_mps, phi2_mps, u1_mps, u2_mps, differentiators

# Outputs “p_1 = p_0 + dt*F(p_0p5_mps)
# F(p) = sum_i { 
#      - u_iHdp/dx_i 
#      + dd/dx_i^2(alphaHp ) 
#      }  
#      + sum_j { 
#        dd/dphi_j^2 [betaHp]                     <-- Artificial dissipation term
#      + d/dphi_j [gamma_jH( {phi_j} - <Phi_j>)Hp]  
#      - d/dphi_j [delta_jHp]
#      }
def timestepFP_init_spat2Dreac2D(p_0_mps, p_0p5_mps, l, c, dt, chi, split, alpha_mps, beta_mps, gamma1_mps,  gamma2_mps, delta1_mps, delta2_mps, u1_mps, u2_mps, phi1_mps, phi2_mps, differentiators, t = None, tol_compression = 1e-16, tol_bond_zipup = 1e-16):    
    
    # Initialise
    K = 4 # x_1, x_2, phi_1, phi_2
    N = p_0p5_mps.L
    NK = N//K
    chiHadInter = 4*chi
    chiOpsInter = 8*chi
    dx = l/2.0**NK
    dphi = c/2.0**NK
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :tol_compression, "max_bond" :chiOpsInter, "method" : 'svd'}#'isvd', "renorm" : True }
    
    # # Compute the norm of p_0p5_mps at every point in x-space, and use it to calculate how much of the pdf has “leaked” out of the domain. This will later be used as the left BC in phi-space.
    # flat_mps = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*N,site_tag_id='s{}')
    # norm_x = mps.averageMpsAccrossLastKDimensions(p_0p5_mps, NK, split, dphi**2, 2, padBack=True)
    # lostNorm = flat_mps.add_MPS(-1*norm_x, inplace=False)
    # deltaLostNorm_phi1 = mps.mpsHadamardZipupProd(createDelta_TOTEST(NK,K,3,split),lostNorm)
    # deltaLostNorm_phi2 = mps.mpsHadamardZipupProd(createDelta_TOTEST(NK,K,4,split),lostNorm)
    
    # Extract the differentiators
    d_dx1       = differentiators[0]
    d_dx2       = differentiators[1]
    dd_dx1Sq    = differentiators[2]
    dd_dx2Sq    = differentiators[3]
    d_dphi_1    = differentiators[4]
    d_dphi_2    = differentiators[5]
    dd_dphi_1Sq = differentiators[6]
    dd_dphi_2Sq = differentiators[7]    
    
    # First compute convection= -sum_k{u_kHdp_0p5/dx}
    dp_dx_1 = d_dx1.apply(p_0p5_mps, **COMP_OPTS)
    convection = mps.mpsHadamardZipupProd(u1_mps, dp_dx_1, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    dp_dx_2 = d_dx2.apply(p_0p5_mps, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(u2_mps, dp_dx_2, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    convection.add_MPS(temp, inplace=True, **COMP_OPTS)
    convection.multiply(-1.0,inplace=True)
    
    # Now compute diffusion = sum_k{ dd/dxSq_k[ alpha (.) ] } + dd/dphiSq [gamma(.)]
    
    # Begin with x1 & x2 diffusion
    temp = mps.mpsHadamardZipupProd(alpha_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    diffusion = dd_dx1Sq.apply(temp, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(alpha_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = dd_dx2Sq.apply(temp, **COMP_OPTS)
    diffusion.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Then phi_1, phi_2 diffusion
    betaHp = mps.mpsHadamardZipupProd(beta_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = dd_dphi_1Sq.apply(betaHp, **COMP_OPTS)
    diffusion.add_MPS(temp, inplace=True, **COMP_OPTS)
    temp = dd_dphi_2Sq.apply(betaHp, **COMP_OPTS)
    diffusion.add_MPS(temp, inplace=True, **COMP_OPTS)
    
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
    F = d_dphi_1.apply(temp, **COMP_OPTS)
    
    # Ditto for phi_2 component
    temp = phi2p.add_MPS(-1*avgphi2p, inplace=False, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(gamma2_mps, temp, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = d_dphi_2.apply(temp, **COMP_OPTS)
    F.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Now add the convection and diffusion terms into F
    F.add_MPS(convection, inplace=True, **COMP_OPTS)
    F.add_MPS(diffusion,  inplace=True, **COMP_OPTS)
    
    # # Add the - d/dphi_j [epsilon_jH(phi_j)Hp] term for “pushing” the pdf out of the phi1,phi2<0 regions.
    # dphi1p_dhi1 = d_dphi_1.apply(phi1p, compress=True, cutoff=1e-16)
    # dphi2p_dhi2 = d_dphi_2.apply(phi2p, compress=True, cutoff=1e-16)
    # F.add_MPS(mps.mpsHadamardZipupProd(epsilon1_mps, dphi1p_dhi1, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup), inplace=True, compress=True, cutoff=1e-16)
    # F.add_MPS(mps.mpsHadamardZipupProd(epsilon2_mps, dphi2p_dhi2, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup), inplace=True, compress=True, cutoff=1e-16)
    
    # Need to add the source term - sum_i d_dphi_i [ delta_i H p]
    temp = mps.mpsHadamardZipupProd(delta1_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    sourceTerm = d_dphi_1.apply(temp, **COMP_OPTS)
    temp = mps.mpsHadamardZipupProd(delta2_mps, p_0p5_mps, max_intermediateBond = chiHadInter, max_finalBond = chi, tol_bond = tol_bond_zipup)
    temp = d_dphi_2.apply(temp, **COMP_OPTS)
    sourceTerm.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Add interaction term into F
    F.add_MPS(-1*sourceTerm, inplace=True, **COMP_OPTS)
    
    # Finish up by multiplying F with dt and add to p_0_mps
    F.multiply(dt, spread_over="all",inplace=True)
    p_1_mps = F.add_MPS(p_0_mps, inplace=False, **COMP_OPTS)
        
    # And finally, return.    
    return p_1_mps, F

# Computes the overlap between d<phi_i>/dt and F_i, with:
# F_i = sum_j{
#             - u_j d<phi_i>/dx_j + d^2/dx_j^2 [alphaH<phi_i>] + <delta_i>
#             }, i,j \in {1,2}.
# The d/dt is done at the midpoint, at t, and the averaging happens
# over phi1, phi2, and also at time t.

def conscCheck_firstMomentEqFP_spat2Dreac2D(p_tmdt, p_t, p_tpdt, l, c, dt, split, alpha_mps, delta1_mps, delta2_mps, u1_mps, u2_mps, phi1_mps, phi2_mps, tol_bond_zipup = 1e-16):    
    
    # Initialise
    K = 4 # x_1, x_2, phi_1, phi_2
    N = p_tmdt.L
    NK = N//K
    dx = l/2.0**NK
    dphi = c/2.0**NK
    spatialBoundryCond = "periodic"
    
    # Produce the spatial differentiation MPOs
    d_dx1    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = 0,      NpadRight=NK*3, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x1',   upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx1Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = 0,      NpadRight=NK*3, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x1',   upIndIds="k{}", downIndIds="b{}",split=split)
    d_dx2    = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK,     NpadRight=NK*2, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x2',   upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx2Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = NK,     NpadRight=NK*2, h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x2',   upIndIds="k{}", downIndIds="b{}",split=split)

    # First compute <phi1> & <phi2> at time t.
    phi1p = mps.mpsHadamardZipupProd(p_t, phi1_mps, tol_bond = tol_bond_zipup)
    phi2p = mps.mpsHadamardZipupProd(p_t, phi2_mps, tol_bond = tol_bond_zipup)
    avgphi1 = mps.averageMpsAccrossLastKDimensions(phi1p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi2 = mps.averageMpsAccrossLastKDimensions(phi2p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    
    # And do the same for <delta1> & <delta2>
    delta1p = mps.mpsHadamardZipupProd(p_t, delta1_mps, tol_bond = tol_bond_zipup)
    delta2p = mps.mpsHadamardZipupProd(p_t, delta2_mps, tol_bond = tol_bond_zipup)
    avgdelta1 = mps.averageMpsAccrossLastKDimensions(delta1p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgdelta2 = mps.averageMpsAccrossLastKDimensions(delta2p, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    
    # Compute the convective terms u_i d<phi_i>/dx_i
    F1_mps =        -1*mps.mpsHadamardZipupProd( u1_mps, d_dx1.apply( avgphi1 ), tol_bond = tol_bond_zipup)
    F1_mps.add_MPS( -1*mps.mpsHadamardZipupProd( u2_mps, d_dx2.apply( avgphi1 ), tol_bond = tol_bond_zipup ), inplace=True, compress=True, cutoff=1e-16 )
    F2_mps =        -1*mps.mpsHadamardZipupProd( u1_mps, d_dx1.apply( avgphi2 ), tol_bond = tol_bond_zipup)
    F2_mps.add_MPS(- 1*mps.mpsHadamardZipupProd( u2_mps, d_dx2.apply( avgphi2 ), tol_bond = tol_bond_zipup ), inplace=True, compress=True, cutoff=1e-16)
    
    # Now do the diffusive term
    F1_mps.add_MPS( dd_dx1Sq.apply( mps.mpsHadamardZipupProd(alpha_mps, avgphi1, tol_bond = tol_bond_zipup) ), inplace=True, compress=True, cutoff=1e-16 )
    F1_mps.add_MPS( dd_dx2Sq.apply( mps.mpsHadamardZipupProd(alpha_mps, avgphi1, tol_bond = tol_bond_zipup) ), inplace=True, compress=True, cutoff=1e-16 )
    F2_mps.add_MPS( dd_dx1Sq.apply( mps.mpsHadamardZipupProd(alpha_mps, avgphi2, tol_bond = tol_bond_zipup) ), inplace=True, compress=True, cutoff=1e-16 )
    F2_mps.add_MPS( dd_dx2Sq.apply( mps.mpsHadamardZipupProd(alpha_mps, avgphi2, tol_bond = tol_bond_zipup) ), inplace=True, compress=True, cutoff=1e-16 )
    
    # Complete the F_i terms by adding-in the reactive terms (deltas)
    F1_mps.add_MPS(avgdelta1, inplace=True, compress=True, cutoff=1e-16)
    F2_mps.add_MPS(avgdelta2, inplace=True, compress=True, cutoff=1e-16)
    
    # Finally, get d<phi_i>/dt terms and compute overlaps
    phi1p_prev = mps.mpsHadamardZipupProd(p_tmdt, phi1_mps, tol_bond = tol_bond_zipup)
    phi1p_next = mps.mpsHadamardZipupProd(p_tpdt, phi1_mps, tol_bond = tol_bond_zipup)
    phi2p_prev = mps.mpsHadamardZipupProd(p_tmdt, phi2_mps, tol_bond = tol_bond_zipup)
    phi2p_next = mps.mpsHadamardZipupProd(p_tpdt, phi2_mps, tol_bond = tol_bond_zipup)
    #
    avgphi1_prev = mps.averageMpsAccrossLastKDimensions(phi1p_prev, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi1_next = mps.averageMpsAccrossLastKDimensions(phi1p_next, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi2_prev = mps.averageMpsAccrossLastKDimensions(phi2p_prev, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    avgphi2_next = mps.averageMpsAccrossLastKDimensions(phi2p_next, NK, split, dvolToAvg= dphi**2, KtoAvg=2, padBack = True)
    #
    davgphi1_dt = avgphi1_next.add_MPS(-1*avgphi1_prev, inplace=False, compress=True, cutoff=1e-16)/(2*dt)
    davgphi2_dt = avgphi2_next.add_MPS(-1*avgphi2_prev, inplace=False, compress=True, cutoff=1e-16)/(2*dt)
    
    overlap1 = fgtn.tnFidelity(davgphi1_dt, F1_mps)
    overlap2 = fgtn.tnFidelity(davgphi2_dt, F2_mps)
    print(overlap1, overlap2, davgphi1_dt.norm(), F1_mps.norm(), davgphi2_dt.norm(), F2_mps.norm())
    
    return overlap1, overlap2