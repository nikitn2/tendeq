#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import quimb as qu
import numpy as np
from config import phiStart

import PDEs.FokkerPlanck.FokkerPlanckGeneral as fpg
import TnMachinery.MPSfunctions as mps

"""
Here lie general functions required for solving the Fokker-Planck equation. 
In these functions, the Focker-Planck pdf p is assumed to be of the form 
p = p(x_1, x_2, ..., x_{K_x}, phi_1 phi_2, ... phi_{K_{\phi}}), with x_1 corresponding
to the first index, and phi_{K_{\phi}} to the last. 
"""

def constDistr_linear(x, pars):
    
    # Initialise
    a = pars[0]; b = pars[1]
    _, xx =  np.meshgrid(x, x)
    
    yy = a*xx + b
    
    return yy

def constDistr_uniform(x, pars):
    
    # Initialise
    A = pars[0]
    _,xx = np.meshgrid(x, x)
    
    uu = xx*0 + A
    
    return uu

def initialDistr_doubleDelta(x, pars):
    
    # Initialise
    x0 = pars[0]; x1 = pars[1]
    A0 = pars[2]; A1 = pars[3]
    M = len(x)
    
    loc0 = int(x0/x[-1]*M)
    loc1 = int(x1/x[-1]*M)
    
    d0 = 0*x; d0[loc0] = A0
    d1 = 0*x; d1[loc1] = A1
    
    d01 = d0 + d1
        
    return d01

def initialDistr_doubleGaussian(x, pars):
    
    # Initialise
    x0 = pars[0]; x1 = pars[1]
    sigma0 = pars[2]; sigma1 = pars[3]
    A0 = pars[4]; A1 = pars[5]
    
    f0 = A0/(sigma0*np.sqrt(2*np.pi))*np.exp(-0.5*( (x-x0)/sigma0 )**2)
    f1 = A1/(sigma1*np.sqrt(2*np.pi))*np.exp(-0.5*( (x-x1)/sigma1 )**2)
    
    f01 = f0 + f1
        
    return f01

# Creates a initial state whose pdf is a symmetric Gaussian in the spatial coords, 
# tensored with another Gaussian symmetric in the phi coordinates.
# Mean and std is given by spatialGussian_pars & concGaussian_pars
def initialDistrFP_gaussianSpaceGaussConc(NK, split, l, c, K_x, K_phi, spatialGaussian_pars, concGaussian_pars):
    
    # Initialise
    siteTagId = "s{}"
    dx   = l/2**NK
    dphi = c/2**NK
        
    # First create a gaussian spatial-distribution mps. WARNING: Inefficient step; costs ~exp(NK). But OK to do in theory because gaussians are very low-chi.
    x = np.linspace(0,l-dx,2**NK)
    v = initialDistr_doubleGaussian(x, [spatialGaussian_pars[0], spatialGaussian_pars[1], spatialGaussian_pars[2], spatialGaussian_pars[3], 0.5, 0.5] )
    gaussX_mps = mps.mpsDecompFlow1D_timestep(v)
    
    # Now create a gaussian phi-distribution mps. WARNING: Inefficient step; costs ~exp(NK). But OK to do in theory because gaussians are very low-chi.
    phi = np.linspace(0+phiStart,c+phiStart-dphi,2**NK) #NikPhi
    v = initialDistr_doubleGaussian(phi, [concGaussian_pars[0], concGaussian_pars[1], concGaussian_pars[2], concGaussian_pars[3], 0.5, 0.5] )
    gaussPhi_mps = mps.mpsDecompFlow1D_timestep(v)
   
    # Tensor the mpses K_x, then K_phi times
    initDistr_mps = gaussX_mps.copy()
    for k in range(1,K_x):
        # First rename site indices and legs of gaussX_mps appropriately
        temp = mps.shiftIndextagMps(gaussX_mps, Nextra = k*NK)
        
        # Now kron-in temp
        initDistr_mps &= temp
        qu.tensor.tensor_core.new_bond( initDistr_mps[siteTagId.format( k*NK -1 )], initDistr_mps[siteTagId.format( k*NK )] )
        initDistr_mps._L += NK
    for k in range(0,K_phi):
        # First rename site indices and legs of gaussPhi_mps appropriately
        temp = mps.shiftIndextagMps(gaussPhi_mps, Nextra = NK*K_x + k*NK)
        
        # Now kron-in temp
        initDistr_mps &= temp
        qu.tensor.tensor_core.new_bond( initDistr_mps[siteTagId.format(NK*K_x + k*NK -1)], initDistr_mps[siteTagId.format(NK*K_x + k*NK )] )
        initDistr_mps._L += NK
    
    # Interleave if necessary
    if split is False: initDistr_mps = mps.mpsoInterleave(mps = initDistr_mps, K = K_x+K_phi)
    
    return initDistr_mps

# Creates a initial K_phi = 2 state where the middle half (along x_1) contains the 
# phi_1 reactant while the top & bottom quarters contains the phi_2 reactant. The initial 
# pdf of this situation is created by defining two sub-pdfs, each of them uniform 
# in the middle and outer halves of the spatial domain, and respectively Gaussian in 
# the phi_1 & phi_2 domains, with the parameters of the Gaussians given through gaussian_pars.

def initialDistrFP_outerInnerMixtureSpatKDConsc2D(NK, K_x, split, c, gaussian_pars, l= None, forcedNorm = True):
    
    # Initialise
    siteTagId = "s{}"
    siteIndId="k{}"
    dphi = c/2**NK
        
    # First create the “x1 – x2 – ... – xK_x” mps for middle & outer uniform spatial distributions
    temp1 = qu.tensor.tensor_builder.MPS_product_state(arrays = [np.array([1.0,0.0])] + [np.array([0.0,1.0])] + [np.array([1.0,1.0])]*(NK*K_x-2),site_tag_id=siteTagId, site_ind_id=siteIndId)
    temp2 = qu.tensor.tensor_builder.MPS_product_state(arrays = [np.array([0.0,1.0])] + [np.array([1.0,0.0])] + [np.array([1.0,1.0])]*(NK*K_x-2),site_tag_id=siteTagId, site_ind_id=siteIndId)
    uniMiddleHalfSpace_mps = temp1.add_MPS(temp2, inplace=False, compress=True, cutoff=1e-16)
    
    temp1 = qu.tensor.tensor_builder.MPS_product_state(arrays = [np.array([1.0,0.0])] + [np.array([1.0,0.0])] + [np.array([1.0,1.0])]*(NK*K_x-2),site_tag_id=siteTagId, site_ind_id=siteIndId)
    temp2 = qu.tensor.tensor_builder.MPS_product_state(arrays = [np.array([0.0,1.0])] + [np.array([0.0,1.0])] + [np.array([1.0,1.0])]*(NK*K_x-2),site_tag_id=siteTagId, site_ind_id=siteIndId)
    uniOuterHalfSpace_mps = temp1.add_MPS(temp2, inplace=False, compress=True, cutoff=1e-16)
    
    # Now create a generic gaussian phi-distribution mps. WARNING: Inefficient step; costs ~exp(NK). But OK to do in theory because gaussians are very low-chi.
    phi = np.linspace(0+phiStart,c+phiStart-dphi,2**NK) #NikPhi
    vHigh = initialDistr_doubleGaussian(phi, [gaussian_pars[0][0], gaussian_pars[0][1], gaussian_pars[0][2], gaussian_pars[0][3], 0.5, 0.5] )
    gaussHighPhi_mps= mps.mpsDecompFlow1D_timestep(vHigh)
    vLow = initialDistr_doubleGaussian(phi, [gaussian_pars[1][0], gaussian_pars[1][1], gaussian_pars[1][2],  gaussian_pars[1][3], 0.5, 0.5]  )
    gaussLowPhi_mps = mps.mpsDecompFlow1D_timestep(vLow)
    
    # Force the Gaussians to be normalised to 1 if requested
    if forcedNorm is True:
        flat_mps = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*NK,site_tag_id='s{}')
        normHigh = (flat_mps @ gaussHighPhi_mps)*dphi
        normLow  = (flat_mps @ gaussLowPhi_mps )*dphi
        gaussHighPhi_mps.multiply(1/normHigh, inplace=True, spread_over="all" )
        gaussLowPhi_mps.multiply( 1/normLow,  inplace=True, spread_over="all" )
    
    # Tensor the gaussian mpses K_phi times with the uniform mpses
    if K_x >= 2:
        initDistrMiddles_mps = mps.mpsKron(mps.mpsKron(uniMiddleHalfSpace_mps, gaussHighPhi_mps), gaussLowPhi_mps)
        initDistrOuter_mps   = mps.mpsKron(mps.mpsKron(uniOuterHalfSpace_mps,  gaussLowPhi_mps),  gaussHighPhi_mps)
    if K_x == 0:
        initDistrMiddles_mps = .5*mps.mpsKron(gaussHighPhi_mps, gaussLowPhi_mps)
        initDistrOuter_mps   = .5*mps.mpsKron(gaussLowPhi_mps,  gaussHighPhi_mps)
    
    # Now add-up the two pdfs.
    initDistr_mps = initDistrMiddles_mps.add_MPS(initDistrOuter_mps, inplace=False, compress=True, cutoff=1e-16)
        
    # Perform smoothing
    if K_x>0: initDistr_mps = fpg.spatialSmoothing(initDistr_mps, K_x, K_phi=2, l=1, smoothFac=0.1/4**NK, split=True, chi=64, spatialBoundryCond="periodic", apps=40)    
    
    # Interleave if necessary
    if split is False: initDistr_mps = mps.mpsoInterleave(mps = initDistr_mps, K = K_x+2)
    
    return initDistr_mps

# Creates a initial state whose pdf is an uncorrelated K_phi-dimensional 
# Gaussian in phi-space.
def initialDistrFP_uncorrelatedGaussConc(NK, K_phi, split, c, mus, sigmas, forcedNorm = True):
    
    # Initialise
    dphi = c/2**NK
            
    for k in range(0,K_phi):
        phi = np.linspace(0+phiStart,c+phiStart-dphi,2**NK) #NikPhi
        v = initialDistr_doubleGaussian(phi, [mus[k], mus[k], sigmas[k], sigmas[k], 0.5, 0.5] )
        gaussPhi_mps = mps.mpsDecompFlow1D_timestep(v)
        
        if k == 0: initDistr_mps = gaussPhi_mps
        else:
            # Keep kroning-in Gaussians
            initDistr_mps = mps.mpsKron(initDistr_mps, gaussPhi_mps)

    # Force the Gaussian to be normalised to 1 if requested
    if forcedNorm is True:
        flat_mps = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*NK*K_phi,site_tag_id='s{}')
        normCurr = (flat_mps @ initDistr_mps)*dphi**K_phi
        initDistr_mps.multiply(1/normCurr, inplace=True, spread_over="all" )

    # Interleave if necessary
    if split is False: initDistr_mps = mps.mpsoInterleave(mpso = initDistr_mps, K = K_phi, NK = NK)
    
    return initDistr_mps

def flowField_TaylorGreenVortexAndJet(NK, K_padLeft, K, K_padRight, split, l, pars ):
    
    # Initialise
    A=pars[0]; B=pars[1]; C=pars[2]
    a=pars[3]; b=pars[4]; c=pars[5]
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :1e-15, "method" : 'svd'}#'isvd', "renorm" : True }

    
    # First create the x coords
    x = np.linspace(0, l*(1-2**-NK), 2**NK)
        
    # Now produce the Taylor-Green vortex building blocks
    Acosax1_mps = mps.mpsDecompFlow1D_timestep(A*np.cos(a*x)); sinbx2_mps = mps.mpsDecompFlow1D_timestep(np.sin(b*x)); sincx3_mps = mps.mpsDecompFlow1D_timestep(np.sin(c*x))
    Bsinax1_mps = mps.mpsDecompFlow1D_timestep(B*np.sin(a*x)); cosbx2_mps = mps.mpsDecompFlow1D_timestep(np.cos(b*x))
    Csinax1_mps = mps.mpsDecompFlow1D_timestep(C*np.sin(a*x));                                                         coscx3_mps = mps.mpsDecompFlow1D_timestep(np.cos(c*x))
    
    # And the jet ones
    A_mps    = mps.mpsDecompFlow1D_timestep(A + 0*x)
    exp_mps  = mps.mpsDecompFlow1D_timestep(np.exp( -0.5*(x-l/2)**2/(l/6)**2 ))
    
    # Create the actual MPSs
    if K==2:
        
        Acosax1sinbx2_mps = mps.mpsKron(Acosax1_mps, sinbx2_mps)
        Bsinax1cosbx2_mps = mps.mpsKron(Bsinax1_mps, cosbx2_mps)
        
        # And pad to the left & right using uniform mpses
        if K_padLeft >0: 
            uniLeft_mps = mps.constDistrUniformFP(NK*K_padLeft)
            Acosax1sinbx2_mps = mps.mpsKron(uniLeft_mps, Acosax1sinbx2_mps)
            Bsinax1cosbx2_mps = mps.mpsKron(uniLeft_mps, Bsinax1cosbx2_mps)
            
        if K_padRight>0: 
            uniRight_mps= mps.constDistrUniformFP(NK*K_padRight)
            Acosax1sinbx2_mps = mps.mpsKron(Acosax1sinbx2_mps, uniRight_mps)
            Bsinax1cosbx2_mps = mps.mpsKron(Bsinax1cosbx2_mps, uniRight_mps)
        
        if split == False:
            Acosax1sinbx2_mps = mps.mpsoInterleave(mps = Acosax1sinbx2_mps, K = K + K_padLeft + K_padRight)
            Bsinax1cosbx2_mps = mps.mpsoInterleave(mps = Bsinax1cosbx2_mps, K = K + K_padLeft + K_padRight)
        
        return Acosax1sinbx2_mps, Bsinax1cosbx2_mps
        
    elif K==3:
        
        U1_mps = mps.mpsKron(mps.mpsKron(Acosax1_mps, sinbx2_mps),sincx3_mps)
        U2_mps = mps.mpsKron(mps.mpsKron(Bsinax1_mps, cosbx2_mps),sincx3_mps)
        U3_mps = mps.mpsKron(mps.mpsKron(Csinax1_mps, sinbx2_mps),coscx3_mps)
        
        jet1_mps = -1*mps.mpsKron(A_mps, mps.mpsKron(exp_mps, exp_mps) )
        U1_mps.add_MPS(jet1_mps, inplace=True, **COMP_OPTS)
        
        if K_padLeft>0: 
            uniLeft_mps = mps.constDistrUniformFP(NK*K_padLeft)
            U1_mps = mps.mpsKron(uniLeft_mps, U1_mps)
            U2_mps = mps.mpsKron(uniLeft_mps, U2_mps)
            U3_mps = mps.mpsKron(uniLeft_mps, U3_mps)
        
        if K_padRight>0:
            uniRight_mps= mps.constDistrUniformFP(NK*K_padRight)
            U1_mps = mps.mpsKron(U1_mps, uniRight_mps)
            U2_mps = mps.mpsKron(U2_mps, uniRight_mps)
            U3_mps = mps.mpsKron(U3_mps, uniRight_mps)
        
        if split == False:
            U1_mps = mps.mpsoInterleave(mps = U1_mps, K = K + K_padLeft + K_padRight)
            U2_mps = mps.mpsoInterleave(mps = U2_mps, K = K + K_padLeft + K_padRight)
            U3_mps = mps.mpsoInterleave(mps = U3_mps, K = K + K_padLeft + K_padRight)
        
        return U1_mps, U2_mps, U3_mps
    
    else:
        print("Error! K must be either 2 or 3. Exiting.")
        return -1

# NOTE: This function uses an inefficient algorithm to generate the initial
# distributions, but it's OK to do this as long as these distributions can be
# efficiently generated in theory.
def timestepFPKD_distributions_initGaussLinM_func(NK, K_phi, split, c, concGaussian_pars, lin_pars, chiGauss = None):
    
    # Initialise
    alpha= lin_pars[0]
    beta = lin_pars[1]
    dphi = c/2.0**NK
    
    # Produce each of the MPSs for the KD FP equation    
    
    # Initial condition f(phi1,phi2,...,phiK)
    mus    = concGaussian_pars[0:K_phi]
    sigmas = concGaussian_pars[K_phi:]
    f_mps = initialDistrFP_uncorrelatedGaussConc(NK, K_phi, split, c, mus, sigmas)
    if chiGauss is not None: f_mps.compress(max_bond=chiGauss)
    
    # Linear drift terms m_k(x,t) = x_k
    m_mpses = []
    for k in range(0,K_phi): m_mpses += [mps.constDistrLinear(NK, k, K_phi-k-1, split, pars= [alpha*phiStart + beta, alpha*(c-dphi+phiStart)+beta ])] #NikPhi
    
    return f_mps, m_mpses

# Requires Nt >= NK
def create_delta_tEq0(NK, Nt, K, siteTagId = "s{}", siteIndId="k{}", split=True):
    
    if Nt < NK: print("Nt must be greater or equal to NK. Exiting."); return -1

    delta_tEq0 = qu.tensor.tensor_builder.MPS_product_state(arrays = [np.array([1.0,1.0])]*NK*K + [np.array([1.0,1.0e-40])]*Nt,site_tag_id=siteTagId, site_ind_id=siteIndId)
    
    if split == False: delta_tEq0 = mps.mpsoInterleave(delta_tEq0, K = K, NK = NK, Nt = Nt)
    
    return delta_tEq0