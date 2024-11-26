#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import quimb as qu
import numpy as np
import time
import warnings

import TnMachinery.MPSfunctions as mps
import PDEs.FokkerPlanck.FokkerPlanckDistrs as fpDis

import config as cfg
import sys

"""
Here lie general functions required for solving the Fokker-Planck equation. 
In these functions, the Focker-Planck pdf p is assumed to be of the form 
p = p(x_1, x_2, ..., x_{K_x}, v_1, v_2, ..., v_{K_v}, phi_1 phi_2, ... phi_{K_{\phi}}),
# with x_1 corresponding to the first index, and phi_{K_{\phi}} to the last. 
"""

# v0_mps is here the t=0 mps defined in x,v,phi space only
def prepDmrgFP(v0_mps, K, NK, Nt, solveDmrgpFP_pde, kwargs_solveDmrgpFP_pde, split):
    
    # Initialise
    if split == True: v_mps = mps.mpsKron(v0_mps, fpDis.create_delta_tEq0(0, Nt, 0, split = True))
    else:
        # It is assumed v0_mps is interleaved. Ensure it stays as such.
        v_mps = mps.mpsoRevInterleave(v0_mps, K, NK, 0)
        v_mps = mps.mpsKron(v0_mps, fpDis.create_delta_tEq0(0, Nt, 0, split = True))
        v_mps = mps.mpsoInterleave(v_mps, K, NK, Nt)
        v_mps.compress(form=0, cutoff_mode = "rel", cutoff = 1e-15)
    
    # Produce the terms needed to compute the cost function.
    Ham = solveDmrgpFP_pde(v_mps, Nt, split, **kwargs_solveDmrgpFP_pde )

    return v_mps, Ham

def solveDmrgFP(v_mps, Ham, K, Nt, split, chi, uu_ket_var = None, maxSweeps = 20, eps_DMRG = 1e-08, numLocIters=2000, eigLocTol = 1e-12, eigLocVec = 8):
        
    # Initialise
    N = Ham.L
    NK= (N-Nt)//K
    
    # Initial guess
    if uu_ket_var is None:
        uu_ket_var = 1e-6*qu.tensor.tensor_builder.MPS_rand_state(L=N, bond_dim=1, phys_dim=2,site_tag_id='s{}')
        uu_ket_var.add_MPS(v_mps,inplace=True, compress=True, cutoff=1e-16)
    
    # Identify groundstate using quimb's DMRG
    dmrg = qu.tensor.tensor_dmrg.DMRG2(ham=Ham, p0 = uu_ket_var)
    dmrg.opts["local_eig_backend"] = "lobpcg" 
    dmrg.opts["local_eig_tol"] = eigLocTol
    dmrg.opts["local_eig_ncv"] = eigLocVec
    dmrg.opts["local_eig_maxiter"] = numLocIters
    #dmrg.opts["local_eig_ham_dense"] = True
    #dmrg.opts["local_eig_norm_dense"] = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        dmrg.solve(tol=eps_DMRG, max_sweeps=maxSweeps, bond_dims=chi, cutoffs=eps_DMRG/2, sweep_sequence ="RL", verbosity=2) #local
    uu_ket_var = dmrg.state; costs = [costEachSweep[-1] for costEachSweep in dmrg.total_energies]

    # Normalise correctly
    delta_tEq0 = fpDis.create_delta_tEq0(NK = NK, Nt = Nt, K = K, split = split)
    normGoal = (v_mps @ delta_tEq0)
    normCurr = (uu_ket_var @ delta_tEq0)
    uu_ket_var.multiply(normGoal/normCurr, spread_over="all",inplace=True);
    
    return uu_ket_var, costs

# Solves the dp/dt = F equation with a RK2 timestepping scheme. The scheme is based upon two Euler steps, each of which
# look like this: p_tpdt = p_t + dt*F.
def solveTimestepFP(p_0_mps, l, v, c, T, Mt, chi, split, K_x, K_v, K_phi, samplingPeriod, solveTimestepFP_pde, kwargs_solveTimestepFP_pde, tol_compression=1e-15):
        
    # Initialise
    dt = T/Mt
    p_t_mps = p_0_mps.copy(); p_t_mpses=[]
    norm0 = p_0_mps.norm()
    probSqAmplfThres = 1e2 # If the Integral of prob^2 increases more than that number, something is likely wrong and the simulation ought to be aborted. 
    
    # Run Runge-Kutta 2
    for timestep in range(Mt+1):
        
        start_time = time.time()
        
        # Do not advance further if simulation about to crash
        if np.abs(p_t_mps.norm() - norm0)/norm0 > probSqAmplfThres*norm0: 
            print("Simulation about to crash. Filling up rest of p_t_mpses with current state and exiting.")
            if len(p_t_mpses)>0: last = p_t_mpses[-1]
            else: last = p_t_mps
            p_t_mpses += [last for _ in range(Mt//samplingPeriod+1 - len(p_t_mpses))]
            break
        
        # Sample if necessary
        if (timestep==0 and samplingPeriod < Mt) or ( timestep % samplingPeriod == 0 ): p_t_mpses +=[p_t_mps]
            
        # Do not advance further if reached last timestep
        if timestep == Mt: break
        
        # First time-step dt/2 forward
        p_midpoint_mps, _ = solveTimestepFP_pde(p_0_mps=p_t_mps, p_0p5_mps=p_t_mps, l=l, c=c, dt=dt/2.0, split=split, t = timestep*dt, tol_compression = tol_compression, **kwargs_solveTimestepFP_pde)
        p_midpoint_mps.compress(max_bond=chi, cutoff = tol_compression)
        
        # Now use the t+dt/2 derivative to compute p at t+dt
        p_t_mps, F_t0p5_mps = solveTimestepFP_pde(p_0_mps=p_t_mps, p_0p5_mps=p_midpoint_mps, l=l, c=c, dt=dt, split=split, t = timestep*dt, tol_compression = tol_compression, **kwargs_solveTimestepFP_pde)
        p_t_mps.compress(max_bond=chi, cutoff_mode='rel', cutoff= tol_compression)
        
        end_time = time.time()
        print("RK2 (MPS) timestep time:{:.2f} seconds; max_bond={}; {:.0f}% done.".format(end_time - start_time,p_t_mps.max_bond() ,timestep/Mt*100), flush=True)
        
    return p_t_mpses

# Smooths the mps u_mps by applying the operator “nu_smoothing*dt/dx**2 dd_dxiSq” on it “apps” times.
def spatialSmoothing(u_mps, K_xv, K_phi, l, smoothFac, split, chi, spatialBoundryCond, apps, tol_zipup = 1e-16, tol_compression = 1e-16):    
    
    # Initialise
    chiOpsInter = 4*chi
    K = K_xv+ K_phi
    N = u_mps.L
    NK = N//K
    dx = l/2.0**NK
    if K_phi<1: print("ERROR: need at least one species (so K_phi > 0)! Exiting.\n"); sys.exit(1)
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :tol_compression, "max_bond" :chiOpsInter, "method" : 'svd'}#'isvd', "renorm" : True }
        
    
    # Produce spatial differentiation MPOs
    dd_dx1Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = 0,    NpadRight=NK*(2+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx2Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = NK,   NpadRight=NK*(1+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    dd_dx3Sq = mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=2, NpadLeft = NK*2, NpadRight=NK*(0+K_phi), h=dx,   boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split)
    
    # Now perform actual smoothing for the mps
    for i in range(apps):
        diffusion =       smoothFac*dd_dx1Sq.apply(u_mps, **COMP_OPTS)
        diffusion.add_MPS(smoothFac*dd_dx2Sq.apply(u_mps, **COMP_OPTS), inplace=True, **COMP_OPTS)
        diffusion.add_MPS(smoothFac*dd_dx3Sq.apply(u_mps, **COMP_OPTS), inplace=True, **COMP_OPTS)
         
        # Sum spatial terms into mps
        u_mps = u_mps.add_MPS(diffusion, inplace = False, **COMP_OPTS)
            
    return u_mps

# # Extracts pdf at certain points in x-space (x_vec).

def getPdfSlice(p_mps, Nt, split, x_vec, K_x, K_vphi):
    
    # Initialise
    N      = p_mps.L
    K_xvphi= K_x+K_vphi
    NK     = (N-Nt)//(K_x+K_vphi)
    vphiInds = [["k{}".format(k) for k in range(NK*(k+K_x), NK*(1+k+K_x))] for k in range(0,K_vphi)]
    tInds = ["k{}".format(k) for k in range(NK*K_xvphi, NK*K_xvphi+Nt)]
    
    # Convert the location into a bitwise string, and convert that again into a list
    xBitwise_string = ""
    for x in x_vec: xBitwise_string += f"{x:0{NK}b}"
    xBitwise = [[0,1] if xBitwise_string[i]=="1" else [1,0] for i in range(len(xBitwise_string)) ]
    
    # Creates a delta functions for the x coordinates
    delta = qu.tensor.tensor_builder.MPS_product_state(arrays = xBitwise, site_tag_id="s{}", site_ind_id="k{}")
    if split == False: print("Split=False not supported, exiting."); return -1
        #delta = mps.mpsInterleave(delta, K_x)
    
    # Extract the pdf now
    # First define the fusemap
    fuseMap = {}
    for k in range(0,K_vphi): fuseMap.update({"vphi{}".format(k+1) : tuple(vphiInds[k])})
    fuseMap.update({"t":tInds})
    
    # And compute the slice
    temp_tensor = p_mps @ delta
    pslice = temp_tensor.fuse(fuseMap).data
    
    return pslice

# Extracts statistical quantities of the pdf
def computeStats(p_mps, Nt, split, v, c, K_x, K_v, K_phi, actualMeans = None, actualStds = None):
    
    # Initialise general
    N      = p_mps.L
    K_vphi = K_v+K_phi
    Kxvphi   = K_x+K_vphi
    NK     = (N-Nt)//(Kxvphi)
    Mt    = 2**Nt
    flatPhiRaw_mps  = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*NK*K_vphi,site_tag_id='s{}')
    flatPhi_mps  = mps.shiftIndextagMps(flatPhiRaw_mps, NK*K_x)
    if Nt >0: flatTime_mps = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*Nt,site_tag_id='s{}')
    else: flatTime_mps = None
    xInds = [["k{}".format(k) for k in range(NK*k, NK*(1+k))] for k in range(0,K_x)]
    tInds = ["k{}".format(k) for k in range(NK*Kxvphi, NK*Kxvphi+Nt)]
    chiIntermed = 4*p_mps.max_bond()
            
    # Reverse-interleave if necessary
    if split == False: p_mps = mps.mpsoRevInterleave(p_mps.copy(), K=Kxvphi, NK=NK, Nt=Nt)
        #pp_rec = mps.mpsInvDecompFlow1D(p_mps,Nt = Nt, split=True)
        #fks.plotFlow1D(pp_rec[:,0:Mt:2**(Nt-NK)])
    
    # Initialise the statistical quantities
    norms   = np.zeros( [2**NK]*K_x + [1]+[K_vphi]*0 + [Mt])
    means   = np.zeros( [2**NK]*K_x + [K_vphi]*1     + [Mt]); stMeans   = means.copy()
    covs    = np.zeros( [2**NK]*K_x + [K_vphi]*2     + [Mt]); stCovs    = covs.copy()
    #coskews = np.zeros( [2**NK]*K_x + [K_vphi]*3     + [Mt]); stCoskews = coskews.copy()
    
    # Compute volume element
    dvphi = (v/2**NK)**K_v*(c/2**NK)**K_phi
    
    # Produce the vphi slopes
    vphiRaw_mpses = []; vphi_mpses = []
    for k in range(0,K_vphi):
        if k<K_v: extent = v
        else: extent = c
        vphiRaw_mps = mps.constDistrLinear(NK, K_left= k, K_right=K_vphi-1-k, split=True,pars=[0.0 +cfg.phiStart, extent*(1-2**-NK)+cfg.phiStart])
        vphiRaw_mpses += [vphiRaw_mps]
        vphi_mpses += [mps.shiftIndextagMps(vphiRaw_mps,NK*K_x)] #NikPhi  
        
    # Now compute moments
    if K_x == 0:
        
        # Initialise the outputer
        out = lambda contrRes, fuseMap: contrRes if isinstance(contrRes,float) else contrRes.fuse(fuseMap).data
        
        # Norm (0th moment)
        temp_tensor = (p_mps @ flatPhi_mps)*dvphi
        norms[:] = out(temp_tensor,{"t" : tuple(tInds)})
        
        # Mean (1st moment)
        for i in range(0,K_vphi):
            temp_tensor = (p_mps @ vphi_mpses[i])*dvphi
            means[i,:] = out(temp_tensor,{"t" : tuple(tInds)})
        
        # Also get fluctuation-mpses. They'll be useful later for computing the 
        # higher moments in a numerically stable manner. And if analytical mean 
        # not provided, estimate it.
        if actualMeans is None: 
            actualMeans= 0*means
            for i in range(0,K_vphi): actualMeans[i,:] = means[i,:]
        fluc_mpses = []
        for i in range(0,K_vphi):
            if len(actualMeans[i,:])>1: temp0 = mps.mpsDecompFlow1D_timestep(actualMeans[i,:])
            else:                       temp0 = actualMeans[i,:][0]
            temp1 = mps.mpsKron(flatPhi_mps,temp0)
            temp2 = mps.mpsKron(vphi_mpses[i],flatTime_mps)
            fluc_mpses += [temp2.add_MPS(-1*temp1)]
            
        # Covariance (2nd moment)
        for i in range(0,K_vphi):
            for j in range(0,K_vphi):
                pHmomentSq = mps.mpsHadamardZipupProd(p_mps, mps.mpsHadamardZipupProd(fluc_mpses[i], fluc_mpses[j], max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed)
                temp_tensor= (pHmomentSq @ flatPhi_mps)*dvphi
                covs[i,j,:] = out(temp_tensor,{"t" : tuple(tInds)})
            
        # # Skew (3rd moment) 
        # for i in range(0,K_vphi):
        #     for j in range(0,K_vphi):
        #         for k in range(0,K_vphi):
        #             pHmomentCub = mps.mpsHadamardZipupProd(p_mps, mps.mpsHadamardZipupProd(fluc_mpses[i],mps.mpsHadamardZipupProd(fluc_mpses[j], fluc_mpses[k], max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed)
        #             temp_tensor= (pHmomentCub @ flatPhi_mps)*dvphi
        #             coskews[i,j,k,:] = out(temp_tensor,{"t" : tuple(tInds)})
                        
        # Standardise the momements. And if analytical std not provided, estimate it.
        if actualStds is None:
            actualStds= 0*means
            for i in range(0,K_vphi): actualStds[i,:]  = np.sqrt(covs[i,i,:])
        
        # Standardised means
        for i in range(0,K_vphi): stMeans[i,:] = (means[i,:] - actualMeans[i,:])/actualStds[i,:]
        
        for i in range(0,K_vphi):
            # Standardised covariances
            for j in range(0,K_vphi):
                stCovs[i,j,:] = (covs[i,j,:])/(actualStds[i,:]*actualStds[j,:])
                # # Standardised skews
                # for k in range(0,K_vphi):
                #     stCoskews[i,j,k,:] = (coskews[i,j,k,:])/(actualStds[i,:]*actualStds[j,:]*actualStds[k,:])
    
    else:
    
        # First define the fusemap
        fuseMap = {}
        for k in range(0,K_x): fuseMap.update({"x{}".format(k+1) : tuple(xInds[k])})
        fuseMap.update({"t":tInds})
        
        # And a uniform function in space
        flatSpatial_mps  = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*NK*K_x,site_tag_id='s{}')
    
        # Norm (0th moment)
        temp_tensor = (p_mps @ flatPhi_mps)*dvphi
        norms = temp_tensor.fuse(fuseMap).data
        
        # Mean (1st moment)
        for i in range(0,K_vphi):
            temp_tensor = (p_mps @ vphi_mpses[i])*dvphi
            means[...,i,:] = temp_tensor.fuse(fuseMap).data
        
        # Also get fluctuation-mpses. They'll be useful later for computing the 
        # higher moments in a numerically stable manner. And if analytical mean 
        # not provided, estimate it.
        if actualMeans is None: actualMeans=means
        
        fluc_mpses = []
        for i in range(0,K_vphi):
            
            temp1 = mps.mpsKron(mps.mpsDecompFlowKD(actualMeans[...,i,:], K_x, Nt, split=True), flatPhiRaw_mps)
            temp2 = mps.mpsKron(flatSpatial_mps,mps.mpsKron(vphiRaw_mpses[i],flatTime_mps))
            fluc_mpses += [temp2.add_MPS(-1*temp1)]
        
        # Covariance (2nd moment)
        for i in range(0,K_vphi):
            for j in range(0,K_vphi):                
                
                # mean_i  = (p_mps @ vphi_mpses[i])*dvphi
                # mean_j  = (p_mps @ vphi_mpses[j])*dvphi
                # mean_ij = (p_mps @ mps.shiftIndextagMps(mps.mpsHadamardZipupProd(vphiRaw_mpses[i],vphiRaw_mpses[j]),NK*K_x) )*dvphi
                # covs[...,i,j,:] = mean_ij.fuse(fuseMap).data - mean_i.fuse(fuseMap).data*mean_j.fuse(fuseMap).data
                
                pHmomentSq = mps.mpsHadamardZipupProd(p_mps, mps.mpsHadamardZipupProd(fluc_mpses[i], fluc_mpses[j], max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed)
                temp_tensor= (pHmomentSq @ flatPhi_mps)*dvphi
                covs[...,i,j,:] = temp_tensor.fuse(fuseMap).data
        
        # # Skew (3rd moment) 
        # for i in range(0,K_vphi):
        #     for j in range(0,K_vphi):
        #         for k in range(0,K_vphi):
        #             pHmomentCub = mps.mpsHadamardZipupProd(p_mps, mps.mpsHadamardZipupProd(fluc_mpses[i],mps.mpsHadamardZipupProd(fluc_mpses[j], fluc_mpses[k], max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed), max_intermediateBond = chiIntermed)
        #             temp_tensor= (pHmomentCub @ flatPhi_mps)*dvphi
        #             coskews[...,i,j,k,:] = temp_tensor.fuse(fuseMap).data
                    
        # Standardise the momements. And if analytical std not provided, estimate it.
        if actualStds is None:
            actualStds= 0*means
            for i in range(0,K_vphi): actualStds[...,i,:]  = np.sqrt(covs[...,i,i,:])
        
        # Standardised means
        for i in range(0,K_vphi): stMeans[...,i,:] = (means[...,i,:] - actualMeans[...,i,:])/actualStds[...,i,:]
        
        for i in range(0,K_vphi):
            # Standardised covariances
            for j in range(0,K_vphi):
                stCovs[...,i,j,:] = (covs[...,i,j,:])/(actualStds[...,i,:]*actualStds[...,j,:])
                # # Standardised skews
                # for k in range(0,K_vphi):
                #     stCoskews[...,i,j,k,:] = (coskews[...,i,j,k,:])/(actualStds[...,i,:]*actualStds[...,j,:]*actualStds[...,k,:])
        
    # Now return the statistics
    #return norms, means, covs, coskews, stMeans, stCovs, stCoskews
    return norms, means, covs, stMeans, stCovs
