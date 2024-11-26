#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import TnMachinery.MPSfunctions as mps
import PDEs.FokkerPlanck.FokkerPlanckDistrs as fpDis

"""
Here are various functions for calculating p(x,t) solutions of the Fokker-Planck equation.
"""

def dmrgFPKDFeynmanKitaevHamiltonian_convecDiffu(v_mps, Nt, split, m_mpses, D_matrix, c, dt, tol_bond_zipup = 1e-16, mainCoeff = 1e0, penCoeff=1e-1, timeAcc = 2):    
    
    # Outputs MPOs neded to calculate the dmrg environments for solving the 
    # Fokker-Planck equation in KD: “ dp/dt + d(m(.)p)/dphi_i - 0.5*dd(D_ijHp)/dphi_idphi_j = 0 ”, with phi being K-dimensional.
    # This is done using the Feynman-Kitaev formalism which projects the FK Eq into the form
    # of a Hamiltonian whose ground-state provides the solution to the FK Eq.
    
    # Initialise
    N = v_mps.L
    K = len(m_mpses)
    NK = (N-Nt)//K
    dphi = c/2.0**NK
    spatialBoundryCond=   "dirichlet"
    temporlBoundaryCond = "dirichlet"
    SPLIT_OPTS = {"K": K, "NK": NK , "Nt": Nt}
    COMP_OPTS = {"cutoff_mode" : 'rel', "cutoff" : 1e-15, "compress" : True}
            
    # Produce the temporal shifting MPO
    sPlus   =  mps.mpoCreatePlusShift1(N=Nt, boundryCond = temporlBoundaryCond, NpadLeft = NK*K, NpadRight=0, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}", split=split, **SPLIT_OPTS)
    
    # Produce the phi-differentiation MPOs
    d_dphis=[]
    for k in range(0,K):
        d_dphis += [mps.mpoCreateAcc2SpatialDiff(N=NK, diffOrder=1, NpadLeft = NK*k,  NpadRight=N - NK*(k+1), h=dphi, boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}", split=split, **SPLIT_OPTS)]
        #print("d_dphi:"); d_dphis[k].show()
    # Create F
    
    # Begin by computing Sum_i{d/dphi_iHm_i}
    F = d_dphis[0].apply( 1*mps.mpsHadamardZipupProd( m_mpses[0], d_dphis[0].identity() ))
    for k in range(1,K):
        temp = d_dphis[k].apply( mps.mpsHadamardZipupProd(m_mpses[k], d_dphis[k].identity()))
        F.add_MPO(temp, inplace=True, **COMP_OPTS)
    
    # Now compute -0.5*Sum_ij{ D_ij dd_dphi_idphi_j }
    for i in range(0,K):
        for j in range(0,K):
            scalar = -0.5*D_matrix[i][j]
            temp = scalar*d_dphis[i].apply(d_dphis[j])
            F.add_MPO(temp, inplace=True, **COMP_OPTS)
    
    # Build up delta_t=0(.)Id:
    delta_tEq0 = fpDis.create_delta_tEq0(NK, Nt, K, siteTagId = "s{}", siteIndId="k{}", split=True)
    Id = F.identity()
    delta_tEq0HId = mps.mpsHadamardZipupProd(delta_tEq0, Id, mpoContractLegs = "upper")
    IdMinusdelta_tEq0HId = delta_tEq0HId.multiply(-1,inplace=False).add_MPO(Id, inplace=False, **COMP_OPTS) 
    
    # First initialise building blocks
    alphaMinu = IdMinusdelta_tEq0HId.add_MPO(-1*sPlus, inplace=False, **COMP_OPTS)
    alphaMinuAdj = alphaMinu.H.partial_transpose([n for n in range(0,N)])
    
    if   timeAcc == 1: alphaPlus = 2*IdMinusdelta_tEq0HId
    elif timeAcc == 2: alphaPlus =   IdMinusdelta_tEq0HId.add_MPO(sPlus, inplace=False, **COMP_OPTS)
    else: print("timeAcc must be either 1 or 2 (1st or second order accurate in time). Exiting."); return -1
    
    alphaPlusF = dt/2*alphaPlus.apply(F, **COMP_OPTS)
    FadjAlphaPlusAdj = alphaPlusF.H.partial_transpose([n for n in range(0,N)])    
    
    # Then create the terms of OadjO
    alphaMinuAdjAlphaMinu = alphaMinuAdj.apply(alphaMinu, **COMP_OPTS)
    alphaMinuAdjAlphaPlusF = alphaMinuAdj.apply(alphaPlusF, **COMP_OPTS)
    FadjAplhaPlusAdjalphaMinu = FadjAlphaPlusAdj.apply(alphaMinu, **COMP_OPTS)
    FadjalphaPlusAdjAlphaPlusF = FadjAlphaPlusAdj.apply(alphaPlusF, **COMP_OPTS)#cutoff=1e-8) # <-- Cheeky compress to keep conditioning number under control ? If so, make cutoff = 1e-16.
    
    # And combine them into OadjO
    OadjO = alphaMinuAdjAlphaMinu
    OadjO.add_MPO( alphaMinuAdjAlphaPlusF,    inplace=True, **COMP_OPTS)
    OadjO.add_MPO( FadjAplhaPlusAdjalphaMinu, inplace=True, **COMP_OPTS)
    OadjO.add_MPO( FadjalphaPlusAdjAlphaPlusF,inplace=True, **COMP_OPTS) #<-- laplace-squared term
    OadjO.multiply(mainCoeff,inplace=True, spread_over="all")
    
    # Create penalty term
    vvAdj = mps.mpoOuterProduct(v_mps, v_mps.H)
    vvAdj.multiply(-1.0/v_mps.norm()**2,inplace=True, spread_over="all")
    penaltyTerm = penCoeff*delta_tEq0HId.add_MPO(vvAdj,inplace=False, **COMP_OPTS)
            
    # Time to create the Feynman-Kitaev Hamiltonian itself    
    Hamiltonian = OadjO.add_MPO(penaltyTerm, inplace=False, **COMP_OPTS) #<-- Feynman-Kitaev Hamiltonian here.
    
    return Hamiltonian

# Outputs p_1 = p_0 + dt*F,
# F = Sum_i{d/dphi_i [ -m_i*p_0p5_mps]} + 0.5*Sum_ij{ D_ij dd_dphi_idphi_j p_0p5_mps } 

def timestepFPKD_convecDiffu(p_0_mps, p_0p5_mps, m_mpses, D_matrix, K_phi, c, dt, split, tol_compression = 1e-15, t=None, l=None):    
    
    # Initialise
    N = p_0p5_mps.L
    NK = N//K_phi
    dphi = c/2.0**NK
    spatialBoundryCond = "dirichlet"
    SPLIT_OPTS = {"K": K_phi, "NK": NK , "Nt": 0}
    COMP_OPTS = {"compress": True, "cutoff_mode" :'rel', "cutoff" :tol_compression, "method" : 'svd'}
        
    # Produce the phi-differentiation MPOs
    d_dphis=[]
    for k in range(0,K_phi): d_dphis += [mps.mpoCreateAcc8SpatialDiff( N=NK, diffOrder=1, NpadLeft = NK*k,  NpadRight=N - NK*(k+1), h=dphi, boundryCond=spatialBoundryCond, siteTagId='s{}', upIndIds="k{}", downIndIds="b{}",split=split, **SPLIT_OPTS)]
    
    # Begin by computing dt*Sum_i{d/dphi_i [ -m_i*p_0p5_mps]}
    scalar = -dt
    temp = mps.mpsHadamardZipupProd(p_0p5_mps, scalar*m_mpses[0])
    F = d_dphis[0].apply(temp, **COMP_OPTS)
    for k in range(1,K_phi):
        temp = mps.mpsHadamardZipupProd(p_0p5_mps, scalar*m_mpses[k])
        F.add_MPS(d_dphis[k].apply(temp, **COMP_OPTS), inplace=True, **COMP_OPTS)
    
    # Now compute dt*0.5*Sum_ij{ D_ij dd_dphi_idphi_j p_0p5_mps }
    for i in range(0,K_phi):
        for j in range(0,K_phi):
            scalar = 0.5*dt*D_matrix[i][j]
            temp = scalar*d_dphis[i].apply(d_dphis[j].apply(p_0p5_mps))
            F.add_MPS(temp, inplace=True, **COMP_OPTS)
    
    # Finish up by adding F to p_0_mps
    p_1_mps = F.add_MPS(p_0_mps, inplace=False, **COMP_OPTS)
    
    return p_1_mps, F
