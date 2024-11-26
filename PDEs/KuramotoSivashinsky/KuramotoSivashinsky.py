#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import quimb as qu 
import TnMachinery.MPSfunctions as mps
import TnMachinery.genTNfunctions as gtn
import PDEs.pdegeneral as pdeg

"""
Here are various functions for calculating u(x,t) solutions of the 1D 
Kuramoto-Sivashinsky equation.
"""

def costKSperiodic1D_init(initCondFunc, initCondPars, split, Nx, Nt, T, l, nu, mu):
    
    # Initialise
    x = np.linspace(0,l,2**Nx)
    dx = l/2.0**Nx
    dt = T/2.0**Nt
    spatialBoundryCond="periodic"
    
    #Create initial field and scale by -1.
    #Note:  This (current implementation) isn't an efficient way of initialising the MPS, 
    #but it's fine for the paper as long as it's theoretically possible to efficiently do it.
    v0_mps = mps.TEMP_mpsInitialCond1D(initCondFunc, initCondPars, x, split) 
    v0_mps.multiply(-1,inplace=True)
    
    # Define the differencing operators
    d_dt =   mps.mpoCreateAcc1TemporalDiff(N=Nt,              NpadLeft = Nx, NpadRight= 0,                                      siteTagId='s{}', temporalTensorsTag='t', upIndIds="k{}", downIndIds="b{}")
    d_dx = mps.mpoCreateAcc2SpatialDiff(   N=Nx, diffOrder=1, NpadLeft = 0,  NpadRight=Nt, h=1, boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x', upIndIds="k{}", downIndIds="b{}")
    dd_dxdx = mps.mpoCreateAcc2SpatialDiff(N=Nx, diffOrder=2, NpadLeft = 0,  NpadRight=Nt, h=1, boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x', upIndIds="k{}", downIndIds="b{}")
    d4_dx4 =  mps.mpoCreateAcc2SpatialDiff(N=Nx, diffOrder=4, NpadLeft = 0,  NpadRight=Nt, h=1, boundryCond=spatialBoundryCond, siteTagId='s{}', spatialTensorsTag ='x', upIndIds="k{}", downIndIds="b{}")
    
    # Scale the terms appropriately
    d_dx.multiply(dt/dx,         inplace=True, spread_over='all')
    dd_dxdx.multiply(nu*dt/dx**2,inplace=True, spread_over='all')
    d4_dx4.multiply(mu*dt/dx**4, inplace=True, spread_over='all')
    
    return v0_mps, d_dt, d_dx, dd_dxdx,d4_dx4, dx, dt

def costKS1D(uu_ket, v_ket, d_dt, d_dx, dd_dxdx, d4_dx4, dx, dt, chi = None, eps = 0.0, C0 = 0.0, Burgers = True, dtSmall = False):
    
    # Now produce the various terms needed to solve the Kuramoto-Sivashinsky equation
    
    # First the linear terms
    du_dt =   qu.tensor.tensor_arbgeom.tensor_network_apply_op_vec(tn_op=d_dt,tn_vec = uu_ket) #<-- The first term is only “V(x,t=0)”. Hence why we need to use v_ket to get “V(x,t=0) - V(x,t=-\Delta t)”.
    ddu_dxdx =qu.tensor.tensor_arbgeom.tensor_network_apply_op_vec(tn_op=dd_dxdx,tn_vec = uu_ket)
    if Burgers is False: d4u_dx4 = qu.tensor.tensor_arbgeom.tensor_network_apply_op_vec(tn_op=d4_dx4,tn_vec = uu_ket)
    # And now the nonlinear one
    
    # udu/dx form: 
    du_dx = qu.tensor.tensor_arbgeom.tensor_network_apply_op_vec(tn_op=d_dx,tn_vec = uu_ket,compress=False)
    convective = gtn.hadamardProdTN(uu_ket, du_dx)
    
    # Gather all the terms of the KS equation into a list and then calculate the cost function and normalise it. 
    # The closer this scalar is to 0, the nearer uu_ket is to the correct solution. Have to set chi & eps to None
    # to maintain numerical stability.
    
    if dtSmall is False:
        if Burgers is False: CF = gtn.squareOfTnSum([v_ket, du_dt, convective, ddu_dxdx, d4u_dx4], chi = None, eps = None)
        else: CF = gtn.squareOfTnSum([v_ket, du_dt, convective, ddu_dxdx], chi = None, eps = None)
    else:
        # Only keep O(dt^1) terms
        if Burgers is False: CF= du_dt @ du_dt + v_ket @ v_ket + 2*(du_dt @ v_ket + du_dt @ convective + du_dt @ ddu_dxdx + du_dt @ d4u_dx4 +  v_ket @ convective + v_ket @ ddu_dxdx + v_ket @ d4u_dx4)
        else: CF = du_dt @ du_dt + v_ket @ v_ket + 2*(du_dt @ v_ket + du_dt @ convective + du_dt @ ddu_dxdx +  v_ket @ convective + v_ket @ ddu_dxdx)
    
    return CF*dx*dt + C0

def computeDistanceKSperiodic1D(uu_ket, initCondFunc, initCondPars, split, Nx, Nt, T, l, nu, mu, C0=0.0, Burgers=True, dtSmall = False):
    
    # Initialise the constants needed for calculating the cost function
    v_ket, d_dt, d_dx, dd_dxdx,d4_dx4, dx, dt = costKSperiodic1D_init(initCondFunc, initCondPars, split, Nx, Nt, T, l, nu, mu)

    # Now compute the distance and return it.
    distance = costKS1D(uu_ket = uu_ket, v_ket = v_ket, d_dt= d_dt,d_dx= d_dx,dd_dxdx= dd_dxdx, d4_dx4= d4_dx4,dx= dx, dt= dt, C0=C0, Burgers=Burgers, dtSmall=dtSmall)
    
    return distance

def varSolveGlobalKSperiodic1D_chiExpand(initCondFunc, initCondPars, l, T, nu, mu, split, chi, chi0=1, M =2**8, nsteps=5000, Burgers=True, dtSmall = False, optimizer = 'nadam', autodiff_backend = 'jax', **optimize_opts):
    
    # Initialise
    chi_temp=chi0
    t = np.linspace(0,T,M); x = np.linspace(0,l,M);
    N2 = int(np.log2(M))
    
    # Initialise the constants needed for calculating the cost function
    v0_mps, d_dt, d_dx, dd_dxdx,d4_dx4, dx, dt = costKSperiodic1D_init(initCondFunc, initCondPars, split, N2, N2, T, l, nu, mu)
    const_terms = {'v_ket' : v0_mps,'d_dt': d_dt, 'd_dx': d_dx, 'dd_dxdx': dd_dxdx, 'd4_dx4': d4_dx4}
    const_simpleTerms = {'dt': dt, 'dx' : dx, 'Burgers': Burgers, 'dtSmall': dtSmall}
    uu_ket_var = v0_mps.copy() 
    
    # Perform minimisation
    while chi_temp <= chi:
        uu_ket_var.expand_bond_dimension(chi_temp, rand_strength=1*1e-2, inplace=True)
        uu_ket_var.compress(form = 0, max_bond = chi_temp, cutoff=1e-15)
        
        # Now minimise the distance using autodiff
        tnopt = qu.tensor.TNOptimizer(uu_ket_var, loss_fn=costKS1D, loss_constants=const_terms,loss_kwargs=const_simpleTerms,optimizer=optimizer, autodiff_backend=autodiff_backend)
        #uu_ket_var = tnopt.optimize_basinhopping(n=nsteps, nhop=10)
        
        uu_ket_var = tnopt.optimize(n=int(chi_temp/chi*nsteps), **optimize_opts)
                
        # Plot & show
        print('\n'); uu_ket_var.show()
        pdeg.plotFlow1D(t=t,x=x,uu=mps.mpsInvDecompFlow1D(uu_ket_var))
        
        chi_temp+=1
    
    # Set the x, t tags
    uu_ket_var = mps.mpsSetTagsInds1D(uu_ket_var, split)
    
    return uu_ket_var, v0_mps

def varSolveGlobalKSperiodic1D_gridExpand(initCondFunc, initCondPars, l, T, nu, mu, split, chi, M_init = 2**4, M_final =2**8 ,optimizer = None, nsteps=5000, Burgers=True, dtSmall = False, **optimizer_opts):
    
    # Initialise
    N0 = int(2*np.log2(M_init))
    Nf = int(2*np.log2(M_final))
    chi_step = chi_temp = min(chi, chi/np.min(N0/Nf) )
    
    # Perform minimisation
    for N2 in range(int(N0/2),int(Nf/2)+1):
        # Initialise the constants needed for calculating the cost function
        t = np.linspace(0,T,2**N2); x = np.linspace(0,l,2**N2);
        
        # Initialise the constants needed for calculating the cost function
        v0_mps, d_dt, d_dx, dd_dxdx,d4_dx4, dx, dt = costKSperiodic1D_init(initCondFunc, initCondPars, split, N2, N2, T, l, nu, mu)
        const_terms = {'v_ket' : v0_mps,'d_dt': d_dt, 'd_dx': d_dx, 'dd_dxdx': dd_dxdx, 'd4_dx4': d4_dx4}
        const_simpleTerms = {'dt': dt, 'dx' : dx, 'Burgers': Burgers, 'dtSmall': dtSmall}
        
        # Initialise variational function
        uu_ket_var = v0_mps.copy()
        
        # Ensure bond-dimension of variational function is chi_temp
        uu_ket_var.expand_bond_dimension(chi_temp, rand_strength=1*1e-1, inplace=True)
        uu_ket_var.compress(form = 0, max_bond = chi_temp, cutoff=1e-15)
        
        # Now minimise the distance using autodiff
        tnopt = qu.tensor.TNOptimizer(uu_ket_var, loss_fn=costKS1D, loss_constants=const_terms, loss_kwargs=const_simpleTerms, **optimizer_opts)
        #uu_ket_var = tnopt.optimize_basinhopping(n=nsteps, nhop=10)
        uu_ket_var = tnopt.optimize(int(chi_temp/chi*nsteps))
        
        # Plot & show
        print('\n'); uu_ket_var.show()
        pdeg.plotFlow1D(t=t,x=x,uu=mps.mpsInvDecompFlow1D(uu_ket_var))
        
        # Increase chi
        chi_temp = min(chi, chi_temp+chi_step)
        
        # Prolongate the variational function
        if N2 != int(Nf/2): uu_ket_var = mps.tnToMps1D(mps.mpsProlongate1D(uu_ket_var, split), split, max_bond = chi_temp)
    
    # Scale v0_mps back and return everything    
    v0_mps[-1].modify(data=v0_mps[-1].data*(-1))    
    return uu_ket_var, v0_mps