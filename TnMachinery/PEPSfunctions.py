#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import quimb as qu
from TnMachinery.MPSfunctions import *
import TnMachinery.genTNfunctions as gtn

"""
Here lie PEPS functions for representing spacetime fluid fields in the form of PEPS networks.
"""

def pepsCalcMaxNecessaryBondDims(uu_peps, chi, rowChiFlat = False, colChiFlat = False):
    # Calculates the maximum necessary horizontal and vertical bond-dimensions at each site
    
    # initialise
    Nrows = uu_peps.Lx # number of row sites
    Ncols = uu_peps.Ly # number of column sites
    
    if isinstance(chi,int):
        chi_row=chi_col=chi
    else:
        chi_row = chi[0]
        chi_col = chi[1]
    
    chiMaxHoriz= (chi_row*np.ones([Nrows,Ncols-1])).astype(int)
    chiMaxVert = (chi_col*np.ones([Nrows-1,Ncols])).astype(int)
    chiMaxHoriz_temp = chiMaxVert_temp = 1
    
    if(rowChiFlat is False):
        # Sweep east.
        for nRow in range(0,Nrows):
            for nCol in range(0,Ncols-1):
                chiMaxHoriz_temp= chiMaxHoriz_temp*uu_peps.phys_dim(nRow,nCol)
                chiMaxHoriz[nRow,nCol] = chiMaxHoriz_temp
            chiMaxHoriz_temp = 1;
        # Sweep west.
        for nRow in range(Nrows-1,-1,-1):
            for nCol in range(Ncols-1,0,-1):
                chiMaxHoriz_temp= chiMaxHoriz_temp*uu_peps.phys_dim(nRow,nCol)
                chiMaxHoriz[nRow,nCol-1] = min(chiMaxHoriz[nRow,nCol-1],chiMaxHoriz_temp, chi_row)
            chiMaxHoriz_temp = 1
    #
    if(colChiFlat is False):   
        # Sweep south
        for nCol in range(0,Ncols):
            for nRow in range(0,Nrows-1):
                chiMaxVert_temp= chiMaxVert_temp*uu_peps.phys_dim(nRow,nCol)
                chiMaxVert[nRow,nCol] = chiMaxVert_temp
            chiMaxVert_temp = 1;
        # Sweep north
        for nCol in range(Ncols-1,-1,-1):
            for nRow in range(Nrows-1,0,-1):
                chiMaxVert_temp= chiMaxVert_temp*uu_peps.phys_dim(nRow,nCol)
                chiMaxVert[nRow-1,nCol] = min(chiMaxVert[nRow-1,nCol],chiMaxVert_temp,chi_col)
            chiMaxVert_temp = 1;
    return chiMaxHoriz, chiMaxVert

def pepsExpandBondDims(uu_peps, chi, rand_strength=0.1, rowChiFlat = False, colChiFlat = False):
    
    uu_peps = uu_peps.copy()
    
    if isinstance(chi,int):
        uu_peps.expand_bond_dimension(chi, rand_strength=rand_strength, inplace=True)
        return uu_peps
    else:
        # chi is a list of two elements, the horizontal and vertical bond dimensions, which means
        # that the different bonds are of various bond dimensions.
        
        # initialise
        Nrows = uu_peps.Lx # number of row sites
        Ncols = uu_peps.Ly # number of column sites
        
        # First calculate the maximum necessary horizontal and vertical bond-dimensions at each site
        (chiMaxHoriz,chiMaxVert) = pepsCalcMaxNecessaryBondDims(uu_peps,chi,rowChiFlat, colChiFlat)
        
        # Now expand each bond:
        for nRow in range(0,Nrows):
            for nCol in range(0,Ncols):
                # Expand horizontal bond
                if(nCol<Ncols-1):
                    qu.tensor.tensor_core.TensorNetwork.expand_bond_dimension(uu_peps,new_bond_dim = chiMaxHoriz[nRow,nCol], rand_strength=rand_strength, inds_to_expand=qu.tensor.tensor_core.bonds(uu_peps[nRow,nCol],uu_peps[nRow,nCol+1]),inplace = True)
                # Expand vertical bonds:
                if(nRow<Nrows-1):
                    qu.tensor.tensor_core.TensorNetwork.expand_bond_dimension(uu_peps,new_bond_dim = chiMaxVert[nRow,nCol], rand_strength=rand_strength, inds_to_expand=qu.tensor.tensor_core.bonds(uu_peps[nRow,nCol],uu_peps[nRow+1,nCol]), inplace = True)
    return uu_peps

def pepsRepresentFlow1D_autodiff(uu, chi = 2, tol_mpsBond = 1e-10, rowChiFlat = False, colChiFlat = False, **kwargs):
    # Recasts the flow field u(x,t) into a PEPS of bond-dimension chi using autodiff. Assumes uu is a matrix 
    # of 2**N elements, with N being an integer, where the fastest varying index corresponds to t
    # (i.e., rows correspond to x and columns to t). This function returns a PEPS 2 x N2 network where 
    # the first row of N2 tensors correspond to the spatial length scales (in decreasing order of size) 
    # and the remainder N correspond to the temporal length scales.
    
    # Initialise
    N = int(2*np.log2(len(uu)))
    N2 = int(N/2)
    
    # Produce an exact (up to tol_mpsBond tolerance) x – t ordered MPS decomposition.
    uu_mps = mpsDecompFlow1D(uu, tol_bond=tol_mpsBond, split=True)
    
    # Produce an initial PEPS of bond-dimension chi=1.
    uu_peps = qu.tensor.PEPS.ones(Lx=2, Ly=N2, bond_dim=1, site_ind_id = 'dummy')
    for n in range(0,N2):            
        uu_peps[0,n].add_tag('x'); uu_peps[0,n].reindex({'dummy': 'k{}'.format(n)},inplace=True)
        uu_peps[1,n].add_tag('t'); uu_peps[1,n].reindex({'dummy': 'k{}'.format(n+N2)},inplace=True)
    uu_peps.site_ind_id='k{}'
    
    # Use autodiff” to train the peps network towards the exact mps for increasing chi.
    if isinstance(chi,int):
       chi_row = chi_col = chi
    else:
        chi_row = chi[0]; chi_col = chi[1]
    chi_temp_row = 1; chi_temp_col = 1
    while chi_temp_row <= chi_row or chi_temp_col <= chi_col:
        uu_peps = qu.tensor.tensor_core.tensor_network_fit_autodiff(tn = uu_peps, tn_target = uu_mps, **kwargs)
        uu_peps.show()
        if chi_temp_row == chi_row and chi_temp_col == chi_col: break
        if chi_temp_row < chi_row: chi_temp_row = chi_temp_row+1
        if chi_temp_col < chi_col: chi_temp_col = chi_temp_col+1
        uu_peps = pepsExpandBondDims(uu_peps, [chi_temp_row, chi_temp_col], 0.1, rowChiFlat, colChiFlat)
    
    # Return uu_peps and exit.
    return uu_peps

def pepsAndMpsFidelity_autodiff(uu, chi_peps, rowChiFlat = False, colChiFlat = False, chi_mps = 1, tol_mpsBond = 1e-10, **kwargs):
    
    uu_mpsExact = mpsDecompFlow1D(uu, tol_bond=tol_mpsBond, split=True)
    #
    uu_pepsChi = pepsRepresentFlow1D_autodiff(uu, chi_peps, tol_mpsBond=tol_mpsBond, rowChiFlat = rowChiFlat, colChiFlat = colChiFlat, **kwargs);
    uu_mpsChi =   mpsRepresentFlow1D_autodiff(uu, split =  True, chi = chi_mps, tol_bond  = tol_mpsBond, **kwargs)
    #
    fidelity_pepsChi = gtn.tnFidelity(uu_pepsChi,uu_mpsExact)
    fidelity_mpsChi  = gtn.tnFidelity(uu_mpsChi,uu_mpsExact )
    
    return fidelity_pepsChi, fidelity_mpsChi, uu_mpsExact, uu_pepsChi, uu_mpsChi

def pepsCreateRand(Nrows, Ncols, chi = 2, phys_dim = 2, site_ind_id = 'k{}{}', rowChiFlat = False, colChiFlat = False):
    
    uu_peps = qu.tensor.PEPS.rand(Lx=Nrows, Ly=Ncols, phys_dim=phys_dim, bond_dim=np.max(chi), site_ind_id = site_ind_id)
    (chiMaxHoriz,chiMaxVert) = pepsCalcMaxNecessaryBondDims(uu_peps,chi, rowChiFlat, colChiFlat)
    for nRow in range(0,Nrows):
        for nCol in range(0,Ncols):
            # compress horizontally
            if(nCol<Ncols-1):
                qu.tensor.tensor_core.tensor_compress_bond(uu_peps[nRow,nCol],uu_peps[nRow,nCol+1],max_bond=chiMaxHoriz[nRow,nCol])
            # compress vertically
            if(nRow<Nrows-1):
                qu.tensor.tensor_core.tensor_compress_bond(uu_peps[nRow,nCol],uu_peps[nRow+1,nCol],max_bond=chiMaxVert[nRow,nCol])
    return uu_peps

def pepsCompressionRatio(uu_peps=None, chi=None, phys_dim=2, Nrows=None,Ncols = None, adjustForGaugeDegsFreedom = True, rowChiFlat = False, colChiFlat = False, cyclic = False):
    
    # extract information and initialise
    if uu_peps is None:
        uu_peps= pepsCreateRand(Nrows, Ncols, chi=chi, phys_dim = phys_dim, rowChiFlat = rowChiFlat, colChiFlat = colChiFlat)
    
    Nrows = uu_peps.Lx # number of row sites
    Ncols = uu_peps.Ly # number of column sites
    pepsNumParas = 0
    fullNumParas = 1
    
    for nRow in range(0,Nrows):
        for nCol in range(0,Ncols):
            # extract bond dimensions around the (nRow,nCol) tensor
            northBondDim = eastBondDim = southBondDim = westBondDim = 1
            if cyclic or nRow>0:       northBondDim= qu.tensor.tensor_core.bonds_size(uu_peps[nRow,nCol], uu_peps[nRow-1,nCol])
            if cyclic or nCol<Ncols-1: eastBondDim = qu.tensor.tensor_core.bonds_size(uu_peps[nRow,nCol], uu_peps[nRow,nCol+1])
            if cyclic or nRow<Nrows-1: southBondDim= qu.tensor.tensor_core.bonds_size(uu_peps[nRow,nCol], uu_peps[nRow+1,nCol])
            if cyclic or nCol>0:       westBondDim = qu.tensor.tensor_core.bonds_size(uu_peps[nRow,nCol], uu_peps[nRow,nCol-1])
            
            # Accommodate both standard “{}{}” convention, as well as Nik's “{}” convention.
            if(uu_peps.site_ind_id[-4:] == '{}{}'):
                physBondDim = uu_peps.phys_dim(nRow,nCol)
            elif(uu_peps.site_ind_id[-2:] == '{}'):
                physInd = uu_peps.site_ind_id.format(nCol + nRow*Ncols); physBondDim = uu_peps[nRow,nCol].ind_size(physInd)
            else:
                raise ValueError('uu_peps site_ind_id format not recognised.')
            
            # Now calculate the number of elements present in each tensor, while adjusting for the gauge degree of freedom in the internal bonds
            fullNumParas = fullNumParas*physBondDim
            numParasInCurrentTensor = northBondDim*eastBondDim*southBondDim*westBondDim*physBondDim
            if nRow < Nrows-1 and adjustForGaugeDegsFreedom:
                numParasInCurrentTensor = numParasInCurrentTensor - southBondDim**2
            if nCol < Ncols-1 and adjustForGaugeDegsFreedom:
                numParasInCurrentTensor = numParasInCurrentTensor - eastBondDim**2
            pepsNumParas += numParasInCurrentTensor
    
    
    compRat = 2**(Nrows*Ncols)/pepsNumParas
    uu_peps.show()
    return compRat

def pepsInvDecompFlow1D(uu_peps):
    
    # Initialise
    N2 = uu_peps.Ly
    N = N2*2
    
    # Contract
    uuT = uu_peps.contract()
    
    # Extract the spatial and temporal indices
    spatialIndices = [f'k{n}' for n in range(0,N2)]
    temporalIndices = [f'k{n}' for n in range(N2,N)]
    
    # Now fuse the relevant legs together
    uuT.fuse(fuse_map = {'Fx': tuple(spatialIndices),'Ft': tuple(temporalIndices)},inplace=True)
    
    # Produce the final array representing u(x,t), and return it.
    uuR = uuT.data.reshape(2**N2,2**N2)#.transpose()
       
    return uuR