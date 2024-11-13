#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import quimb as qu
from TnMachinery.MPSfunctions import * 
import TnMachinery.genTNfunctions as gtn

"""
Here lie various tree tensor network (TTN) functions that deal with spacetime fluid fields.
"""

def is_power_of_2(x):
    return np.log2(x) % 1 == 0.0

def ttnTensorSplit(tensor, leftPhysInds, rightPhysInds, max_bond = None, cutoff = 1e-10):
    # This function decomposes “tensor” into a little tree of tensors. No effort is made to keep
    # the truncations globally optimal, as that would require maintaing a canonical form and
    # keeping track of its canonical centre, which is too much hassle.
    
    # Perform the truncating SVDs.
    (left_tensor, tensor) = tensor.split(left_inds = leftPhysInds, method='svd', max_bond = max_bond, cutoff = cutoff, absorb = 'right', get = 'tensors')
    (centre_tensor, right_tensor) = tensor.split(left_inds= None, right_inds = rightPhysInds, method='svd', max_bond = max_bond, cutoff = cutoff, absorb = 'left', get = 'tensors')
    
    return (left_tensor, centre_tensor, right_tensor)    

def ttnRepresentFlow1D_autodiff(uu, max_bond = None, tol_bond = 1e-10, **kwargs):
    # TTN representation of flow field u(x,t) whose accuracy is given by the provided bond dimension/tolerance.
    # Assumes uu is a matrix of 2**N elements, with N being an integer, where the fastest varying index 
    # corresponds to t (i.e., rows correspond to x and columns to t). This function returns an N-site TTN network,
    # of height ~log2(N).
   
    # Initialise
    uu = uu.reshape(-1,) # first flatten uu into a row-array (row-major ordering).
    N = int(np.log2(len(uu)))
    
    nlayers = int(np.floor(np.log2(N)))
    
    # Initialise the tensor
    inds = ['k{}'.format(i) for i in range(0,N)]
    tensor = qu.tensor.Tensor(data = uu.reshape([2]*N), inds= inds)
    
    # Perform the first ttn tensor-split manually to produce the top layer:
    n2 = N//2
    leftPhysInds = ['k{}'.format(i) for i in range(0,n2)]
    rightPhysInds = ['k{}'.format(i) for i in range(n2,N)]
    (left_tensor, centre_tensor, right_tensor) = ttnTensorSplit(tensor, leftPhysInds, rightPhysInds,max_bond= max_bond, cutoff = tol_bond)
    left_tensor.modify(tags=qu.oset(['x','tosplit']))
    centre_tensor.modify(tags=qu.oset(['layer{}'.format(nlayers)]))
    right_tensor.modify(tags=qu.oset(['t','tosplit']))
    #
    # Store the tensors in a tensor network
    uu_ttn = left_tensor & centre_tensor & right_tensor
    
    for i in range(1,nlayers):
        layer = nlayers-i
        tensors_to_split = uu_ttn['tosplit']
        uu_ttn.delete('tosplit')
        
        for j in range(0,len(tensors_to_split)):
            tensor = tensors_to_split[j]
            n = len(tensor.inds) -1 # number of physical indices
            #
            if n > 1:
                n2 = n//2
                if j % 2 == 0:
                    leftPhysInds = [tensor.inds[i] for i in range(0,n2)]; rightPhysInds = [tensor.inds[i] for i in range(n2,n)]
                else:
                    leftPhysInds = [tensor.inds[i] for i in range(1,n2+1)]; rightPhysInds = [tensor.inds[i] for i in range(n2+1,n+1)]
                #
                (left_tensor, centre_tensor, right_tensor) = ttnTensorSplit(tensor, leftPhysInds, rightPhysInds, max_bond, cutoff = tol_bond)
                centre_tensor.retag({'tosplit':'layer{}'.format(layer)},inplace=True)
                subtree = left_tensor & centre_tensor & right_tensor
                uu_ttn.add_tensor_network(subtree)
            else:
                tensor.retag({'tosplit':'layer{}'.format(layer)})
                uu_ttn.add_tensor(tensor)
    
    uu_ttn.retag({'tosplit':'layer0'},inplace=True)
    
    # The ttn representation should be decently accurate, but since the truncations weren't 
    # globally optimal, the ttn needs to be fine-tuned. Let us do this with autodiff.
    
    # First produce an exact (up to tol_bond tolerance) x – t ordered MPS decomposition.
    uu_mps = mpsDecompFlow1D(uu, tol_bond=tol_bond, split=True)
    
    # Now use autodiff to maximise the fit between the ttn and the exact rep.
    uu_ttn = qu.tensor.tensor_core.tensor_network_fit_autodiff(tn = uu_ttn, tn_target = uu_mps, **kwargs)
    
    return uu_ttn

def ttnAndMpsFidelity_autodiff(uu, chi_ttn=1, chi_mps = 1, tol_bond = 1e-10, **kwargs):
    
    uu_mpsExact_split = mpsDecompFlow1D(uu, tol_bond=tol_bond, split=True)
    uu_mpsExact_interleaved = mpsDecompFlow1D(uu, tol_bond=tol_bond, split=False)

    #
    uu_ttnChi = ttnRepresentFlow1D_autodiff(uu, chi_ttn, tol_bond=tol_bond, **kwargs);
    uu_mpsChi_split = uu_mpsExact_split.copy(); uu_mpsChi_split.compress(max_bond=chi_mps);#mpsRepresentFlow1D_autodiff(uu, split =  True, chi = chi_mps, tol_bond  = tol_bond, **kwargs)
    uu_mpsChi_interleaved = uu_mpsExact_interleaved.copy(); uu_mpsChi_interleaved.compress(max_bond=chi_mps);#mpsRepresentFlow1D_autodiff(uu, split =  False, chi = chi_mps, tol_bond  = tol_bond, **kwargs)
    #
    fidelity_ttnChi             = gtn.tnFidelity(uu_ttnChi,uu_mpsExact_split)
    fidelity_mpsChi_split       = gtn.tnFidelity(uu_mpsChi_split,uu_mpsExact_split)
    fidelity_mpsChi_interleaved = gtn.tnFidelity(uu_mpsChi_interleaved,uu_mpsExact_interleaved)
    
    return fidelity_ttnChi, fidelity_mpsChi_split, fidelity_mpsChi_interleaved, uu_ttnChi#, uu_mpsExact_split, uu_ttnChi, uu_mpsChi_split, uu_mpsChi_interleaved

def ttnInvDecompFlow1D(uu_ttn):
    
    # Initialise
    N = int(uu_ttn['layer0'][-1].inds[-1][1:]) +1 #len(uu_ttn['layer0'])
    N2 = N//2
    
    # Contract
    uuT = uu_ttn.contract()
    
    # Extract the spatial and temporal indices
    spatialIndices = [f'k{n}' for n in range(0,N2)]
    temporalIndices = [f'k{n}' for n in range(N2,N)]
    
    # Now fuse the relevant legs together
    uuT.fuse(fuse_map = {'Fx': tuple(spatialIndices),'Ft': tuple(temporalIndices)},inplace=True)
    
    # Produce the final array representing u(x,t), and return it.
    uuR = uuT.data.reshape(2**N2,2**N2)
       
    return uuR