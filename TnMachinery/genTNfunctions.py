#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import quimb as qu

"""
Here lie various tensor network operations that deal with fields in space & time
"""

def tnNorm_sum(tn):
    
    N = tn.nsites
    
    flat_mps = qu.tensor.MPS_product_state([np.array([1.0,1.0])]*N,site_tag_id='s{}')
    
    nfact = tn.H @ flat_mps
    
    return tn.multiply(1 / nfact, spread_over='all')

def tnNorm(tn):
    
    nfact = (tn.H @ tn)**0.5
    
    return tn.multiply(1 / nfact, spread_over='all')

def tnFidelity(tn1, tn2):
    
    overlap = (tn1.H & tn2).contract(backend = "numpy")
    
    return ( overlap/(tn1.norm()*tn2.norm()) )**2

# Connects u1_ket & u2_ket with kronecker delta tensors such that the contraction
# of this tensor network results in the Hadamard (elementwise) product between 
# u1_ket and u2_ket. u1_ket and u2_ket must have the same number of outer inds.
def hadamardProdTN(u1_ket, u2_ket, site_ind_id = 'k{}'):
    
    # Initialise
    N = len(u1_ket.outer_inds())
    
    # Define the underlying array of the kronecker delta tensor
    delta_array = np.array([0.0]*8)
    delta_array[0 + 2*0 + 4*0] = 1
    delta_array[1 + 2*1 + 4*1] = 1
    
    # Reindex the sites appropriately
    u1_ket = u1_ket.reindex_all('b{}',inplace=False)
    u2_ket = u2_ket.reindex_all('c{}', inplace=False)
    
    tn = qu.tensor.TensorNetwork(u1_ket.tensors)
    # Loop through and build up the network
    for i in range(0,N):
        delta_i = qu.tensor.Tensor(data=delta_array.reshape(2,2,2,order='F'), inds = ['b{}'.format(i),'c{}'.format(i),site_ind_id.format(i)])
        
      #  # Now add necessary tags to delta_i
      #  delta_i.add_tag('s{}'.format(i))
      #  if(i < N//2): delta_i.add_tag(['x','x{}'.format(i)])
      #  else: delta_i.add_tag(['t','t{}'.format(i)])
        
        tn &=delta_i
        
    tn &= u2_ket
    
    return tn

def tnEnvsSum(tnlet, tnEnv_list, allTags=['left','right','var'] ):
    # Assumes tn_env_list contains the same tags every time, and that allTags
    # include every tag seen in the tensor networks tn_env_list and tnlet.
    
    # Initialise
    result = 0.0
    N = len(tnEnv_list)
    
    # Calculate SUM_k < tn_env[k] | tnlet>
    for i in range(0,N):
        tnFull = tnEnv_list[i] & tnlet
        result += tnFull.contract(tags=allTags, output_inds=qu.oset([]))#(tags=allTags, output_inds=qu.oset([]))
        
    return -result/tnlet.norm()

def squareOfTnSum(terms, chi = None, eps = None, areReal = True):
    
    # Initialise
    result=0.0
    n = len(terms)
    
    #Calculate the square of the sum of the terms
    for i in range(0,n):
          for j in range(i,n):
              overlap_tn = terms[i] & terms[j]
              if (chi == None and eps == None):
                  overlap = overlap_tn^...
              else:
                  overlap = overlap_tn.contract(tags=..., max_bond = 2, cutoff=0.0)
              #
              if   i!=j and     areReal: overlap = 2.0*overlap
              elif i!=j and not areReal: overlap = overlap + np.conj(overlap)
              result += overlap
              #print("for i,j=({},{}): {}".format(int(i),int(j),overlap))
    
    return result