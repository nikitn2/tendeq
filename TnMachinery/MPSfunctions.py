#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
from jax import config;

config.update("jax_enable_x64", True)
import TnMachinery.genTNfunctions as gtn

"""
Here lie various MPS and MPO functions that deal with discretised functions in space/time.
The first part of the file deals with the MPS functions, the other part with the MPO ones.
"""

def primefact(number):
    factors = []
    if number < 2: return [1]
    f = 2
    while f ** 2 <= number:
        while number % f == 0:
            factors.append(f)
            number //= f
        f += 1
    if number > 1: factors.append(number)
    return factors


# Dmrg based on autodifferencing. Requires that loss_fn = loss_fn(mps, ...).
def mpsDmrg_autodiff(mps_init, chi_max, loss_fn, loss_kwargs, eps=1e-15,
                     optimizer_opts={"optimizer": 'l-bfgs-b', "autodiff_backend": 'jax'},
                     optimize_opts={'jac': True, 'hessp': False, 'ftol': 1e-12, 'gtol': 1e-12, 'eps': 1e-5, "n": 1000}):
    # Initialise
    sti = mps_init.site_tag_id
    chi_new = mps_init.max_bond()
    mps_new = mps_init.copy()
    loss_new = loss_fn(mps_var=mps_init, **loss_kwargs)
    loss_prev = 10 * loss_new

    # Perform DMRG until convergence
    while loss_prev - loss_new > eps:

        # Update previous loss
        loss_prev = loss_new

        # Run dmrg
        for i in range(0, mps_new.L, 3):

            # Prep mps for optimisation
            if chi_new < chi_max: chi_new += 1
            mps_new.expand_bond_dimension(chi_new, rand_strength=1e-14, inplace=True);
            mps_new.compress(form='right', cutoff_mode="rel", cutoff=1e-15)
            mps_new.canonize(i)

            # Prep optimiser
            tagsToOpti = [sti.format(j) for j in range(i, min(i + 3, mps_new.L))]
            tnopt = qu.tensor.TNOptimizer(mps_new, tags=tagsToOpti, loss_fn=loss_fn, loss_constants=loss_kwargs,
                                          **optimizer_opts)

            # Optimise
            mps_new = tnopt.optimize(**optimize_opts)

        # Update new loss
        loss_new = loss_fn(mps_var=mps_new, **loss_kwargs)

    # Ensure mps is writeable by using a deep-copy, then compress and return
    mps_new = mps_new.copy(deep=True)
    mps_new.compress(form='right', cutoff_mode="rel", cutoff=1e-15)
    return mps_new


# Calculates the Hadamard (elementwise) product between the mps-vector u1_mps
# and u2_mpso, which can either be an mps or an mpo (if an MPO, the element
# -wise multiplication can go either into *lower* or *upper* legs).
def mpsHadamardZipupProd(u1_mps, u2_mpso, mpoContractLegs="lower", max_intermediateBond=None, max_finalBond=None,
                         tol_bond=1e-15):
    # Check whether any of the factors are None
    if u1_mps is None or u2_mpso is None: return None

    # Check whether any of the factors are scalars
    if isinstance(u1_mps, float) or isinstance(u2_mpso, float): return u1_mps * u2_mpso
    if isinstance(u1_mps, int) or isinstance(u2_mpso, int):   return u1_mps * u2_mpso

    # Initialise
    N = u2_mpso.L  # number of sites of the vectors

    # Reindex in preperation
    if isinstance(u2_mpso, qu.tensor.tensor_1d.MatrixProductState):
        remainderIndices = (u2_mpso.site_ind_id,)
        uC2_mpso = u2_mpso.reindex_sites('hadamard_mpsHadamardZipupProd{}',
                                         inplace=False)  # <-- This is the index which will be contracted over.
    else:
        if mpoContractLegs == "lower":
            remainderIndices = (u2_mpso.lower_ind_id, u2_mpso.upper_ind_id)
            uC2_mpso = u2_mpso.reindex_lower_sites('hadamard_mpsHadamardZipupProd{}',
                                                   inplace=False)  # <-- This is the index which will be contracted over.
        else:
            remainderIndices = (u2_mpso.upper_ind_id, u2_mpso.lower_ind_id)
            uC2_mpso = u2_mpso.reindex_upper_sites('hadamard_mpsHadamardZipupProd{}',
                                                   inplace=False)  # <-- This is the index which will be contracted over.

    uC1_mps = u1_mps.reindex_sites('temp_mpsHadamardZipupProd{}', inplace=False)

    # Put the canonical centres of the MPS vectors on the leftmost sites
    uC1_mps.right_canonize()
    uC2_mpso.right_canonize()

    ## Now turn uC1_mps into a diagonal MPO using Kronecker-delta tensors

    # First define the underlying Kronecker-delta tensor
    delta_array = np.array([0.0] * 8)
    delta_array[0 + 2 * 0 + 4 * 0] = 1
    delta_array[1 + 2 * 1 + 4 * 1] = 1

    # Now build up the copy-tensors and contract with the sites of uC1_mps
    u1_o_tensors = []
    for i in range(0, N):
        site_i = uC1_mps[i]
        site_index = mpsRecoverSiteIndex(site_i, "temp_mpsHadamardZipupProd{}")

        delta_i = qu.tensor.Tensor(data=delta_array.reshape(2, 2, 2, order='F'),
                                   inds=[remainderIndices[0].format(site_index),
                                         'hadamard_mpsHadamardZipupProd{}'.format(site_index),
                                         'temp_mpsHadamardZipupProd{}'.format(site_index)])
        u1_o_tensors.append((delta_i & site_i) ^ ...)
    u1_o = qu.tensor.tensor_core.TensorNetwork(u1_o_tensors)

    # Perform the zipup contraction

    # Connect
    uOut_mpso = uC2_mpso
    uC2_mpso |= u1_o

    # Perform the first contraction manually
    uOut_mpso.contract(tags='s0', inplace=True)

    # Get the left inds
    left_inds = [remainderIndx.format(mpsRecoverSiteIndex(uOut_mpso['s0'], remainderIndices[0])) for remainderIndx in
                 remainderIndices]

    # And SVD rightwards
    uOut_mpso.split_tensor(tags='s0', left_inds=left_inds, method='svd', max_bond=max_intermediateBond, cutoff=tol_bond,
                           absorb='right', rtags='triangle')
    triangle_tensor = uOut_mpso['triangle'];
    triangle_tensor.drop_tags();
    triangle_tensor.add_tag('triangle')

    # And then sweep through the remaining sites and zip them up.
    for i in range(1, N):
        prevSite_tag = 's{}'.format(i - 1)
        currSite_tag = 's{}'.format(i)

        # First contract the "triangle" tensor rightwards
        uOut_mpso.contract(tags=['triangle', currSite_tag], inplace=True)
        triangle_tensor = uOut_mpso['triangle'];
        triangle_tensor.drop_tags('triangle')

        # Then move the canonical centre rightwards
        if i < N - 1:
            # Get the left inds
            left_inds = [remainderIndx.format(mpsRecoverSiteIndex(uOut_mpso[currSite_tag], remainderIndices[0])) for
                         remainderIndx in remainderIndices]
            left_inds += list(qu.tensor.tensor_core.bonds(uOut_mpso[prevSite_tag], uOut_mpso[currSite_tag]))

            # And SVD rightwards
            uOut_mpso.split_tensor(tags=currSite_tag, left_inds=left_inds, method='svd', max_bond=max_intermediateBond,
                                   cutoff=tol_bond, absorb='right', rtags='triangle')
            triangle_tensor = uOut_mpso['triangle'];
            triangle_tensor.drop_tags();
            triangle_tensor.add_tag('triangle')

    # Now do a backwards compression-sweep, and return.
    uOut_mpso.compress(max_bond=max_finalBond, cutoff=tol_bond, renorm=0)

    return uOut_mpso


# Calculates the Hadamard product between u_mps and v_tn, which can be some sort of
# tree tensor network. Keeps the tags of v_tn. This operation can be very expensive because
# it costs O(chi_u^2 chi_v^2), if v is a mps/mpo of bond-dimension chi_v, and the final
# state will be of bond-dimension chi_u*chi_v.
def mpsHadamardProd(u_mps, v_tn, contractingIndId="k{}"):
    # Initialise
    N = u_mps.L
    physDim = v_tn.phys_dim()
    uC_mps = u_mps.reindex_sites('temp{}', inplace=False)
    vC_tn = v_tn.copy()

    for n in range(0, N):
        copyTensor = qu.tensor.tensor_core.COPY_tensor(d=physDim, inds=[contractingIndId.format(n), 'temp{}'.format(n),
                                                                        'prelower{}'.format(n)])
        uC_mps &= copyTensor
        uC_mps.contract_ind('temp{}'.format(n))
    vC_tn &= uC_mps
    for n in range(0, N):
        vC_tn.contract_ind(contractingIndId.format(n))
        vC_tn.reindex({'prelower{}'.format(n): contractingIndId.format(n)}, inplace=True)
    vC_tn.fuse_multibonds(inplace=True)

    return vC_tn


# Warning: This one scales exponentially in the number of tensors! Should be replaced with TT-cross/variational-contruction/prolongation-construction
def mpsSqrt_direct(mps, chi):
    uu = mpsInvDecompFlow1D_timestep(mps)
    uu = np.sqrt(np.abs(uu))
    mps = mpsDecompFlow1D_timestep(uu)
    mps.compress(max_bond=chi)

    # print("max[sqrt_uu] =", np.max(uu[:]))

    return mps


# Compute the square-root thru DMRG. Warning: randomly returns either the positive or negative of the square-root.
def mpsSqrt_dmrg(mps, chi=None, optimizer_opts={"optimizer": 'l-bfgs-b', "autodiff_backend": 'jax'},
                 optimize_opts={'jac': True, 'hessp': False, 'ftol': 1e-12, 'gtol': 1e-12, 'eps': 1e-5, "n": 1000}):
    # Initialise
    if chi == None: chi = 4 * mps.max_bond()

    # Define loss function for square-root loss
    def mpsSqrt_loss(mps_var, mps_targ):
        loss_mps = mpsHadamardProd(mps_var, mps_var)
        loss_mps.add_MPS(-1 * mps_targ, inplace=True, compress=False)

        return loss_mps @ loss_mps

    # Minimise the loss and return
    mps_sqrt = mpsDmrg_autodiff(mps_init=mps, chi_max=chi, loss_fn=mpsSqrt_loss, loss_kwargs={"mps_targ": mps},
                                optimizer_opts=optimizer_opts, optimize_opts=optimize_opts)

    return mps_sqrt


# Squeezes all “superfluous” legs of tensors in the mps. Note that this
# object ceases to be an mps after the operation.
def mpsSqueeze(u_mps):
    # Initialise
    N = u_mps.L
    u_tn = u_mps.copy()

    for n in range(0, N):
        u_tn[n].squeeze(inplace=True)

    u_tn = qu.tensor.tensor_core.TensorNetwork(u_tn)
    return u_tn


# Add new tags to the first mps according to the ordering of the physical
# indices of the second mps.
def mpsAddTagsAccToPhysIndex(u_mps, v_mps, new_tag_ind_id='a{}', site_ind_id='k{}'):
    # Initialise
    N = u_mps.L

    # Generate a dict-map between the physical indices and tags of v_mps
    v_tagPhys_dict = {}
    for n in range(0, N):
        physIndx_n = mpsRecoverSiteIndex(v_mps[n], site_ind_id)
        v_tagPhys_dict[physIndx_n] = n

    # Now add the tags
    for n in range(0, N):
        u_tensor = u_mps['s{}'.format(n)]
        u_physIndx_n = mpsRecoverSiteIndex(u_tensor, site_ind_id)
        u_tensor.add_tag(new_tag_ind_id.format(v_tagPhys_dict[u_physIndx_n]))

        v_tensor = v_mps['s{}'.format(n)]
        v_physIndx_n = mpsRecoverSiteIndex(v_tensor, site_ind_id)
        v_tensor.add_tag(new_tag_ind_id.format(v_tagPhys_dict[v_physIndx_n]))


def mpsRecoverSiteIndex(t, site_ind_id='k{}'):
    # First identify the letters of the site id:
    letterLen = site_ind_id.index('{')
    letterFormat = site_ind_id[0:letterLen]

    index = np.nan
    for ind in t.inds:
        if ind[0:letterLen] == letterFormat:
            index = ind[letterLen:]
    return int(index)


def mpsCompressionRatio(uu_mps=None, N=None, chi=None, phys_dim=2, adjustForGaugeDegsFreedom=True):
    # extract information and initialise
    if uu_mps is None:
        uu_mps = qu.tensor.MPS_rand_state(L=N, bond_dim=chi, phys_dim=phys_dim)
        uu_mps.compress(cutoff=1e-16)
    N = uu_mps.L  # number of sites
    mpsNumParas = 0
    fullNumParas = 1

    for n in range(0, N):
        # extract bond dimensions around the nth tensor
        leftBondDim = qu.tensor.tensor_core.bonds_size(uu_mps[n - 1], uu_mps[n])  if n>1 and N>1 or uu_mps.cyclic == True else 1
        rightBondDim = qu.tensor.tensor_core.bonds_size(uu_mps[n], uu_mps[n + 1]) if n<N and N>1 or uu_mps.cyclic == True else 1
        physBondDim = uu_mps.phys_dim(n)

        fullNumParas = fullNumParas * physBondDim
        if n < N - 1 and adjustForGaugeDegsFreedom:
            mpsNumParas = mpsNumParas + leftBondDim * rightBondDim * physBondDim - rightBondDim ** 2  # the "-rightBondDim**2" comes from removing the spurious gauge degrees of freedom.
        else:
            mpsNumParas = mpsNumParas + leftBondDim * rightBondDim * physBondDim

    compRat = fullNumParas / mpsNumParas  # ; uu_mps.show()
    return compRat


def mpsComputeMutualInf(uu_mps):
    # Computes the mutual information (MI) between every site in the MPS and plots
    # the resulting MI matrix.
    N = uu_mps.L
    mis = np.zeros([N, N])
    uu_mpsC = uu_mps.copy();
    uu_mpsC.normalize();
    qdat = qu.qu((uu_mpsC ^ ...).data.transpose().reshape(-1, ))
    for i in range(0, N):
        for j in range(0, N):
            mis[i, j] = qu.mutinf_subsys(qdat, [2] * N, i, j)

    ticks = [i for i in range(1, N, 2)]
    plt.imshow(mis, cmap=plt.cm.jet, origin='lower')
    plt.colorbar()
    plt.title("Mutual information between all sites")
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.show()

    return mis


# Transforms the indices of a split D-dimensional mps/mpo chain into
# those of an interleaved one. Assumes N % D = 0. Inplace operation.
def mpsInterleaveInds(uu_mpso, D):
    # Initialise
    N = uu_mpso.L
    ND = N // D

    for n in range(0, ND):
        for d in range(0, D):
            if isinstance(uu_mpso, qu.tensor.tensor_1d.MatrixProductState):
                uu_mpso[D * n + d].reindex(
                    {uu_mpso.site_ind_id.format(D * n + d): uu_mpso.site_ind_id.format(n + ND * d)}, inplace=True)
            else:
                uu_mpso[D * n + d].reindex(
                    {uu_mpso.upper_ind_id.format(D * n + d): uu_mpso.upper_ind_id.format(n + ND * d)}, inplace=True)
                uu_mpso[D * n + d].reindex(
                    {uu_mpso.lower_ind_id.format(D * n + d): uu_mpso.lower_ind_id.format(n + ND * d)}, inplace=True)
    return


def mpsRepresentFlow1D_als(uu, split, chi=16, tol_bond=1e-10, **kwargs):
    # Recasts the flow field u(x,t) into a MPS of bond-dimension chi. Assumes uu is a matrix
    # of 2**N elements, with N being an integer, where the fastest varying index corresponds to t
    # (i.e., rows correspond to x and columns to t). This function returns an N-length MPS network where either:
    # (1) if split = True, the first N2 = N/2 tensors correspond to the spatial length scales
    # (in decreasing order of size) and the remainder N2 correspond to the temporal length scales;
    # (2) otherwise, the N tensors store the spatial and temporal length scales
    # in an interleaved fashion that goes from largest scales to finest.

    # Initialise
    N = int(2 * np.log2(len(uu)))

    # Produce an exact (up to tol_bond tolerance) MPS representation of uu
    uu_mpsExact = mpsDecompFlow1D(uu, tol_bond=tol_bond, split=split)

    # Now find the best MPS representation that has a bond dimension chi using the method of “alternating least squares”.
    uu_mps_var = qu.tensor.MPS_rand_state(L=N, bond_dim=1)
    for chi_temp in range(1, chi + 1):
        uu_mps_var.expand_bond_dimension(chi_temp, rand_strength=0.1, inplace=True)
        uu_mps_var.compress(form='left', cutoff=1e-16)
        uu_mps_var = qu.tensor.tensor_core.tensor_network_fit_als(tn=uu_mps_var, tn_target=uu_mpsExact, **kwargs);

    # Set the x, t tags
    uu_mps_var = mpsSetTags1D(uu_mps_var, split)

    # Return the solution and exit
    return uu_mps_var


def mpsRepresentFlow1D_autodiff(uu, split, chi=16, tol_bond=1e-10, **kwargs):
    # Recasts the flow field u(x,t) into a MPS of bond-dimension chi using autodiff. Assumes uu is a matrix
    # of 2**N elements, with N being an integer, where the fastest varying index corresponds to t
    # (i.e., rows correspond to x and columns to t). This function returns an N-length MPS network where either:
    # (1) if split = True, the first N2 = N/2 tensors correspond to the spatial length scales
    # (in decreasing order of size) and the remainder N2 correspond to the temporal length scales;
    # (2) otherwise, the N tensors store the spatial and temporal length scales
    # in an interleaved fashion that goes from largest scales to finest.

    # Produce an exact (up to tol_bond tolerance) MPS representation of uu
    uu_mpsExact = mpsDecompFlow1D(uu, tol_bond=tol_bond, split=split)

    # Now find the best MPS representation that has a bond dimension chi using the method of “alternating least squares”.
    uu_mps_var = uu_mpsExact.copy();
    uu_mps_var.compress(max_bond=chi)
    # uu_mps_var = qu.tensor.MPS_rand_state(L=N, bond_dim=1)

    # Set the x, t tags and site inds
    uu_mps_var = mpsSetTags1D(uu_mps_var, split)

    uu_mps_var = qu.tensor.tensor_core.tensor_network_fit_autodiff(tn=uu_mps_var, tn_target=uu_mpsExact, **kwargs);

    # Return the solution and exit
    return uu_mps_var


# Prolongates both time and space dimensions of the tensor network (which should be MPS-like).
# Periodic boundary conditions are assumed. Bonds are created between neighbouring sites,
# if they did not previously exist. Expands the bond-dimension between the neighbouring sites.

def mpsProlongateKD(orig_mps, K, NK, boundryCond="periodic", split=True, chi=None):
    # Initialise
    prolo_mps = orig_mps.copy()
    N = prolo_mps.L
    Nt = N - K * NK
    COMP_OPTS = {"cutoff_mode": 'rel', "cutoff": 1e-15, "max_bond": chi}

    def siteSplitQR(mps, n, siteTagId):
        lsite = siteTagId.format(n - 1);
        csite = siteTagId.format(n)
        lInds = ['k{}'.format(n)] + [*mps[csite].bonds(mps[lsite])]
        newLTag = set([siteTagId.format(n)]);
        remTags = set(mps[csite].tags) - newLTag
        mps.split_tensor(tags=mps[csite].tags, left_inds=lInds, ltags="tempLeft", rtags="tempRight", method='qr')
        Tl = mps["tempLeft"];
        Tl.drop_tags();
        Tl.add_tag(newLTag)
        Tr = mps["tempRight"];
        Tr.drop_tags();
        Tr.add_tag(remTags)
        return

    # Create the prolongator
    prolo_mpo = mpoCreateAcc2ProlongatorKD(K, NK, Nt, boundryCond=boundryCond, split=split)

    # Use the prolongator to prolong the mps by connecting & contracting each prolongator in turn

    # First prep indices for contraction
    prolo_mps.reindex_all('b{}', inplace=True)
    prolo_mps.drop_tags()

    # Connect MPS with prolongator
    prolo_mps &= prolo_mpo

    # Contract site-by-site
    for n in range(prolo_mps.L): prolo_mps.contract_ind("b{}".format(n))
    prolo_mps.reindex_all('k{}', inplace=True);
    prolo_mps.retag_all(orig_mps.site_tag_id, inplace=True)

    # Remove loops that may have appeared in the network
    if split == False:
        Keff = K + int(Nt > 0)
        specialSites = [orig_mps.site_tag_id.format(n) for n in range(Keff * NK - 1, Keff * NK + K + 1)]
        prolo_mps.contract_tags(specialSites, inplace=True);
        prolo_mps.fuse_multibonds(inplace=True)
        #
        for n in range(Keff * NK - 1, Keff * NK + K): siteSplitQR(prolo_mps, n, orig_mps.site_tag_id)

    else:
        for k in range(1, K + 1):
            specialSites = [orig_mps.site_tag_id.format(n) for n in range(k * NK + k - 2, k * NK + k)]
            prolo_mps.contract_tags(specialSites, inplace=True);
            prolo_mps.fuse_multibonds(inplace=True)
            #
            siteSplitQR(prolo_mps, k * NK + k - 2, orig_mps.site_tag_id)

    prolo_mps._L = prolo_mpo.L

    # Ccompress, & return
    prolo_mps.compress(**COMP_OPTS)
    prolo_mps.compress(max_bond=chi)

    return prolo_mps


def tnToMps1D(uu_tn, NK, Nt, split, physInds='k{}', max_bond=None, tol_bond=1e-16):
    # Takes the 1D MPS uu_tn, where every site is a physical site and
    # tagged using the standard x{},t{},s{}, and outputs a MPS of
    # bond-dimension max_bond. Assumes MPS has open boundary conditions.

    # Initialise
    N = len(uu_tn.tensors)
    # N2 = int(N/2)
    arrays = []

    # First reshape all tensors in uu_tn to plr structure
    for n in range(0, N):
        plr = qu.utils.oset([physInds.format(n)])
        if n > 0:
            leftBond = uu_tn['s{}'.format(n)].bonds(uu_tn['s{}'.format(n - 1)])
            plr.add(*leftBond)
        if n < N - 1:
            rightBond = uu_tn['s{}'.format(n)].bonds(uu_tn['s{}'.format(n + 1)])
            plr.add(*rightBond)
        #
        arrays += [uu_tn['s{}'.format(n)].transpose(*plr, inplace=True).data]

    uu_mps = qu.tensor.tensor_1d.MatrixProductState(arrays, shape='plr', site_ind_id='k{}', site_tag_id='s{}')

    # Add tags
    for n in range(0, NK): uu_mps['s{}'.format(n)].add_tag(['x{}'.format(n), 'x'])
    for n in range(0, Nt): uu_mps['s{}'.format(n + NK)].add_tag(['t{}'.format(n), 't'])

    # Compress
    uu_mps.compress(max_bond=max_bond, cutoff=tol_bond, renorm=0)

    return uu_mps


# NOTE NOT EFFICIENT: cost is poly(M)
def TEMP_mpsInitialCond1D(initCondFunc, initCondPars, x, split, physInd='k{}'):
    # Generates the initial function v(x,t=0) in mps form (where t is padded)
    # with an equal number of sites to x).

    # Initialise
    M = len(x)
    uu = np.zeros([M, M])
    uu[:, 0] = initCondFunc(x, initCondPars)
    v_mps = mpsDecompFlow1D(uu, split=split)

    return v_mps


def mpsoShiftTensorLeft(mpso, n, max_bond=None, cutoff=1e-16):
    # Initialise
    lTen = mpso[n - 1]
    rTen = mpso[n]
    COMP_OPTS = {"cutoff_mode": 'rel', "cutoff": 1e-15, "max_bond": max_bond}

    if isinstance(mpso, qu.tensor.tensor_1d.MatrixProductState):
        physIndId = (mpso.site_ind_id,)
    else:
        physIndId = (mpso.lower_ind_id, mpso.upper_ind_id)
    lenPhysInd = len(physIndId)

    # Create what will be the new pair of left inds
    leftInds = {mpso.bond(n - 2, n - 1), *[physIndId[i].format(n) for i in range(lenPhysInd)]}
    leftInds = (set(lTen.inds) | set(rTen.inds)) & leftInds

    # Store original tags before cleaning the tensors of tags
    leftOrigTags = mpso[n - 1].tags;
    lTen.drop_tags();
    lTen.add_tag("tocontr")
    rightOrigTags = mpso[n].tags;
    rTen.drop_tags();
    rTen.add_tag("tocontr")

    # Contract left and righ tensors and split such that the formerly right index ends up in the left tensor
    mpso.contract_tags(tags="tocontr", inplace=True)
    mpso.split_tensor(tags="tocontr", left_inds=leftInds, method='svd', absorb="left", max_bond=max_bond, cutoff=cutoff,
                      ltags=leftOrigTags, rtags=rightOrigTags)
    mpso.drop_tags("tocontr")

    # And flip the index names of the n-1, n indices
    for i in range(lenPhysInd):
        mpso[n - 1].reindex({physIndId[i].format(n): physIndId[i].format(n - 1)}, inplace=True)
        mpso[n].reindex({physIndId[i].format(n - 1): physIndId[i].format(n)}, inplace=True)

    # Compress and finish.
    mpso.compress(**COMP_OPTS);
    mpso.compress(max_bond=max_bond)

    return


def mpsoMoveTensorLeft(mpso, n, numShifts, max_bond=None, cutoff=1e-16):
    # Shift the nth tensor leftwards numShifts times
    while numShifts > 0:
        mpsoShiftTensorLeft(mpso, n, max_bond=None, cutoff=1e-16)
        n -= 1
        numShifts -= 1

    return


# Shifts the NK[i] tensors of dim K0[i] leftwards NK[i]*dK[i] times (note that NK,
# K0, dk are all lists here.
def mpsoMoveDimsLeft(mpso, NKs, K0s, dKs):
    # Initialise
    numDims = len(NKs)

    for i in range(0, numDims):
        for n in range(NKs[i] * K0s[i], NKs[i] * (K0s[i] + 1)):
            mpsoMoveTensorLeft(mpso, n, NKs[i] * dKs[i] - 1)
    return


# Takes an input mps/mpo “x1_0 – x1_1 ... – x2_0 – x2_1 ... xK_0 ... – xK_NK – t_0 – t_1 – ... t_{Nt-1}” and
# interleaves into “x1_0 – x2_0 – ... – xK_0 – t_0 – x1_1 – ... x{K-1}_NK – xK_NK – t_{Nt-1}”
def mpsoInterleave(mpso, K, NK=None, Nt=None, max_bond=None, cutoff=1e-16):
    # Precheck
    if NK <= 2: print("NK must be >2. Exiting"); sys.exit(1)

    # Initialise
    mpsoI = mpso.copy()
    COMP_OPTS = {"cutoff_mode": 'rel', "cutoff": 1e-15, "max_bond": max_bond}

    # If NK not provided, assume all dimensions are spatial and possess the same number of sites
    if NK == None: NK = mpso.L // K

    # In case there are only spatial dimensions/all dimensions are of equal length
    if Nt == None or Nt == 0:
        if K > 1:
            # Interleave the MPS by sorting the tensors first by lengthscale, and then dimension
            for nk in range(NK, 1, -1):
                for k in range(1, K):
                    n = NK * K - nk * (K - k)
                    numShifts = (nk - 1) * k
                    mpsoMoveTensorLeft(mpsoI, n, numShifts, max_bond=max_bond, cutoff=cutoff)
                    mpsoI.compress(**COMP_OPTS)

        else:
            pass

    # Seperate time/spatial dimensions requires special consideration
    else:
        if Nt >= NK:
            mpsoI = mpsoInterleave(mpso=mpsoI, K=K + 1, NK=NK, max_bond=None, cutoff=cutoff)
        else:
            print("Nt<NK not currently supported. Exiting.\n"); sys.exit(1)

    # Clean & return
    mpsoI.fuse_multibonds(inplace=True)
    if mpsoI.max_bond() > 16: mpsoI.compress(**COMP_OPTS); mpsoI.compress(max_bond=max_bond)
    return mpsoI


# Takes an input mps/mpo “x1_0 – x2_0 – ... – xK_0 – t_0 – x1_1 – ... x{K-1}_NK – xK_NK – t_{Nt-1}” and
# reverse-interleaves into “x1_0 – x1_1 ... – x2_0 – x2_1 ... xK_0 ... – xK_NK – t_0 – t_1 – ... t_{Nt-1}”
def mpsoRevInterleave(mpso, K, NK, Nt=None, max_bond=None, cutoff=1e-16):
    # Precheck
    if NK <= 2: print("NK must be >2. Exiting"); sys.exit(1)

    # Initialise
    mpsoRI = mpso.copy()
    COMP_OPTS = {"cutoff_mode": 'rel', "cutoff": 1e-15, "max_bond": max_bond}

    # If NK not provided, assume all dimensions are spatial and possess the same number of sites
    if NK == None: NK = mpso.L // K

    # In case there are only spatial dimensions/all dimensions are of equal length
    if Nt == None or Nt == 0:
        if K > 1:
            # Reverse-interleave the MPS by sorting the tensors first by dimension, then lengthscale
            for k in range(1, K):
                for nk in range(1, NK):
                    n = K * nk + k - 1
                    numShifts = (K - 1) * nk
                    mpsoMoveTensorLeft(mpsoRI, n, numShifts, max_bond=max_bond, cutoff=cutoff)
                    mpsoRI.compress(**COMP_OPTS)
        else:
            pass
    # Seperate time/spatial dimensions requires special consideration
    else:
        if Nt >= NK:
            mpsoRI = mpsoRevInterleave(mpso, K + 1, NK, max_bond=None, cutoff=cutoff)
        else:
            print("Nt<NK not currently supported. Exiting.\n"); sys.exit(1)

    # Clean & return
    mpsoRI.fuse_multibonds(inplace=True)
    if mpsoRI.max_bond() > 16: mpsoRI.compress(cutoff=cutoff); mpsoRI.compress(max_bond=max_bond)
    return mpsoRI


# Shifts indices and tags of mpsOrig
def shiftIndextagMps(mpsOrig, Nextra, start=0, shifted_ind_id=None, tag_shift=True, N=None):
    # Initialise
    if N == None: N = mpsOrig.L
    siteTagId = mpsOrig.site_tag_id
    if isinstance(mpsOrig, qu.tensor.tensor_1d.MatrixProductState):
        if shifted_ind_id == None:
            siteIndId = (mpsOrig.site_ind_id,)
        else:
            siteIndId = (shifted_ind_id,)
    else:
        if shifted_ind_id == None:
            siteIndId = (mpsOrig.lower_ind_id, mpsOrig.upper_ind_id)
        else:
            siteIndId = (shifted_ind_id["lower"], shifted_ind_id["upper"])

    reindexSiteIndMap = {}
    reindexSiteTagMap = {}
    for n in range(start, N):
        for i in range(len(siteIndId)): reindexSiteIndMap[siteIndId[i].format(n)] = siteIndId[i].format(Nextra + n)
        reindexSiteTagMap[siteTagId.format(n)] = siteTagId.format(Nextra + n)

    mpsNew = mpsOrig.reindex(reindexSiteIndMap, inplace=False)
    if tag_shift: mpsNew.retag(reindexSiteTagMap, inplace=True)
    if shifted_ind_id != None:
        if isinstance(mpsOrig, qu.tensor.tensor_1d.MatrixProductState):
            mpsOrig.reindex_all(shifted_ind_id, inplace=True)
        else:
            mpsOrig.reindex_lower_sites(shifted_ind_id["lower"], inplace=True)
            mpsOrig.reindex_upper_sites(shifted_ind_id["upper"], inplace=True)

    return mpsNew


# Computes the tensor product between mps1 (left) and mps2 (right).
# Note: these mpses both need to have the same indices.
def mpsKron(mps1, mps2):
    # Do a few prechecks
    if mps1 is None:
        return mps2
    elif mps2 is None:
        return mps1

    if isinstance(mps1, float) or isinstance(mps2, float): return mps1 * mps2
    if isinstance(mps1, int) or isinstance(mps2, int):   return mps1 * mps2

    # Initialise
    N = mps1.L
    siteTagId = mps1.site_tag_id
    mpsOut = mps1.copy()

    # Reindex & retag mps2
    mps2 = shiftIndextagMps(mps2, N)

    # Now tensor mps1 & mps2
    mpsOut &= mps2
    qu.tensor.tensor_core.new_bond(mpsOut[siteTagId.format(N - 1)], mpsOut[siteTagId.format(N)]);
    mpsOut._L += mps2.L

    return mpsOut


# Creates a linear function in K-dimensional space
# Note: scales O(2**NK) ! But that's OK because in theory linear functions can
# be easily initialised with low chi.
def constDistrLinear(NK, K_left, K_right, split, pars):
    # Initialise
    siteTagId = "s{}"
    siteIndId = "k{}"

    # Create the 1D linear function and turn it into an MPS
    a = pars[0];
    b = pars[1]
    lin_mps = mpsDecompFlow1D_timestep(np.linspace(a, b, 2 ** NK))
    # print(np.linspace(a,b,2**NK))

    if K_left > 0:
        uniSpace_mps = qu.tensor.tensor_builder.MPS_product_state(arrays=[np.array([1.0, 1.0])] * NK * K_left,
                                                                  site_tag_id=siteTagId, site_ind_id=siteIndId)
        lin_mps = shiftIndextagMps(lin_mps, Nextra=NK * K_left)
        lin_mps &= uniSpace_mps
        qu.tensor.tensor_core.new_bond(lin_mps[siteTagId.format(NK * K_left - 1)],
                                       lin_mps[siteTagId.format(NK * K_left)]);
        lin_mps._L += NK * K_left

    if K_right > 0:
        uniSpace_mps = qu.tensor.tensor_builder.MPS_product_state(arrays=[np.array([1.0, 1.0])] * NK * K_right,
                                                                  site_tag_id=siteTagId, site_ind_id=siteIndId)
        uniSpace_mps = shiftIndextagMps(uniSpace_mps, Nextra=NK * (K_left + 1))
        lin_mps &= uniSpace_mps
        qu.tensor.tensor_core.new_bond(lin_mps[siteTagId.format(NK * (K_left + 1) - 1)],
                                       lin_mps[siteTagId.format(NK * (K_left + 1))]);
        lin_mps._L += NK * K_right

    # Interleave if necessary
    if split == False: lin_mps = mpsoInterleave(mpso=lin_mps, K=K_left + K_right, NK=NK)

    return lin_mps


# Creates a uniform function
def constDistrUniformFP(N, volume=1.0):
    # Initialise
    siteTagId = "s{}"
    siteIndId = "k{}"

    uni_mps = qu.tensor.tensor_builder.MPS_product_state(arrays=[np.array([1, 1])] * N, site_tag_id=siteTagId,
                                                         site_ind_id=siteIndId)
    uni_mps.multiply(1.0 / volume, spread_over="all", inplace=True)

    return uni_mps


# Averages the function over the last KtoAvg dimensions
def averageMpsAccrossLastKDimensions(mps, NK, split, dvolToAvg, KtoAvg, padBack=False):
    if split is False: print("split=False is currently not supported in this function. Exiting."); return -1

    # Initialise
    N = mps.L
    K = N // NK
    mpsAvgd = mps.copy()

    # Get integrator
    flat_mps = qu.tensor.MPS_product_state([np.array([1.0, 1.0])] * NK * KtoAvg, site_tag_id="s{}")
    flat_mps = shiftIndextagMps(flat_mps, Nextra=NK * (K - KtoAvg))
    integrator = flat_mps.multiply(dvolToAvg, spread_over="all", inplace=False)

    # Integrate
    if K == KtoAvg:  # <-- returns a scalar
        return integrator @ mps
    else:  # <-- returns an mps
        mpsAvgd &= integrator
        lastTags = ["s{}".format(n) for n in range(NK * (K - KtoAvg) - 1, N)]
        mpsAvgd = mpsAvgd.contract(tags=lastTags)

        # Retag the final tensor and adjust length
        finTen = mpsAvgd[N - 1];
        finTen.drop_tags();
        finTen.add_tag("s{}".format(NK * (K - KtoAvg) - 1))
        mpsAvgd._L -= NK * (K - KtoAvg)

        # If padding back to old length is wanted, do so
        if padBack is True:
            mpsAvgd &= flat_mps
            qu.tensor.tensor_core.new_bond(mpsAvgd["s{}".format(NK * (K - KtoAvg) - 1)],
                                           mpsAvgd["s{}".format(NK * (K - KtoAvg))])
            mpsAvgd._L = N

        return mpsAvgd


# Integrates mps across dimension k.
def integrateMpsAcrossDim(mps, NK, Dim, split, l, padBack=False):
    if split is False: print("split=False is currently not supported in this function. Exiting."); return -1

    # Initialise
    N = mps.L
    K = N // NK
    mpsInteg = mps.copy()
    dvol = l / 2 ** NK

    # Get integrator
    flat_mps = qu.tensor.MPS_product_state([np.array([1.0, 1.0])] * NK, site_tag_id="s{}")
    flat_mps = dvol * shiftIndextagMps(flat_mps, Nextra=NK * Dim)
    integrator = flat_mps.multiply(dvol, spread_over="all", inplace=False)

    # Integrate
    if K == 1:
        # returns a scalar
        return integrator @ mps
    else:
        # returns an mps
        mpsInteg &= integrator
        contrTags = ["s{}".format(n) for n in range(NK * Dim, NK * (Dim + 1))]
        if Dim == 0:
            absrbTag = "s{}".format(NK * (Dim + 1))
        else:
            absrbTag = "s{}".format(NK * Dim - 1)
        contrTags += [absrbTag]
    mpsInteg = mpsInteg.contract(tags=contrTags)

    # Retag the integral-containing tensor
    integTen = mpsInteg[NK * Dim];
    integTen.drop_tags();
    integTen.add_tag(absrbTag)

    # Reindex/tag mps
    reindexSiteIndMap = {}
    reindexSiteTagMap = {}
    for n in range(NK * (Dim + 1), N):
        reindexSiteIndMap["k{}".format(n)] = "k{}".format(n - NK)
        reindexSiteTagMap["s{}".format(n)] = "s{}".format(n - NK)
    mpsInteg.reindex(reindexSiteIndMap, inplace=True)
    mpsInteg.retag(reindexSiteTagMap, inplace=True)

    # Update length and return
    mpsInteg._L -= NK * (K - 1)

    return mpsInteg


def padAtEnd(mps, NtoPad):
    # Initialise flat mps and kron it with the mps
    flat_mps = qu.tensor.MPS_product_state([np.array([1.0, 1.0])] * NtoPad, site_tag_id=mps.site_tag_id)
    mpsPadded = mpsKron(mps, flat_mps)

    return mpsPadded


## 1D functions

def mpsSetTags1D(uu_ket, split):
    # Initialise
    N = uu_ket.L
    N2 = int(N / 2)

    if (split == True):
        # In case a x – t ordering of the MPS chain:
        for n in range(0, N2):
            uu_ket[n].add_tag(['x', 'x{}'.format(n)])
            uu_ket[n + N2].add_tag(['t', 't{}'.format(n)])
    else:
        # In case an interleaved ordering of the MPS chain:
        for n in range(0, N2):
            uu_ket[2 * n].add_tag(['x', 'x{}'.format(n)])
            # uu_ket[2*n].reindex({'k{}'.format(2*n) : 'k{}'.format(n)}, inplace=True)
            #
            uu_ket[2 * n + 1].add_tag(['t', 't{}'.format(n)])
            # uu_ket[2*n+1].reindex({'k{}'.format(2*n+1) : 'k{}'.format(N2+n)}, inplace=True)
    return uu_ket


# Creates a K-dimensional delta function for dim k \in{1,2,...,K}
def createDelta(NK, K, k, split):
    if split is True:
        delta_k = qu.tensor.tensor_builder.MPS_product_state(
            arrays=[np.array([1.0, 1.0])] * NK * (k - 1) + [np.array([1.0, 1e-16])] * NK + [
                np.array([1.0, 1.0])] * NK * (K - k), site_tag_id="s{}", site_ind_id="k{}")
    else:
        arrays = [[np.array([1.0, 1e-16])] if (n - k + 1) % K == 0 else [np.array([1, 1])] for n in range(0, NK * K)]
        delta_k = qu.tensor.tensor_builder.MPS_product_state(arrays=arrays, site_tag_id="s{}", site_ind_id="k{}")

    return delta_k

# Creates a delta-function at index n of the m-sized dimension
def createDeltaIndex(n, m, split=True):
    # Initialise
    dims = primefact(m)
    locs = []
    arrays = []

    # Compute location of 1 in radix system
    for dim in dims:
        locs.append(n % dim)
        n //= dim

    # Build up MPS arrays
    for i in range(len(dims)):
        dim = dims[i];
        loc = locs[i]
        array = np.zeros(dim);
        array[loc] = 1
        arrays.append(array)

    # Now create the delta-function
    if split == True:
        delta_mps = qu.tensor.tensor_builder.MPS_product_state(arrays[::-1], site_tag_id="s{}", site_ind_id="k{}")
        if len(arrays) == 1 and len(arrays[0].shape) == 1: delta_mps.squeeze(inplace=True)
    else:
        delta_mps = None  # Todo.
    
    return delta_mps

# Concatenates a list of mpses into a single new mps.
def mpsConcatenate(mpses_in, max_bond=None, cutoff=1e-14):
    # Initialise
    m = len(mpses_in)
    mpses_out = None
    chi_intermed = None if max_bond == None else max_bond * 4

    for i, mps_in in enumerate(mpses_in):
        index_mps = createDeltaIndex(i, m)
        mps_out = mpsKron(mps_in, index_mps)
        if mpses_out == None: mpses_out = mps_out
        else:                 mpses_out.add_MPS(mps_out, inplace=True, compress=False)
        if i % 20 == 0:
            print("MPS-concat {:.1f}% complete.".format(i / len(mpses_in) * 100.0),flush=True)
            mpses_out.compress(max_bond=max_bond, cutoff=cutoff)
            mpses_out.show()

    mpses_out.compress(max_bond=max_bond, cutoff=cutoff)
    return mpses_out

# Useful MPS decomp/inverseDecomp function wrappers
def mpsDecompFlow1D(uu, tol_bond=1e-16, split=True): return mpsDecompFlowKD(uu, K=1, Nt=int(np.log2(uu.shape[-1])), split=split, tol_bond=tol_bond)
def mpsInvDecompFlow1D(uu_mps, Nt=None, split=True): return mpsInvDecompFlowKD(uu_mps, K=1, Nt=Nt, split=split)
def mpsDecompFlow1D_timestep(u, tol_bond=1e-16): return mpsDecompFlowKD_timestep(u, K=1, split=True, tol_bond=tol_bond)
def mpsInvDecompFlow1D_timestep(u_mps): return mpsInvDecompFlowKD_timestep(u_mps, K=1, split=True)
def mpsDecompFlowKD_timestep(u, K, split, tol_bond=1e-16): return mpsDecompFlowKD(u, K, Nt=0, split=split, tol_bond=tol_bond)

def mpsInvDecompFlowKD_timestep(u_mps, K, split):
    # Assume u_mps is of shape (2**NK,2**NK,...2**NK)
    N = u_mps.L
    NK = N // K
    Ms = (2 ** NK,) * K

    return mpsInvDecompFlowKD(u_mps, Ms, split=split)


def mpsSetTagsIndsKD(u_mps, K, Nt, split):
    # Initialise
    N = u_mps.L
    NK = (N - Nt) // K
    if Nt > 0:
        Ktot = K + 1
    else:
        Ktot = K

    for k in range(0, Ktot):
        for n in range(0, NK):
            # In case a x1 – x2 ... – x_K – t ordering of the MPS chain:
            if (split == True):
                tensorNum = k * NK + n
            # In case an interleaved ordering of the MPS chain:
            else:
                tensorNum = n * Ktot + k

            if k == K:
                tag = "t_{}".format(n)
            else:
                tag = 'x{}_{}'.format(k + 1, n)
            u_mps[tensorNum].add_tag(tag)

    for n in range(NK * (Ktot), N): u_mps[n].add_tag("t_{}".format(n - NK * K))

    return u_mps


def mpsDecompFlowKD_uneven(u, split, tol_bond=1e-16):
    # MPS decomposes flow field u(x_1, x_2,...,x_K) exactly (up to tol_bond tolerance). Assumes u is a K-dimensional object of
    # type (M_0, M_1,...,M_K) elements, where the first dimension (rows) corresponds to x_1, then the second
    # (columns) to x_2 and so on until the x_Kth dimension, which corresponds to the slowest varying index.

    # Initialise
    primes = [primefact(dim)[::-1] for dim in u.shape]
    dims = [prime for prime_group in primes for prime in prime_group]
    NKs = [len(prime) for prime in primes];
    KNs = [];
    Ks_valid = []

    # Get breakdown of Ks and Ns
    while sum(len(prime) for prime in primes) > 0:
        ks_valid = [];
        k = 0
        for prime in primes:
            if len(prime) == 0:
                k += 1
            elif len(prime) > 0:
                ks_valid.append(k); prime.pop(0); k += 1
        Ks_valid.append(ks_valid)
        KNs.append(len(ks_valid))

    # Define new_inds for reshaping as appropriate
    old_inds = np.arange(len(dims));
    new_inds = 0 * old_inds.copy()
    it = 0
    for n in range(max(NKs)):
        for k in Ks_valid[n]:
            # print("n={}".format(n),"k={}".format(k), ":" , it, " ->", n + np.sum(NKs[:k],dtype=int))
            if split == True:
                new_inds[it] = old_inds[it]
            else:
                new_inds[it] = n + np.sum(NKs[:k], dtype=int)
            it += 1

    # Reshape the array and update dims after
    u = u.reshape(dims).transpose(new_inds).flatten()
    dims = [dims[i] for i in new_inds]

    # And now return the final MPS
    return qu.tensor.MatrixProductState.from_dense(u, dims, cutoff=tol_bond, site_ind_id='k{}', site_tag_id='s{}')


def mpsDecompFlowKD(u, K, Nt, split, tol_bond=1e-16):
    # MPS decomposes flow field u(x_1, x_2,...,x_K, t) exactly (up to tol_bond tolerance). Assumes u is a K+1-dimensional object of
    # type (2**N, 2**N,...,2**N, 2**N_t) elements, where the first dimension (rows) corresponds to x_1, then the second
    # (columns) to x_2 and so on until the x_Kth dimension and then finally the t-dimension, which corresponds to the slowest varying index.

    u_mps = mpsDecompFlowKD_uneven(u, split, tol_bond)
    u_mps = mpsSetTagsIndsKD(u_mps, K, Nt, split)

    return u_mps


def mpsInvDecompFlowKD(u_mps, Ms, split):
    # Returns dense tensor u(x_1, x_2,...,x_K) from its MPS form. Assumes u is a K-dimensional object of
    # type (M_0, M_1,...,M_K) elements, where the first dimension (rows) corresponds to x_1, then the second
    # (columns) to x_2 and so on until the x_Kth dimension, which corresponds to the slowest varying index.

    # Initialise
    primes = [primefact(dim)[::-1] for dim in Ms]  # if dim>1]
    NKs = [len(prime) for prime in primes];
    KNs = [];
    Ks_valid = []
    sii = u_mps.site_ind_id

    # Get breakdown of Ks and Ns
    while sum(len(prime) for prime in primes) > 0:
        ks_valid = [];
        k = 0
        for prime in primes:
            if len(prime) == 0:
                k += 1
            elif len(prime) > 0:
                ks_valid.append(k); prime.pop(0); k += 1
        Ks_valid.append(ks_valid)
        KNs.append(len(ks_valid))

    # First contract the MPS and extract the underlying data
    uT = u_mps.contract()

    ## Fuse the respective legs together.

    # Build up fuse map
    fuse_map = {};
    it = 0
    for k in range(max(KNs)): fuse_map['F{}'.format(k)] = []
    if split == True:
        for k in range(max(KNs)):
            for n in range(NKs[k]): fuse_map['F{}'.format(k)].append(sii.format(it)); it += 1
    else:
        for n in range(max(NKs)):
            for k in Ks_valid[n]: fuse_map['F{}'.format(k)].append(sii.format(it)); it += 1

    # Now fuse the relevant legs together
    uT.fuse(fuse_map=fuse_map, inplace=True)

    # Produce the final array representing u(x,t), and return it.
    uR = uT.data.reshape(Ms)  # .transpose()

    return uR


def mpsGetFidelityCompRatio(u, split, chi, cutoff):
    u_mps = mpsDecompFlowKD_uneven(u, split)
    u_mps_compr = u_mps.copy();
    u_mps_compr.compress(max_bond=chi, cutoff=cutoff)
    u_compr = mpsInvDecompFlowKD(u_mps_compr, u.shape, split)

    fid = gtn.tnFidelity(u_mps, u_mps_compr)
    CR = mpsCompressionRatio(u_mps_compr)

    return (u_compr, u_mps_compr), fid, CR


def mpoOuterProduct(u_ket, v_bra, new_ind_id="b{}", compress=True, tol=1e-16):
    # Assumes u_ket and v_bra have the same site_ind_id and site_tag

    # Initialise
    N = u_ket.L

    vC_bra = v_bra.reindex_all(new_ind_id, inplace=False)
    outer_ketbra = u_ket & vC_bra

    # Contract the up-down tensors
    for i in outer_ketbra.gen_sites_present():
        outer_ketbra ^= u_ket.site_tag(i)

    if isinstance(outer_ketbra, qu.tensor.Tensor):
        outer_ketbra = outer_ketbra.as_network()

    outer_ketbra.view_as_(
        qu.tensor.tensor_1d.MatrixProductOperator,
        cyclic=(u_ket.cyclic or vC_bra.cyclic), L=N, site_tag_id=u_ket.site_tag_id,
        lower_ind_id=vC_bra.site_ind_id, upper_ind_id=u_ket.site_ind_id,
    )
    outer_ketbra.fuse_multibonds_()

    if compress is True: outer_ketbra.compress(cutoff=tol)

    return outer_ketbra


# Differentiates every element except the t=0 (does nothing there).
# Note: this function hard-locks dt = 1. This is necessary to ensure the initial state remains unchanged.
def mpoCreateAcc1TemporalDiff(N, NpadLeft=0, NpadRight=0, siteTagId='s{}', temporalTensorsTag='t', upIndIds="k{}",
                              downIndIds="b{}", split=True, **splitOpts):
    ## First define the leftmost array /DUR
    left_array = np.array([0.0] * 8)

    # Sending in 0 from right: do nothing.
    left_array[0 + 2 * (0) + 4 * (0)] = 1.0
    left_array[1 + 2 * (1) + 4 * (0)] = 1.0

    # Sending in 1 from right: add upwards (substract downwards).
    left_array[0 + 2 * (1) + 4 * (1)] = 1.0

    ## Now build-up the underlying arrays of the bulk tensors.

    # Create the bulk nodes and the two last nodes /DLUR
    bulk_array = np.array([0.0] * 16)

    # When sending in 0 from the right: do nothing.
    bulk_array[0 + 2 * 0 + 4 * 0 + 8 * 0] = 1.0
    bulk_array[1 + 2 * 0 + 4 * 1 + 8 * 0] = 1.0

    # When sending in 1 from the right: add upwards (substract downwards).
    bulk_array[1 + 2 * 1 + 4 * 0 + 8 * 1] = 1.0
    bulk_array[0 + 2 * 0 + 4 * 1 + 8 * 1] = 1.0

    # Temporal stepsize must be locked to one.
    h = 1.0

    # Now do the rightmost array
    right_array = np.array([1.0, -1.0]) / h

    # Finally, create the identity tensor that will be used to pad-out the MPO.
    id_array = np.array([0.0] * 4)
    id_array[0 + 2 * 0] = 1.0
    id_array[1 + 2 * 1] = 1.0

    ## Now create the tensors and use them as the basis of the MPO tensor network.
    bulk_tensor = qu.tensor.Tensor(data=bulk_array.reshape(2, 2, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    #
    left_tensor = qu.tensor.Tensor(data=left_array.reshape(2, 1, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    #
    right_tensor = qu.tensor.Tensor(data=right_array.reshape(2, 1, order='F'), inds=['R', 'dummy'])
    right_tensor = (right_tensor & bulk_tensor) ^ ...
    right_tensor = qu.tensor.Tensor(data=right_tensor.data.reshape(2, 2, 2, 1), inds=['D', 'L', 'U', 'R'])
    #
    idBulk_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 1, 2, 1, order='F'), inds=['D', 'L', 'U', 'R'])
    idLeft_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 2, 1, order='F'), inds=['D', 'U', 'R'])
    idRight_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 1, 2, order='F'), inds=['D', 'L', 'U'])

    # Remove phantom indices
    if NpadLeft == 0: left_tensor.squeeze(inplace=True)
    if NpadRight == 0: right_tensor.squeeze(inplace=True)

    # Now produce the MPO
    diff_tn = qu.tensor.tensor_1d.MatrixProductOperator(
        arrays=[idLeft_tensor.data] * int(NpadLeft > 0) + [idBulk_tensor.data for _ in range(NpadLeft - 1)] + [
            left_tensor.data] + [bulk_tensor.data for _ in range(N - 2)] + [right_tensor.data] + [idBulk_tensor.data for
                                                                                                  _ in range(
                NpadRight - 1)] + [idRight_tensor.data] * int(NpadRight > 0),
        shape='dlur', site_tag_id=siteTagId, upper_ind_id=upIndIds, lower_ind_id=downIndIds, bond_name='internal')

    # And tag the temporal tensors appropriately
    for i in range(NpadLeft, NpadLeft + N):
        diff_tn.tensors[i].add_tag(temporalTensorsTag)

    # And if necessary, interleave
    if split is False: diff_tn = mpsoInterleave(diff_tn, **splitOpts)

    # And return it.
    return diff_tn


# Shifts the contents of the vector rightwards/downwards (if going along rows/cols) in a periodic manner by default
def mpoCreatePlusShift1(N, boundryCond="periodic", NpadLeft=0, NpadRight=0, siteTagId='s{}', upIndIds="k{}",
                        downIndIds="b{}", split=True, **splitOpts):
    ## First define the leftmost array /DUR
    left_array = np.array([0.0] * 8)

    # Sending in 0 from right: do nothing.
    left_array[0 + 2 * (0) + 4 * (0)] = 1.0
    left_array[1 + 2 * (1) + 4 * (0)] = 1.0

    # Sending in 1 from right: add upwards (substract downwards).
    left_array[0 + 2 * (1) + 4 * (1)] = 1.0

    # If reached boundary, act
    if boundryCond == "periodic":
        left_array[1 + 2 * (0) + 4 * (1)] = 1.0
    elif boundryCond == "dirichlet":
        pass  # or boundryCond == "normconserving": pass# do nothing
    else:
        print("The provided boundary conditions are not currently supported. Exiting.\n"); sys.exit(1)

    ## Now build-up the underlying arrays of the bulk tensors.

    # Create the bulk nodes and the two last nodes /DLUR
    bulk_array = np.array([0.0] * 16)

    # When sending in 0 from the right: do nothing.
    bulk_array[0 + 2 * 0 + 4 * 0 + 8 * 0] = 1.0
    bulk_array[1 + 2 * 0 + 4 * 1 + 8 * 0] = 1.0

    # When sending in 1 from the right: add upwards (substract downwards).
    bulk_array[1 + 2 * 1 + 4 * 0 + 8 * 1] = 1.0
    bulk_array[0 + 2 * 0 + 4 * 1 + 8 * 1] = 1.0

    # Now do the rightmost array
    right_array = np.array([.0, +1.0])

    # Finally, create the identity tensor that will be used to pad-out the MPO.
    id_array_chi1 = np.array([0.0] * 4)
    id_array_chi1[0 + 2 * 0] = 1.0;
    id_array_chi1[1 + 2 * 1] = 1.0

    ## Now create the tensors and use them as the basis of the MPO tensor network.
    bulk_tensor = qu.tensor.Tensor(data=bulk_array.reshape(2, 2, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    #
    left_tensor = qu.tensor.Tensor(data=left_array.reshape(2, 1, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    #
    right_tensor = qu.tensor.Tensor(data=right_array.reshape(2, 1, order='F'), inds=['R', 'dummy'])
    right_tensor = (right_tensor & bulk_tensor) ^ ...
    right_tensor = qu.tensor.Tensor(data=right_tensor.data.reshape(2, 2, 2, 1), inds=['D', 'L', 'U', 'R'])

    # Remove phantom indices
    if NpadLeft == 0:  left_tensor.squeeze(inplace=True)
    if NpadRight == 0: right_tensor.squeeze(inplace=True)

    # Build up the identity tensors and produce the MPO
    idLeft_tensor = qu.tensor.Tensor(data=id_array_chi1.reshape(2, 2, 1, order='F'), inds=['D', 'U', 'R'])
    idRight_tensor = qu.tensor.Tensor(data=id_array_chi1.reshape(2, 1, 2, order='F'), inds=['D', 'L', 'U'])
    idBulkChi1_tensor = qu.tensor.Tensor(data=id_array_chi1.reshape(2, 1, 2, 1, order='F'), inds=['D', 'L', 'U', 'R'])

    # Split MPO structure is straightforward.
    arrays = [idLeft_tensor.data] * int(NpadLeft > 0) + [idBulkChi1_tensor.data] * (NpadLeft - 1) + [
        left_tensor.data] + [bulk_tensor.data] * (N - 2) + [right_tensor.data] + [idBulkChi1_tensor.data] * (
                         NpadRight - 1) + [idRight_tensor.data] * int(NpadRight > 0)

    # Now finally form the MPO
    PlusShift1_tn = qu.tensor.tensor_1d.MatrixProductOperator(arrays=arrays, shape='dlur', site_tag_id=siteTagId,
                                                              upper_ind_id=upIndIds,
                                                              lower_ind_id=downIndIds)  # , bond_name='internal')

    # And if necessary, interleave
    if split is False: PlusShift1_tn = mpsoInterleave(PlusShift1_tn, **splitOpts)

    # Before returning it
    return PlusShift1_tn


def mpoCreateAcc8SpatialDiff(N, diffOrder, boundryCond="periodic", NpadLeft=0, NpadRight=0, h=1.0, siteTagId='s{}',
                             upIndIds="k{}", downIndIds="b{}", split=True, **splitOpts):
    # Initialise
    COMP_OPTS = {"compress": True, "cutoff_mode": 'rel', "cutoff": 1e-16}
    zero = 1e-40

    # Ensure there are more than 1 site.
    if N < 2: print("N must be at least 2. Exiting.\n"); sys.exit(1)

    # Create the building blocks
    S_p1 = mpoCreatePlusShift1(N, boundryCond, NpadLeft, NpadRight, siteTagId, upIndIds, downIndIds, split=True)
    S_m1 = S_p1.H.partial_transpose(list(range(NpadLeft + N + NpadRight)), inplace=False)
    S_p2 = S_p1.apply(S_p1, **COMP_OPTS)
    S_m2 = S_m1.apply(S_m1, **COMP_OPTS)
    S_p3 = S_p1.apply(S_p2, **COMP_OPTS)
    S_m3 = S_m1.apply(S_m2, **COMP_OPTS)
    S_p4 = S_p1.apply(S_p3, **COMP_OPTS)
    S_m4 = S_m1.apply(S_m3, **COMP_OPTS)
    Id = S_p1.identity()

    # Now build the differentiators
    if diffOrder == 1:
        diff = Id.multiply(zero, inplace=False)
        diff.add_MPO(-4 / (5 * h) * S_p1, inplace=True, **COMP_OPTS)
        diff.add_MPO(+4 / (5 * h) * S_m1, inplace=True, **COMP_OPTS)
        diff.add_MPO(+1 / (5 * h) * S_p2, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (5 * h) * S_m2, inplace=True, **COMP_OPTS)
        diff.add_MPO(-4 / (105 * h) * S_p3, inplace=True, **COMP_OPTS)
        diff.add_MPO(+4 / (105 * h) * S_m3, inplace=True, **COMP_OPTS)
        diff.add_MPO(+1 / (280 * h) * S_p4, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (280 * h) * S_m4, inplace=True, **COMP_OPTS)
    elif diffOrder == 2:
        diff = Id.multiply(-205 / (72 * h ** 2), inplace=False)
        diff.add_MPO(+8 / (5 * h ** 2) * S_p1, inplace=True, **COMP_OPTS)
        diff.add_MPO(+8 / (5 * h ** 2) * S_m1, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (5 * h ** 2) * S_p2, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (5 * h ** 2) * S_m2, inplace=True, **COMP_OPTS)
        diff.add_MPO(+8 / (315 * h ** 2) * S_p3, inplace=True, **COMP_OPTS)
        diff.add_MPO(+8 / (315 * h ** 2) * S_m3, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (560 * h ** 2) * S_p4, inplace=True, **COMP_OPTS)
        diff.add_MPO(-1 / (560 * h ** 2) * S_m4, inplace=True, **COMP_OPTS)

    # And if necessary, interleave
    if split is False: diff = mpsoInterleave(diff, **splitOpts)

    return diff


def mpoCreateAcc2SpatialDiff(N, diffOrder, NpadLeft=0, NpadRight=0, h=1.0, boundryCond="periodic", siteTagId='s{}',
                             upIndIds="k{}", downIndIds="b{}", split=True, **splitOpts):
    # Initialise
    COMP_OPTS = {"compress": True, "cutoff_mode": 'rel', "cutoff": 1e-16}
    zero = 1e-40

    # Ensure there are more than 1 site.
    if N < 2: print("N must be at least 2. Exiting.\n"); sys.exit(1)

    # Create the building blocks
    if boundryCond == "normconserving" or boundryCond == "zerograd":
        S_p1 = mpoCreatePlusShift1(N, "dirichlet", NpadLeft, NpadRight, siteTagId, upIndIds, downIndIds)
    else:
        S_p1 = mpoCreatePlusShift1(N, boundryCond, NpadLeft, NpadRight, siteTagId, upIndIds, downIndIds)

    S_m1 = S_p1.H.partial_transpose(list(range(NpadLeft + N + NpadRight)), inplace=False)
    Id = S_p1.identity()

    # Now build the differentiators
    if diffOrder == 1:
        diff = Id.multiply(zero, inplace=False)
        diff.add_MPO(-1 / (2 * h) * S_p1, inplace=True, **COMP_OPTS)
        diff.add_MPO(+1 / (2 * h) * S_m1, inplace=True, **COMP_OPTS)
    elif diffOrder == 2:
        diff = Id.multiply(-2 / (h ** 2), inplace=False)
        diff.add_MPO(+1 / (h ** 2) * S_p1, inplace=True, **COMP_OPTS)
        diff.add_MPO(+1 / (h ** 2) * S_m1, inplace=True, **COMP_OPTS)

    else:
        print("only diffOrder = 1,2 currently supported. Exiting.\n"); sys.exit(1)

    # Finally, if the “normconserving” boundary condition is used, then apply it on the MPO before returning
    if boundryCond == "normconserving":
        delta0, deltaF = create_deltas0andF(N, NpadLeft=NpadLeft, NpadRight=NpadRight, siteTagId=siteTagId,
                                            siteIndId=upIndIds)
        if diffOrder == 1:
            adjust0 = (+1 / (2 * h)) * mpsHadamardZipupProd(delta0, Id, mpoContractLegs="upper")
            adjustF = (-1 / (2 * h)) * mpsHadamardZipupProd(deltaF, Id, mpoContractLegs="upper")
        elif diffOrder == 2:
            adjust0 = (1 / h ** 2) * mpsHadamardZipupProd(delta0, Id, mpoContractLegs="upper")
            adjustF = (1 / h ** 2) * mpsHadamardZipupProd(deltaF, Id, mpoContractLegs="upper")
        elif diffOrder == 4:
            print("normconserving BCs aren't supported with 4th order differencing. Exiting.\n")
            sys.exit(1)
        diff.add_MPO(adjust0, inplace=True, **COMP_OPTS)
        diff.add_MPO(adjustF, inplace=True, **COMP_OPTS)
    elif boundryCond == "zerograd":
        delta0, deltaF = create_deltas0andF(N, NpadLeft=NpadLeft, NpadRight=NpadRight, siteTagId=siteTagId,
                                            siteIndId=upIndIds)

        if diffOrder == 1:
            adjust0 = (-1 / (2 * h)) * mpsHadamardZipupProd(delta0, S_m1, mpoContractLegs="upper")
            adjustF = (+1 / (2 * h)) * mpsHadamardZipupProd(deltaF, S_p1, mpoContractLegs="upper")
        elif diffOrder == 2:
            adjust0 = (1 / h ** 2) * mpsHadamardZipupProd(delta0, S_m1, mpoContractLegs="upper")
            adjustF = (1 / h ** 2) * mpsHadamardZipupProd(deltaF, S_p1, mpoContractLegs="upper")
        diff.add_MPO(adjust0, inplace=True, **COMP_OPTS)
        diff.add_MPO(adjustF, inplace=True, **COMP_OPTS)

    # And if necessary, interleave
    if split is False: diff = mpsoInterleave(diff, **splitOpts)

    return diff


# Creates a semi-uniform mps which equals one everywhere except at the first and final element
def create_uniExcept0andF(N, NpadLeft=0, NpadRight=0, siteTagId="s{}", siteIndId="k{}", split=True, **splitOpts):
    delta0, deltaF = create_deltas0andF(N, NpadLeft, NpadRight, siteTagId, siteIndId)
    uni = qu.tensor.tensor_builder.MPS_product_state(arrays=[np.array([1, 1]) for _ in range(NpadLeft + N + NpadRight)],
                                                     site_tag_id=siteTagId, site_ind_id=siteIndId)

    semiUni_mps = uni.add_MPS(-1 * delta0, compress=True, cutoff_mode='rel', cutoff=1e-16)
    semiUni_mps.add_MPS(-1 * deltaF, inplace=True, compress=True, cutoff_mode='rel', cutoff=1e-16)

    # And if necessary, interleave
    if split is False: semiUni_mps = mpsoInterleave(semiUni_mps, **splitOpts)

    return semiUni_mps


# Creates a delta at the first and final element
def create_deltas0andF(N, NpadLeft=0, NpadRight=0, siteTagId="s{}", siteIndId="k{}", split=True, **splitOpts):
    delta0 = qu.tensor.tensor_builder.MPS_product_state(
        arrays=[np.array([1.0, 1.0]) for _ in range(NpadLeft)] + [np.array([1.0, 1e-40]) for _ in range(N)] + [
            np.array([1.0, 1.0]) for _ in range(NpadRight)], site_tag_id=siteTagId, site_ind_id=siteIndId)
    deltaF = qu.tensor.tensor_builder.MPS_product_state(
        arrays=[np.array([1.0, 1.0]) for _ in range(NpadLeft)] + [np.array([1e-40, 1.0]) for _ in range(N)] + [
            np.array([1.0, 1.0]) for _ in range(NpadRight)], site_tag_id=siteTagId, site_ind_id=siteIndId)

    # And if necessary, interleave
    if split == False:
        delta0 = mpsoInterleave(delta0, **splitOpts)
        deltaF = mpsoInterleave(deltaF, **splitOpts)

    return delta0, deltaF


# Creates a prolongator (or restrictor) that assumes periodic boundary conditions for one dimension.
def mpoCreateAcc2Prolongator1D(N, boundryCond="periodic", NpadLeft=0, NpadRight=0, siteTagId='s{}', tensorTag=[],
                               upIndIds="k{}", downIndIds="b{}"):
    ## First define the leftmost array /DUR
    left_array = np.array([0.0] * 8)

    # Sending in 0 from right: do nothing.
    left_array[0 + 2 * (0) + 4 * (0)] = 1.0
    left_array[1 + 2 * (1) + 4 * (0)] = 1.0

    # Sending in 1 from right: substract upwards (add downwards).
    left_array[1 + 2 * (0) + 4 * (1)] = 1.0

    # If reached boundary, act
    if boundryCond == "periodic":
        left_array[0 + 2 * (1) + 4 * (1)] = 1.0
    elif boundryCond == "dirichlet":
        pass  # do nothing
    else:
        print("The provided boundary conditions <{}> are not currently supported. Exiting.\n".format(
            boundryCond)); sys.exit(1)

    ## Now build-up the underlying arrays of the bulk tensors.

    # Create the bulk nodes and the two last nodes /DLUR
    bulk_array = np.array([0.0] * 16)

    # When sending in 0 from the right: do nothing.
    bulk_array[0 + 2 * 0 + 4 * 0 + 8 * 0] = 1.0
    bulk_array[1 + 2 * 0 + 4 * 1 + 8 * 0] = 1.0

    # When sending in 1 from the right: substract upwards (add downwards).
    bulk_array[0 + 2 * 1 + 4 * 1 + 8 * 1] = 1.0
    bulk_array[1 + 2 * 0 + 4 * 0 + 8 * 1] = 1.0

    # Now do the right array /LU
    right_array = np.array([0.0] * 4)
    right_array[0 + 2 * 0] = 1.0
    right_array[0 + 2 * 1] = 0.5
    right_array[1 + 2 * 1] = 0.5

    # Finally, create the identity tensor that will be used to pad-out the MPO.
    id_array = np.array([0.0] * 4)
    id_array[0 + 2 * 0] = 0.0
    id_array[1 + 2 * 1] = 0.0

    ## Now create the tensors and use them as the basis of the MPO tensor network.
    bulk_tensor = qu.tensor.Tensor(data=bulk_array.reshape(2, 2, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    left_tensor = qu.tensor.Tensor(data=left_array.reshape(2, 1, 2, 2, order='F'), inds=['D', 'L', 'U', 'R'])
    right_tensor = qu.tensor.Tensor(data=right_array.reshape(1, 2, 2, 1, order='F'), inds=['D', 'L', 'U', 'R'])
    #
    idBulk_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 1, 2, 1, order='F'), inds=['D', 'L', 'U', 'R'])
    idLeft_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 2, 1, order='F'), inds=['D', 'U', 'R'])
    idRight_tensor = qu.tensor.Tensor(data=id_array.reshape(2, 1, 2, order='F'), inds=['D', 'L', 'U'])

    # Remove phantom indices
    if NpadLeft == 0:  left_tensor.squeeze(inplace=True)
    if NpadRight == 0: right_tensor.squeeze(include=['R'], inplace=True);

    # Now produce the MPO
    prolo_tn = qu.tensor.tensor_1d.MatrixProductOperator(
        arrays=[idLeft_tensor.data] * int(NpadLeft > 0) + [idBulk_tensor.data for _ in range(NpadLeft - 1)] + [
            left_tensor.data] + [bulk_tensor.data for _ in range(N - 1)] + [right_tensor.data] + [idBulk_tensor.data for
                                                                                                  _ in range(
                NpadRight - 1)] + [idRight_tensor.data] * int(NpadRight > 0),
        shape='dlur', site_tag_id=siteTagId, upper_ind_id=upIndIds, lower_ind_id=downIndIds, bond_name='internal')

    # Remove the dummy down leg of the right tensor
    prolo_tn[NpadLeft + N].squeeze(include=['b{}'.format(NpadLeft + N)], inplace=True)

    # And tag the tensors appropriately
    for i in range(NpadLeft, NpadLeft + N + 1): prolo_tn.tensors[i].add_tag(tensorTag)

    # Finally, return it.
    return prolo_tn


# KD Prolongator
def mpoCreateAcc2ProlongatorKD(K, NK, Nt, boundryCond="periodic", siteTagId='s{}', tensorTag=[], upIndIds="k{}",
                               downIndIds="b{}", split=True):
    # Get 1D space-prolongator
    proloSpace1D = mpoCreateAcc2Prolongator1D(NK, boundryCond, 0, 0, siteTagId, tensorTag, upIndIds, downIndIds)
    proloKD = proloSpace1D.copy()

    # Kron it K times with itself to get spatial dims in
    for k in range(K - 1): proloKD = mpsKron(proloKD, proloSpace1D)

    # And potentially kron it for temporal
    if Nt != None and Nt != 0:
        proloSpace1DTime = mpoCreateAcc2Prolongator1D(Nt, boundryCond, 0, 0, siteTagId, tensorTag, upIndIds, downIndIds)
        proloKD = mpsKron(proloKD, proloSpace1DTime)

    # Do interleaving and reindexing of the down-sites appropriately
    reindexSiteIndMap = {}
    if split == True:
        for n in range(0, K * (NK + 1)):         reindexSiteIndMap[downIndIds.format(n)] = downIndIds.format(
            int(n - n // (NK + 1)))
        for n in range(K * (NK + 1), proloKD.L): reindexSiteIndMap[downIndIds.format(n)] = downIndIds.format(int(n - K))
    elif split == False:
        proloKD = mpsoInterleave(proloKD, K, NK + 1, Nt + 1)
        if Nt > 0:
            Keff = K + 1
        else:
            Keff = K
        for n in range(Keff * NK, proloKD.L):  reindexSiteIndMap[downIndIds.format(n)] = downIndIds.format(int(n - K))
    proloKD.reindex(reindexSiteIndMap, inplace=True)

    # Clean up and return
    proloKD.fuse_multibonds(inplace=True)
    proloKD.compress(cutoff=1e-16)

    return proloKD