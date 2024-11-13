#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import config as cfg
import time
import itertools
from scipy.ndimage import gaussian_filter

import PDEs.PdeGeneral as pdeg

"""
Here lie functions for directly solving the Fokker-Planck equations
"""

def getTGVSmag(TGVpars, Mx, l, gamma, spatDim = 3 ):
    
    # Initialise
    dx = l/Mx
    
    # Construct the TaylorGreen vortex and Smagorinsky stress tensor
    
    #Prepare
    A = TGVpars[0]; B = TGVpars[1]; C = TGVpars[2]
    a = TGVpars[3]; b = TGVpars[4]; c = TGVpars[5]
    x = np.linspace(0,l-dx,Mx)
    Acosax = A*np.cos(a*x); sinbx = np.sin(b*x); sincx = np.sin(c*x)
    Bsinax = B*np.sin(a*x); cosbx = np.cos(b*x)
    Csinax = C*np.sin(a*x);                      coscx = np.cos(c*x)
    
    # Create the velocity field
    if spatDim == 2:
        # Taylor-Green vortex
        Acosax1sinbx2 =np.tensordot(Acosax,sinbx,0)
        Bsinax1cosbx2 =np.tensordot(Bsinax,cosbx,0)
        U = [Acosax1sinbx2, Bsinax1cosbx2]
        
    else:
        # Taylor-Green vortex
        Acosax1sinbx2sincx3 =np.tensordot(np.tensordot(Acosax,sinbx,0),sincx,0)
        Bsinax1cosbx2sincx3 =np.tensordot(np.tensordot(Bsinax,cosbx,0),sincx,0)
        Csinax1sinbx2coscx3 =np.tensordot(np.tensordot(Csinax,sinbx,0),coscx,0)
        
        # Add a jet
        uni = A + 0*x
        exp = np.exp( -0.5*(x-l/2)**2/(l/6)**2 )
        jet1_gauss = -np.tensordot(np.tensordot(uni,exp,0),exp,0)
        jet2_gauss =  0
        jet3_gauss =  0
        #
        U = [Acosax1sinbx2sincx3 + jet1_gauss, Bsinax1cosbx2sincx3 + jet2_gauss, Csinax1sinbx2coscx3 + jet3_gauss]
    
    # And construct the Smagorinsky stress tensor
    
    # First get the derivatives
    DU=[[None]*spatDim for _ in range(spatDim)]
    for i, j in itertools.product(range(spatDim),range(spatDim)): 
        DU[i][j] = 1/dx*pdeg.differencer(U[j], i, 1, "periodic",method="central2")
    
    # Then compute np.sqrt{ Sum_ij (DU[i][j] + DU[j][i])^2 }
    temp = 0
    for i, j in itertools.product(range(spatDim),range(spatDim)):
        temp += (DU[i][j] + DU[j][i])**2
    
    # And finally get Smagorinsky model
    S = gamma*np.sqrt(temp)
    
    return U, S

# RK2 solver for the PDE: 
# df/dt = 
#         - u_i df/dx_i + d/dx_i [(nu+S)df/dx_i f] 
#         + d/dphi_j {[Omega*(phi_j - <Phi_j>) + alpha*d/dphi_j] f}
#         - d/dphi_j (S_j*f), 
# with S_j = -consts_reac[j]*phi_1*phi_2*...

# Input: f0 ~ ([Mx]*spatDim + [Mphi]*reacDim)

def directSolveFP(f0, TGVpars, spatDim, Mt, samplingPeriod, l, T, nu, gamma, const_omega, alpha, consts_reac,c=1, w=0):
    
    # Initialise
    Mx   = f0.shape[0]
    Mphi = f0.shape[-1]
    dt        = T/Mt
    dx        = l/Mx
    dphi      = 1/Mphi
    reacDim   = len(f0.shape)-spatDim
    MtSamples = Mt//samplingPeriod
    f = np.zeros( [Mx]*spatDim + [Mphi]*reacDim + [MtSamples+1])
    
    # Check provided input is sensible
    if spatDim != 2 and spatDim != 3: raise ValueError('Error: spatial dimension must be either 2 or 3! Exiting.')
   
    # Insert into f
    f[...,0] = f0
    
    # Get the TaylorGreen vortex and Smagorinsky stress tensor
    U, S = getTGVSmag(TGVpars, Mx, l, gamma, spatDim) # U ~ (spatDim, [Mx]*spatDim) ; S ~ ([Mx]*spatDim)
    
    # Pre for time-march
    
    # Define needed variables
    phi = np.linspace(0,1,Mphi)
    phis = [phi.reshape( (1,)*spatDim + (1,)*(j-spatDim) + (Mphi,) + (1,)*(spatDim-j+reacDim-1)  ) for j in range(spatDim, spatDim+reacDim)]
    multPhis = 1
    for j in range(reacDim): multPhis = multPhis*phis[j]
    compSpaceBoundryCond = "normconserving" # Should be “norm-conserving"; periodic only for testing
        
    # Define dt propagator
    def propagator(g,coswt=1):
        
        # Initialise
        dg_dt = 0
        visc = nu+coswt*S
        O = const_omega*visc
        
        # Convection in real-space
        for i in range(0,spatDim): dg_dt -= coswt*U[i]*1/dx*pdeg.differencer(g, i) - 1/dx**2*pdeg.differencer(visc*pdeg.differencer(g, i), i)
            
        # Convection in composition-space
        for j in range(spatDim, spatDim+reacDim):
            phi_j = phis[j-spatDim]
            EPHI_j = np.sum(phi_j*g, axis = tuple([-(i+1) for i in range(reacDim)]),keepdims=True)*dphi**2
                    
            dg_dt += 1/dphi*pdeg.differencer(O*(phi_j-EPHI_j)*g,j, boundaryCond = compSpaceBoundryCond)
            dg_dt += alpha/dphi**2*pdeg.differencer(g, j, order=2, boundaryCond = compSpaceBoundryCond)
            dg_dt += 1/dphi*pdeg.differencer(consts_reac[j-spatDim]*multPhis*g,j, boundaryCond = compSpaceBoundryCond)
            
        return dg_dt
    
    # Start time-marching
    f_t = f0.copy()
    for t in range(1,Mt+1):
        start_time = time.time()
        
        prev = f_t.copy()
        coswt = np.cos(w*t/Mt*T) + cfg.zero
                
        # Compute dt/2 forward in time
        half= +0.5*dt*propagator(prev,coswt) + prev
        
        # Now use the above to perform a full step
        f_t = +1.0*dt*propagator(half,coswt) + prev
        
        #print(f_t.sum()*dphi**2*dx**3) #norm
        
        end_time = time.time()
        print("RK2 (direct) timestep time:{:.2f} seconds; {:.0f}% done.".format(end_time - start_time, t/Mt*100), flush=True)
            
        if t % samplingPeriod == 0: f[...,t//samplingPeriod] = f_t

    return f

# RK2 solver for the PDE: 
# d<phi_i>/dt = F_i(<phi_i>)
#             = sum_j{- u_j d<phi_i>/dx_j + d^2/dx_j^2 [alphaH<phi_i>]}. i,j = either 2 or 3.
# Here the static velocity field u_i is simply the Taylor-green vortex

def directSolveFPfirstMomentEqsNoReac(EPHI0, TGVpars, Mt, samplingPeriod, l, T, nu, gamma, w = 0, scheme="RK2"):
        
    # Initialise
    Mx = EPHI0.shape[0]
    dt        = T/Mt
    dx        = l/Mx
    spatDim   = len(EPHI0.shape)-1
    reacDim   = EPHI0.shape[-1]
    MtSamples = Mt//samplingPeriod
    EPHI = np.zeros( [Mx]*spatDim + [reacDim] + [MtSamples+1])

    # Check provided input is sensible
    if spatDim != 2 and spatDim != 3: raise ValueError('Error: spatial dimension must be either 2 or 3! Exiting.')

    # # Presmooth EPHI0
    # nu_smoothing = l/Mx
    # for t in range(int(0.01*T/dt)):
    #     for j in range(spatDim): EPHI0 = EPHI0 - nu_smoothing*dt/dx**2*pdeg.differencer(EPHI0, j, 2, "periodic","central2")

    # Insert into EPHI
    for i in range(0,reacDim): EPHI[...,i,0] = EPHI0[...,i]
    
    # Get the TaylorGreen vortex and Smagorinsky stress tensor
    U, S = getTGVSmag(TGVpars, Mx, l, gamma, spatDim)
    
    # Time-march forward now
    EPHI_t = EPHI0.copy()
    # Define dt propagator
    propagator = lambda j, orig, method: -coswt * U[j] * 1 / dx * pdeg.differencer(orig, j, 1, "periodic", method) + 1 / dx ** 2 * pdeg.differencer((nu + coswt * S) * pdeg.differencer(orig, j, 1, "periodic", method), j, 1, "periodic", method)

    for t in range(1,Mt+1):
        
        start_time = time.time()
        for i in range(0,reacDim):
            prev = EPHI_t[...,i].copy()
            coswt = np.cos(w*t/Mt*T) + cfg.zero

            if scheme == "RK2":
                # Set finite difference method
                method = "central2"
                
                # Compute dt/2 forward in time
                half = prev.copy()
                for j in range(spatDim): half += + 0.5*dt*propagator(j,prev,method)
                
                # Now use the above to perform a full step
                for j in range(spatDim): EPHI_t[...,i] += 1*dt*propagator(j,half,method)
            
            elif scheme == "MacCormack":

                print("Warning: MacCormack implementation is untested.")
                
                #Compute predictor step
                pred = prev.copy()
                for j in range(spatDim): pred += dt*propagator(j,prev, method="forward1")
   
                # Now use the above to perform the full step                
                EPHI_t[...,i] = 0.5*(prev + pred)
                for j in range(spatDim): EPHI_t[...,i] += 0.5*dt*propagator(j,pred, method = "backward1")
        
        end_time = time.time()
        print("RK2 (direct) timestep time:{:.2f} seconds; {:.0f}% done.".format(end_time - start_time, t/Mt*100))
            
        if t % samplingPeriod == 0: EPHI[...,t//samplingPeriod] = EPHI_t
    
    return EPHI