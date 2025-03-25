#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

"""
Here are various functions for directly calculating u(x,t) solutions of the 1D 
Kuramoto-Sivashinsky equation.
"""

def directComputeDistanceKSperiodic1D(uu, u0, l = 1.0, T =100.0, nu = 1.0, mu= 0.0,C0=0.0):
    
    # initialise
    Mx=uu.shape[0]
    Mt = uu.shape[1]
    dt = T/Mt
    dx = l/Mx
    uu0 = np.zeros([Mx, Mt]); uu0[:,0] =u0
        
    d_dt = np.zeros((Mt, Mt))
    d_dx = np.zeros((Mx, Mx))
    dd_dxdx = np.zeros((Mx, Mx))
    dddd_dx4 = np.zeros((Mx, Mx))
    for i in range(0,Mx):
        d_dx[i,(i+1)%Mx] = 0.5; d_dx[i,i-1] = -0.5
        dd_dxdx[i,i]=-2.0; dd_dxdx[i,(i+1)%Mx] = 1.0; dd_dxdx[i,i-1] = 1.0
        dddd_dx4[i,i]=6.0; dddd_dx4[i,(i+1)%Mx] = -4.0; dddd_dx4[i,i-1] = -4.0;
        dddd_dx4[i,(i+2)%Mx] = 1.0; dddd_dx4[i,i-2] = 1.0;
        
    for i in range(0,Mt):
        d_dt[i,i] = 1.0
        if i>0: d_dt[i,i-1] = -1.0
    
    # Sparsify
    d_dt = sp.sparse.coo_matrix(d_dt)
    d_dx = sp.sparse.coo_matrix(d_dx)
    dd_dxdx = sp.sparse.coo_matrix(dd_dxdx)
    dddd_dx4 = sp.sparse.coo_matrix(dddd_dx4)
    
    # Scale appropriately
    d_dx    = d_dx*dt/dx
    dd_dxdx = dd_dxdx*dt*nu/dx**2
    dddd_dx4= dddd_dx4*dt*mu/dx**4
    
    # Compute cost/loss/distance:
    convective = uu*((d_dx @ uu))
    diffusive = (dd_dxdx @ uu)
    fourthOrder = (dddd_dx4 @ uu)
    
    uu_temp = (uu @ d_dt.transpose()) - uu0 + convective + diffusive + fourthOrder
    CF = sum(sum(uu_temp**2))
    
    return CF*dt*dx + C0

def directSolveKSeqs1D(initialFlow, pars, nx = 2**10, L = 32.0, dt = 0.25, t0=0.0, tF =100, nu = 1.0, mu = 1.0):
    
    #Solution of Kuramoto-Sivashinsky equation:
    # u_t + u*u_x + nu*u_xx + mu*u_xxxx = 0, periodic boundary conditions on [0,L]
    # computation is based on v = fft(u), so linear term is diagonal
   
    nt = int(1+(tF - t0) / dt)
    
    # wave number mesh
    k = np.arange(-nx/2, nx/2, 1)
    
    x = np.linspace(start=0, stop=L, num=nx)
    
    # solution mesh in real space
    uu = np.ones((nx, nt))
    # solution mesh in Fourier space
    u_hat = np.ones((nx, nt), dtype=complex)
    
    u_hat2 = np.ones((nx, nt), dtype=complex)
    
    # initial condition 
    u0 = initialFlow(x,pars)
    
    # Fourier transform of initial condition
    u0_hat = (1 / nx) * np.fft.fftshift(np.fft.fft(u0))
    
    u0_hat2 = (1 / nx) * np.fft.fftshift(np.fft.fft(u0**2))
    
    # set initial condition in real and Fourier mesh
    uu[:,0] = u0
    u_hat[:,0] = u0_hat
    
    u_hat2[:,0] = u0_hat2
    
    # Fourier Transform of the linear operator
    FL = nu*(((2 * np.pi) / L) * k) ** 2 - mu * (((2 * np.pi) / L) * k) ** 4
    # Fourier Transform of the non-linear operator
    FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)
    
    # resolve EDP in Fourier space
    for j in range(0,nt-1):
      uhat_current = u_hat[:,j]
      uhat_current2 = u_hat2[:,j]
      if j == 0:
        uhat_last2 = u_hat2[:,0]
      else:
        uhat_last2 = u_hat2[:,j-1]
      
      # compute solution in Fourier space through a finite difference method
      # Cranck-Nicholson + Adam 
      u_hat[:,j+1] = (1 / (1 - (dt / 2) * FL)) * ( (1 + (dt / 2) * FL) * uhat_current + ( ((3 / 2) * FN) * (uhat_current2) - ((1 / 2) * FN) * (uhat_last2) ) * dt )
      # go back in real space
      uu[:,j+1] = np.real(nx * np.fft.ifft(np.fft.ifftshift(u_hat[:,j+1])))
      
      # clean the imaginary parts of u_hat and u_hat2 to help maintain numerical stability during long time evolutions
      u_hat[:,j+1] = (1 / nx) * np.fft.fftshift(np.fft.fft(uu[:,j+1]))
      u_hat2[:,j+1] = (1 / nx) * np.fft.fftshift(np.fft.fft(uu[:,j+1]**2))
      
    return uu, u0
    
def directSolveBurgersPeriodic1D(initialFlow, pars, Mx, Mt, l = 1.0, T =100.0, nu = 1.0):
    # Solves the Burgers equation directly through Euler's method. The spatial
    # derivative is represented through a second-order accurate central finite stencil.
    # Sparseness of the differencing stencils is not exploited. 
    
    # initialise
    dt = T/Mt
    dx = l/Mx
    uu = np.zeros((Mx, Mt))
    
    # temporal and spatial grids
    #t = np.linspace(0,T,Mt) 
    x = np.linspace(0,l,Mx)
    
    # initial condition 
    u0 = initialFlow(x,pars)
    
    # produce the differencing matrices d_dx and dd_dxdx, using periodic 
    # boundary conditions.
    d_dx = np.zeros((Mx, Mx))
    dd_dxdx = np.zeros((Mx, Mx))
    for i in range(0,Mx):
        d_dx[i,(i+1)%Mx] = 1.0; d_dx[i,i-1] = -1.0
        dd_dxdx[i,i]=-2.0; dd_dxdx[i,(i+1)%Mx] = 1.0; dd_dxdx[i,i-1] = 1.0
    # Scale appropriately
    d_dx    = sp.sparse.coo_matrix( d_dx*dt/(2*dx) )
    dd_dxdx = sp.sparse.coo_matrix( dd_dxdx*dt*nu/dx**2 )
    
    # Time-march forward now
    for i in range(0,Mt):
        if i == 0: u_prev = u0
        else: u_prev = uu[:,i-1]
        convective = u_prev*(d_dx @ u_prev)
        diffusive = dd_dxdx @ u_prev
        uu[:,i] = u_prev - convective - diffusive
    
    return uu, u0
