#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from util.bipolar import *

"""
Here lie generic functions for pde visualisation or computation
"""

# Stats must be of shape [[Mx]*2, [K_phi]*o, T], with o being the moment-order.
def plotStatsAcrossTime2D(stats, moments, t_plot=None, titleBases=None, vlabel = r"$x_1/l_0$", hlabel = r"$x_2/l_0$", figName = None, fixColorbar = True, figsize = None, logTransform = False, colorbar = False, spacing = (-0.1,0), cbar_ticks_and_labels = None, loc_colorbar=[0.83, 0.153, 0.01, 0.7], vrange = [-np.inf, +np.inf]):
    
    # Initialise
    numRows = stats.shape[-1]
    numCols = len(moments)
    
    if fixColorbar == True: vmin = max(vrange[0],np.nanmin(stats)); vmax = min(vrange[1],np.nanmax(stats))
    else:                   vmin = None;        vmax = None
    
    # Plot the statistic across time in a single plot
    if figsize is None: figsize = (2*numCols, 1.5*numRows)
    fig, axs = plt.subplots(numRows, numCols, figsize=figsize, sharex=True, dpi=300)
    
    # Adjust spacing
    fig.subplots_adjust(wspace=spacing[0])
    fig.subplots_adjust(hspace=spacing[1]) 
    
    for row in range(numRows):
        
        # Timestamp row   
        if numCols>1: temp = (row,0)
        else: temp = row
        
        if t_plot is not None: axs[temp].text(-0.4, 0.9, f't = \n{t_plot[row]}', va='center', ha='left', transform=axs[temp].transAxes, fontsize=8)
        
        # Set title
        if titleBases is not None:
            titles = []
            if row == 0:
                for titleB in titleBases: titles = titleBases
            else:                         titles = [None]*len(titleBases)
        else: titles = [None]*numCols
        
        # And plot
        for col in range(numCols): 
            
            # Pick axis
            if numCols>1: ax = axs[row,col]; 
            else: ax = axs[row]
            
            # Forced equal aspect ratio
            ax.set_aspect('equal', 'box')
            
            # Set vertical label & ticks
            if col == 0: vlabelTemp = vlabel; vtickLabel = True
            else:        vlabelTemp = "";     vtickLabel = False
            
            # Set horizontal label & ticks
            if row == numRows-1: hlabelTemp = hlabel; htickLabel = True
            else:                hlabelTemp = "";     htickLabel = False
            
            stat = stats[(...,)+ (*moments[col],) + (row,)]
            
            if fixColorbar == False:
                if vmin == None: vmin = np.nanmin(stat)
                if vmax == None: vmax = np.nanmax(stat)
            
            im = plotFlow1D(stat, figName = figName,
                           t_label=hlabelTemp, x_label=vlabelTemp, title = titles[col], ax = ax, fig=fig, t_tickLabel = htickLabel, x_tickLabel = vtickLabel, vmin = vmin, vmax=vmax, colorbar = colorbar and not fixColorbar, logTransform=logTransform )
    
    # Create a colorbar
    if colorbar and fixColorbar:
        cbar_ax = fig.add_axes(loc_colorbar)  # [left, bottom, width, height]
        cbar=fig.colorbar(im, cax=cbar_ax, extend='max')
        fig.subplots_adjust(right=0.82)
        if cbar_ticks_and_labels is not None: 
            cbar.set_ticks(cbar_ticks_and_labels[0])
            cbar.set_ticklabels(cbar_ticks_and_labels[1])
            cbar.set_label(cbar_ticks_and_labels[2],labelpad=-9)

    return fig, axs

# In this function, the rows of uu are plotted along the vertical axis (such that
# the [0,0] element is in the lower-left corner), which pyplot calls the 
# “y-axis”, while the columns are plotted along the horizontal axis, which pyplot
# refers to as the “x-axis”. In my case, I refer to the vertical axis as x-axis 
# [plots rows of uu <--> u(x,t), i.e. x-dimension], and horizontal axis as the 
# t-axis (plots columns of uu, i.e. t-dimension).
def plotFlow1D(uu,t=None,x=None, t_label = None, x_label=None, title = None, figName = "spacetimeSolution", surfplot=False, ax = None, fig = None, t_tickLabel = True, x_tickLabel = True, vmin=None, vmax=None, colorbar = True, logTransform = False):
    
    # Initialise colormap
    cm = bipolar(neutral=0.0); cm = cm(np.linspace(0, 1, 256))
    #cm = plt.cm.gist_ncar(np.linspace(0, 1, 256))
    alphas = (np.linspace(0, 1, 256))
    cm[:, -1] = alphas  # Set the alpha channel
    cmap = colors.LinearSegmentedColormap.from_list("custom_jet", cm)
    
    # Set font size
    fontsize = plt.rcParams["font.size"]
        
    # plot
    M = uu.shape[0]
    if t is None: t= np.linspace(0,1-1/M,M)
    if x is None: x= np.linspace(0,1-1/M,M)
    if t_label is None: t_label = "t"
    if x_label is None: x_label = "x"
    
    tt, xx = np.meshgrid(t, x)
    
    #Log-transform if needed
   # if logTransform == True: uu = np.log10(uu)
    
    # surface plot
    if surfplot is True:
        if ax is not None: 
            print("Can only do surfplot when ax/fig is NOT given! Exiting."); return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(tt, xx, uu, cmap='jet')
        if logTransform == True: ax.set_zlabel('Log10 scaled')

    else:
        if ax is None or fig is None: fig, ax = plt.subplots()
        if logTransform == True: 
            
            if vmin == None: vmin = max(np.min(uu.flatten()), 1e-2)
            if vmax == None: vmax = np.max(uu.flatten())
            im = ax.pcolormesh(tt,xx,uu,cmap=cmap,shading='auto', norm = colors.LogNorm(vmin=vmin, vmax=vmax))
            
            
        else: im = ax.pcolormesh(tt,xx,uu,cmap=cmap,shading='auto', vmin=vmin, vmax=vmax)
        if colorbar == True: #fig.colorbar(im, ax = ax)
            fig.colorbar(im, ax=ax, extend='max')
         
    # Set ticks along t,x axes
    ax.set_xticks( np.round([t[M//5], t[M//2], t[M*4//5+1]], 1))
    ax.set_yticks( np.round([x[M//5], x[M//2], x[M*4//5+1]], 1))
    
    ax.set_xlabel(t_label, fontsize = fontsize, labelpad=-1)
    ax.set_ylabel(x_label, fontsize = fontsize, labelpad=0)
    if title is not None: ax.set_title(title, fontsize=fontsize)
    
    ax.tick_params(labelbottom=t_tickLabel, labelleft=x_tickLabel, labelsize=fontsize)
    
    if figName is not None: fig.savefig(figName + ".png", format="png", bbox_inches="tight", dpi=300)

    return im

def initialFlow_sin(x,pars):
    omega = pars[0]
    A = pars[1]
    B = pars[2]
    return A*np.sin(omega*x) + B

def initialFlow_gauss(x,pars):
    x0 = pars[0]; sigma = pars[1]; A = pars[2]
    f = A/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(x-x0)**2/(sigma**2))
    return f

def differencer(u, axis, order = 1, boundaryCond = "periodic", method = "central2"):
    
    if   method=="central2": return differencer_central(u, axis, order, boundaryCond)
    elif method=="forward1": return differencer_forward(u, axis, order, boundaryCond)
    elif method=="backward1": return differencer_backward(u, axis, order, boundaryCond)
    else: print("Unsupported method requested. Exiting!"); sys.exit(1)

# Differentiates u along given axis to 1st or second order. Differencing is
# performed according to a second-order accurate central finite differencing
# stencil. Periodic or norm-conserving boundary conditions only.

def differencer_central(u, axis, order = 1, boundaryCond = "periodic"):
    
    # Initialise
    u_m1 = np.roll(u, -1 , axis)
    u_p1 = np.roll(u, +1 , axis)
    if boundaryCond == "normconserving":
        index0 = [slice(None)]*u.ndim; index0[axis] = 0
        indexF = [slice(None)]*u.ndim; indexF[axis] = -1
    if boundaryCond != "periodic" and boundaryCond != "normconserving": 
        print("Only periodic and normconserving boundary conditions supported. Exiting!"); sys.exit(1)
    
    # Compute derivative
    if order   == 1:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": 
            # 1D special case: u_p1[0] = -u[0]; u_m1[-1] = -u[-1]
            u_p1[tuple(index0)] = -u[tuple(index0)]
            u_m1[tuple(indexF)] = -u[tuple(indexF)]
        du = (u_m1 - u_p1)/2.0
    elif order == 2:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": 
            # 1D special case: u_p1[0] = +u[0]; u_m1[-1] = +u[-1]
            u_p1[tuple(index0)] = +u[tuple(index0)]
            u_m1[tuple(indexF)] = +u[tuple(indexF)]
        
        du = u_m1 -2*u + u_p1
        
    return du

# Differentiates u along given axis to 1st or second order. Differencing is
# performed according to a first-order accurate forward finite differencing
# stencil. Periodic or norm-conserving boundary conditions only.

def differencer_forward(u, axis, order = 1, boundaryCond = "periodic"):
    
    # Initialise
    u_m1 = np.roll(u, -1 , axis)
    if order == 2: u_p2 = np.roll(u, -2 , axis)
    if boundaryCond != "periodic" and boundaryCond != "normconserving": 
        print("Only periodic and normconserving boundary conditions supported. Exiting!"); sys.exit(1)
    
    # Compute derivative
    if order   == 1:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": print("normconserving BCs not currently supported for forward/backward differencing. Exiting!"); sys.exit(1)
        du = u_m1 - u
    elif order == 2:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": print("normconserving BCs not currently supported for forward/backward differencing. Exiting!"); sys.exit(1)
        du = u_p2 -2*u_m1 +u 
        
    return du

# Same as above, but now using a backwards stencil.

def differencer_backward(u, axis, order = 1, boundaryCond = "periodic"):
    
    # Initialise
    u_p1 = np.roll(u, +1 , axis)
    if order == 2: u_m2 = np.roll(u, +2 , axis)
    if boundaryCond != "periodic" and boundaryCond != "normconserving": 
        print("Only periodic and normconserving boundary conditions supported. Exiting!"); sys.exit(1)
    
    # Compute derivative
    if order   == 1:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": print("normconserving BCs not currently supported for forward/backward differencing. Exiting!"); sys.exit(1)
        du = u - u_p1
    elif order == 2:
        if boundaryCond == "periodic": pass
        if boundaryCond == "normconserving": print("normconserving BCs not currently supported for forward/backward differencing. Exiting!"); sys.exit(1)
        du = u -2*u_p1 + u_m2
        
    return du