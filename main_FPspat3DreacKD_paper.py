# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
from util.bipolar import *

import config as cfg
import config_FP3D as cfgFP3
import PDEs.FokkerPlanck.FokkerPlanckGeneral as fpg
import PDEs.PdeGeneral as pdeg



def simComputeVariances(loadname, normalise = (False,False)):
    
    try: 
        # First load data
        with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname + "_stats.pkl", "rb") as f: varsLoaded = pickle.load(f)
        dx = varsLoaded["dx"]
        
        # Extract means
        if "means_mps" in varsLoaded: means = varsLoaded["means_mps"]
        else: means = varsLoaded["means_exact"]
        
        # If accessible, extract covs
        if "covs_mps" in varsLoaded: covs  = varsLoaded["covs_mps"]
        
        # Normalise if necessary
        if any(normalise):
            # Extract norms across x-space and use them to normalise means
            if normalise[0]==True: 
                if "norms_mps" in varsLoaded: norms = varsLoaded["norms_mps"]
                else: norms = 1+0*varsLoaded["means_exact"]
                normaliser = norms[...,np.newaxis,:]
            else: 
                if "norms_mps" in varsLoaded: norms_integ = varsLoaded["norms_integ"]
                else: norms_integ = np.ones(NUM_SAMPLES)
                normaliser = norms_integ
            # And normalise
            means /= normaliser
            covs  /= normaliser[...,np.newaxis,:]**2
        
        # Get RA means:
        phi_T = np.sum(means,  axis=(0,1,2))*dx**3
        
        # Get RA variance:
        R_T = np.zeros([K_phi,K_phi,NUM_SAMPLES+1])
        for i in range(K_phi):
            for j in range(K_phi):
                phi_ij_T = np.sum(means[...,i,:]*means[...,j,:],axis=(0,1,2))*dx**3
                R_T[i,j,:] =  phi_ij_T - phi_T[i,:]*phi_T[j,:]
        
        # Compute SGS variance:
        if "covs_mps" in varsLoaded: tau_T = np.sum(covs, axis=(0,1,2))*dx**3
        else: tau_T = np.nan
        
        # Extract timepoints and exit
        return phi_T, R_T, tau_T
    
    except FileNotFoundError:
        print("File " + cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname + "_stats.pkl" + " doesn't exist. Exiting simPlot.")
        
        return np.nan,np.nan,np.nan    

def simNonReacCompare(loadnameExact, loadnameMps):
    
    try: 
        # First load data
        with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadnameMps+ "_stats.pkl", "rb")   as f: varsLoadedMps   = pickle.load(f)
        with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadnameExact+ "_stats.pkl", "rb") as f: varsLoadedExact = pickle.load(f)
        
        # Initialise    
        if varsLoadedMps["T"] != varsLoadedExact["T"]: print("T of the MPS and Exact sims don't match; exiting."); sys.exit(1) 
        
        # Extract exact and mps means, and dx, chi
        means_exact = varsLoadedExact["means_exact"]
        means_mps   = varsLoadedMps["means_mps"]
        #norms_integ_mps = varsLoadedMps["norms_integ"]
        dx          = varsLoadedMps["dx"]
        
        # Compute fidelities
                
        # Get total mps reacs
        phi_T_mps   = np.sum(means_mps[...,:,:],  axis=(0,1,2))*dx**3#/norms_integ_mps
        
        # Get total exact reacs
        means_exact_init = means_exact[...,:,0]
        phi_T_exact = np.tile(np.sum(means_exact_init[...,0],axis=(0,1,2))*dx**3, (means_exact.shape[-2],means_exact.shape[-1]))

        # Compute errors & return
        error_mean     = np.sqrt(np.sum((means_exact - means_mps)**2)/means_exact.size)
        error_totReacs = np.sqrt(np.sum((phi_T_exact - phi_T_mps)**2)/phi_T_mps.size  )
        return error_mean, error_totReacs
        
    except FileNotFoundError:
        print("File " + loadnameMps + " and/or " + loadnameExact + " doesn't exist. Exiting simNonReacCompare.")
                
        return np.nan, np.nan

def simReacsCompare(loadname):
    
    try: 
        # First load data
        with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname+ "_stats.pkl", "rb") as f: varsLoaded = pickle.load(f)
        dx = varsLoaded["dx"]
        norms_integ = varsLoaded["norms_integ"]
        
        # Extract means & norms
        if "norms_mps" in varsLoaded: 
            means = varsLoaded["means_mps"]
            norms_integ = varsLoaded["norms_integ"]
        else: 
            means = varsLoaded["means_exact"]
            norms_integ = np.zeros(means.shape[-1])*np.nan
        
        # Integrated reactant means
        phi1_T = np.sum(means[...,0,:],  axis=(0,1,2))*dx**3
        phi2_T = np.sum(means[...,1,:],  axis=(0,1,2))*dx**3
        
        # Compute errors & return
        
        # Consumption and norm errors (RMS)
        error_reac  = np.sqrt(np.sum((phi1_T - phi2_T  )**2)/len(phi1_T)     )
        error_norms = np.sqrt(np.sum((1.0 - norms_integ)**2)/len(norms_integ))

        return max(1e-16, error_reac), max(1e-16, error_norms)
    
    except FileNotFoundError:
        print("File " + cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname+ "_stats.pkl" + " doesn't exist. Exiting simPlot.")
        
        return (np.nan, np.nan)

### Figures:
# Nature standard figure sizes: single-column is 90mm = 3.55in, double-column is 170mm = 6.7in.
# Max size = 180mm (w) & 215mm (h) => 7.089in & 8.465in. 

# Sted default plot-settings
fontszDefault = 8
plt.rcParams.update({'font.size': fontszDefault,"font.family" : "Times New Roman" })
plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4  # Adjust for x-axis major ticks
plt.rcParams['ytick.major.width'] = 0.4  # Adjust for y-axis major ticks
plt.rcParams['xtick.minor.width'] = 0.2  # Adjust for x-axis minor ticks
plt.rcParams['ytick.minor.width'] = 0.2  # Adjust for y-axis minor ticks
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.markeredgewidth'] = 0.25
mpl.pyplot.viridis()

## Figure 1, 3D-visualisation of first moment + plots of pdfs, for illustrating the simulation case.

# # Extracts 3D data required for creating Figure 1a in paraview.
def paperFig1(samples_plot, omega_mps, chi_mps,logTransform = True):
    
    plt.rcParams.update({'font.size': 7})
    
    with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + cfgFP3.savename_mps.format("LinLin",0,omega_mps,NK,chi_mps).replace('.', 'p') + "_stats.pkl", "rb") as f: 
                
        # Initialise
        varsLoaded = pickle.load(f)
        l = varsLoaded["l"]
        
        # Extract and clean ratio between phi1, phi2 moments
        mratios = varsLoaded["means_mps"][...,0,:]/varsLoaded["means_mps"][...,1,:]
        print("Min/Max of <Phi1>/<Phi2> are {} and {} @ Omega={},chi={}.".format(mratios.min(), mratios.max(), omega_mps, chi_mps))
        mratios[mratios<0] = np.nan
        mratios[mratios>10]= 10
        
        # Set up coordinates
        x = y = z = np.linspace(0,l*(1-1/2**NK),2**NK)
        t = np.arange(0,5,1)
        
        # Save 3D data in VTK form
        os.makedirs(cfg.dir_paper + "visu3D/", exist_ok=True)
        for t in samples_plot:
            gridToVTK(
                "paper/visu3D/t{}".format(t),
                x,
                y,
                z,
                pointData={"log10avgPhi1/avgPhi2": np.log10(mratios[...,t]+1e-8).ravel()},
            )
            
        # Extract PDFs
        p_t_mpses = varsLoaded["p_t_mpses"]
        p_t_mpses = [p_t_mpses[i] for i in samples_plot]
        pp_mps = np.zeros([2**NK]*(2) + [2] + [len(p_t_mpses)])
        coords_x1 = [2**(NK-1)-1, 0] # x1= l*(1/2, 0)
        coords_x2 = [2**(NK-0)-1, 2**(NK-1) -1] # x2= l*(1,   1/2)
        coords_x3 = [2**(NK-1)-1, 2**(NK-0)-1] # x3= l*(1/2, 1)
        
        for t in range(len(p_t_mpses)):
            for i in range(2): pp_mps[...,i,t] = fpg.getPdfSlice(p_t_mpses[t], 0, True, [coords_x1[i],coords_x2[i], coords_x3[i]], K_x, K_phi)[...,0]
            
            
            print("{:.2f}".format(0.5*2**-14*np.sum(pp_mps[...,0,t]+pp_mps[...,1,t],axis=(0,1))))
        
        
        if logTransform:
            top = 100
            bottom = 1e-2
            
            pp_mps[pp_mps<bottom] = np.nan
            cbar_ticks = [0.01, 0.1, 1, 10, top]
            cbar_labels= ["0.01", "0.1", "1", "10", "{:.0f}".format(top)]
            cbar_labels[0] = r"$\leq$" + cbar_labels[0]
            cbar_ticks_and_labels=(cbar_ticks,cbar_labels, r"$f(\varphi_1,\varphi_2;\mathbf{x},t)$")
        else: cbar_ticks_and_labels = None
        
        
        fig, axs = pdeg.plotStatsAcrossTime2D(pp_mps, [[0],[1]],None, ["",""],figsize = (2.65, 4), spacing = (-0.3,0.03), vlabel = r"$\varphi_1$", hlabel = r"$\varphi_2$", figName = None, fixColorbar=True, logTransform = logTransform, colorbar=True, cbar_ticks_and_labels = cbar_ticks_and_labels, loc_colorbar=[0.78, 0.153, 0.01, 0.7])
        fig.text(0.05, .91, r"$C_{{\Omega}}$={:.2f}, $\chi$={}, t = 0, 0.25, 1, 2".format(omega_mps/OMEGA_MAX,chi_mps), ha='left', va='top', fontsize = 8)
        
        # Save figure
        os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
        fig.savefig(cfg.dir_paper+"figures/fig1.png", format="png", bbox_inches="tight", dpi=cfg.dpi)
        
        plt.rcParams.update({'font.size': fontszDefault})

## Figure 2, outlining the results for the reac = 0 case.

# Extracts data needed to produce Figure 2
def paperFig2_prep(chisOmegas, samples_plot):
    
    # A useful support-function
    def extractRatio(means):
        means = means[:,:,2**(NK-1),...]
        mratio = means[...,0,:]/means[...,1,:]        
        return mratio
    
    ## Fig 2a:
    
    # Initialise and load exact data
    with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + cfgFP3.savename_exactMoms.format(NK) + "_stats.pkl", "rb") as f: 
    
        # Initialise
        varsLoaded = pickle.load(f)
        #MtSamples = varsLoaded["MtSamples"]
        mratios = np.zeros([2**NK, 2**NK, 1+len(chisOmegas), len(samples_plot)])
        
        # Get exact ratio between phi1, phi2 moments
        mratios[...,0,:] = extractRatio(varsLoaded["means_exact"])[...,samples_plot]
    
    # Then load MPS data
    for i, chiOmega in enumerate(chisOmegas):
        chi   = chiOmega[0]
        omega = chiOmega[1]
        loadname_chi_omega = cfgFP3.savename_mps.format("LinLin",0, omega, NK, chi).replace('.', 'p')
        
        with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname_chi_omega + "_stats.pkl", "rb") as f: 
            mratios[...,i+1,:] = extractRatio(pickle.load(f)["means_mps"])[...,samples_plot]
    
    ## Fig 2b & 2c:
    # Define nonreac, inconsistency & norm error-matrix (function of Da, omega, chi) and fill it
    errors = np.nan*np.zeros( [2,1+len(reacs), len(OMEGAS), len(chis)] )
    
    # Fig 2b: fill nonreac comparison with exact part
    for j, omega in enumerate(OMEGAS):
        for k, chi in enumerate(chis):
            #
            error_mean, error_totReacs = simNonReacCompare(cfgFP3.savename_exactMoms.format(NK), cfgFP3.savename_mps.format("LinLin", 0, omega,NK,chi).replace(".","p"))
            errors[0,0,j, k] = error_totReacs
            errors[1,0,j, k] = error_mean
    
    # Fig 2c: fill inconsistency/norm error parts
    for i, reac in enumerate(reacs):
        for j, omega in enumerate(OMEGAS):
            for k, chi in enumerate(chis):
                error_reac, error_norm = simReacsCompare(cfgFP3.savename_mps.format("LinLin",reac, omega, NK, chi).replace('.', 'p'))
                errors[0,i+1, j, k] = error_reac
                errors[1,i+1, j, k] = error_norm
    
    return mratios, errors
    
# Creates subfigure 2a, illustrating both the numerically-exact first moment and corresponding MPS solutions
def paperFig2_draw_a(chisOmegas,mratios):
    
    # First define colorbar ticks & labels:
    cbar_ticks = [1/MAX_MRATIO, 1, MAX_MRATIO]#np.round(np.linspace( np.nanmin(mratios), np.nanmax(mratios),13), 1)
    cbar_labels= ["{:.1f}".format(tick) for tick in cbar_ticks]
    cbar_labels[0] = r"$\leq$" + cbar_labels[0]
    cbar_labels[-1] = r"$\geq$" + cbar_labels[-1]
    cbar_ticks_and_labels=(cbar_ticks,cbar_labels, r"$\langle\Phi_1\rangle/\langle\Phi_2\rangle$")
    loc_colorbar=[0.83, 0.153, 0.01, 0.7]
    
    # Mark-out mratio plotting domain:
    mratios[mratios<=1/MAX_MRATIO] = 1/MAX_MRATIO
    mratios[mratios>MAX_MRATIO] = MAX_MRATIO
    
    # Now plot
    fig, axs = pdeg.plotStatsAcrossTime2D(mratios, [[i] for i in range(len(chisOmegas)+1)], fixColorbar=True, colorbar=True, spacing=(+0.02,-0.15), 
                                       figsize = (7.089, 8.465*2.02/5), logTransform = True,cbar_ticks_and_labels=cbar_ticks_and_labels, vrange = [1/MAX_MRATIO, MAX_MRATIO], loc_colorbar = loc_colorbar )
    
    # Denote that this is subplot “A”
    height = 0.945
    fig.text(0.074, height+0.002, "A", ha='left', va='top', weight='bold')#, fontsize = 8)
    
    # Denote chi,omega of each column
    wlocs = [0.15, 0.22, 0.305, 0.393, 0.482, 0.565, 0.653,0.74]
    fig.text(wlocs[0],height*.985, "Exact", ha='left', va='top', fontsize = 8)
    for i,pair in enumerate(chisOmegas): fig.text(wlocs[i+1],height, r"$\chi={}$".format(pair[0]) +",\n" + r"$C_\Omega={:.2f}$".format(pair[1]/OMEGA_MAX), ha='left', va='top', fontsize = 8)

    # Save & return handles
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/fig2a.png", format="png", bbox_inches="tight", dpi=cfg.dpi)
    return fig, axs

# Draws subfigures 2b and 2c, plotting errors
def paperFig2_draw_bc(errors, xvals = None):
    
    # Is xaxis chis or single-core CPU-times?
    if type(xvals) == dict:
        xlabel = 'CPU-time (s)'
        figname= "figErrvsCPU"
        CHI0=8
        xvals =[xvals[chi] for chi in [chi for chi in chis if chi>=CHI0]]
    else: 
        xlabel = r"$\chi$"
        figname= "fig2bc"
        xvals = chis
    
    # Produce plot
    fig, axs = plt.subplots(2, 1+len(reacs), figsize=(7.089*.7335, 8.465*1.0/5), sharex=False, sharey = False, dpi=cfg.dpi)
    
    for h in range(2):
        for i in range(1+len(reacs)):
            for j in range(len(OMEGAS)): 
                errs = errors[h,i,j,:]
                axs[h,i].plot(xvals, errs[len(errs)-len(xvals):], "*--", color=cm[j])#, label = r"C_\Omega={:.2f}$".format(OMEGAS[k]/OMEGA_MAX))
    
    # Set axis properties
    for h in range(2):
        for i in range(1+len(reacs)):
            ax = axs[h,i]
            ax.set_yscale('log')
            
            if xlabel == 'CPU-time (s)':
                ax.set_xscale('log', base=10); ax.set_xticks(CPU_time_ticks)
                if i == 0 and h==1: ax.set_xticklabels([r"$1$"] + [r"$10^{:.0f}$".format(i) for i in range(1,len(CPU_time_ticks))])
                if i == len(reacs) and h==1: ax.set_xticklabels([r""] + [r"$10^{:.0f}$".format(i) for i in range(1,len(CPU_time_ticks))])
                elif i == 1: ax.set_xticklabels([r"$1$"] + [r"$10^{:.0f}$".format(i) for i in range(1,len(CPU_time_ticks[:-1]))] + [r""])
                else: ax.set_xticklabels([r""] + [r"$10^{:.0f}$".format(np.log10(time)) if time<CPU_time_ticks[-1] else r"" for time in CPU_time_ticks[1:]])
            else: 
                ax.set_xscale('log', base=2); ax.set_xticks(chis)
                ax.set_xticklabels([r"$2^{:.0f}$".format(np.log2(chi)) if np.log2(chi)%2==0 else r"" for chi in chis])

            if h==1:ax.set_xlabel(xlabel, labelpad=-1)

            # Set major and minor ticks
            if h ==0 and i==0: ax.set_yticks([10**-(1*i+1) for i in range(3)])
            if h ==0 and i >0:
                ax.set_yticks([10**-(1*i+1) for i in range(4)])
                ax.set_ylim([0.5*1e-4,0.15])
            if h ==1 and i >0:
                ax.set_yticks([10**-(1*i+1) for i in range(4)])
                ax.set_ylim([1e-4,0.75])
            if           i >1: ax.yaxis.set_ticklabels([])
            
    # Set labels
    labelpad = -0.5
    axs[0,0].set_ylabel(r"$\overline{\langle \Phi_\alpha\rangle}$ Error", labelpad=labelpad)
    axs[1,0].set_ylabel(r"$\langle \Phi_\alpha\rangle$ Error", labelpad=labelpad)
    axs[0,1].set_ylabel(r"$\overline{\langle \Phi_1\rangle-\langle\Phi_2\rangle}$"+"\nError", labelpad=labelpad) #axs[0,1].set_ylabel("Consump. Error", labelpad=labelpad)    
    axs[1,1].set_ylabel(r"$\overline{\langle1\rangle}$ Error", labelpad=labelpad) #axs[1,1].set_ylabel("Norm Error", labelpad=labelpad)
    fig.subplots_adjust(hspace=+0.025, wspace =+0.015)
    
    minor_locator = mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=10)
    [axs[0,1+i].yaxis.set_minor_locator(minor_locator) for i in range(len(reacs))]
    
    # Create legend
    legend_handles = [mpl.lines.Line2D([], [], color=cm[i], marker="*", linestyle='None', 
                                markersize=2, label = r"$C_\Omega$={:.2f}".format(OMEGAS[i]/OMEGA_MAX)) for i, omega in enumerate(OMEGAS)]
    axs[0,4].legend(handles=legend_handles, loc=[1.05,-0.03], handlelength=1, handletextpad=0.1, borderpad=0.1, ncol = 1)

    # Denote these as subplot “b” and "c"
    height = 0.97
    if xlabel == r"$\chi$": labels=["B","C"]
    else: labels=["A","B"]
    fig.text(-0.095, height, labels[0], ha='left', va='top',weight='bold')#, fontsize = 8)
    fig.text(0.19,   height, labels[1], ha='left', va='top',weight='bold')#, fontsize = 8)
    
    # Denote Damkohler numbers of each column
    fig.text(+0.04,  height-0.02, r"Da$=0$", ha='left', va='top')#, fontsize = 8)
    fig.text(+0.33,  height-0.02, r"Da$=0$", ha='left', va='top')#, fontsize = 8)
    fig.text(+0.47, height-0.02, r"Da$=0.5$", ha='left', va='top')#, fontsize = 8)
    fig.text(+0.63, height-0.02, r"Da$=1$", ha='left', va='top')#, fontsize = 8)
    fig.text(+0.78, height-0.02, r"Da$=1.5$", ha='left', va='top')#, fontsize = 8)
    
    # Seperate first two columns slightly
    pos0 = axs[0,0].get_position().bounds
    pos1 = axs[1,0].get_position().bounds       
    axs[0,0].set_position([pos0[0] - 0.13, pos0[1], pos0[2], pos0[3]]) 
    axs[1,0].set_position([pos1[0] - 0.13, pos1[1], pos1[2], pos1[3]]) 
    
    # Save & return handles
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/" + figname+ ".png", format="png", bbox_inches="tight", dpi=cfg.dpi)
    return fig, axs
    

## Figure 3, outlining the results for the reac >= 0 case.

# Extracts data needed to produce Figure 3$\langle\Phi_1\rangle/\langle \Phi_2\rangle
def paperFig3_prep(chi):
    
    ## Fig 3b
    
    # Define extraction coordinates
    coords_x1 = 2**(NK-1) -1    # x1=l/2
    coords_x2 = 2**(NK-1) -1    # x2=l/2
    coords_x3 = 2**(NK-1) -1    # x3=l/2
    
    # Initialise pdf-matrix
    pp_mps = np.zeros([2**NK, 2**NK, len(reacs),len(OMEGAS) ])
    for i,reac in enumerate(reacs):
        for j,omega in enumerate(OMEGAS):
            loadname_chi_reac_omega = cfgFP3.savename_mps.format("LinLin",reac , omega, NK, chi).replace('.', 'p')
            with open(cfg.dir_data + cfgFP3.dir_results+ stats_dir + loadname_chi_reac_omega + "_stats.pkl", "rb") as f: 
                p_t_mps = pickle.load(f)["p_t_mpses"][-1]
                pp_mps[...,i,j] = fpg.getPdfSlice(p_t_mps, 0, True, [coords_x1,coords_x2, coords_x3], K_x, K_phi)[...,0]
    
    return pp_mps

# Plots Fig 3
def paperFig3_draw(pp_mps, logTransform = False):
    
    plt.rcParams.update({'font.size': 10})
    
    # First define colorbar ticks & labels:
    if logTransform:
        top = np.ceil(np.round(pp_mps.max()/100,1)*100)
        bottom = 1e-2
        
        pp_mps[pp_mps<bottom] = np.nan
        cbar_ticks = [0.01, 0.1, 1, 10, top]
        cbar_labels= ["0.01", "0.1", "1", "10", "{:.0f}".format(top)]
        cbar_labels[0] = r"$\leq$" + cbar_labels[0]
        cbar_ticks_and_labels=(cbar_ticks,cbar_labels,r"$f(\varphi_1,\varphi_2;\mathbf{x},t)$")
    else: cbar_ticks_and_labels = None
    loc_colorbar = [0.83, 0.153, 0.01, 0.7]
    
    # Now plot
    fig, axs = pdeg.plotStatsAcrossTime2D(pp_mps, [[i] for i in range(pp_mps.shape[2])], vlabel = r"$\varphi_1$", hlabel = r"$\varphi_2$", fixColorbar=True, colorbar=True, spacing=(0.030,-0.15), 
                                       figsize = (7.089*0.5, 8.465*0.402), logTransform = logTransform,cbar_ticks_and_labels=cbar_ticks_and_labels, loc_colorbar=loc_colorbar)
    
    # Denote that this is subplot “b”
    wloc = -0.1
    hloc = 0.905

    wlocs_reac = [.15, .305, .50, .655]
    hlocs_omega = [.8, .62, .43, .26]#, .212]
    
    # Place Omegas on each row
    for i,omega in enumerate(OMEGAS): fig.text(wloc, hlocs_omega[i], r"$C_\Omega=$" +"\n{:.2f}".format(omega/OMEGA_MAX), ha='left', va='top')#, fontsize = 8)
    
    # Denote Damkohler numbers of each column
    for i,reac in enumerate(reacs): fig.text(wlocs_reac[i], hloc, r"Da$={}$".format(reac), ha='left', va='top')#, fontsize = 8)
   
    # Save & return handles
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/fig3.png", format="png", bbox_inches="tight", dpi=cfg.dpi)
    plt.rcParams.update({'font.size': fontszDefault})
    return fig, axs

def paperFig4_prep(chi, normalise = (False,False)):
    
    # Initialise
    phi_Ts_exact = np.zeros([K_phi,       NUM_SAMPLES])
    R_Ts_exact   = np.zeros([K_phi,K_phi, NUM_SAMPLES])
    phi_Ts_mps   = np.zeros([K_phi,       NUM_SAMPLES+1, len(reacs), len(OMEGAS)])
    R_Ts_mps     = np.zeros([K_phi,K_phi, NUM_SAMPLES+1, len(reacs), len(OMEGAS)])
    tau_Ts_mps   = np.zeros([K_phi,K_phi, NUM_SAMPLES+1, len(reacs), len(OMEGAS)])
        
    # Fill above objects, first from exact sims, then mps ones:
    phi_Ts_exact, R_Ts_exact, _ = simComputeVariances(cfgFP3.savename_exactMoms.format(NK))
    for i, reac in enumerate(reacs):
        for j,omega in enumerate(OMEGAS): 
            phi_Ts_mps[...,i,j], R_Ts_mps[...,i,j], tau_Ts_mps[...,i,j] = simComputeVariances(cfgFP3.savename_mps.format("LinLin", reac, omega,NK,chi).replace('.', 'p'), normalise = normalise)
        
    # Done    
    return [phi_Ts_exact, R_Ts_exact], [phi_Ts_mps, R_Ts_mps, tau_Ts_mps]


# Plot fig 4
def paperFig4_draw(stats_exac, stats_mps):
    
    plt.rcParams.update({'font.size': 10})
    
    # Produce plot
    t_plot = np.linspace(0,T,NUM_SAMPLES+1)
    fig, axs = plt.subplots(3, len(reacs), figsize=(7.089*.5, 8.465*.25), sharex="col", sharey = "row", dpi=cfg.dpi)
    for i in range(3):
        for j in range(len(reacs)):
            for k in range(len(OMEGAS)):
                
                # Define what to plot 
                if i==0: stats = stats_mps[0][0,:,j,k];   ylabel = r"$ \overline{\langle \Phi_1\rangle}$"
                if i==1: stats =-stats_mps[1][0,1,:,j,k]; ylabel = r"$-R_{12}$"
                if i==2: stats =-stats_mps[2][0,1,:,j,k]; ylabel = r"$-\overline{\Upsilon_{12}}$"

                # Plot & set ylabel
                axs[i,j].plot(t_plot[:], stats, "-",color = cm[k] )#, label = r"$C_\Omega={:.2f}$".format(OMEGAS[k]/OMEGA_MAX))
                if j==0: axs[i,j].set_ylabel(ylabel, labelpad=1.0)

                # Set axis properties
                if i == 0:
                    axs[i, j].set_yticks(ticks=[0.5, 0.4, 0.3])
                    axs[i, j].set_ylim([0.225, 0.5125])
                if i == 1:
                    axs[i, j].set_yticks(ticks=[0.05, 0.03, 0.01])
                    axs[i, j].set_ylim([0.005, 0.055])
                if i == 2:
                    axs[i, j].set_yticks(ticks=[0.02, 0.01, 0])
                    axs[i, j].set_ylim([0, 0.025])

                if j<len(reacs)-1: axs[i,j].set_xticks(ticks = [0,1,2], labels = ["0","1"," "])
                else:              axs[i,j].set_xticks(ticks = [0,1,2], labels = ["0","1","2"])
                if i==2: axs[i,j].set_xlabel(r"$t/T_0$", labelpad=0)
    
    fig.subplots_adjust(hspace=+0.025, wspace =+0.025)
    
    # Create legend
    legend_handles = [mpl.lines.Line2D([], [], color=cm[i], marker="*", linestyle='None', 
                                markersize=2, label = r"$C_\Omega={:.2f}$".format(OMEGAS[i]/OMEGA_MAX)) for i, omega in enumerate(OMEGAS)]
    axs[0,len(reacs)-1].legend(handles=legend_handles, loc=[1.05,-.14], handlelength=1, handletextpad=0.1, borderpad=0.02, ncol = 1, labelspacing=0.1,columnspacing=-0.1)
    
    # Denote Damkohler numbers of each column
    h = 0.95
    wlocs = [.16, .34, .55, .73]
    for i, reac in enumerate(reacs): fig.text(wlocs[i], h, r"Da$={}$".format(reac), ha='left', va='top')#, fontsize = 8)
    
    
    # Save & return handles
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/fig4.png", format="png", bbox_inches="tight", dpi=cfg.dpi)
    plt.rcParams.update({'font.size': fontszDefault})
    return fig, axs
    
# Plot fig S1
def paperFigS1_draw(chis_times_dict, CHI0, CHI1, cm):
    
    # Extract chis and times
    x_values = list(chis_times_dict.keys())   # chis
    y_values = list(chis_times_dict.values()) # times
    
    # Fit linear regression models
    start  = x_values.index(CHI0)
    cutoff = x_values.index(CHI1)
    coeffs_1 = np.polyfit(np.log2(x_values[start:cutoff+1]), np.log2(y_values[start:cutoff+1]),  1)
    coeffs_2 = np.polyfit(np.log2(x_values[cutoff:]), np.log2(y_values[cutoff:]),  1)
    poly_1   = np.poly1d(coeffs_1)
    poly_2   = np.poly1d(coeffs_2)
    
    # Generate x-values for the fitted curve
    x_fit_1 = np.linspace(min(x_values[start:cutoff+1]), max(x_values[start:cutoff+1]), 400)
    y_fit_1 = 2**poly_1(np.log2(x_fit_1))  # Calculate y-values using the polynomial model
    x_fit_2 = np.linspace(min(x_values[cutoff:]), max(x_values[cutoff:]), 400)
    y_fit_2 = 2**poly_2(np.log2(x_fit_2))  # Calculate y-values using the polynomial model

    # Create the plot
    fig = plt.figure(figsize=(7.089*0.5, 8.465*0.2))
    plt.plot(x_values, y_values, 'o', label='Measured', color=cm[0])  # Plot the original data points
    plt.plot(x_fit_1, y_fit_1, '-', label=r'$\sim \chi^{{{:.1f}}}$ fit'.format(coeffs_1[0]),color=cm[1])  # Plot regression line 1
    plt.plot(x_fit_2, y_fit_2, '-', label=r'$\sim \chi^{{{:.1f}}}$ fit'.format(coeffs_2[0]),color=cm[2])  # Plot regression line 1
    
    plt.xscale('log', base=2)  # Set x-axis to log scale with base 2
    plt.yscale('log', base=10)  # Set y-axis to log scale with base 2
    plt.xlabel('χ', labelpad=-1)
    plt.ylabel('CPU-time (s)', labelpad=-1)
    plt.xticks(ticks = x_values[::2], labels = [r"${}$".format(x) for x in x_values[::2]])
    if "CPU_time_ticks" in vars() or "CPU_time_ticks" in globals(): plt.yticks(ticks = CPU_time_ticks, labels = [r"$1$", r"$10^1$", r"$10^2$", r"$10^3$"])
    plt.legend()
    plt.show()
    
    # Save figure
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/figCPUvsChi.png", format="png", bbox_inches="tight", dpi=cfg.dpi)
    return fig

# Plot fig S2
def paperFigS2_draw(chisN_times_dict, N_dnstimes_dict):
        
    # Extract chis and times
    xDNS_values = list(N_dnstimes_dict.keys()); numN = len(xDNS_values)   # N = log2(M)
    yDNS_values = list(N_dnstimes_dict.values()) # times of DNS
    xMPS_values = list(chisN_times_dict.keys()); numChi = int(len(xMPS_values)/numN); chis = [xMPS_values[c][0] for c in range(0,numN*numChi, numN) ]
    yMPS_values = list(chisN_times_dict.values())
    
    # Fit linear regression models
    coeffs_loglogDNS =   np.polyfit(xDNS_values[1:-1], np.log2(yDNS_values[1:-1]),  1)
    poly_DNS   = np.poly1d(coeffs_loglogDNS)
        
    # Generate x-values for the fitted curve
    x_fit = 2**np.linspace(min(xDNS_values), max(xDNS_values), 64)
    y_fit_DNS = 2**poly_DNS(np.log2(x_fit))  # Calculate y-values using the polynomial model
        
    # Create the plot
    fig = plt.figure(figsize=(7.089*0.5, 8.465*0.2))
    plt.plot([2**val for val in xDNS_values], yDNS_values, 'o', label='FD, ' +r'$\sim M^{{{:.1f}}}$ fit'.format(coeffs_loglogDNS[0]), color=cm[0])  # Plot the original data points
    plt.plot(x_fit, y_fit_DNS, '-',color=cm[0])  # Plot regression line 1
    [plt.plot([2**val for val in xDNS_values], yMPS_values[c*numN:(c+1)*numN], 'o-.', label= r"MPS, $\chi={}$".format(chis[c]), color=cm[1+(numChi-1-c)]) for c in range(numChi-1,-1,-1)]  # Plot regression line 1
    
    print("Estimated DNS time@N=7:", int(y_fit_DNS[-1]), "s")
    print("Actual chi=32 MPS time@N=7:", int(chisN_times_dict[(32,7)]), "s")
    
    plt.xscale('log', base=2)  # Set x-axis to log scale with base 2
    plt.yscale('log', base=10)  # Set y-axis to log scale with base 2
    plt.xlabel('M', labelpad=-0.5)
    plt.ylabel('CPU-time (s)', labelpad=-1.5)
    plt.xticks(ticks = [2**val for val in xDNS_values], labels = [r"${}$".format(2**x) for x in xDNS_values])
    plt.yticks(ticks = [0.01] + CPU_time_ticks[::2] + [10**4], labels = [r"$10^{-2}$", r"$1$", r"$10^2$", r"$10^4$"])
    plt.legend(loc=[1.015,+0.3], handlelength=0.5, handletextpad=0.9, borderpad=0.1, ncol = 1)
    plt.show()
    
    # Save figure
    os.makedirs(cfg.dir_paper + "figures/", exist_ok=True)
    fig.savefig(cfg.dir_paper+"figures/fig_CPUvsM.png", format="png", bbox_inches="tight", dpi=cfg.dpi)    
    
    return fig

if __name__ == '__main__':
    
    # Set colormap
    #cm = bipolar(neutral=0.0); cm = cm(np.linspace(0, 1, 5))
    cm = plt.cm.nipy_spectral(np.linspace(0.1, 1, 5))
    #alphas = (np.linspace(0.5, 1, n))
    #cm[:, -1] = alphas  # Set the alpha channel
    
    # Define dimensionality & set split-option
    K_x = 3
    K_phi = 2
    reacs = [0,0.5,1,1.5]
    NUM_SAMPLES = 16; T = 2
    OMEGA_MAX = 1814
    stats_dir = "stats/"
    MAX_MRATIO = 5
    OMEGAS = [449, 904, 1359, 1814] #
    CHI0=8; CHI1=48

    # Measured using the ARC-HTC clusters of Oxford Universty
    chis_times_dict = {8:3.19,12:3.78,16:4.57,24:6.91,32:11.32,48:25.29,64:52.82,96:143.53,128:369.98,192: 995.40, 256: 2551.01}#, 384: 7574.71, 512:16732.50 }
    chisN_times_dict= {(8,3):1.30, (8,4):1.78, (8,5):2.27, (8,6):2.88, (8,7):3.19, (32,3):1.85, (32,4):3.63, (32,5):6.02, (32,6):8.99, (32,7):11.32, (128,3):2.84, (128,4):33.40, (128,5):119.90, (128,6):246.18, (128,7):369.98 }
    N_dnstimes_dict = {3:0.01,4:0.44,5:16.41,6:695.32, 7: np.nan}

    CPU_time_ticks = [1, 10, 100, 1000]
    
    # Select grid and chi resolutions, and the omegas and reacs for plotting
    NK = 7
    chis=[2,4, 8,16,32,64,96,128]

    # For figure 1
    samples_plot_fig1 = [0, 1, 8, 16]

    #For figure 2
    samples_plot_fig2 = [1, 2, 8, 16]

    chisOMEGAS = [(chis[-1],OMEGAS[0]),(chis[-1],OMEGAS[1]) ,(chis[-2],OMEGAS[1]), (chis[-2],OMEGAS[2]), (chis[-3],OMEGAS[2]), (chis[-3],OMEGAS[3]),(chis[-4],OMEGAS[3])]
    
    # Create figure 1
    paperFig1(samples_plot_fig1, omega_mps = OMEGAS[-1], chi_mps = chis[-1], logTransform=True)

    # Create figure 2 & S2
    mratios_a, errors_fig2 = paperFig2_prep(chisOmegas = chisOMEGAS, samples_plot=samples_plot_fig2)
    fig2a,_ = paperFig2_draw_a(chisOMEGAS,mratios_a)
    fig2bc,_ = paperFig2_draw_bc(errors_fig2)
    
    # Create figure 3
    pp_mps = paperFig3_prep(chis[-1])
    fig3a,_ = paperFig3_draw(pp_mps, logTransform=True)
    
    # Create figure 4
    stats_exac, stats_mps = paperFig4_prep(chis[-1], normalise = (True,False))
    paperFig4_draw(stats_exac, stats_mps)
    
    # Create figure S1
    figS1 = paperFigS1_draw(chis_times_dict = chis_times_dict, CHI0=CHI0, CHI1=CHI1, cm=cm)
    
    # Create figure S2
    figS2 = paperFigS2_draw(chisN_times_dict, N_dnstimes_dict)
    
    # Create figure S3
    figS3, _ = paperFig2_draw_bc(errors_fig2, xvals =chis_times_dict )        
    