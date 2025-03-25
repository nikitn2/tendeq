# -*- coding: utf-8 -*-

import os
import pickle
import PDEs.FokkerPlanck.FokkerPlanck_spat3DreacKD as fpKDs3rK
import PDEs.FokkerPlanck.FokkerPlanckGeneral as fpg
import PDEs.FokkerPlanck.FokkerPlanckDirect as fpd
import PDEs.FokkerPlanck.FokkerPlanckDistrs as fpDis
import PDEs.PdeGeneral as pdeg
import config as cfg
import config_FP3D as cfgFP3
import itertools
import argparse
import TnMachinery.MPSfunctions as mps
from numba import jit, prange
import numpy as np
import sys

## 3D spatial simulation with K_phi reacting chemicals

def simParams(NK, T, reac = cfg.zero, reacType = "LinLin", omega = 200, l=1.0, c=1.0):
    
    # Initialise
    Mt  = 2**int(np.log2(T) +(NK+2))
    dphi= c/2**NK
    dx  = l/2**NK
    U   = 1.0 #1 default
    k   = 2.0 #2 default
    
    # Check that there are at least 16 temporal and 8 spatial gridpoints
    if Mt<16 or NK <3: print("Mt must be >= 16, and NK>=3 (=> Mspatial >=2^3=8) ! Exiting."); sys.exit(1)
    
    # Physical parameters
    const_nu    =     1.00*1e-3 #<- Spatial diffusion aka “Gamma” in paper # 1.0*1e-3
    const_gamma =     0.75*1e-3/25 #<- Smagorinsky constant. const_gamma = 0.5*C_s*(n*dx)**2. #0.75*1e-3/25
    const_omega =     omega #<- Mixing constant. const_omega = C_omega/(n*dx)**2.
    reac_pars   =     K_phi*[reac] + [reacType] #<- Chemical reaction consts + reaction function to use: "LinLin" or "Lin".
    consts_TGV  = np.array([U, U, -2*U ,k*2*np.pi/l, k*2*np.pi/l, k*2*np.pi/l ])
    const_w     = cfg.zero*2*np.pi/T
       
    # Artificial diffusion for maintaining numerical stability
    const_alpha =     4*1e-3*omega/OMEGA_MAX# 4*1e-3*omega/omega_max
        
    ins_spread    = c*2**-3 # c*2**-3
    ins_bdrySep   = 0.25 #0.25
    ins_gaussPars = [ [(1-ins_bdrySep)*c, (1-ins_bdrySep)*c, ins_spread, ins_spread ], [ins_bdrySep*c, ins_bdrySep*c, ins_spread, ins_spread ] ]
    
    # Set sampling rate
    samplingPeriod = Mt//NSAMPLES #16
    MtSamples      = Mt//samplingPeriod
        
    # Get grid parameters 
    M_xphi = 2**NK
    dt = T/Mt;    t = np.linspace(0,T,Mt+1)
    dx = l/M_xphi; x = np.linspace(0,l-dx,M_xphi)
    dphi=c/M_xphi
    
    varsToSave = {"NK":NK,"Nt":int(np.log2(Mt)), "M_xphi": M_xphi, "samplingPeriod": samplingPeriod, "MtSamples":MtSamples, "T":T,"l":l,"c":c, "dt":dt, "t":t, "dx": dx, "x":x, "dphi": dphi,
                  "const_alpha":const_alpha,"const_nu":const_nu,"const_gamma":const_gamma,"const_omega":const_omega,"reac_pars":reac_pars,
                  "consts_TGV":consts_TGV, "instate_gaussPars":ins_gaussPars, "const_w": const_w, "ins_bdrySep":ins_bdrySep}
    
    return NK, Mt, T, l, c, const_alpha, const_nu, const_gamma, const_omega, reac_pars, consts_TGV, const_w, ins_gaussPars, samplingPeriod, MtSamples, M_xphi, dt, dx, dphi, ins_bdrySep, varsToSave

def simParseArgs():
    
    # Initialise
    parser = argparse.ArgumentParser(description="Required inputs: “--T, --NKs, --Chis”.")
    def stringParser_setType_float(arg_string):
        if arg_string.find(" ") != -1: return [float(x) for x in arg_string.split()]
        else: return float(arg_string)
    def stringParser_setType_int(arg_string):
        if arg_string.find(" ") != -1: return [int(x) for x in arg_string.split()]
        else: return int(arg_string)
    def stringParser_delist(arg_list):
        if any(isinstance(x, list) for x in arg_list): return arg_list[0]
        else:                                          return arg_list
    # Add arguments
    parser.add_argument("--T",                   type=float, required=True)
    parser.add_argument("--reacs",  nargs = "+", type=stringParser_setType_float, required=True)
    parser.add_argument("--reacType",            type=str, dest="reacType",       required=True)
    parser.add_argument("--omegas", nargs = "+", type=stringParser_setType_float, required=True)
    parser.add_argument("--NKs",    nargs = "+", type=stringParser_setType_int,   required=True)
    parser.add_argument("--Chis",   nargs = "+", type=stringParser_setType_int,   required=True)
    parser.add_argument("--exactRunMeansToo",   dest="exactRunMeans",   action="store_const", const=True, default=False)
    parser.add_argument("--exactRunFPsToo",     dest="exactRunFPs",     action="store_const", const=True, default=False)
    parser.add_argument("--statsPlotsToo", dest="statsPlots", action="store_const", const=True, default=False)
    parser.add_argument('--mpsRun', default=True, action=argparse.BooleanOptionalAction)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    return args.T, stringParser_delist(args.reacs), args.reacType, stringParser_delist(args.omegas), stringParser_delist(args.NKs), stringParser_delist(args.Chis), args.exactRunMeans, args.exactRunFPs, args.statsPlots, args.mpsRun

def simRunMps(chi, *args_simParams):
    
    # Load parameters
    NK, Mt, T, l, c, const_alpha, const_nu, const_gamma, const_omega, reac_pars, consts_TGV, const_w, ins_gaussPars, samplingPeriod, MtSamples, M_xphi, dt, dx, dphi, ins_bdrySep, varsToSave = simParams(*args_simParams)
        
    # Set MPS parameters
    tol_compression = 1e-15; tol_zipup = 1e-15
    varsToSave.update({"split":split, "tol_compression":tol_compression, "tol_zipup":tol_zipup})
    
    # Create initial state and constant distributions/functions
    pInit_mps, nu_mps, gamma_mps, reac_mpses, phi_mpses, u1_mps, u2_mps, u3_mps, d_dx1, d_dx2, d_dx3, d_dphis, dd_dphisSq = fpKDs3rK.timestepFP_distributions_spat3DreacKD(NK, K_phi, l, c, split, const_nu, const_gamma, const_omega, reac_pars, fpDis.flowField_TaylorGreenVortexAndJet, consts_TGV, 
                                                                                                                     initialDistrFP = fpDis.initialDistrFP_outerInnerMixtureSpatKDConsc2D,
                                                                                                                     kwargs_initialDistrFP = {"K_x": K_x, "gaussian_pars": ins_gaussPars, "forcedNorm": True})
    
    # And solve the Fokker-Planck equation    
    print("\nBeginning simulation. NK={},chi={},reac=({},{},{}),omega={}; nu={},gamma={},alpha={},spread={},initBdrySep={}.".format(NK,chi,reac_pars[0],reac_pars[1], reac_pars[2],const_omega,const_nu,const_gamma,const_alpha , ins_gaussPars[-1][-1],ins_bdrySep) )
    p_t_mpses = fpg.solveTimestepFP(
                                p_0_mps=pInit_mps, l=l, v=1.0, c=c, T=T, Mt=Mt, chi=chi, split=split, K_x=K_x,K_v=0, K_phi=K_phi, samplingPeriod = samplingPeriod, tol_compression = tol_compression,
                                solveTimestepFP_pde=fpKDs3rK.timestepFP_init_spat3DreacKD,
                                kwargs_solveTimestepFP_pde={"chi": chi, "const_alpha" : const_alpha, "nu_mps" : nu_mps, "gamma_mps" : gamma_mps, "const_omega" : const_omega,  "S_mpses" : reac_mpses, "u1_mps" : u1_mps, "u2_mps" : u2_mps, "u3_mps" : u3_mps, "phi_mpses" : phi_mpses, "d_dx1": d_dx1, "d_dx2":d_dx2, "d_dx3" :d_dx3, "d_dphis" :d_dphis, "dd_dphisSq" : dd_dphisSq, "w":const_w, "tol_zipup" : tol_zipup})
    
    varsToSaveChi = varsToSave.copy()
    varsToSaveChi["p_t_mpses"] = p_t_mpses
    varsToSaveChi["chi"] =chi
    
    # Save
    savename = cfgFP3.savename_mps.format(reacType,reac_pars[0], const_omega, NK, chi).replace('.', 'p')
    os.makedirs(cfg.dir_data + cfgFP3.dir_results + "mpses/", exist_ok=True)
    with open(cfg.dir_data + cfgFP3.dir_results +"mpses/" + savename + "_mpses.pkl", "wb") as f: pickle.dump(varsToSaveChi, f)
        
    # Printout
    print("Simulations succcesful. Compression ratio of MPS vs DNS: {:.2e}".format(mps.mpsCompressionRatio(p_t_mpses[-1])))
    
    return

def simStatsMps(loadname, samples = None):
    
    try:
        
        # Load parameters
        with open(cfg.dir_data + cfgFP3.dir_results +"mpses/"+ loadname+ "_mpses.pkl", "rb") as f: varsLoaded = pickle.load(f)
        
        # Initialise
        varsLoaded.update({"chi":chi})        
        p_t_mpses = varsLoaded["p_t_mpses"]
        samplingPeriod = varsLoaded["samplingPeriod"]
        M_xphi = varsLoaded["M_xphi"]
        dt = varsLoaded["dt"]
        c = varsLoaded["c"]
        const_alpha = varsLoaded["const_alpha"]
        print("\nBeginning data extraction (avisc={}) from {}.".format(const_alpha,loadname) )
        
        if samples is None: 
            MtSamples = len(p_t_mpses)
            samples   = np.arange(0,MtSamples)
        else: MtSamples = len(samples)
    
        # Create containers for statistics
        norms_mps     = np.zeros([M_xphi]*3 + [K_phi]*0 + [MtSamples])
        norms_integ   = np.zeros([MtSamples])
        means_mps     = np.zeros([M_xphi]*3 + [K_phi]*1 + [MtSamples])
        covs_mps      = np.zeros([M_xphi]*3 + [K_phi]*2 + [MtSamples])
        #coskews_mps   = np.zeros([M_xphi]*3 + [K_phi]*3 + [MtSamples])
        stMeans_mps   = np.zeros([M_xphi]*3 + [K_phi]*1 + [MtSamples])
        stCovs_mps    = np.zeros([M_xphi]*3 + [K_phi]*2 + [MtSamples])
        #stCoskews_mps = np.zeros([M_xphi]*3 + [K_phi]*3 + [MtSamples])
                
        for i,t in enumerate(samples):
            # Get stats
            print("Computing stats for t={}.".format(t*samplingPeriod*dt))
            #norm_mps, mean_mps, cov_mps, coskew_mps, stMean_mps, stCov_mps, stCoskew_mps = fpg.computeStats(p_mps = p_t_mpses[t], Nt = 0, split = split, v = 0, c = c, K_x = K_x, K_v = 0, K_phi = K_phi)
            norm_mps, mean_mps, cov_mps, stMean_mps, stCov_mps = fpg.computeStats(p_mps = p_t_mpses[t], Nt = 0, split = split, v = 0, c = c, K_x = K_x, K_v = 0, K_phi = K_phi)

            
            norms_mps[...,i] = norm_mps[...,0]; norms_integ[i] = norm_mps.sum()/M_xphi**3
            means_mps[...,i] = mean_mps[...,0]
            covs_mps[...,i] = cov_mps[...,0]
            #coskews_mps[...,i] = coskew_mps[...,0]
            stMeans_mps[...,i] = stMean_mps[...,0]
            stCovs_mps[...,i] = stCov_mps[...,0]
            #stCoskews_mps[...,i] = stCoskew_mps[...,0]
        
        # Save and exit
        varsLoaded.update({"p_t_mpses": p_t_mpses, "norms_mps":norms_mps, "means_mps":means_mps, "covs_mps":covs_mps, "stMeans_mps":stMeans_mps, "stCovs_mps":stCovs_mps, "samples": samples, "norms_integ": norms_integ})
        os.makedirs(cfg.dir_data + cfgFP3.dir_results + "stats/", exist_ok=True)
        with open(cfg.dir_data + cfgFP3.dir_results + "stats/" + loadname + "_stats.pkl", "wb") as f: pickle.dump(varsLoaded, f)
    
    except FileNotFoundError:
        print("File " + cfg.dir_data + cfgFP3.dir_results +"mpses/"+ loadname+ "_mpses.pkl" + " doesn't exist. Exiting simStatsMps.")
    
    return

def simRunExactFP(*args_simParams, samples):
    
    # Load parameters
    NK, Mt, T, l, c, const_alpha, const_nu, const_gamma, const_omega, reac_pars, consts_TGV, const_w, ins_gaussPars, samplingPeriod, MtSamples, M_xphi, dt, dx, dphi, ins_bdrySep, varsToSave = simParams(*args_simParams)
        
    # Get initial PDF 
    pInit_mps = fpKDs3rK.timestepFP_distributions_spat3DreacKD(NK, K_phi, l, c, split, const_nu, const_gamma, const_omega, reac_pars, fpDis.flowField_TaylorGreenVortexAndJet, consts_TGV, 
                                                                     initialDistrFP = fpDis.initialDistrFP_outerInnerMixtureSpatKDConsc2D,
                                                                     kwargs_initialDistrFP = {"K_x": K_x, "gaussian_pars": ins_gaussPars, "forcedNorm": True})[0]
    p0 = mps.mpsInvDecompFlowKD_timestep(pInit_mps, K_x+K_phi, split=True)
    
    # If the reaction rate is zero, an exact solution is available for the first moment. Compute it
    ps_exact = fpd.directSolveFP(p0, consts_TGV, 3, Mt, samplingPeriod, l, T, const_nu, const_gamma, const_omega, const_alpha, reac_pars[:-1], w=0)
    
    if samples is not None: ps_exact = ps_exact[..., samples]
    else:                   samples = np.arange(0,MtSamples+1)
    
    # Convert to MPS for compatibility with the stats-MPS function
    p_t_Exactmpses = [ mps.mpsDecompFlowKD_timestep(ps_exact[...,t],K_x+K_phi,split=True) for t in range(MtSamples)]
    
    # Save
    varsToSave.update({"p_t_mpses":p_t_Exactmpses, "samples" :samples})
    savename = cfgFP3.savename_exactFPs.format(reacType,reac_pars[0],const_omega, NK).replace('.', 'p')
    os.makedirs(cfg.dir_data + cfgFP3.dir_results +"mpses/", exist_ok=True)
    with open(cfg.dir_data + cfgFP3.dir_results +"mpses/" +savename + "_mpses.pkl", "wb") as f: pickle.dump(varsToSave, f)
    
    return

def simRunExactFirstMoms(*args_simParams, samples):
    
    # Load parameters
    NK, Mt, T, l, c, _, const_nu, const_gamma, _, _, consts_TGV, const_w, ins_gaussPars, samplingPeriod, MtSamples, _, _, _, _,_, varsToSave = simParams(*args_simParams)
    
    # Initialise the first moment
    p0_mps = fpDis.initialDistrFP_outerInnerMixtureSpatKDConsc2D(NK, K_x, True, c, ins_gaussPars)
    _, mean0_mps, _, _, _ = fpg.computeStats(p_mps = p0_mps, Nt = 0, split = True, v = 0, c = c, K_x = K_x, K_v = 0, K_phi = K_phi)
    mean0_mps = mean0_mps[...,0]
        
    # If the reaction rate is zero, an exact solution is available for the first moment. Compute it
    means_exact = fpd.directSolveFPfirstMomentEqsNoReac(mean0_mps, TGVpars=consts_TGV, Mt=Mt, samplingPeriod = samplingPeriod, l=l, T=T, nu=const_nu, gamma = const_gamma, w = const_w, scheme = "RK2")#"RK2")
    if samples is not None: 
        means_exact = means_exact[..., samples]
    else: samples = np.arange(0,MtSamples+1)
    
    # Save
    varsToSave.update({"means_exact":means_exact, "samples" :samples})
    savename = cfgFP3.savename_exactMoms.format(NK)
    os.makedirs(cfg.dir_data + cfgFP3.dir_results +"stats/", exist_ok=True)
    with open(cfg.dir_data + cfgFP3.dir_results +"stats/" +savename + "_stats.pkl", "wb") as f: pickle.dump(varsToSave, f)

def simPlot(loadname):
    
    try: 
        # First load data
        with open(cfg.dir_data + cfgFP3.dir_results+ "stats/" + loadname+ "_stats.pkl", "rb") as f: varsLoaded = pickle.load(f)
        
        # Initialise
        reac_pars = varsLoaded["reac_pars"]
        const_omega = varsLoaded["const_omega"]
        samples = varsLoaded["samples"]
        samplingPeriod = varsLoaded["samplingPeriod"]
        dt = varsLoaded["dt"]
        t_plot = np.array(samples)*samplingPeriod*dt
        M_xphi = varsLoaded["M_xphi"]
        dividerX3= 2
        plot_args = {"fixColorbar": True, "colorbar": True, "spacing": (-0.3,0.05), "loc_colorbar" :[0.77, 0.153, 0.01, 0.7] }
                
        # Plot whatever stats are present
        absLoc3= M_xphi//dividerX3
        prefix = "figures/" + cfgFP3.dir_results; os.makedirs(prefix, exist_ok=True)
        if "norms_mps" in varsLoaded: #any( r > 2*cfg.zero for r in np.abs(reac_pars)): 
            
            # Define filename
            prefix += cfgFP3.savename_mps.format(reacType,reac_pars[0], const_omega, varsLoaded["NK"],varsLoaded["chi"]).replace('.', 'p')        
    
            # Extract & plot pdf
            if "p_t_mpses" in varsLoaded: p_t_mpses = varsLoaded["p_t_mpses"]
            pp_mps = np.zeros([M_xphi]*(K_phi) + [2] + [len(samples)])
            
            # Pick coordinates to plot PDF at
            coords_x1 = [0 , M_xphi//2 -1 ]      # x1=(0, l/2)
            coords_x2 = M_xphi//2        -1      # x2=l/2
            coords_x3 = M_xphi//dividerX3-1      # x3=l/dividerX3
            for i,t in enumerate(samples):
                for j in range(2): pp_mps[...,j,i] = fpg.getPdfSlice(p_t_mpses[t], 0, split, [coords_x1[j],coords_x2, coords_x3], K_x, K_phi)[...,0]
            pdeg.plotStatsAcrossTime2D(pp_mps, [[0],[1]], t_plot, [r"$p(\phi_1,\phi_2; \mathbf{{x}} = (0, 0, l/{})$,".format(dividerX3), r"$p(\phi_1,\phi_2; \mathbf{{x}} = (l/2, 0, l/{})$,".format(dividerX3)],vlabel = r"$\phi_1/c$", hlabel = r"$\phi_2/c$", figName = prefix + "pdfs", **plot_args)
            
            # Extract & plot stats
            # if "norms_mps"   in varsLoaded:   pdeg.plotStatsAcrossTime2D(varsLoaded["norms_mps"][:,:,absLoc3,...], [[]], t_plot,                                  [r"Norm($x_1,x_2; x_3 = l/{}$)".format(dividerX3)],vlabel = r"$x_1/l$", hlabel = r"$x_2/l$", figName = prefix + "norms" , fixColorbar=True)        
            if "means_mps" in varsLoaded: pdeg.plotStatsAcrossTime2D(varsLoaded["means_mps"][:,:,absLoc3,...],   [[0],[1]], t_plot,                           [r"$E[\Phi_1;x_3=l/{}]$,".format(dividerX3), r"$E[\Phi_2;x_3=l/{}]$".format(dividerX3)],vlabel = r"$x_1/l$", hlabel = r"$x_2/l$", figName = prefix + "means", **plot_args)
            if "covs_mps"    in varsLoaded:   pdeg.plotStatsAcrossTime2D(varsLoaded["covs_mps"][:,:,absLoc3,...],    [[0,0],[1,1],[0,1] ], t_plot,                [r"Var[$\Phi_1;x_3=l/{}$],".format(dividerX3), r"Var[$\Phi_2;x_3=l/{}$],".format(dividerX3),r"Cov[$\Phi_1,\Phi_2;x_3=l/{}$],".format(dividerX3) ], 0.21, figName = prefix + "Covs",  **plot_args)
            #if "stCovs_mps"  in varsLoaded:   pdeg.plotStatsAcrossTime2D(varsLoaded["stCovs_mps"][:,:,absLoc3,...],  [[0,1]], t_plot,                             [r"$\frac{{\mathrm{{Cov}}[\Phi_1,\Phi_2;x_3=l/{0}]}}{{\sqrt{{ \mathrm{{Var}} [\Phi_1;x_3=l/{0}] \mathrm{{Var}}[\Phi_2;x_3=l/{0}]}}}}$,".format(dividerX3) ], 0.21, figName = prefix + "stCovs", **plot_args)
            #if "coskews_mps" in varsLoaded:   pdeg.plotStatsAcrossTime2D(varsLoaded["coskews_mps"][:,:,absLoc3,...], [[0,0,0],[0,0,1],[0,1,1],[1,1,1] ] , t_plot, [r"Coskew[$\Phi_1$,$\Phi_1$,$\Phi_1;x_3=l/{}$],".format(dividerX3), r"Coskew[$\Phi_1$,$\Phi_1$,$\Phi_2;x_3=l/{}$],".format(dividerX3), r"Coskew[$\Phi_1$,$\Phi_2$,$\Phi_2;x_3=l/{}$],".format(dividerX3), r"Coskew[$\Phi_2$,$\Phi_2$,$\Phi_2;x_3=l/{}$],".format(dividerX3) ], 0.21, figName = prefix + "Coskews", **plot_args)
            #if "stcoskews_mps" in varsLoaded: pdeg.plotStatsAcrossTime2D(varsLoaded["stCoskews_mps"][:,:,absLoc3,...],  [[0,0,1],[0,1,1]], [r"$\frac{\mathrm{Coskew}[\Phi_1,\Phi_1, \Phi_2]}{\sqrt{ \mathrm{Var} [\Phi_1]^2 \mathrm{Var}[\Phi_2]}}$, x_3=l/{}".format(dividerX3), r"$\frac{\mathrm{Coskew}[\Phi_1,\Phi_2, \Phi_2]}{\sqrt{ \mathrm{Var} [\Phi_1] \mathrm{Var}[\Phi_2]^2}}$, x_3=l/{}".format(dividerX3)], 0.21, figName = prefix + "stCoskews", **plot_args)
            
        else:
            prefix += cfgFP3.savename_exactMoms.format(varsLoaded["NK"])
            if "means_exact" in varsLoaded:pdeg.plotStatsAcrossTime2D(varsLoaded["means_exact"][:,:,absLoc3,...], [[0],[1]], t_plot, [r"$E[\Phi_1;x_3=l/{}]$,".format(dividerX3), r"$E[\Phi_2;x_3=l/{}]$".format(dividerX3)],vlabel = r"$x_1/l$", hlabel = r"$x_2/l$", figName = prefix + "means", **plot_args)

    except FileNotFoundError: 
        print("File " + cfg.dir_data + cfgFP3.dir_results+ "stats/" + loadname+ "_stats.pkl" + " OR some folder (eg in data, or figures) doesn't exist. Exiting simPlot.")
    
    return

if __name__ == '__main__':
    
    # Set global variables
    
    # Define dimensionality & set split-option
    K_x = 3
    K_phi = 2
    split= True # <-- This MPS parameter must be set to true, for now.
    
    # Define domain and select system sizes. If arguments have been parsed in, use them, and otherwise use default defined here.
    if len(sys.argv)>1:
        T_in, reacs_in, reacType, omegas_in, NKs_in, chis_in, exactRunMeans, exactRunFPs, statsPlots, mpsRun = simParseArgs()
        samples = None
    else:
        reacs_in = [0,0.5,1,1.5] #0 to 1.5
        reacType = "LinLin" # "LinLin" vs "Lin".
        omegas_in = [449, 904, 1359, 1814] # Equivalent to C_omega = .25, .50, .75, 1.0 in the paper (up to two digits of precision): omega_in = C_omega/Δ_l^2.
        NKs_in = [7]
        chis_in = [2,4,8,16, 32, 64, 96, 128]
        #        
        T_in=2.0
        samples = None
        #
        exactRunFPs = True
        exactRunMeans = True
        mpsRun = True
        statsPlots = True
    
    NSAMPLES = 16
    OMEGA_MAX= 1814
    
    # If reac = 0, run exact simulation & plot the mean, and compare against MPS
    if exactRunMeans and any(np.isclose(reacs_in, 0)):
        for NK in NKs_in:
            simRunExactFirstMoms(NK, T_in, samples= samples,)
            if statsPlots: simPlot(cfgFP3.savename_exactMoms.format(NK))
    
    # If necessary, run exact FP simulation (warning: extremely expensive for large NK)
    if exactRunFPs:
        for NK, reac, omega in itertools.product(NKs_in, reacs_in, omegas_in):
            savename = cfgFP3.savename_exactFPs.format(reacType,reac,omega, NK).replace('.', 'p'); chi = 0
            simRunExactFP(NK, T_in, reac, reacType, omega, samples = samples)
            if statsPlots:
                simStatsMps(savename, samples)
                simPlot(savename)
    
    # Run MPS simulation and plot resulting moments
    for chi, NK, reac, omega in itertools.product(chis_in,NKs_in, reacs_in, omegas_in):
        savename = cfgFP3.savename_mps.format(reacType,reac, omega, NK, chi).replace('.', 'p')
        if mpsRun: simRunMps(chi, NK, T_in, reac,reacType, omega)
        if statsPlots: 
            simStatsMps(savename, samples)
            simPlot(savename)