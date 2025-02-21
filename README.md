![Package Logo](misc/logo-banner.jpg)

[`tendeq`](https://github.com/nikitn2/tendeq) is a python package for using TEnsor Networks to solve general Differential EQuations (with [`quimb`](https://github.com/jcmgray/quimb) providing the tensor networks backend).

Functionality is split in two parts:

---

The core `TnMachinery` module contains tools for working with various tensor network geometries. With it, you can:

- Use matrix product states and operators (MPSs, MPOs) to recast solutions into MPS/MPO form, and time-evolve them with finite difference schemes under some differential equation. It is possible to solve both using a classical time-stepping format, and by variationally optimising in space-time in a DMRG-inspired manner. 
- Both dimension-by-dimension (split ordering) and lengthscale-by-lengthscale (also known as interleaved ordering) orderings of the MPS tensors are supported.
- It is possible to construct MPS representations of solutions on extremely dense grids through prolongation schemes. In the future, it is planned to incorporate the option of using the tensor cross sampling algorithm for constructing MPS representations of solutions.
- Explore how effective two different ansatze are for compressing solutions: the projected entangled pair state (PEPS) and tree tensor network (TTN) representations.

---

The `PDEs` module contains specific implementations for various differential equations:
- The Kuramoto-Sivashinsky equation (of which the Burgers' equation is a special case).
- Different Fokker-Planck equations that model turbulent combustion.

---

To install tendeq:

1. Download it: eg, using git, run `git clone https://github.com/nikitn2/tendeq` in the terminal.
2. Create & activate a valid environment within which to run the package: eg, using a Conda package manager and the provided .yml file, run `conda env create -f environment.yml` in the terminal after navigating to the tendeq directory.

To reproduce the results of the paper from which this code hails, "Tensor networks enable the calculation of turbulence probability distributions":

1. To perform the simulations and save the resulting mpses (sampled at 17 timepoints during the simulation), run `python main_FPspat3DreacKD_timestep.py --T 2.0 --reacs 0 0.5 1.0 1.5 --omegas 449 904 1359 1814 --NKs 7 --Chis 2 4 8 16 32 64 96 128 --reacType "LinLin" --exactRunMeansToo`. Keep in mind that performing all the simulations across the 128 different combinations of chi, C_omega and Da is likely to take a significant amount of time; consider running the simulations in parallel for at least the different (C_omega, Da) combinations. Also, note that the omega in the code equals `C_omega/Δ_l**2` of the paper (see the comments in `main_FPspat3DreacKD_timestep.py` for more details).
2. To extract statistics from the mpses, run the script again, but with the `--no-mpsRun --statsPlotsToo` options: `python main_FPspat3DreacKD_timestep.py --T 2.0 --reacs 0 0.5 1.0 1.5 --omegas 449 904 1359 1814 --NKs 7 --Chis 2 4 8 16 32 64 96 128 --reacType "LinLin" --no-mpsRun --statsPlotsToo`.
3. The figures of the paper can then be drawn by running `python main_FPspat3DreacKD_paper.py`.

Alternatively, if the MPS data is available in the data/results_FPspat3DreacKD_timestep/mpses folder, simply run `python main_FPspat3DreacKD_timestep.py --T 2.0 --reacs 0 0.5 1.0 1.5 --omegas 449 904 1359 1814 --NKs 7 --Chis 2 4 8 16 32 64 96 128 --reacType "LinLin" --mpsRun --statsPlotsToo`, and then step 3. 

––––––

If you find this code useful, please cite the paper for which it was written: "Tensor networks enable the calculation of turbulence probability distributions" (2025); N. Gourianov, P. Givi, D. Jaksch, S. B. Pope."
