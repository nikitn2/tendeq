import numpy as np

N = 256                           # number of grid points = 2^8
L = 1.0                           # domain length
dx = L / N                        # grid spacing
x = np.linspace(0, L, N, endpoint=False)
nu = 0.002                        # viscosity
dt = 0.002                        # time-step (satisfies Δt ≲ Δx for stability)
T0 = 5.0                          # convective timescale (fastest parcel crosses domain in time T0)
steps = int(T0 / dt)              # number of RK2 steps to reach one convective time
u = np.sin(2*np.pi * x)           # initial condition u(x,0) = sin(2πx)

def tt_svd_compress(vec, chi_max):
    """Compress 256-vector to MPS with max bond dimension chi_max."""
    A = vec.reshape([2]*8)               # reshape into 8-dimensional tensor (2x2x...x2)
    cores = []
    r = 1                                # initial bond dimension
    for i in range(7):                  # for each split between 8 sites
        # Merge current bond dim and physical dim for SVD
        A = A.reshape((r*2, -1))
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        r_new = min(chi_max, U.shape[1]) # truncate to chi_max
        # truncate SVD components
        U = U[:, :r_new]
        s = s[:r_new];  Vt = Vt[:r_new, :]
        # absorb singular values into U
        U = U * s 
        # store core tensor (reshape U back to (r, 2, r_new))
        cores.append(U.reshape((r, 2, r_new)))
        r = r_new
        A = Vt                            # carry Vt as next A
    # Final core (reshape last A to (r, 2, 1))
    cores.append(A.reshape((r, 2, -1)))
    return cores

def mps_to_vector(cores):
    """Reconstruct full state vector from MPS cores."""
    # Start with first core (shape 1 × 2 × χ1) – use index 0 since left bond is size 1
    psi = cores[0][0]  # shape = (2, χ1)
    for core in cores[1:]:
        # core has shape (χ_prev, 2, χ_next); contract with psi on matching bond
        psi = np.tensordot(psi, core, axes=([psi.ndim-1], [0]))  # result shape: (..., 2, χ_next)
    return psi.reshape(-1)             # flatten final tensor to 1D length N

def burgers_rhs(u_vec):
    """Compute RHS F(u) = -u * u_x + nu * u_xx for vector u_vec (length N)."""
    # periodic finite differences:
    u_x = (np.roll(u_vec, -1) - np.roll(u_vec, 1)) / (2*dx)
    u_xx = (np.roll(u_vec, -1) - 2*u_vec + np.roll(u_vec, 1)) / (dx**2)
    return -u_vec * u_x + nu * u_xx

# Compress initial state to MPS with desired bond dimension
mps = tt_svd_compress(u, chi_max=3)
# Time-march with RK2:
for n in range(steps):
    u_vec = mps_to_vector(mps)                        # decompress to full vector
    k1 = burgers_rhs(u_vec)                           # stage-1 slope
    u_half = u_vec + 0.5*dt*k1
    k2 = burgers_rhs(u_half)                          # stage-2 slope
    u_vec_new = u_vec + dt * k2                       # RK2 update
    mps = tt_svd_compress(u_vec_new, chi_max=3) # compress back to MPS

u_final = mps_to_vector(mps)
print("Final L2 norm:", np.linalg.norm(u_final))
