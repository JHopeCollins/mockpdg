from math import pi
import numpy as np
from mpi4py import MPI
from scipy.fft import fft, ifft

import spatial as space
import temporal as time
from common import print_in_order, print_once  # noqa: F401
from common import l2norm, errnorm  # noqa: F401


def problem_functions(xl, yl):
    f = (1 + 8*pi*pi)*np.cos(2*pi*xl)*np.cos(2*pi*yl)
    uexact = np.cos(2*pi*xl)*np.cos(2*pi*yl)
    uinit = np.zeros_like((uexact))
    return f, uexact, uinit


def timestep_residual(comm, dt, h, nu, u1, u0, res):
    assert u0.shape == u1.shape
    assert u0.shape == res.shape
    space.laplacian(comm, h, u1, res, fac=-nu)
    dt1 = 1/dt
    interior = np.s_[1:-1, 1:-1]
    res[interior] += dt1*(u1[interior] - u0[interior])


# set up parallelism

global_comm = MPI.COMM_WORLD

nranki = 2
nrankj = 2
nrankt = 2

space_comm, time_comm = time.space_time_comms(global_comm,
                                              nrankj, nrankj,
                                              nrankt)

# problem parameters

L = 1
n = 2**5
nu = 1

h = L/n

cfl = 2

dt = h*h*cfl/nu

omega = 1
sigma = 1/dt

space_its = 10
time_its = 4
print_once(global_comm, f"({n}, {space_its}, {time_its})")

# set up spatial problem

xl, yl = space.local_grid(space_comm, L, n)

f, uexact, uinit = problem_functions(xl, yl)
uinit[:] = f[:]
u1 = uinit.copy()   # timestep owned by this rank
u0 = uinit.copy()   # timestep owned by previous rank
res = np.zeros_like(uinit)  # residual
cres = np.zeros_like(uinit, dtype=complex)  # residual
ucpx = np.zeros_like(uinit, dtype=complex)  # complex sol

cwrk = np.zeros_like(uinit, dtype=complex)  # working array

bcs = space.neumann_bcs

# set up temporal problem

nx = np.prod(u1.shape)
transfer = time.transpose_transfer(time_comm, nx)
transpose_src = np.zeros(transfer.subshapeA, dtype=complex)
transpose_dst = np.zeros(transfer.subshapeB, dtype=complex)

# residual for local timestep

for i in range(time_its):

    time.update_time_halo(time_comm, u1, u0)
    if time_comm.rank == 0:
        u0[:] = uinit
    timestep_residual(space_comm, dt, h, nu, u1, u0, res)
    
    # fft residual
    
    transpose_src[:] = res.reshape(1, nx)

    transfer.forward(transpose_src, transpose_dst)
    transpose_dst[:] = fft(transpose_dst, axis=0)
    transfer.backward(transpose_dst, transpose_src)

    cres[:] = transpose_src.reshape(cres.shape)
    
    # block solve
    
    ucpx[:] = 0
    space.solve_helmholtz(space_comm, bcs,
                          omega, space_its,
                          h, sigma, cres, ucpx, cwrk)
    
    # ifft residual
    
    transpose_src[:] = ucpx.reshape(1, nx)

    transfer.forward(transpose_src, transpose_dst)
    transpose_dst[:] = ifft(transpose_dst, axis=0)
    transfer.backward(transpose_dst, transpose_src)

    res[:] = transpose_src.reshape(cres.shape).real
    
    # update solution
    
    u1 += res
