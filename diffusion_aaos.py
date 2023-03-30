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
    space.laplacian(comm, h, u1, res, nu=-nu)
    dt1 = 1/dt
    interior = np.s_[1:-1, 1:-1]
    res[interior] += dt1*(u1[interior] - u0[interior])


def allatonce_residual(time_comm, space_comm, dt, h, nu, uinit, u1, u0, res):
    time.update_time_halo(time_comm, u1, u0)
    if time_comm.rank == 0:
        u0[:] = uinit
    timestep_residual(space_comm, dt, h, nu, u1, u0, res)
    space.dirichlet_bcs(space_comm, res)


# set up parallelism

global_comm = MPI.COMM_WORLD

nranki = 2
nrankj = 2
nt = 32

space_comm, time_comm = time.space_time_comms(global_comm,
                                              nrankj, nrankj, nt)

grank = global_comm.rank
trank = time_comm.rank
srank = space_comm.rank
print_in_order(global_comm, f"Global rank {grank}, time rank {trank}, space rank {srank}")

# problem parameters

alpha = 1e-3

L = 1
n = 2**9
nu = 5

h = L/n

cfl = 5

dt = h*h*cfl/nu

omega = 1

stol=1e-5
space_its = 2e3
atol = 7e-10
time_its = 4
print_once(global_comm, f"({n}, {space_its}, {time_its}, {alpha}, {h}, {dt})")

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

exponents = np.arange(nt)/nt
gamma_full = alpha**exponents
gamma = gamma_full[time_comm.rank]
gamma1 = 1/gamma

c1col = np.array([1, -1, *[0 for _ in range(nt-2)]])/dt
c2col = np.array([1,  0, *[0 for _ in range(nt-2)]])

eigvals1 = fft(gamma_full*c1col, norm='backward')
eigvals2 = fft(gamma_full*c2col, norm='backward')

eigval1 = eigvals1[time_comm.rank]
eigval2 = eigvals2[time_comm.rank]

allatonce_residual(time_comm, space_comm, dt, h, nu, uinit, u1, u0, res)
unorm = l2norm(global_comm, u1)
resnorm = l2norm(global_comm, res)
print_once(global_comm, f" 0 | {unorm} | {resnorm}")
resnorm0 = resnorm

for i in range(time_its):

    # residual of local timestep of all-at-once system

    allatonce_residual(time_comm, space_comm, dt, h, nu, uinit, u1, u0, res)

    # fft residual

    res*=gamma
    transpose_src[:] = res.reshape(transpose_src.shape)
    transfer.forward(transpose_src, transpose_dst)

    transpose_dst[:] = fft(transpose_dst, axis=0)

    transfer.backward(transpose_dst, transpose_src)
    cres[:] = transpose_src.reshape(cres.shape)

    # block solve

    ucpx[:] = 0
    ni, r0, r = space.solve_helmholtz(space_comm, omega, space_its, stol,
                                      h, bcs, eigval1, nu, cres, ucpx, cwrk)
    if space_comm.rank == 0:
        print_in_order(time_comm, f"{ni}, {r/r0}")

    # ifft residual

    transpose_src[:] = ucpx.reshape(transpose_src.shape)
    transfer.forward(transpose_src, transpose_dst)

    transpose_dst[:] = ifft(transpose_dst, axis=0)

    transfer.backward(transpose_dst, transpose_src)
    res[:] = transpose_src.reshape(cres.shape).real
    res*=gamma1

    # update solution

    u1 -= res

    unorm = l2norm(global_comm, u1)
    resnorm = l2norm(global_comm, res)
    print_once(global_comm, f" {i+1} | {unorm} | {resnorm}")
    if resnorm/resnorm0 < atol:
        break
    #print_once(global_comm, "")
