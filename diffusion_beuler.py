from math import pi
import numpy as np
from mpi4py import MPI
import spatial as space

from common import print_once, print_in_order
from common import l2norm, errnorm


def problem_functions(xl, yl):
    from spatial import northg, southg, eastg, westg
    f = (1 + 8*pi*pi)*np.cos(2*pi*xl)*np.cos(2*pi*yl)
    uexact = np.cos(2*pi*xl)*np.cos(2*pi*yl)
    uinit = np.zeros_like((uexact))
    return f, uexact, uinit

global_comm = MPI.COMM_WORLD

nrank = 2

assert global_comm.size == nrank*nrank

cartcomm = global_comm.Create_cart((nrank, nrank), periods=[True, True])

L = 1
n = 2**7

h = L/n

show_plot = False

omega = 1
sigma = 1
nu = 1

nt = 10
cfl = 5
dt = cfl*h*h/nu

niterations = 1e3
tol = 1e-4

xl, yl = space.local_grid(cartcomm, L, n)

print_once(cartcomm, f"({n}, {niterations})")

f, uexact, uinit = problem_functions(xl, yl)
uinit[:] = f

u0 = uinit.copy()
u1 = uinit.copy()
uwrk = uinit.copy()

for i in range(nt):
    nit, res0, res = space.solve_helmholtz(cartcomm, omega, niterations, tol, h,
                                           space.neumann_bcs, sigma, dt*nu,
                                           u0, u1, uwrk)
    print_once(cartcomm, f"{i}: {nit}, {res/res0}")
    u0[:] = u1[:]

    if show_plot:
        uglob = space.global_array(cartcomm, u1, n)
        if cartcomm.rank == 0:
            import matplotlib.pyplot as plt
            x = np.linspace(0, L, n+1)
            y = np.linspace(0, L, n+1)
            x, y = np.meshgrid(x, y)
            plt.contourf(x, y, uglob, cmap='viridis')
            plt.colorbar()
            plt.show()

uorig = u1.copy()
u0[:] = uinit
u1[:] = uinit

print_once(cartcomm, "\n")

metrics = space.diffusion_timesteps(cartcomm, omega, niterations, tol, h,
                                    nt, space.neumann_bcs, dt, nu, u0, u1, uwrk)

print_once(cartcomm, l2norm(cartcomm, uorig-u1))
