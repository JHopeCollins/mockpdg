from math import pi
import numpy as np
from mpi4py import MPI
import spatial as space

from common import print_once, print_in_order
from common import l2norm, errnorm


def problem_functions(xl, yl):
    from spatial import northg, southg, eastg, westg
    f = (1 + 8*pi*pi)*np.cos(2*pi*xl)*np.cos(2*pi*yl)
    f[northg] = 0; f[southg] = 0; f[eastg] = 0; f[westg] = 0
    uexact = np.cos(2*pi*xl)*np.cos(2*pi*yl)
    uinit = np.zeros_like((uexact))
    return f, uexact, uinit

global_comm = MPI.COMM_WORLD

nrank = 2

assert global_comm.size == nrank*nrank

cartcomm = global_comm.Create_cart((nrank, nrank), periods=[True, True])

L = 1
n = 2**6

omega = 1
sigma = 1
nu = 1

niterations = 2e3
tol = 1e-3

h = L/n

xl, yl = space.local_grid(cartcomm, L, n)

print_once(cartcomm, f"({n}, {niterations})")

f, uexact, uinit = problem_functions(xl, yl)
uinit[:] = 0

uwrk = uinit.copy()
usol = uinit.copy()

print_once(cartcomm, errnorm(cartcomm, uexact, usol))
nit, res0, res = space.solve_helmholtz(cartcomm, omega, niterations, tol, h,
                                 space.neumann_bcs, sigma, nu, f,
                                 usol, uwrk)
print_once(cartcomm, errnorm(cartcomm, uexact, usol))
print_once(cartcomm, f"{nit}, {res0}, {res}")


uglob = space.global_array(cartcomm, usol, n)
#if cartcomm.rank == 0:
if False:
    import matplotlib.pyplot as plt
    x = np.linspace(0, L, n+1)
    y = np.linspace(0, L, n+1)
    x, y = np.meshgrid(x, y)
    plt.contourf(x, y, uglob, cmap='viridis')
    plt.colorbar()
    plt.show()
