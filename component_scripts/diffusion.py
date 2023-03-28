import numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt


def l2norm(u):
    sm = np.sum(u*u)
    sz = np.prod(u.shape)
    return sqrt(sm)/sz


def errnorm(uexact, u):
    return l2norm(u - uexact)


def neumann_bcs(u):
    u[0, :] = u[1, :]
    u[:, 0] = u[:, 1]
    u[-1, :] = u[-2, :]
    u[:, -1] = u[:, -2]


def jacobi_iteration(h, omega, sigma, f, u0, u1):
    fi = f[1:-1, 1:-1]

    ut = u1[1:-1, 1:-1]

    un = u0[2:, 1:-1]
    us = u0[:-2, 1:-1]

    ue = u0[1:-1, 2:]
    uw = u0[1:-1, :-2]

    h2 = h*h
    d = sigma*h2 + 4
    d1 = 1/d

    ut[...] = d1*(h2*fi + un + us + ue + uw)
    u0[...] = (1 - omega)*u0[...] + omega*u1[...]


def solve_helmholtz(bcs, omega, niterations, sigma, f, ucurr, uwrk):
    for i in range(niterations):
        jacobi_iteration(h, omega, sigma, f, ucurr, uwrk)
        bcs(ucurr)

L = 1
n = 2**8

niterations = 10000

omega = 1
sigma = 1

print(f"({n}, {niterations})")

h = L/n

x = np.linspace(-h/2, L+h/2, n+2)
y = np.linspace(-h/2, L+h/2, n+2)
x, y = np.meshgrid(x, y)

f = (1 + 8*pi*pi)*np.cos(2*pi*x)*np.cos(2*pi*y)

uexact = np.cos(2*pi*x)*np.cos(2*pi*y)

uinit = np.zeros_like((uexact))
uwrk = uinit.copy()
usol = uinit.copy()

interior = np.s_[1:-1, 1:-1]

def err(u):
    return errnorm(uexact[interior], u[interior])

print(err(usol))
solve_helmholtz(neumann_bcs, omega, niterations, sigma, f, usol, uwrk)
print(err(usol))

# plt.contourf(x, y, usol, cmap='viridis')
# plt.colorbar()
# plt.show()
