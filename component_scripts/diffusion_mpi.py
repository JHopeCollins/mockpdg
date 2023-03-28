import numpy as np
from math import pi, sqrt
from mpi4py import MPI


def print_in_order(comm, text):
    comm.Barrier()
    for i in range(comm.size):
        rank = comm.rank
        if rank == i:
            print(f"Rank {rank}: {text}")
        comm.Barrier()
    comm.Barrier()


def print_once(comm, text):
    comm.Barrier()
    if comm.rank == 0:
        print(text)
    comm.Barrier()


def halo_swap(ccomm, shift, data, sendbuf, recvbuf, send_idx, recv_idx):
    src, dst = ccomm.Shift(*shift)
    rank = ccomm.rank

    sendbuf[:] = data[send_idx]

    sendtag = dst
    recvtag = rank

    ccomm.Sendrecv(sendbuf, dst, sendtag=sendtag,
                   recvbuf=recvbuf, source=src, recvtag=recvtag)

    data[recv_idx] = recvbuf[:]


def update_halos(ccomm, data):
    sendbuf = np.zeros(ucurr.shape[0])
    recvbuf = np.zeros(ucurr.shape[0])

    halo_swap(ccomm, (0, 1), data, sendbuf, recvbuf, np.s_[:,-2], np.s_[:,0])
    halo_swap(ccomm, (0, -1), data, sendbuf, recvbuf, np.s_[:,1], np.s_[:,-1])
    
    halo_swap(ccomm, (1, 1), data, sendbuf, recvbuf, np.s_[1,:], np.s_[-1,:])
    halo_swap(ccomm, (1, -1), data, sendbuf, recvbuf, np.s_[-2,:], np.s_[0,:])


def l2norm(comm, u):
    local_sum = np.sum(u*u)
    local_size = np.prod(u.shape)
    glob = np.array([local_sum, local_size])
    comm.Allreduce(MPI.IN_PLACE, glob)
    global_sum = glob[0]
    global_size = glob[1]
    return sqrt(global_sum)/global_size


def errnorm(comm, uexact, u):
    return l2norm(comm, u - uexact)


def neumann_bcs(ccomm, u):
    coordi, coordj = cartcomm.coords
    dimi, dimj = cartcomm.dims

    if coordi == 0:
        u[:, 0] = u[:, 1]

    if coordi == dimi-1:
        u[:, -1] = u[:, -2]

    if coordj == 0:
        u[-1, :] = u[-2, :]

    if coordj == dimj-1:
        u[0, :] = u[1, :]


def jacobi_iteration(h, omega, sigma, f, ucurr, uwrk):
    fi = f[1:-1, 1:-1]

    ut = uwrk[1:-1, 1:-1]

    un = ucurr[2:, 1:-1]
    us = ucurr[:-2, 1:-1]

    ue = ucurr[1:-1, 2:]
    uw = ucurr[1:-1, :-2]

    h2 = h*h
    d = sigma*h2 + 4
    d1 = 1/d

    ut[...] = d1*(h2*fi + un + us + ue + uw)
    ucurr[...] = (1 - omega)*ucurr[...] + omega*uwrk[...]


def solve_helmholtz(comm, bcs, omega, niterations, h, sigma, f, ucurr, uwrk):
    for i in range(niterations):
        update_halos(comm, ucurr)
        bcs(comm, ucurr)
        jacobi_iteration(h, omega, sigma, f, ucurr, uwrk)


def local_grid(ccomm, L, n):
    coordi, coordj = ccomm.coords
    assert ccomm.dims[0] == ccomm.dims[1]
    nrank = ccomm.dims[0]

    nl = n//nrank
    Ll = L/nrank
    originx = coordi*Ll
    originy = coordj*Ll

    xl = np.linspace(originx-h/2, originx+Ll+h/2, nl+2)
    yl = np.linspace(originy-h/2, originy+Ll+h/2, nl+2)
    return np.meshgrid(xl, yl)


def global_array(ccomm, arr, n):
    coordi, coordj = ccomm.coords
    assert ccomm.dims[0] == ccomm.dims[1]
    nrank = ccomm.dims[0]

    nl = n//nrank
    iloc = coordi*nl
    jloc = coordj*nl

    garr = np.zeros((n+1, n+1))
    garr[iloc:iloc+nl, jloc:jloc+nl] = arr[1:-1, 1:-1]
    cartcomm.Allreduce(MPI.IN_PLACE, garr)

    return garr


global_comm = MPI.COMM_WORLD

nrank = 2

assert global_comm.size == nrank*nrank

cartcomm = global_comm.Create_cart((nrank, nrank), periods=[True, True])

L = 1
n = 2**8

omega = 1
sigma = 1

niterations = 10000

h = L/n

xl, yl = local_grid(cartcomm, L, n)

print_once(cartcomm, f"({n}, {niterations})")

f = (1 + 8*pi*pi)*np.cos(2*pi*xl)*np.cos(2*pi*yl)

uexact = np.cos(2*pi*xl)*np.cos(2*pi*yl)

uinit = np.zeros_like((uexact))
uwrk = uinit.copy()
usol = uinit.copy()

interior = np.s_[1:-1, 1:-1]

def err(u):
    return errnorm(cartcomm, uexact[interior], u[interior])

print_once(cartcomm, err(usol))
solve_helmholtz(cartcomm, neumann_bcs, omega, niterations, h, sigma, f, usol, uwrk)
print_once(cartcomm, err(usol))


uglob = global_array(cartcomm, usol, n)
# if cartcomm.rank == 0:
if False:
    import matplotlib.pyplot as plt
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    x, y = np.meshgrid(x, y)
    plt.contourf(x, y, uglob, cmap='viridis')
    plt.colorbar()
    plt.show()
