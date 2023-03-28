import numpy as np
from mpi4py import MPI


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
    sendbuf = np.zeros(data.shape[0], dtype=data.dtype)
    recvbuf = np.zeros(data.shape[0], dtype=data.dtype)

    halo_swap(ccomm, (0, 1), data, sendbuf, recvbuf, np.s_[:,-2], np.s_[:,0])
    halo_swap(ccomm, (0, -1), data, sendbuf, recvbuf, np.s_[:,1], np.s_[:,-1])
    
    halo_swap(ccomm, (1, 1), data, sendbuf, recvbuf, np.s_[1,:], np.s_[-1,:])
    halo_swap(ccomm, (1, -1), data, sendbuf, recvbuf, np.s_[-2,:], np.s_[0,:])


def neumann_bcs(cart_comm, u):
    coordi, coordj = cart_comm.coords
    dimi, dimj = cart_comm.dims

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


def laplacian(comm, h, u, lap, fac=1):
    assert u.shape == lap.shape
    update_halos(comm, u)
    lap[...] = 0
    ui = u[1:-1, 1:-1]

    un = u[2:, 1:-1]
    us = u[:-2, 1:-1]

    ue = u[1:-1, 2:]
    uw = u[1:-1, :-2]

    h2i = fac/(h*h)

    lap[1:-1, 1:-1] = h2i*(un + us + ue + uw - 4*ui)


def local_grid(ccomm, L, n):
    coordi, coordj = ccomm.coords
    assert ccomm.dims[0] == ccomm.dims[1]
    nrank = ccomm.dims[0]

    h = L/n
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
