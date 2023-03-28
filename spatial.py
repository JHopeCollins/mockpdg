import numpy as np
from mpi4py import MPI
from common import l2norm

# index ranges for interior nodes and neighbours
interior = np.s_[1:-1, 1:-1]
northi = np.s_[2:, 1:-1]
southi = np.s_[:-2, 1:-1]
easti = np.s_[1:-1, 2:]
westi = np.s_[1:-1, :-2]

# index ranges of interior boundary nodes
northb = np.s_[1, :]
southb = np.s_[-2, :]
eastb = np.s_[:, -2]
westb = np.s_[:, 1]

# index ranges of ghost nodes
northg = np.s_[0, :]
southg = np.s_[-1, :]
eastg = np.s_[:, -1]
westg = np.s_[:, 0]


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

    east_shift = (0, 1)
    west_shift = (0, -1)
    north_shift = (1, 1)
    south_shift = (1, -1)

    halo_swap(ccomm, east_shift, data, sendbuf, recvbuf, eastb, westg)
    halo_swap(ccomm, west_shift, data, sendbuf, recvbuf, westb, eastg)
    
    halo_swap(ccomm, north_shift, data, sendbuf, recvbuf, northb, southg)
    halo_swap(ccomm, south_shift, data, sendbuf, recvbuf, southb, northg)


def neumann_bcs(cart_comm, u):
    coordi, coordj = cart_comm.coords
    dimi, dimj = cart_comm.dims

    # set ghost node equal to boundary node

    if coordi == 0:
        u[westg] = u[westb]

    if coordi == dimi-1:
        u[eastg] = u[eastb]

    if coordj == 0:
        u[southg] = u[southb]

    if coordj == dimj-1:
        u[northg] = u[northb]


def jacobi_iteration(h, omega, sigma, nu, f, ucurr, uwrk):
    fi = f[interior]

    ut = uwrk[interior]
    uc = ucurr[interior]

    un = ucurr[northi]
    us = ucurr[southi]

    ue = ucurr[easti]
    uw = ucurr[westi]

    h2 = h*h
    d = sigma*h2 + 4*nu
    d1 = 1/d

    ut[...] = d1*(h2*fi + nu*(un + us + ue + uw))
    uc[...] = (1 - omega)*uc[...] + omega*ut[...]


def solve_helmholtz(comm, omega, niterations, tol, h,
                    bcs, sigma, nu, f, ucurr, uwrk):
    def residual(u):
        nonlocal uwrk
        helmholtz(comm, h, u, uwrk, sigma, nu)
        uwrk-=f
        return l2norm(comm, uwrk)

    res0 = residual(ucurr)
    res = res0
    nit = 0
    while (nit < niterations and res > res0*tol):
        update_halos(comm, ucurr)
        bcs(comm, ucurr)
        jacobi_iteration(h, omega, sigma, nu, f, ucurr, uwrk)
        res = residual(ucurr)
        nit += 1
    return nit, res0, res


def laplacian(comm, h, u, lap, nu=1):
    assert u.shape == lap.shape
    update_halos(comm, u)
    lap[...] = 0
    ui = u[interior]

    un = u[northi]
    us = u[southi]

    ue = u[easti]
    uw = u[westi]

    h2i = nu/(h*h)

    lap[interior] = h2i*(un + us + ue + uw - 4*ui)


def helmholtz(comm, h, u, helm, sigma=1, nu=1):
    assert u.shape == helm.shape
    laplacian(comm, h, u, helm, -nu)
    helm[interior] += sigma*u[interior]

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


def global_array(cart_comm, arr, n):
    coordi, coordj = cart_comm.coords
    assert cart_comm.dims[0] == cart_comm.dims[1]
    nrank = cart_comm.dims[0]

    nl = n//nrank
    iloc = coordi*nl
    jloc = coordj*nl

    garr = np.zeros((n+1, n+1))
    garr[iloc:iloc+nl, jloc:jloc+nl] = arr[interior]
    cart_comm.Allreduce(MPI.IN_PLACE, garr)

    return garr
