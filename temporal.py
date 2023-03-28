import numpy as np
from mpi4py import MPI
from pencil import Pencil, Subcomm
from common import flatten


def space_time_comms(global_comm, ni, nj, nt):
    ns = ni*nj
    assert global_comm.size == ns*nt
    grank = global_comm.rank

    space_comm = global_comm.Split(color=(grank//ns), key=grank)
    cart_comm = space_comm.Create_cart((ni, nj), periods=[True, True])
    space_comm.Free()

    cart_idx = flatten(cart_comm.coords, cart_comm.dims)
    time_comm = global_comm.Split(color=cart_idx, key=grank)

    return cart_comm, time_comm


def update_time_halo(tcomm, ucurr, uprev):
    rank = tcomm.rank
    size = tcomm.size

    src = (rank - 1) % size
    dst = (rank + 1) % size

    sendtag = dst
    recvtag = rank

    sendbuf = ucurr
    recvbuf = uprev

    tcomm.Sendrecv(sendbuf, dst, sendtag,
                   recvbuf, src, recvtag)


def transpose_transfer(comm, n, dtype=complex):
    subcomm = Subcomm(comm, [0, 1])
    shape = [comm.size, n]
    pencil0 = Pencil(subcomm, shape, axis=1)
    pencil1 = pencil0.pencil(0)
    return pencil0.transfer(pencil1, dtype)
