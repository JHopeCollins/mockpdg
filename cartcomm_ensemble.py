import numpy as np
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


def flatten(idx, sizes):
    assert len(idx) == len(sizes)
    for i, s in zip(idx, sizes):
        assert i < s

    flat = 0
    off = 1
    for i in range(len(idx)-1, -1, -1):
        flat += off*idx[i]
        off *= sizes[i]
    return flat


def space_time_comms(gcomm, ni, nj, nt):
    ns = ni*nj
    assert gcomm.size == ns*nt
    grank = gcomm.rank

    scomm = global_comm.Split(color=(grank//ns), key=grank)
    ccomm = scomm.Create_cart((ni, nj), periods=[True, True])
    scomm.Free()

    cidx = flatten(ccomm.coords, ccomm.dims)
    tcomm = gcomm.Split(color=cidx, key=grank)

    return ccomm, tcomm


global_comm = MPI.COMM_WORLD

nranki = 2
nrankj = 2
nrankt = 2

cart_comm, time_comm = space_time_comms(global_comm, nrankj, nrankj, nrankt)

crank = cart_comm.rank
trank = time_comm.rank

print_in_order(global_comm, f"{crank}: {trank}: {cart_comm.coords}")
