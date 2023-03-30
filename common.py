from math import sqrt
import numpy as np
from mpi4py import MPI


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


def l2norm(comm, u):
    local_sum = np.sum(u*np.conj(u)).real
    local_size = np.prod(u.shape)
    glob = np.array([local_sum, local_size])
    comm.Allreduce(MPI.IN_PLACE, glob)
    global_sum = glob[0]
    global_size = glob[1]
    return sqrt(global_sum)/global_size


def errnorm(comm, uexact, u):
    return l2norm(comm, u - uexact)
