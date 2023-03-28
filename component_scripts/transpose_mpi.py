import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Pencil, Subcomm


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


def transpose_transfer(comm, n):
    subcomm = Subcomm(comm, [0, 1])
    shape = [comm.size, n]
    pencil0 = Pencil(subcomm, shape, axis=1)
    pencil1 = pencil0.pencil(0)
    return pencil0.transfer(pencil1, int)


global_comm = MPI.COMM_WORLD

n = 8

transfer = transpose_transfer(global_comm, n)

# pencil subcommunicator distributed in dim 0 (time) not in dim 1 (space)

a0 = np.zeros(transfer.subshapeA, int)
a1 = np.zeros(transfer.subshapeB, int)
print_once(global_comm, transfer.subshapeA)
print_once(global_comm, transfer.subshapeB)

a0[:] = global_comm.rank
print_in_order(global_comm, a0)

transfer.forward(a0, a1)

print_in_order(global_comm, f"\n{a1}")
a1+=10

transfer.backward(a1, a0)

print_in_order(global_comm, a0)
