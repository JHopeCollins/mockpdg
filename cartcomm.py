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

def update_halos(ccomm, data, sendbuf, recvbuf):
    halo_swap(ccomm, (0, 1), data, sendbuf, recvbuf, np.s_[:,-2], np.s_[:,0])
    halo_swap(ccomm, (0, -1), data, sendbuf, recvbuf, np.s_[:,1], np.s_[:,-1])
    
    halo_swap(ccomm, (1, 1), data, sendbuf, recvbuf, np.s_[1,:], np.s_[-1,:])
    halo_swap(ccomm, (1, -1), data, sendbuf, recvbuf, np.s_[-2,:], np.s_[0,:])

def halo_swap(ccomm, shift, data, sendbuf, recvbuf, send_idx, recv_idx):
    src, dst = ccomm.Shift(*shift)
    rank = ccomm.rank

    sendbuf[:] = data[send_idx]

    sendtag = dst
    recvtag = rank

    ccomm.Sendrecv(sendbuf, dst, sendtag=sendtag,
                   recvbuf=recvbuf, source=src, recvtag=recvtag)

    data[recv_idx] = recvbuf[:]

n = 4

global_comm = MPI.COMM_WORLD

nranki = 2
nrankj = 2

assert global_comm.size == nranki*nrankj

cartcomm = global_comm.Create_cart((nranki, nrankj), periods=[True, True])

data = cartcomm.rank*np.ones((n+2, n+2))

sendbuf = np.zeros(n+2)
recvbuf = np.zeros(n+2)

print_in_order(global_comm, f"\n{data}\n")

update_halos(cartcomm, data, sendbuf, recvbuf)

print_in_order(global_comm, f"\n{data}\n")
