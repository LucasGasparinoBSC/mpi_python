# Small test for mpi4py
# Rank 0 creates 2 arrays, and scatter them to the other ranks
# All ranks sum the partial arrays

from mpi4py import MPI
import numpy as np

# Start the MPI environment
comm = MPI.COMM_WORLD  # Communicator
rank = comm.Get_rank() # Rank ID
size = comm.Get_size() # Number of ranks

# Rank 0 creates the numpy arrays (floats)
arrSize = size*100
arr1 = np.empty(1, dtype=np.float32)
arr2 = np.empty(1, dtype=np.float32)
arr3 = np.empty(arrSize, dtype=np.float32)
if rank == 0:
    # Resize arr1 and arr2
    arr1 = np.empty(arrSize, dtype=np.float32)
    arr2 = np.empty(arrSize, dtype=np.float32)
    # Fill the arrays with some values
    for i in range(arrSize):
        arr1[i] = i
        arr2[i] = arrSize-i
        arr3[i] = 0

# Partition the arrays using Scatter
nPart = arrSize//size
recvbuf1 = np.empty(nPart, dtype=np.float32)
recvbuf2 = np.empty(nPart, dtype=np.float32)
partSum = np.empty(nPart, dtype=np.float32)
comm.Scatter([arr1, MPI.FLOAT], [recvbuf1, MPI.FLOAT])
comm.Scatter([arr2, MPI.FLOAT], [recvbuf2, MPI.FLOAT])

# Sum the partial arrays
for i in range(nPart):
    partSum[i] = recvbuf1[i] + recvbuf2[i]
    
# Gather the partial sums
comm.Gather([partSum, MPI.FLOAT], [arr3, MPI.FLOAT])

# Rank 0 prints the final array
if rank == 0:
    print(arr3)
