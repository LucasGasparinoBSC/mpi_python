from mpi4py import MPI
import numpy as np

# Start the MPI environment
comm = MPI.COMM_WORLD  # Communicator
rank = comm.Get_rank() # Rank ID
size = comm.Get_size() # Number of ranks

arrSize = 1000;
ratio = arrSize / size
if ratio < 1.0:
    print("Error: Array size must be greater than or equal to the number of ranks")
    exit(1)

if rank == 0:
    arr1 = np.empty(arrSize, dtype=np.int32)
    arr2 = np.empty(arrSize, dtype=np.int32)
    for i in range(arrSize):
        arr1[i] = i
        arr2[i] = arrSize - i

# Start partition timer
comm.Barrier()
t1 = MPI.Wtime()

# If decimal point of ratio is greater than 0.5, round up, else round down
if ratio - int(ratio) > 0.5:
    nPart = int(ratio) + 1
else:
    nPart = int(ratio)

# Array relating rank with chunk size:
# All entries except the last are nPart.
# The last entry is arrSize - ((size-1)*nPart)

# Allocate the array with "size" entries
chunkSize = np.empty(size, dtype=np.int32)
for i in range(size-1):
    chunkSize[i] = nPart
chunkSize[size-1] = arrSize - ((size-1)*nPart)

# Print thhe array from rank 0
if rank == 0:
    print(chunkSize)

# Each rank allocates its own chunk
chunk1 = np.empty(chunkSize[rank], dtype=np.int32)
chunk2 = np.empty(chunkSize[rank], dtype=np.int32)

# Rank 0 sends the data to all other ranks, including itself
if rank == 0:
    print("Sending data")
    for i in range(0,size):
        istart = chunkSize[0]*i
        iend = istart + chunkSize[i]
        print("Rank", i, ":", istart, iend)
        comm.Send(arr1[istart:iend], dest=i, tag=11)
        comm.Send(arr2[istart:iend], dest=i, tag=12)

# All ranks receive the data
comm.Recv(chunk1, source=0, tag=11)
comm.Recv(chunk2, source=0, tag=12)

# End partition timer
comm.Barrier()
t2 = MPI.Wtime()
if rank == 0:
    print("Partition time:", t2-t1)

# start sum timer
comm.Barrier()
t1 = MPI.Wtime()

# Sum the arrays in each rank
for i in range(chunkSize[rank]):
    chunk1[i] += chunk2[i]

# End sum timer
comm.Barrier()
t2 = MPI.Wtime()
if rank == 0:
    print("Sum time:", t2-t1)

# Start gather timer
comm.Barrier()
t1 = MPI.Wtime()

# Each rank sends the data back to rank 0
comm.Send(chunk1, dest=0, tag=11)
if rank == 0:
    arr3 = np.empty(arrSize, dtype=np.int32)
    for i in range(0,size):
        istart = chunkSize[0]*i
        iend = istart + chunkSize[i]
        comm.Recv(arr3[istart:iend], source=i, tag=11)

# End gather timer
comm.Barrier()
t2 = MPI.Wtime()
if rank == 0:
    print("Gather time:", t2-t1)