#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

    // Init mpi env
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Declare the global arrays
    const int arrSize = 500000000;
    float *arr1;
    float *arr2;
    float *arr3;

    // Check that no rank has 0 entries
    float ratio = (float)arrSize / (float)size;
    if (ratio < 1.0f) {
        printf("Error: arrSize must be divisible by the number of ranks\n");
        MPI_Finalize();
        return 1;
    }

    // Rank 0 allocates and fills arr1 and arr2 (timed)
    float t1 = MPI_Wtime();
    if (rank == 0) {
        arr1 = (float *)malloc(arrSize * sizeof(float));
        arr2 = (float *)malloc(arrSize * sizeof(float));
        arr3 = (float *)malloc(arrSize * sizeof(float));
        for (int i = 0; i < arrSize; i++) {
            arr1[i] = i;
            arr2[i] = arrSize - i;
            arr3[i] = 0.0f;
        }
    }
    float t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Time to allocate and fill arrays: %f\n", t2 - t1);
    }

    // If decimal part of ratio > 0.5, then round up, otherwise round down
    int localSize = (int)ratio;
    if (ratio - (float)localSize > 0.5f) {
        localSize++;
    }
    if (rank == 0) {
        printf("Ratio: %f, localSize: %d\n", ratio, localSize);
    }

    // Compute the local size for each rank
    int *localSizes = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size-1; i++) {
        localSizes[i] = localSize;
    }
    localSizes[size-1] = arrSize - (size-1) * localSize;

    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d: localSize = %d\n", rank, localSizes[rank]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Allocate the local arrays
    float *localArr1 = (float *)malloc(localSizes[rank] * sizeof(float));
    float *localArr2 = (float *)malloc(localSizes[rank] * sizeof(float));

    // Wiith Send and Recv, rank 0 sends parts off arr1 and arr2 to all ranks, including itself
    t1 = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                for (int j = 0; j < localSizes[i]; j++) {
                    localArr1[j] = arr1[j];
                    localArr2[j] = arr2[j];
                }
            } else {
                MPI_Send(&arr1[i * localSize], localSizes[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&arr2[i * localSize], localSizes[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(localArr1, localSizes[rank], MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localArr2, localSizes[rank], MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Time to send and receive arrays: %f\n", t2 - t1);
    }

    // test: each rank prints its local arrays
#ifdef DEBUG
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d: localArr1 = ", rank);
            for (int j = 0; j < localSizes[rank]; j++) {
                printf("%f ", localArr1[j]);
            }
            printf("\n");
            printf("Rank %d: localArr2 = ", rank);
            for (int j = 0; j < localSizes[rank]; j++) {
                printf("%f ", localArr2[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    // Dot product the local arrays into partSum
    t1 = MPI_Wtime();
    double sum = 0.0;
    double partSum = 0.0;
    for (int i = 0; i < localSizes[rank]; i++) {
        partSum += (double)(localArr1[i]/10000000.0 * localArr2[i]/10000000.0);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Sum the partSums into sum using MPI_Allreduce
    MPI_Allreduce(&partSum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    if (rank == 0) {
        printf("Time to compute dot product: %f\n", t2 - t1);
        printf("Dot product: %f\n", sum);
    }

    // Sum the local arrays into localArr1
    t1 = MPI_Wtime();
    for (int i = 0; i < localSizes[rank]; i++) {
        localArr1[i] += localArr2[i];
    }
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Time to sum arrays: %f\n", t2 - t1);
    }

    // Gather the local arrays into arr3 using Send and Recv
    t1 = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                for (int j = 0; j < localSizes[i]; j++) {
                    arr3[j] = localArr1[j];
                }
            } else {
                MPI_Recv(&arr3[i * localSize], localSizes[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Send(localArr1, localSizes[rank], MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Time to gather arrays: %f\n", t2 - t1);
    }

    // Print the final array
#ifdef DEBUG
    if (rank == 0)
    {
        for (int i = 0; i < arrSize; i++) {
            printf("%f\n", arr3[i]);
        }
    }
#else
    if (rank == 0)
    {
        printf("arr3[0] == %f\n", arr3[0]);
        printf("arr3[arrSize-1] == %f\n", arr3[arrSize-1]);
        printf("Done\n");
    }
#endif

    // Finalize mpi env
    MPI_Finalize();
    return 0;
}