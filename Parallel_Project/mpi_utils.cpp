#include "mpi_utils.h"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

#define TAG_WORK   1
#define TAG_RESULT 2
#define TAG_DONE   3
#define TAG_UP     10
#define TAG_DOWN   11

// ============================================================
Block getBlock(int rank, int size, int totalRows) {
    int base = totalRows / size;
    int rem = totalRows % size;
    int start = rank * base + min(rank, rem);
    int count = base + (rank < rem ? 1 : 0);
    return { start, count };
}

// ============================================================
void buildScatterV(int rows, int cols, int size,
    vector<int>& sendcounts,
    vector<int>& displs) {

    sendcounts.resize(size);
    displs.resize(size);
    int base = rows / size, rem = rows % size, offset = 0;

    for (int i = 0; i < size; i++) {
        int localRows = base + (i < rem ? 1 : 0);
        sendcounts[i] = localRows * cols;
        displs[i] = offset;
        offset += localRows * cols;
    }
}


//  GAME OF LIFE

vector<int> game_of_life(int rows, int cols, int iterations,
    const vector<int>& initialGrid,
    MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    
    int activeSize = min(size, rows);
    int color = (rank < activeSize) ? 0 : MPI_UNDEFINED;
    MPI_Comm active_comm;
    MPI_Comm_split(comm, color, rank, &active_comm);

    
    if (color == MPI_UNDEFINED) {
        return vector<int>();
    }

    
    int activeRank, activeTotal;
    MPI_Comm_rank(active_comm, &activeRank);
    MPI_Comm_size(active_comm, &activeTotal);

    Block block = getBlock(activeRank, activeTotal, rows);
    int localRows = block.numRows;

    vector<int> localGrid((localRows + 2) * cols, 0);
    vector<int> newGrid((localRows + 2) * cols, 0);

    
    vector<int> sendcounts(activeTotal), displs(activeTotal);
    {
        int offset = 0;
        for (int p = 0; p < activeTotal; p++) {
            Block b = getBlock(p, activeTotal, rows);
            sendcounts[p] = b.numRows * cols;
            displs[p] = offset;
            offset += b.numRows * cols;
        }
    }

    MPI_Scatterv(
        activeRank == 0 ? initialGrid.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_INT,
        &localGrid[1 * cols],
        localRows * cols,
        MPI_INT, 0, active_comm
    );

    if (activeRank == 0) {
        cout << "\n=== Initial State (Iteration 0) ===\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                cout << initialGrid[i * cols + j] << " ";
            cout << "\n";
        }
        cout << "\n";
    }

    int upRank = (activeRank == 0) ? MPI_PROC_NULL : activeRank - 1;
    int downRank = (activeRank == activeTotal - 1) ? MPI_PROC_NULL : activeRank + 1;

    vector<int> fullGrid(rows * cols, 0);

    for (int iter = 0; iter < iterations; iter++) {

        
        MPI_Request reqs[4];

        MPI_Isend(&localGrid[1 * cols], cols, MPI_INT, upRank, TAG_DOWN, active_comm, &reqs[0]);
        MPI_Irecv(&localGrid[0], cols, MPI_INT, upRank, TAG_UP, active_comm, &reqs[1]);
        MPI_Isend(&localGrid[localRows * cols], cols, MPI_INT, downRank, TAG_UP, active_comm, &reqs[2]);
        MPI_Irecv(&localGrid[(localRows + 1) * cols], cols, MPI_INT, downRank, TAG_DOWN, active_comm, &reqs[3]);

        
        for (int i = 2; i <= localRows - 1; i++) {
            for (int j = 0; j < cols; j++) {
                int alive = 0;
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = (j + dj + cols) % cols;
                        alive += localGrid[ni * cols + nj];
                    }
                int cell = localGrid[i * cols + j];
                newGrid[i * cols + j] =
                    (cell == 1 && (alive == 2 || alive == 3)) ? 1 :
                    (cell == 0 && alive == 3) ? 1 : 0;
            }
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

       
        for (int i : {1, localRows}) {
            for (int j = 0; j < cols; j++) {
                int alive = 0;
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = (j + dj + cols) % cols;
                        alive += localGrid[ni * cols + nj];
                    }
                int cell = localGrid[i * cols + j];
                newGrid[i * cols + j] =
                    (cell == 1 && (alive == 2 || alive == 3)) ? 1 :
                    (cell == 0 && alive == 3) ? 1 : 0;
            }
        }

        swap(localGrid, newGrid);

     
    }

    
    MPI_Gatherv(
        &localGrid[1 * cols],
        localRows * cols,
        MPI_INT,
        activeRank == 0 ? fullGrid.data() : nullptr,
        sendcounts.data(), displs.data(),
        MPI_INT, 0, active_comm
    );

    if (activeRank == 0) {
        cout << "=== Final State (after " << iterations << " iterations) ===\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                cout << fullGrid[i * cols + j] << " ";
            cout << "\n";
        }
        cout << "\n";
    }

    MPI_Comm_free(&active_comm);
    return fullGrid;
}


//  MATRIX MULTIPLICATION 

void matrix_multiply_2D(int N, int K, int M,
    double* A, double* B, double* C,
    MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int color = (rank == 0) ? 0 : 1;
    MPI_Comm group_comm;
    MPI_Comm_split(comm, color, rank, &group_comm);

    vector<double> fullB(K * M);
    if (rank == 0)
        copy(B, B + K * M, fullB.begin());
    MPI_Bcast(fullB.data(), K * M, MPI_DOUBLE, 0, comm);

    int numWorkers = size - 1;
    int CHUNK_ROWS = max(1, N / (4 * numWorkers));

    if (rank == 0) {
        cout << "[Coordinator] Master-Worker Matrix Multiply\n";
        cout << "[Coordinator] Workers: " << numWorkers
            << ", Chunk size: " << CHUNK_ROWS << " rows\n";

        int nextRow = 0, activeJobs = 0;

        for (int w = 1; w <= numWorkers && nextRow < N; w++) {
            int chunkSize = min(CHUNK_ROWS, N - nextRow);
            int header[2] = { nextRow, chunkSize };
            MPI_Send(header, 2, MPI_INT, w, TAG_WORK, comm);
            MPI_Send(A + nextRow * K, chunkSize * K,
                MPI_DOUBLE, w, TAG_WORK, comm);
            nextRow += chunkSize;
            activeJobs++;
        }

        while (activeJobs > 0) {
            MPI_Status status;
            int resultHeader[2];
            MPI_Recv(resultHeader, 2, MPI_INT,
                MPI_ANY_SOURCE, TAG_RESULT, comm, &status);

            int workerRank = status.MPI_SOURCE;
            int startRow = resultHeader[0];
            int chunkSize = resultHeader[1];

            MPI_Recv(C + startRow * M, chunkSize * M,
                MPI_DOUBLE, workerRank, TAG_RESULT,
                comm, MPI_STATUS_IGNORE);
            activeJobs--;

            if (nextRow < N) {
                int newChunk = min(CHUNK_ROWS, N - nextRow);
                int header[2] = { nextRow, newChunk };
                MPI_Send(header, 2, MPI_INT, workerRank, TAG_WORK, comm);
                MPI_Send(A + nextRow * K, newChunk * K,
                    MPI_DOUBLE, workerRank, TAG_WORK, comm);
                nextRow += newChunk;
                activeJobs++;
            }
        }

        for (int w = 1; w <= numWorkers; w++) {
            int header[2] = { -1, -1 };
            MPI_Send(header, 2, MPI_INT, w, TAG_DONE, comm);
        }
        cout << "[Coordinator] All chunks processed\n";
    }
    else {
        while (true) {
            MPI_Status status;
            int header[2];
            MPI_Recv(header, 2, MPI_INT, 0,
                MPI_ANY_TAG, comm, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            int startRow = header[0];
            int chunkSize = header[1];

            vector<double> localA(chunkSize * K);
            MPI_Recv(localA.data(), chunkSize * K,
                MPI_DOUBLE, 0, TAG_WORK, comm, MPI_STATUS_IGNORE);

            vector<double> localC(chunkSize * M, 0.0);
            for (int i = 0; i < chunkSize; i++)
                for (int j = 0; j < M; j++)
                    for (int k = 0; k < K; k++)
                        localC[i * M + j] +=
                        localA[i * K + k] * fullB[k * M + j];

            int resultHeader[2] = { startRow, chunkSize };
            MPI_Send(resultHeader, 2, MPI_INT, 0, TAG_RESULT, comm);
            MPI_Send(localC.data(), chunkSize * M,
                MPI_DOUBLE, 0, TAG_RESULT, comm);
        }
    }

    MPI_Comm_free(&group_comm);
}