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

// ============================================================
//  HEAT DIFFUSION
// ============================================================
vector<double> heat_diffusion(int rows, int cols, int iterations,
    double alpha, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Process organization: boundary vs interior ranks
    int isBoundary = (rank == 0 || rank == size - 1) ? 0 : 1;
    MPI_Comm boundary_comm;
    MPI_Comm_split(comm, isBoundary, rank, &boundary_comm);

    // Distribute rows evenly across processes
    Block block = getBlock(rank, size, rows);
    int localRows = block.numRows;

    // +2 for ghost rows (top and bottom)
    int totalLocalRows = localRows + 2;
    vector<double> localGrid(totalLocalRows * cols, 0.0);
    vector<double> newGrid(totalLocalRows * cols, 0.0);

    // ── Initialize and scatter grid ──────────────────────────
    vector<double> fullGrid;
    if (rank == 0) {
        fullGrid.resize(rows * cols, 0.0);

        // Fixed boundary conditions
        for (int j = 0; j < cols; j++) {
            fullGrid[0 * cols + j] = 100.0; // top
            fullGrid[(rows - 1) * cols + j] = 100.0; // bottom
        }
        for (int i = 0; i < rows; i++) {
            fullGrid[i * cols + 0] = 50.0;  // left
            fullGrid[i * cols + (cols - 1)] = 50.0;  // right
        }
    }

    vector<int> sendcounts(size), displs(size);
    {
        int offset = 0;
        for (int p = 0; p < size; p++) {
            Block b = getBlock(p, size, rows);
            sendcounts[p] = b.numRows * cols;
            displs[p] = offset;
            offset += b.numRows * cols;
        }
    }

    MPI_Scatterv(
        rank == 0 ? fullGrid.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_DOUBLE,
        &localGrid[1 * cols],
        localRows * cols,
        MPI_DOUBLE, 0, comm
    );

    // Neighbour ranks (MPI_PROC_NULL = no neighbour at boundary)
    int upRank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int downRank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // ── Iteration loop ───────────────────────────────────────
    for (int iter = 0; iter < iterations; iter++) {

        // Non-blocking ghost row exchange
        MPI_Request reqs[4];

        MPI_Isend(&localGrid[1 * cols], cols, MPI_DOUBLE,
            upRank, TAG_DOWN, comm, &reqs[0]);
        MPI_Irecv(&localGrid[0 * cols], cols, MPI_DOUBLE,
            upRank, TAG_UP, comm, &reqs[1]);
        MPI_Isend(&localGrid[localRows * cols], cols, MPI_DOUBLE,
            downRank, TAG_UP, comm, &reqs[2]);
        MPI_Irecv(&localGrid[(localRows + 1) * cols], cols, MPI_DOUBLE,
            downRank, TAG_DOWN, comm, &reqs[3]);

        // Compute interior rows while ghost rows travel
        for (int i = 2; i <= localRows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                double center = localGrid[i * cols + j];
                double up = localGrid[(i - 1) * cols + j];
                double down = localGrid[(i + 1) * cols + j];
                double left = localGrid[i * cols + (j - 1)];
                double right = localGrid[i * cols + (j + 1)];
                newGrid[i * cols + j] =
                    center + alpha * (up + down + left + right
                        - 4.0 * center);
            }
        }

        // Wait for ghost rows to arrive
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Compute boundary rows (need ghost rows)
        for (int i : {1, localRows}) {
            for (int j = 1; j < cols - 1; j++) {
                double center = localGrid[i * cols + j];
                double up = localGrid[(i - 1) * cols + j];
                double down = localGrid[(i + 1) * cols + j];
                double left = localGrid[i * cols + (j - 1)];
                double right = localGrid[i * cols + (j + 1)];
                newGrid[i * cols + j] =
                    center + alpha * (up + down + left + right
                        - 4.0 * center);
            }
        }

        // Re-enforce fixed boundary conditions
        if (rank == 0)
            for (int j = 0; j < cols; j++)
                newGrid[1 * cols + j] = 100.0;

        if (rank == size - 1)
            for (int j = 0; j < cols; j++)
                newGrid[localRows * cols + j] = 100.0;

        for (int i = 1; i <= localRows; i++) {
            newGrid[i * cols + 0] = 50.0;
            newGrid[i * cols + (cols - 1)] = 50.0;
        }

        swap(localGrid, newGrid);
    }

    // ── Gather final grid to root ────────────────────────────
    MPI_Gatherv(
        &localGrid[1 * cols],
        localRows * cols,
        MPI_DOUBLE,
        rank == 0 ? fullGrid.data() : nullptr,
        sendcounts.data(), displs.data(),
        MPI_DOUBLE, 0, comm
    );

    // Print sample on root
    if (rank == 0) {
        cout << "\n[Heat Diffusion] Final grid sample (top-left 5x5):\n";
        int printRows = min(5, rows);
        int printCols = min(5, cols);
        for (int i = 0; i < printRows; i++) {
            for (int j = 0; j < printCols; j++)
                cout << fullGrid[i * cols + j] << "\t";
            cout << "\n";
        }
    }

    MPI_Comm_free(&boundary_comm);

    // ✅ Return grid — only rank 0 has data, others return empty
    return fullGrid;
}

// ============================================================
//  MATRIX MULTIPLICATION — Master Worker
// ============================================================
void matrix_multiply_2D(int N, int K, int M,
    double* A, double* B, double* C,
    MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Process organization: coordinator vs workers
    int color = (rank == 0) ? 0 : 1;
    MPI_Comm group_comm;
    MPI_Comm_split(comm, color, rank, &group_comm);

    // Broadcast full B to all processes
    vector<double> fullB(K * M);
    if (rank == 0)
        copy(B, B + K * M, fullB.begin());
    MPI_Bcast(fullB.data(), K * M, MPI_DOUBLE, 0, comm);

    int numWorkers = size - 1;
    int CHUNK_ROWS = max(1, N / (4 * numWorkers));

    // ── Coordinator ──────────────────────────────────────────
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
                MPI_Send(header, 2, MPI_INT,
                    workerRank, TAG_WORK, comm);
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

    // ── Workers ──────────────────────────────────────────────
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