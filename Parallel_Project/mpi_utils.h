#pragma once
#include "config.h"
#include <mpi.h>
#include <vector>

struct Block {
    int startRow;
    int numRows;
};

Block getBlock(int rank, int size, int totalRows);

void buildScatterV(int rows, int cols, int size,
    std::vector<int>& sendcounts,
    std::vector<int>& displs);

void matrix_multiply_2D(int N, int K, int M,
    double* A, double* B, double* C,
    MPI_Comm comm);

std::vector<double> heat_diffusion(int rows, int cols, int iterations,
    double alpha, MPI_Comm comm);