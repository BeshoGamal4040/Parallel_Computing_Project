#pragma once
#include <vector>
#include <string>
#include <mpi.h>

struct Matrix {
    int rows, cols;
    std::vector<double> data;
};

enum AlgorithmType {
    HEAT_DIFFUSION = 1,
    GAME_OF_LIFE = 2,
    MATRIX_MULTIPLICATION = 3
};