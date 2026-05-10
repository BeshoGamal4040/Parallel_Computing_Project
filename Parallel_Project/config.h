#pragma once
#include <vector>
#include <string>
#include <mpi.h>

struct Matrix {
    int rows, cols;
    std::vector<double> data;
};
struct IntMatrix {
    int rows, cols;
    std::vector<int> data;
};
enum AlgorithmType {
    GAME_OF_LIFE = 1,
    MATRIX_MULTIPLICATION = 2
};