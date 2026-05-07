#pragma once
#include "config.h"
#include <string>

Matrix readMatrixFromFile(const std::string& filename);
void writeMatrixToFile(const std::string& filename, double* data, int rows, int cols);
void writeIntMatrixToFile(const std::string& filename, int* data, int rows, int cols);
IntMatrix readIntMatrixFromFile(const std::string& filename);