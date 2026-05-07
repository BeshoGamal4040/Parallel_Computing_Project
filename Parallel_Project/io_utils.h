#pragma once
#include "config.h"
#include <string>

Matrix readMatrixFromFile(const std::string& filename);
void writeMatrixToFile(const std::string& filename, double* data, int rows, int cols);