#include "io_utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace std;

Matrix readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);

    // ✅ Throw instead of exit so main can catch and recover
    if (!file)
        throw std::runtime_error("File not found: " + filename);

    std::vector<double> values;
    int cols = 0, rows = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        double val;
        int lineCount = 0;

        while (ss >> val) {
            values.push_back(val);
            lineCount++;
        }

        if (lineCount == 0) continue;

        if (rows == 0)
            cols = lineCount;
        else if (lineCount != cols)
            throw std::runtime_error(
                "Inconsistent row sizes in file: " + filename);
        rows++;
    }

    if (rows == 0 || cols == 0)
        throw std::runtime_error("Empty or invalid matrix file: " + filename);

    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = values;
    return mat;
}

IntMatrix readIntMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error("File not found: " + filename);

    std::vector<int> values;
    int cols = 0, rows = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        int val;
        int lineCount = 0;

        while (ss >> val) {
            if (val != 0 && val != 1)
                throw std::runtime_error(
                    "Game of Life file must contain only 0 and 1: "
                    + filename);
            values.push_back(val);
            lineCount++;
        }

        if (lineCount == 0) continue;

        if (rows == 0)
            cols = lineCount;
        else if (lineCount != cols)
            throw std::runtime_error(
                "Inconsistent row sizes in file: " + filename);
        rows++;
    }

    if (rows == 0 || cols == 0)
        throw std::runtime_error("Empty or invalid pattern file: " + filename);

    IntMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = values;
    return mat;
}

void writeMatrixToFile(const std::string& filename, double* data, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            file << data[i * cols + j] << " ";
        file << "\n";
    }
    file.close();
}

void writeIntMatrixToFile(const std::string& filename, int* data, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            file << data[i * cols + j] << " ";
        file << "\n";
    }
    file.close();
}