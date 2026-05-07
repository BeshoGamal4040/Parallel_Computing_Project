#include "io_utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

Matrix readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::vector<double> values;
    int cols = 0;
    int rows = 0;
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

        if (rows == 0)
            cols = lineCount;
        else if (lineCount != cols) {
            std::cerr << "Error: inconsistent row sizes in matrix file\n";
            exit(1);
        }
        rows++;
    }

    if (rows == 0 || cols == 0) {
        std::cerr << "Error: empty or invalid matrix file\n";
        exit(1);
    }

    Matrix mat;
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