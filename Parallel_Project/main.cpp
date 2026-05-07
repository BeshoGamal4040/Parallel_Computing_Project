#include <mpi.h>
#include <iostream>
#include "config.h"
#include "io_utils.h"
#include "mpi_utils.h"

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            cerr << "Error: need at least 2 processes.\n";
        MPI_Finalize();
        return 1;
    }

    int choice = 0;
    int rows = 0, cols = 0;
    int iterations = 0;
    double alpha = 0.1;
    Matrix A_mat, B_mat;
    string fileA, fileB;
    double* A_data = nullptr;
    double* B_data = nullptr;
    double* C_data = nullptr;

    double start, end;

    if (rank == 0) {
        cout << "============================\n";
        cout << "MPI Parallel Processing System\n";
        cout << "Running on " << size << " processes\n";
        cout << "============================\n";

        cout << "Choose Algorithm:\n";
        cout << "1. Heat Diffusion\n";
        cout << "2. Game of Life\n";
        cout << "3. Matrix Multiplication\n";
        cin >> choice;

        if (choice == HEAT_DIFFUSION) {
            cout << "Enter grid rows and cols:\n";
            cin >> rows >> cols;
            cout << "Enter number of iterations:\n";
            cin >> iterations;
            cout << "Enter diffusion coefficient alpha (e.g. 0.1):\n";
            cin >> alpha;
        }
        else if (choice == MATRIX_MULTIPLICATION) {
            cout << "Enter file name for Matrix A:\n";
            cin >> fileA;
            cout << "Enter file name for Matrix B:\n";
            cin >> fileB;

            A_mat = readMatrixFromFile(fileA);
            B_mat = readMatrixFromFile(fileB);

            if (A_mat.cols != B_mat.rows) {
                cout << "Error: matrix Multiplication can't be computed.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            cout << "A: " << A_mat.rows << "x" << A_mat.cols
                << ", B: " << B_mat.rows << "x" << B_mat.cols << "\n";

            A_data = A_mat.data.data();
            B_data = B_mat.data.data();
            C_data = new double[A_mat.rows * B_mat.cols]();
        }
    }

    // Broadcast all inputs to every process
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int N = 0, Km = 0, Mc = 0;
    if (rank == 0 && choice == MATRIX_MULTIPLICATION) {
        N = A_mat.rows;
        Km = A_mat.cols;
        Mc = B_mat.cols;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Km, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Mc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Heat Diffusion ──────────────────────────────────────
    if (choice == HEAT_DIFFUSION) {
        start = MPI_Wtime();

        vector<double> result = heat_diffusion(
            rows, cols, iterations, alpha, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();

        // ✅ Only rank 0 does I/O — no cin inside MPI functions
        if (rank == 0) {
            cout << "Execution Time: " << end - start << " seconds\n";

            string outputFile;
            cout << "Enter output file name:\n";
            cin >> outputFile;

            writeMatrixToFile(outputFile, result.data(), rows, cols);
            cout << "Result written to " << outputFile << "\n";
        }
    }

    // ── Matrix Multiplication ───────────────────────────────
    if (choice == MATRIX_MULTIPLICATION) {
        start = MPI_Wtime();

        matrix_multiply_2D(N, Km, Mc,
            A_data, B_data, C_data,
            MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();

        if (rank == 0) {
            cout << "Execution Time: " << end - start << " seconds\n";

            if (N <= 5 && Mc <= 5) {
                cout << "\nResult Matrix:\n";
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < Mc; j++)
                        cout << C_data[i * Mc + j] << " ";
                    cout << "\n";
                }
            }

            string outputFile;
            cout << "Enter output file name:\n";
            cin >> outputFile;

            writeMatrixToFile(outputFile, C_data, N, Mc);
            cout << "Result written to " << outputFile << "\n";

            delete[] C_data;
        }
    }

    if (rank == 0)
        cout << "\n[INFO] Finished.\n";

    MPI_Finalize();
    return 0;
}