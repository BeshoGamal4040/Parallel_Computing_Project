#include <mpi.h>
#include <iostream>
#include <limits>
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

    double start, end;

    while (true) {

        int choice = 0;
        int rows = 0, cols = 0;
        int iterations = 0;
        Matrix A_mat, B_mat;
        IntMatrix GOL_mat;
        string fileA, fileB, fileGOL;
        double* A_data = nullptr;
        double* B_data = nullptr;
        double* C_data = nullptr;
        bool validInput = true;

        if (rank == 0) {
            cout << "\n============================\n";
            cout << "MPI Parallel Processing System\n";
            cout << "Running on " << size << " processes\n";
            cout << "============================\n";

            cout << "Choose Algorithm:\n";
            cout << "1. Game of Life\n";
            cout << "2. Matrix Multiplication\n";
            cout << "3. Quit\n";

            
            while (true) {
                cout << "> ";
                if (!(cin >> choice)) {
                    cout << "[Error] Invalid input. Please enter 1, 2, or 3.\n";
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    continue;
                }
                if (choice < 1 || choice > 3) {
                    cout << "[Error] Please enter 1, 2, or 3.\n";
                    continue;
                }
                break; 
            }
        }

        MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        
        if (choice == 3) {
            if (rank == 0)
                cout << "\n[INFO] Goodbye!\n";
            break;
        }

        
        if (choice == GAME_OF_LIFE && rank == 0) {

            
            while (true) {
                cout << "Enter pattern file name (e.g. glider.txt):\n> ";
                cin >> fileGOL;
                try {
                    GOL_mat = readIntMatrixFromFile(fileGOL);
                    rows = GOL_mat.rows;
                    cols = GOL_mat.cols;
                    cout << "Pattern loaded: " << rows << "x" << cols << " grid\n";
                    break; 
                }
                catch (const exception& e) {
                    cout << "[Error] " << e.what() << "\n";
                    cout << "[Info] Please try again.\n";
                }
            }

            
            while (true) {
                cout << "Enter number of iterations (must be > 0):\n> ";
                if (!(cin >> iterations) || iterations <= 0) {
                    cout << "[Error] Iterations must be a positive integer.\n";
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    continue;
                }
                break;
            }
        }

        
        else if (choice == MATRIX_MULTIPLICATION && rank == 0) {

            
            while (true) {
                cout << "Enter file name for Matrix A:\n> ";
                cin >> fileA;
                try {
                    A_mat = readMatrixFromFile(fileA);
                    cout << "Matrix A loaded: "
                        << A_mat.rows << "x" << A_mat.cols << "\n";
                    break;
                }
                catch (const exception& e) {
                    cout << "[Error] " << e.what() << "\n";
                    cout << "[Info] Please try again.\n";
                }
            }

            
            while (true) {
                cout << "Enter file name for Matrix B:\n> ";
                cin >> fileB;
                try {
                    B_mat = readMatrixFromFile(fileB);
                    cout << "Matrix B loaded: "
                        << B_mat.rows << "x" << B_mat.cols << "\n";
                    break;
                }
                catch (const exception& e) {
                    cout << "[Error] " << e.what() << "\n";
                    cout << "[Info] Please try again.\n";
                }
            }

            
            while (A_mat.cols != B_mat.rows) {
                cout << "[Error] Incompatible dimensions: A is "
                    << A_mat.rows << "x" << A_mat.cols
                    << " but B is "
                    << B_mat.rows << "x" << B_mat.cols << "\n";
                cout << "[Error] A cols (" << A_mat.cols
                    << ") must equal B rows (" << B_mat.rows << ").\n";

                
                while (true) {
                    cout << "Enter file name for Matrix A:\n> ";
                    cin >> fileA;
                    try {
                        A_mat = readMatrixFromFile(fileA);
                        cout << "Matrix A loaded: "
                            << A_mat.rows << "x" << A_mat.cols << "\n";
                        break;
                    }
                    catch (const exception& e) {
                        cout << "[Error] " << e.what() << "\n";
                        cout << "[Info] Please try again.\n";
                    }
                }

                while (true) {
                    cout << "Enter file name for Matrix B:\n> ";
                    cin >> fileB;
                    try {
                        B_mat = readMatrixFromFile(fileB);
                        cout << "Matrix B loaded: "
                            << B_mat.rows << "x" << B_mat.cols << "\n";
                        break;
                    }
                    catch (const exception& e) {
                        cout << "[Error] " << e.what() << "\n";
                        cout << "[Info] Please try again.\n";
                    }
                }
            }

            cout << "A: " << A_mat.rows << "x" << A_mat.cols
                << ", B: " << B_mat.rows << "x" << B_mat.cols << "\n";

            A_data = A_mat.data.data();
            B_data = B_mat.data.data();
            C_data = new double[A_mat.rows * B_mat.cols]();
        }

        
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

        // Game of Life 
        if (choice == GAME_OF_LIFE) {
            vector<int> initialGrid(rows * cols, 0);
            if (rank == 0)
                initialGrid = GOL_mat.data;
            MPI_Bcast(initialGrid.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

            start = MPI_Wtime();

            vector<int> result = game_of_life(
                rows, cols, iterations, initialGrid, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            end = MPI_Wtime();

            if (rank == 0) {
                cout << "\nTotal Execution Time: " << end - start << " seconds\n";

                string outputFile;
                cout << "Enter output file name for final grid:\n> ";
                cin >> outputFile;

                writeIntMatrixToFile(outputFile, result.data(), rows, cols);
                cout << "Final grid written to " << outputFile << "\n";
            }
        }

        //  Matrix Multiplication
        if (choice == MATRIX_MULTIPLICATION) {
            start = MPI_Wtime();

            matrix_multiply_2D(N, Km, Mc,
                A_data, B_data, C_data,
                MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            end = MPI_Wtime();

            if (rank == 0) {
                cout << "Execution Time: " << end - start << " seconds\n";

                string outputFile;
                cout << "Enter output file name:\n> ";
                cin >> outputFile;

                writeMatrixToFile(outputFile, C_data, N, Mc);
                cout << "Result written to " << outputFile << "\n";

                delete[] C_data;
                C_data = nullptr;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

    } 

    MPI_Finalize();
    return 0;
}