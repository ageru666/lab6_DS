#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>


std::vector<std::vector<int>> MatrixMultiplySequential(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    int numRowsA = A.size();
    int numColsA = A[0].size();
    int numColsB = B[0].size();
    std::vector<std::vector<int>> C(numRowsA, std::vector<int>(numColsB, 0));

    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numColsB; ++j) {
            for (int k = 0; k < numColsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

std::vector<std::vector<int>> MatrixMultiplyRowwise(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int numRows, int numCols, int numColsB, int rank, int size) {
    std::vector<std::vector<int>> result(numRows, std::vector<int>(numColsB, 0));

  
    std::vector<std::vector<int>> localA(numRows, std::vector<int>(numCols, 0));
    MPI_Scatter(A.data(), numRows * numCols / size, MPI_INT, localA.data(), numRows * numCols / size, MPI_INT, 0, MPI_COMM_WORLD);

   
    std::vector<std::vector<int>> localB(numCols, std::vector<int>(numColsB / size, 0));
    MPI_Scatter(B.data(), numCols * numColsB / size, MPI_INT, localB.data(), numCols * numColsB / size, MPI_INT, 0, MPI_COMM_WORLD);

  
    std::vector<std::vector<int>> localC(numRows, std::vector<int>(numColsB / size, 0));
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColsB / size; ++j) {
            for (int k = 0; k < numCols; ++k) {
                localC[i][j] += localA[i][k] * localB[k][j];
            }
        }
    }

    MPI_Gather(localC.data(), numRows * numColsB / size, MPI_INT, result.data(), numRows * numColsB / size, MPI_INT, 0, MPI_COMM_WORLD);

    return result;
}

void MatrixMultiplyFox(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C, int numRows, int numCols, int numColsB, int rank, int size) {
    int blockSize = numRows / size;
    std::vector<int> localA(blockSize * blockSize);
    std::vector<int> localB(blockSize * blockSize);
    std::vector<int> localC(blockSize * blockSize, 0);

    MPI_Scatter(A.data(), blockSize * blockSize, MPI_INT, localA.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), blockSize * blockSize, MPI_INT, localB.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    for (int step = 0; step < size; ++step) {
        int pivot = (rank + step) % size;
        MPI_Bcast(localA.data(), blockSize * blockSize, MPI_INT, pivot, MPI_COMM_WORLD);

        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                for (int k = 0; k < blockSize; ++k) {
                    localC[i * blockSize + j] += localA[i * blockSize + k] * localB[k * blockSize + j];
                }
            }
        }

        int nextProcess = (rank + 1) % size;
        int prevProcess = (rank - 1 + size) % size;
        MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_INT, nextProcess, 0, prevProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

   
    MPI_Gather(localC.data(), blockSize * blockSize, MPI_INT, C.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);
}

void MatrixMultiplyKannon(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C, int numRows, int numCols, int numColsB, int rank, int size) {
    int blockSize = numRows / size; 
    std::vector<int> localA(blockSize * blockSize);
    std::vector<int> localB(blockSize * blockSize);
    std::vector<int> localC(blockSize * blockSize, 0);

    MPI_Scatter(A.data(), blockSize * blockSize, MPI_INT, localA.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), blockSize * blockSize, MPI_INT, localB.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> tempA(blockSize * blockSize);
    int destProcess, sourceProcess;

    for (int step = 0; step < size; ++step) {
        destProcess = (rank + step) % size;
        sourceProcess = (rank - step + size) % size;

        MPI_Sendrecv(localA.data(), blockSize * blockSize, MPI_INT, destProcess, 0, tempA.data(), blockSize * blockSize, MPI_INT, sourceProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                for (int k = 0; k < blockSize; ++k) {
                    localC[i * blockSize + j] += tempA[i * blockSize + k] * localB[k * blockSize + j];
                }
            }
        }

        if (step < size - 1) {
            MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_INT, destProcess, 0, sourceProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Gather(localC.data(), blockSize * blockSize, MPI_INT, C.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int numRows = 4;
    const int numCols = 4;
    const int numColsB = 4;

    std::vector<std::vector<int>> A(numRows, std::vector<int>(numCols, rank + 1));
    std::vector<std::vector<int>> B(numCols, std::vector<int>(numColsB, rank + 1));
    std::vector<std::vector<int>> result;

    if (rank == 0) {
        result = std::vector<std::vector<int>>(numRows, std::vector<int>(numColsB, 0));
    }

    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        result = MatrixMultiplySequential(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Sequential Algorithm Duration: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        result = MatrixMultiplyRowwise(A, B, numRows, numCols, numColsB, rank, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Row-wise Algorithm Duration: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        MatrixMultiplyFox(A, B, result, numRows, numCols, numColsB, rank, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Fox Algorithm Duration: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        MatrixMultiplyKannon(A, B, result, numRows, numCols, numColsB, rank, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Kannon Algorithm Duration: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
