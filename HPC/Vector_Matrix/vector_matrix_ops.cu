#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <iomanip>
#include <numeric>

using namespace std;

/**
 * @brief Macro for checking CUDA errors and aborting on failure.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_SIZE 16

// Number of times to repeat GPU kernel timing for stable averages (avoids cold-start noise)
#define TIMING_REPEATS 5

/**
 * @brief Helper: time a kernel using CUDA Events for precise GPU-only measurement.
 *        Runs the kernel TIMING_REPEATS times and returns the average duration (ms).
 */
template <typename KernelFunc>
double measureKernelMs(KernelFunc kernel) {
    // Warm-up to ensure GPU is at full speed before we measure
    kernel();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalMs = 0.0f;
    for (int r = 0; r < TIMING_REPEATS; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(totalMs) / TIMING_REPEATS;
}

/**
 * @brief Displays detailed information about the CUDA-enabled device.
 */
void printCudaDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    cout << "\n===== CUDA Device Information =====" << endl;
    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        cout << "Device " << d << ": " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
        cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << endl;
        cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    }
    cout << "===================================\n" << endl;
}

// ================== GPU KERNELS ==================

/**
 * @brief CUDA Kernel for Vector Addition (Parallel).
 * @param A Input vector A.
 * @param B Input vector B.
 * @param C Output vector C = A + B.
 * @param n Size of the vectors.
 */
__global__ void vectorAddKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief CUDA Kernel for Matrix Multiplication using Global Memory (Naive Parallel).
 *        Each thread computes one output element C[row][col] by reading
 *        rows from A and columns from B directly from global (device) memory.
 * @param A Input matrix A (N x N).
 * @param B Input matrix B (N x N).
 * @param C Output matrix C = A * B.
 * @param N Dimension of the square matrices.
 */
__global__ void matMulGlobalKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief CUDA Kernel for Matrix Multiplication using Shared Memory (Tiled Parallel).
 *        Tiles of A and B are loaded into fast shared memory, dramatically reducing
 *        costly global memory accesses. Each TILE_SIZE x TILE_SIZE thread block
 *        cooperatively loads one tile per iteration and computes a partial sum.
 * @param A Input matrix A (N x N).
 * @param B Input matrix B (N x N).
 * @param C Output matrix C = A * B.
 * @param N Dimension of the square matrices.
 */
__global__ void matMulSharedKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory (with boundary check)
        sA[ty][tx] = (row < N && (t * TILE_SIZE + tx) < N) ? A[row * N + t * TILE_SIZE + tx] : 0.0f;
        // Load tile of B into shared memory (with boundary check)
        sB[ty][tx] = (col < N && (t * TILE_SIZE + ty) < N) ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;

        __syncthreads(); // Ensure tile is fully loaded before computing

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads(); // Ensure all threads done before loading next tile
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ================== CPU (SEQUENTIAL) FUNCTIONS ==================

/**
 * @brief Sequential CPU implementation of Vector Addition.
 */
void cpuVectorAdd(const vector<float>& A, const vector<float>& B, vector<float>& C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

/**
 * @brief Sequential CPU implementation of Matrix Multiplication (triple loop).
 */
void cpuMatMul(const vector<float>& A, const vector<float>& B, vector<float>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Main benchmarking entry point.
 *
 * Benchmarks:
 *   1. Vector Addition: Sequential CPU vs Parallel GPU
 *   2. Matrix Multiplication: Sequential CPU vs Global GPU vs Shared GPU
 *
 * GPU timing uses CUDA Events for precise kernel-only measurement and averages
 * over TIMING_REPEATS runs to eliminate first-run cold-start noise.
 */
int main() {
    printCudaDeviceInfo();

    // Write separate CSV files for vector and matrix benchmarks
    ofstream matFile("vector_matrix_result.txt");
    matFile << "SIZE,CPU_SEQ_TIME,GPU_GLOBAL_TIME,GPU_SHARED_TIME,SPEEDUP_GLOBAL,SPEEDUP_SHARED,SHARED_ADVANTAGE\n";

    ofstream vecFile("vector_result.txt");
    vecFile << "SIZE,CPU_SEQ_TIME,GPU_PARALLEL_TIME,SPEEDUP\n";

    vector<int> matrixSizes = {256, 512, 1024, 2048};

    for (int N : matrixSizes) {
        cout << "\n>>> Benchmarking N = " << N << " <<<" << endl;

        // -------- VECTOR ADDITION --------
        int vecSize = N * 1024;
        size_t vecBytes = vecSize * sizeof(float);
        float *d_vecA, *d_vecB, *d_vecC;

        CUDA_CHECK(cudaMalloc(&d_vecA, vecBytes));
        CUDA_CHECK(cudaMalloc(&d_vecB, vecBytes));
        CUDA_CHECK(cudaMalloc(&d_vecC, vecBytes));

        vector<float> h_vecA(vecSize, 1.0f), h_vecB(vecSize, 2.0f), h_vecC(vecSize);
        CUDA_CHECK(cudaMemcpy(d_vecA, h_vecA.data(), vecBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vecB, h_vecB.data(), vecBytes, cudaMemcpyHostToDevice));

        // Sequential CPU timing
        auto t0 = chrono::high_resolution_clock::now();
        cpuVectorAdd(h_vecA, h_vecB, h_vecC, vecSize);
        auto t1 = chrono::high_resolution_clock::now();
        double cpuVecMs = chrono::duration<double, milli>(t1 - t0).count();

        // Parallel GPU timing (CUDA Events, averaged)
        dim3 vecBlock(256);
        dim3 vecGrid((vecSize + vecBlock.x - 1) / vecBlock.x);
        double gpuVecMs = measureKernelMs([&]() {
            vectorAddKernel<<<vecGrid, vecBlock>>>(d_vecA, d_vecB, d_vecC, vecSize);
        });

        double vecSpeedup = cpuVecMs / gpuVecMs;
        cout << fixed << setprecision(3);
        cout << "  [Vector Add]  Sequential CPU = " << cpuVecMs
             << " ms  |  Parallel GPU = " << gpuVecMs
             << " ms  |  Speedup = " << vecSpeedup << "x" << endl;

        vecFile << vecSize << "," << cpuVecMs << "," << gpuVecMs << "," << vecSpeedup << "\n";

        CUDA_CHECK(cudaFree(d_vecA));
        CUDA_CHECK(cudaFree(d_vecB));
        CUDA_CHECK(cudaFree(d_vecC));

        // -------- MATRIX MULTIPLICATION --------
        size_t matSize = (size_t)N * N;
        size_t matBytes = matSize * sizeof(float);
        float *d_matA, *d_matB, *d_matC;

        CUDA_CHECK(cudaMalloc(&d_matA, matBytes));
        CUDA_CHECK(cudaMalloc(&d_matB, matBytes));
        CUDA_CHECK(cudaMalloc(&d_matC, matBytes));

        vector<float> h_matA(matSize, 1.0f), h_matB(matSize, 2.0f), h_matC(matSize);
        CUDA_CHECK(cudaMemcpy(d_matA, h_matA.data(), matBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matB, h_matB.data(), matBytes, cudaMemcpyHostToDevice));

        // Sequential CPU matrix multiply timing
        t0 = chrono::high_resolution_clock::now();
        cpuMatMul(h_matA, h_matB, h_matC, N);
        t1 = chrono::high_resolution_clock::now();
        double cpuMatMs = chrono::duration<double, milli>(t1 - t0).count();

        dim3 matBlock(TILE_SIZE, TILE_SIZE);
        dim3 matGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        // Global Memory GPU timing (CUDA Events, averaged)
        double globalMatMs = measureKernelMs([&]() {
            matMulGlobalKernel<<<matGrid, matBlock>>>(d_matA, d_matB, d_matC, N);
        });

        // Shared Memory GPU timing (CUDA Events, averaged)
        double sharedMatMs = measureKernelMs([&]() {
            matMulSharedKernel<<<matGrid, matBlock>>>(d_matA, d_matB, d_matC, N);
        });

        double speedupGlobal = cpuMatMs / globalMatMs;
        double speedupShared = cpuMatMs / sharedMatMs;
        double sharedAdvantage = globalMatMs / sharedMatMs;

        cout << "  [Matrix Mul]  Sequential CPU  = " << cpuMatMs << " ms" << endl;
        cout << "                Global GPU       = " << globalMatMs
             << " ms  |  Speedup = " << speedupGlobal << "x" << endl;
        cout << "                Shared GPU       = " << sharedMatMs
             << " ms  |  Speedup = " << speedupShared << "x"
             << "  |  Shared vs Global = " << sharedAdvantage << "x" << endl;

        matFile << N << "," << cpuMatMs << "," << globalMatMs << "," << sharedMatMs
                << "," << speedupGlobal << "," << speedupShared << "," << sharedAdvantage << "\n";

        CUDA_CHECK(cudaFree(d_matA));
        CUDA_CHECK(cudaFree(d_matB));
        CUDA_CHECK(cudaFree(d_matC));
    }

    matFile.close();
    vecFile.close();
    cout << "\nBenchmarks completed. Results saved." << endl;

    return 0;
}
