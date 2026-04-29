#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <float.h>

/**
 * @brief Displays CUDA device information for debugging and optimization.
 */
void display_cuda_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "\n===== CUDA Architecture Information =====" << std::endl;
    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::cout << "Device " << d << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Shared Memory per Block (KB): " << (prop.sharedMemPerBlock / 1024) << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

// Functors for different reduction operations.
struct SumOp { 
    __device__ float operator()(float a, float b) const { return a + b; } 
    __device__ static float identity() { return 0.0f; } 
};

struct MinOp { 
    __device__ float operator()(float a, float b) const { return fminf(a, b); } 
    __device__ static float identity() { return FLT_MAX; } 
};

struct MaxOp { 
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); } 
    __device__ static float identity() { return -FLT_MAX; } 
};

/**
 * @brief Optimized reduction kernel using shared memory and warp-level primitives.
 * @tparam Op The reduction operation (Sum, Min, Max).
 * @param blockPartialResults Output array where each block writes its partial result.
 * @param deviceInputValues Input array of floats to be reduced.
 * @param totalElements Total number of elements in the input.
 * @param op Functor defining the reduction logic.
 */
template <typename Op>
__global__ void optimized_reduction_kernel(float* blockPartialResults, const float* deviceInputValues, int totalElements, Op op) {
    // Dynamic shared memory allocated at kernel launch
    extern __shared__ float sharedMemoryBuffer[];

    unsigned int threadInBlockIdx = threadIdx.x;
    unsigned int globalDataIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Load input into shared memory. 
    // If the index is out of bounds, use the identity value for the operation.
    sharedMemoryBuffer[threadInBlockIdx] = (globalDataIdx < totalElements) ? deviceInputValues[globalDataIdx] : Op::identity();
    __syncthreads();

    // Phase 2: Interleaved reduction in shared memory.
    // We reduce within the block until only one value remains.
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadInBlockIdx < stride) {
            sharedMemoryBuffer[threadInBlockIdx] = op(sharedMemoryBuffer[threadInBlockIdx], sharedMemoryBuffer[threadInBlockIdx + stride]);
        }
        __syncthreads();
    }

    // Phase 3: Warp-level reduction (optimization).
    // For the last 32 elements (one warp), we can avoid __syncthreads() 
    // by using volatile shared memory or shuffle instructions.
    if (threadInBlockIdx < 32) {
        volatile float* warpShared = sharedMemoryBuffer;
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 32]);
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 16]);
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 8]);
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 4]);
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 2]);
        warpShared[threadInBlockIdx] = op(warpShared[threadInBlockIdx], warpShared[threadInBlockIdx + 1]);
    }

    // Write the result of the entire block to the output array.
    if (threadInBlockIdx == 0) {
        blockPartialResults[blockIdx.x] = sharedMemoryBuffer[0];
    }
}

/**
 * @brief Orchestrates a full GPU-side reduction across multiple kernel passes.
 */
template <typename Op>
float launch_multi_pass_reduction(const float* deviceInputBuffer, int elementCount, Op reductionOp) {
    const int threadsPerBlock = 256;
    int remainingElements = elementCount;
    const float* currentInputRef = deviceInputBuffer;
    
    // Allocate space for intermediate results between passes.
    // Each pass reduces N elements to M blocks.
    float* deviceIntermediateBuffer;
    int maxBlockCount = (elementCount + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&deviceIntermediateBuffer, maxBlockCount * sizeof(float));
    
    float* currentOutputRef = deviceIntermediateBuffer;

    while (remainingElements > 1) {
        int blockCount = (remainingElements + threadsPerBlock - 1) / threadsPerBlock;
        
        optimized_reduction_kernel<Op><<<blockCount, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            currentOutputRef, currentInputRef, remainingElements, reductionOp
        );
        
        cudaDeviceSynchronize();

        // The output of this pass becomes the input for the next pass.
        currentInputRef = currentOutputRef;
        remainingElements = blockCount;
    }

    float finalResult;
    cudaMemcpy(&finalResult, currentInputRef, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceIntermediateBuffer);
    
    return finalResult;
}

/**
 * @brief Benchmarks the reduction algorithms and compares CPU vs GPU.
 */
void execute_reduction_benchmark(int size, std::ofstream& csvFile) {
    size_t dataSizeBytes = size * sizeof(float);
    std::vector<float> hostInputValues(size);
    
    // Data Generation:
    // Using (rand() % size) ensures that as N increases, the range of values also increases.
    // This is why previous runs had the same MAX/MIN (they were clamped to 0-99).
    for (int i = 0; i < size; i++) hostInputValues[i] = (float)(rand() % size);

    float *deviceInputValues;
    cudaMalloc(&deviceInputValues, dataSizeBytes);
    cudaMemcpy(deviceInputValues, hostInputValues.data(), dataSizeBytes, cudaMemcpyHostToDevice);

    // --- CPU Reduction ---
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    float cpuSumResult = 0;
    for (int i = 0; i < size; i++) cpuSumResult += hostInputValues[i];
    auto cpuEndTime = std::chrono::high_resolution_clock::now();
    double cpuDurationMs = std::chrono::duration<double, std::milli>(cpuEndTime - cpuStartTime).count();

    // --- GPU Reduction ---
    auto gpuStartTime = std::chrono::high_resolution_clock::now();
    float gpuSumResult = launch_multi_pass_reduction(deviceInputValues, size, SumOp());
    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    double gpuDurationMs = std::chrono::duration<double, std::milli>(gpuEndTime - gpuStartTime).count();

    // Perform other ops entirely on GPU
    float gpuMaxResult = launch_multi_pass_reduction(deviceInputValues, size, MaxOp());
    float gpuMinResult = launch_multi_pass_reduction(deviceInputValues, size, MinOp());
    
    // Average is derived from Sum
    float gpuAverageResult = gpuSumResult / size;

    double calculatedSpeedup = cpuDurationMs / gpuDurationMs;
    double hardwareEfficiency = calculatedSpeedup / 2560.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "N=" << std::setw(8) << size 
              << " | Sum: " << std::setw(12) << gpuSumResult 
              << " | Avg: " << std::setw(8) << gpuAverageResult
              << " | Max: " << std::setw(10) << gpuMaxResult 
              << " | Min: " << std::setw(5) << gpuMinResult 
              << " | Speedup: " << calculatedSpeedup << "x" << std::endl;

    csvFile << size << "," << cpuDurationMs << "," << gpuDurationMs << "," << calculatedSpeedup << "," << hardwareEfficiency << "\n";

    cudaFree(deviceInputValues);
}

int main() {
    display_cuda_info();

    // Seed the random number generator
    srand(time(NULL));

    std::ofstream csvFile("reduction_result.txt");
    csvFile << "N,SERIAL,PARALLEL,SPEEDUP,EFFICIENCY\n";

    int benchmarkSizes[] = {10000, 100000, 1000000, 5000000, 10000000};

    for (int size : benchmarkSizes) {
        execute_reduction_benchmark(size, csvFile);
    }

    csvFile.close();
    std::cout << "\nReduction benchmarks finalized. Data exported to reduction_result.txt" << std::endl;

    return 0;
}
