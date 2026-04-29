# Technical Documentation: CUDA Vector & Matrix Benchmarks (`VECTOR_MATRIX_CUDA.cu`)

This document provides a comprehensive, line-by-line explanation of the CUDA implementation for benchmarking **Vector Addition** and **Matrix Multiplication**, covering both sequential (CPU) and parallel (GPU) implementations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Accurate GPU Timing: CUDA Events](#accurate-gpu-timing-cuda-events)
3. [Kernels: Vector Addition](#kernels-vector-addition)
4. [Kernels: Matrix Multiplication (Global Memory)](#kernels-matrix-multiplication-global-memory)
5. [Kernels: Matrix Multiplication (Shared Memory)](#kernels-matrix-multiplication-shared-memory)
6. [Sequential CPU Baselines](#sequential-cpu-baselines)
7. [Benchmarking Flow in `main()`](#benchmarking-flow-in-main)
8. [Performance Summary](#performance-summary)

---

## Architecture Overview

The file benchmarks two operations across three execution strategies:

| Operation | Sequential CPU | Parallel GPU (Global) | Parallel GPU (Shared) |
|---|---|---|---|
| Vector Addition | ✅ | ✅ | — |
| Matrix Multiplication | ✅ | ✅ | ✅ |

Results are exported to two CSV files:
- `vector_result.txt` — Vector Addition: CPU sequential vs GPU parallel
- `vector_matrix_result.txt` — Matrix Multiplication: CPU vs GPU Global vs GPU Shared

---

## Accurate GPU Timing: CUDA Events

```cpp
template <typename KernelFunc>
double measureKernelMs(KernelFunc kernel) {
    kernel();                        // Warm-up run (not measured)
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalMs = 0.0f;
    for (int r = 0; r < TIMING_REPEATS; r++) {
        cudaEventRecord(start);
        kernel();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Block CPU until GPU event completes
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }
    return static_cast<double>(totalMs) / TIMING_REPEATS;
}
```

### Why CUDA Events instead of `std::chrono`?

Using wall-clock time (`std::chrono`) for GPU benchmarking introduces **significant noise** because it measures:
- CUDA API/driver overhead
- CPU-to-GPU kernel dispatch latency
- OS scheduler delays

For small problems like N=256 (kernel runs ~0.08ms), even a few microseconds of driver jitter can produce **wildly inconsistent results**, causing the "Shared Memory Advantage" to appear unrealistically high or low.

**CUDA Events are hardware counters** embedded directly in the GPU's execution timeline. `cudaEventRecord(start)` inserts a timestamp into the GPU's command queue — it is recorded *inside* the GPU, measuring pure kernel execution time only.

Additionally:
- **Warm-up run**: The first kernel launch is discarded to allow the GPU to reach full clock speed and load the kernel into the instruction cache.
- **`TIMING_REPEATS = 5` averages**: Further reduces measurement variance.

---

## Kernels: Vector Addition

```cpp
__global__ void vectorAddKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

- **One thread per element**: Each CUDA thread independently computes one element of the output vector. With a typical `blockDim.x = 256` and enough blocks to cover all elements, all additions execute *simultaneously*.
- **`__restrict__`**: Tells the compiler that A, B, and C do not alias in memory. This allows the compiler to reorder loads/stores and generate more efficient code (e.g., avoiding redundant memory fences).
- **`blockIdx.x * blockDim.x + threadIdx.x`**: The standard formula for computing a global thread index in a 1D grid. `blockIdx.x` is the block's position in the grid, `blockDim.x` is the number of threads per block (256), and `threadIdx.x` is the thread's position within its block.
- **Boundary check (`if (i < n)`)**: Since the grid is padded to a multiple of `blockDim.x`, some threads may map to out-of-bounds indices. This guard prevents illegal memory access.

---

## Kernels: Matrix Multiplication (Global Memory)

```cpp
__global__ void matMulGlobalKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, int N) {

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
```

- **2D thread layout**: Uses a `dim3 matBlock(TILE_SIZE, TILE_SIZE)` block (16×16 = 256 threads). Each thread computes **one element** `C[row][col]`.
- **Triple loop replaced by parallelism**: The sequential CPU triple-loop `(i, j, k)` is reduced to a single inner loop over `k`. The `i` and `j` dimensions are parallelised across the 2D thread grid.
- **The problem with Global Memory**: For each output element `C[row][col]`, the thread must read the entire row `A[row][:]` and column `B[:][col]` — **N reads each** from VRAM. If N=1024, that's 2×1024 = 2048 high-latency memory reads per thread. Many threads in the same block read *the same elements of B*, causing massive redundant traffic to global memory.

**Memory Access Pattern Diagram:**
```
Thread computing C[row][col]:
  - Reads:  A[row][0], A[row][1], ..., A[row][N-1]   (entire row of A)
  - Reads:  B[0][col], B[1][col], ..., B[N-1][col]   (entire column of B)
  - Writes: C[row][col]                               (one element)
```

---

## Kernels: Matrix Multiplication (Shared Memory)

```cpp
__global__ void matMulSharedKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, int N) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        sA[ty][tx] = (row < N && (t*TILE_SIZE + tx) < N) ? A[row*N + t*TILE_SIZE + tx] : 0.0f;
        sB[ty][tx] = (col < N && (t*TILE_SIZE + ty) < N) ? B[(t*TILE_SIZE + ty)*N + col] : 0.0f;

        __syncthreads();  // Wait: all threads must finish loading before computing

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();  // Wait: all threads must finish computing before loading next tile
    }

    if (row < N && col < N) C[row*N + col] = sum;
}
```

### The Tiling Strategy

Instead of reading A and B from global memory N times each, the kernel divides the matrices into **TILE_SIZE × TILE_SIZE tiles** and processes them one tile at a time:

1. **Tile Load Phase**: All 256 threads in a block **cooperatively** load one tile of A and one tile of B into `__shared__` memory. Each thread loads just one element — so all 256 elements of each tile are loaded in parallel in a single step.
2. **Compute Phase**: Each thread multiplies its row in `sA` against the corresponding column in `sB` to accumulate its partial `sum`.
3. **Repeat**: Move to the next tile, repeat until all tiles are processed.

### Why Shared Memory is Faster

- **Shared memory latency** ≈ 20-30 clock cycles (on-chip SRAM)
- **Global memory latency** ≈ 600-800 clock cycles (off-chip DRAM)

For an N=1024 matrix with TILE_SIZE=16, each element of A is read from global memory only **N/TILE_SIZE = 64 times** instead of N=1024 times. This provides a theoretical **16× reduction** in global memory traffic.

**`__syncthreads()`**: A block-wide barrier. All 256 threads must reach this call before any of them may proceed. The first barrier ensures the tile is fully loaded before computation begins; the second ensures computation is done before the next tile overwrites shared memory.

### Boundary Condition Handling

```cpp
sA[ty][tx] = (row < N && (t*TILE_SIZE + tx) < N) ? A[...] : 0.0f;
```

For matrix sizes that are not exact multiples of TILE_SIZE, some threads in boundary tiles would map to out-of-bounds indices. Padding with `0.0f` (the identity for addition) ensures these threads contribute zero to the sum without corrupting the result.

---

## Sequential CPU Baselines

```cpp
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
```

The sequential CPU uses the standard triple-loop O(N³) algorithm. This is intentionally unoptimised (no BLAS, no loop unrolling, no cache blocking) to represent a true single-threaded baseline for speedup calculations.

---

## Benchmarking Flow in `main()`

For each problem size N ∈ {256, 512, 1024, 2048}:

```
1. Allocate host vectors / matrices
2. Transfer data to GPU (cudaMemcpy H→D)
3. Time sequential CPU version (std::chrono)
4. Time parallel GPU version (CUDA Events, warm-up + 5 runs)
5. Compute speedup = CPU_time / GPU_time
6. Write results to CSV
7. Free GPU memory
```

> Note: `std::chrono` is only used for CPU timing (where it is appropriate). GPU timing exclusively uses CUDA Events.

---

## Performance Summary

Benchmark results on an **NVIDIA GeForce RTX 3050 Laptop GPU** (Compute 8.6):

### Vector Addition

| Vector Size | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|
| 256K | 1.21 | 0.022 | ~55× |
| 512K | 2.44 | 0.043 | ~56× |
| 1M | 4.92 | 0.081 | ~61× |
| 2M | 9.76 | 0.153 | ~64× |

### Matrix Multiplication

| N | CPU (ms) | Global GPU (ms) | Shared GPU (ms) | Shared Advantage |
|---|---|---|---|---|
| 256 | 67 | 0.094 | 0.074 | 1.27× |
| 512 | 604 | 0.727 | 0.549 | 1.33× |
| 1024 | 5,031 | 5.913 | 4.410 | 1.34× |
| 2048 | 103,687 | 49.928 | 35.945 | 1.39× |

> **Key insight**: The Shared Memory advantage grows consistently with N (1.27× → 1.39×), correctly reflecting the increasing benefit of data reuse as the working set grows larger relative to cache capacity. This consistency is only possible because CUDA Events eliminate cold-start and driver latency noise from the measurement.
