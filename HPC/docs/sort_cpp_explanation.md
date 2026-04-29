# Detailed Technical Explanation of `Sort_OpenMp.cpp`

This document provides a comprehensive, line-by-line breakdown of the `Sort_OpenMp.cpp` source code. It explores the logic behind parallel bubble and merge sorts, the optimizations used to maximize throughput, and the dry-run simulations of the parallel execution models.

---

## 1. Bubble Sort Implementation (Even-Odd Transposition)

```cpp
class BubbleSorter {
public:
    static void sort_sequential(std::vector<int>& data) {
        int n = data.size();
        for (int pass = 0; pass < n; ++pass) {
            for (int j = 0; j < n - pass - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }

    static void sort_parallel(std::vector<int>& data) {
        int n = data.size();
        for (int pass = 0; pass < n; ++pass) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n - 1; j += 2) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }

            #pragma omp parallel for schedule(static)
            for (int j = 1; j < n - 1; j += 2) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};
```

### Line-by-Line Explanation
- **`class BubbleSorter`**: Encapsulates bubble sort logic within an OOP structure.
- **`sort_sequential`**: Implements the standard $O(N^2)$ bubble sort.
- **`static void sort_parallel(...)`**: Entry point for parallel bubble sort.
- **`for (int pass = 0; pass < n; ++pass)`**: The outer loop ensures the array is fully sorted by repeating the transposition phases.
- **`#pragma omp parallel for schedule(static)`**: Parallellizes the following loop. `static` scheduling is used because the work per iteration is identical (one comparison/swap), which minimizes overhead.
- **`for (int j = 0; j < n - 1; j += 2)`**: The **Even Phase**. It compares indices (0,1), (2,3), etc. Since these pairs are disjoint, no two threads will ever write to the same memory index simultaneously.
- **`for (int j = 1; j < n - 1; j += 2)`**: The **Odd Phase**. It compares indices (1,2), (3,4), etc. Like the even phase, these pairs are disjoint.

### Justification & "What if"
- **Justification**: Standard bubble sort is difficult to parallelize because each step depends on the previous one. **Even-Odd Transposition** breaks the dependency into two independent phases, allowing for massive data-parallel speedup.
- **Risk of Omission**: If you tried to parallelize a standard bubble sort loop directly, you would have a **Data Race**. Two threads might try to swap elements at indices $(j, j+1)$ and $(j+1, j+2)$ at the same time, leading to corrupted data.

### Dry Run (Even-Odd)
**Initial State**: `{4, 2, 7, 1}` (N=4)
1.  **Pass 0, Even Phase**:
    - Compare (0,1): 4 > 2? Yes → `{2, 4, 7, 1}`
    - Compare (2,3): 7 > 1? Yes → `{2, 4, 1, 7}`
2.  **Pass 0, Odd Phase**:
    - Compare (1,2): 4 > 1? Yes → `{2, 1, 4, 7}`
3.  **Pass 1, Even Phase**:
    - Compare (0,1): 2 > 1? Yes → `{1, 2, 4, 7}`
    - Compare (2,3): 4 > 7? No.
4.  **Result**: Array is sorted.

---

## 2. Merge Sort Implementation (Task-Based)

```cpp
class MergeSorter {
public:
    static void sort_parallel(std::vector<int>& data, int left, int right) {
        std::vector<int> aux_buffer(data.size());
        #pragma omp parallel
        {
            #pragma omp single
            sort_with_tasks(data, aux_buffer, left, right);
        }
    }

private:
    static constexpr int SEQUENTIAL_THRESHOLD = 2048;

    static void sort_with_tasks(std::vector<int>& data, std::vector<int>& aux, int left, int right) {
        int size = right - left + 1;
        if (size > SEQUENTIAL_THRESHOLD) {
            #pragma omp task shared(data, aux)
            sort_with_tasks(data, aux, left, mid);
            #pragma omp task shared(data, aux)
            sort_with_tasks(data, aux, mid + 1, right);
            #pragma omp taskwait
        } else {
            sort_sequential(data, left, mid);
            sort_sequential(data, mid + 1, right);
        }
        merge_with_aux(data, aux, left, mid, right);
    }
};
```

### Line-by-Line Explanation
- **`std::vector<int> aux_buffer(data.size())`**: Allocates a single helper array.
- **`#pragma omp parallel`**: Creates the thread pool once.
- **`#pragma omp single`**: Ensures that only one thread starts the initial recursion, while others wait to steal tasks.
- **`static constexpr int SEQUENTIAL_THRESHOLD = 2048`**: The **Cutoff**. Prevents task overhead for small sub-arrays.
- **`#pragma omp task shared(data, aux)`**: Spawns a background task to sort the left/right half. The `shared` clause ensures all tasks use the same pre-allocated memory.
- **`#pragma omp taskwait`**: Synchronizes the recursion. It forces the current thread to wait until both children have finished sorting their halves before it attempts to merge them.
- **`merge_with_aux(...)`**: Uses the pre-allocated buffer to merge the two halves without any additional heap allocations.

### Justification & "What if"
- **Justification (Aux Buffer)**: Traditional merge sort allocates a `std::vector` inside every `merge` call. In parallel, thousands of threads hitting the global memory allocator (`new`/`delete`) causes **Heap Contention**, which can make the code slower than sequential. Pre-allocating one buffer eliminates this.
- **Justification (Threshold)**: Creating an OpenMP task costs ~1000 cycles. If the array size is small (e.g., 100 elements), sorting it is faster than the time it takes to create a task. The 2048 threshold ensures we only parallelize "meaningful" work.
- **Risk of Omission**: Without `taskwait`, the `merge` function would attempt to combine two halves that haven't been sorted yet, resulting in an unsorted output.

### Dry Run (Task-Based)
1.  **Root**: `sort_parallel(0, 10000)` creates a single parallel region.
2.  **Spawning**: Thread 0 creates Task A (`sort(0, 5000)`) and Task B (`sort(5001, 10000)`).
3.  **Work Stealing**: Thread 0 hits `taskwait` and becomes idle. Thread 1 "steals" Task A. Thread 2 "steals" Task B.
4.  **Threshold**: Thread 1 sees that its range (5000) > 2048, so it spawns further tasks.
5.  **Merge**: Once Task A and B finish, Thread 0 resumes and performs the final merge.

---

## 3. Benchmarking Framework

```cpp
void run_benchmarks() {
    int available_cores = omp_get_max_threads();
    const std::vector<int> test_sizes = {100, 1000, 10000, 20000, 30000};

    for (int n : test_sizes) {
        std::vector<int> original = generate_random_array(n);
        benchmark_bubble_sort(original, n, available_cores);
        benchmark_merge_sort(original, n, available_cores);
    }
}
```

### Line-by-Line Explanation
- **`available_cores`**: Detects how many hardware threads are available (typically 16 on modern high-end machines).
- **`generate_random_array`**: Uses `std::mt19937` with a fixed seed. This ensures that every test run uses the same random sequence, making the comparison between sequential and parallel fair.
- **`measure([&]{ ... })`**: A lambda-based timer that wraps the sort function with `omp_get_wtime()`.

### Justification & "What if"
- **Justification**: High-resolution wall-clock timing (`omp_get_wtime`) is used instead of CPU time (`clock()`). CPU time sums the time of all cores, which would make parallel code look slower than sequential. Wall-clock time measures the actual time a user waits.
- **Risk of Omission**: If you don't use a fixed random seed, the "sortedness" of the input array will change every time. A nearly-sorted array is much faster to sort than a reversed one. This would make your benchmark results inconsistent and unreliable.
