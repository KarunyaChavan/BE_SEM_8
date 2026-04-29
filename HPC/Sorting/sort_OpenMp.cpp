#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <random>
#include <omp.h>

/**
 * @brief Displays OpenMP runtime configuration and available hardware resources.
 */
void display_openmp_info() {
    std::cout << "\n===== OpenMP Environment =====\n";
#ifdef _OPENMP
    std::cout << std::left << std::setw(28) << "OpenMP Version:" << _OPENMP << "\n";
#endif
    std::cout << std::left << std::setw(28) << "Logical Processors:" << omp_get_num_procs() << "\n";
    std::cout << std::left << std::setw(28) << "Max Threads:" << omp_get_max_threads() << "\n";
    std::cout << std::left << std::setw(28) << "Dynamic Threads:" << omp_get_dynamic() << "\n";
    std::cout << "==============================\n\n";
}

/**
 * @brief Provides sequential and parallel implementations of Bubble Sort.
 * Uses the Even-Odd Transposition Sort as the standard parallel strategy.
 */
class BubbleSorter {
public:
    /**
     * @brief Standard sequential bubble sort.
     * @param data The array to be sorted in-place.
     */
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

    /**
     * @brief Parallel bubble sort using Even-Odd Transposition.
     * Even and odd phases compare non-overlapping adjacent pairs, making
     * each phase fully data-race free and safe for parallel execution.
     * @param data The array to be sorted in-place.
     */
    static void sort_parallel(std::vector<int>& data) {
        int n = data.size();
        for (int pass = 0; pass < n; ++pass) {
            // Even phase: independent pairs (0,1), (2,3), (4,5), ...
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n - 1; j += 2) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }

            // Odd phase: independent pairs (1,2), (3,4), (5,6), ...
            #pragma omp parallel for schedule(static)
            for (int j = 1; j < n - 1; j += 2) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

/**
 * @brief Provides sequential and parallel implementations of Merge Sort.
 *
 * The parallel variant solves two common performance pitfalls:
 *  1. Repeated heap allocation per merge (replaced by a single pre-allocated buffer).
 *  2. Creating multiple parallel regions (replaced by one region with task parallelism).
 */
class MergeSorter {
public:
    /**
     * @brief Sequential recursive merge sort.
     * @param data The array to sort.
     * @param left Left boundary index (inclusive).
     * @param right Right boundary index (inclusive).
     */
    static void sort_sequential(std::vector<int>& data, int left, int right) {
        if (left >= right) return;
        int mid = (left + right) / 2;
        sort_sequential(data, left, mid);
        sort_sequential(data, mid + 1, right);
        merge_inplace(data, left, mid, right);
    }

    /**
     * @brief Parallel merge sort entry point.
     * Allocates a single auxiliary buffer once, then launches tasks within
     * a single parallel region. All recursive tasks share this buffer through
     * disjoint index ranges, avoiding any race conditions.
     * @param data The array to sort.
     * @param left Left boundary index (inclusive).
     * @param right Right boundary index (inclusive).
     */
    static void sort_parallel(std::vector<int>& data, int left, int right) {
        // Allocate auxiliary buffer ONCE to eliminate per-call heap contention
        std::vector<int> aux_buffer(data.size());

        #pragma omp parallel
        {
            #pragma omp single
            sort_with_tasks(data, aux_buffer, left, right);
        }
    }

private:
    // Minimum sub-array size before falling back to sequential sort.
    // Balances task granularity: too small = task overhead dominates.
    static constexpr int SEQUENTIAL_THRESHOLD = 2048;

    /**
     * @brief Internal recursive implementation using OpenMP tasks.
     * @param data The array to sort.
     * @param aux Pre-allocated auxiliary buffer for merging (disjoint regions per task).
     * @param left Left boundary index (inclusive).
     * @param right Right boundary index (inclusive).
     */
    static void sort_with_tasks(std::vector<int>& data, std::vector<int>& aux,
                                int left, int right) {
        if (left >= right) return;

        int mid  = (left + right) / 2;
        int size = right - left + 1;

        if (size > SEQUENTIAL_THRESHOLD) {
            // Left and right halves operate on disjoint index ranges — no data race
            #pragma omp task shared(data, aux)
            sort_with_tasks(data, aux, left, mid);

            #pragma omp task shared(data, aux)
            sort_with_tasks(data, aux, mid + 1, right);

            // Ensure both halves are fully sorted before merging
            #pragma omp taskwait
        } else {
            // Sequential fallback for small sub-arrays
            sort_sequential(data, left, mid);
            sort_sequential(data, mid + 1, right);
        }

        // Merge using the pre-allocated buffer — no heap allocation here
        merge_with_aux(data, aux, left, mid, right);
    }

    /**
     * @brief Merges two sorted halves using the pre-allocated auxiliary buffer.
     * Uses the caller's disjoint region of `aux`, so concurrent merges are safe.
     */
    static void merge_with_aux(std::vector<int>& data, std::vector<int>& aux,
                               int left, int mid, int right) {
        // Snapshot the region into aux
        for (int i = left; i <= right; ++i) aux[i] = data[i];

        int i = left, j = mid + 1, k = left;
        while (i <= mid && j <= right) {
            data[k++] = (aux[i] <= aux[j]) ? aux[i++] : aux[j++];
        }
        while (i <= mid)   data[k++] = aux[i++];
        while (j <= right) data[k++] = aux[j++];
    }

    /**
     * @brief Standard in-place merge (used by sequential variant).
     */
    static void merge_inplace(std::vector<int>& data, int left, int mid, int right) {
        std::vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;

        while (i <= mid && j <= right) {
            temp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
        }
        while (i <= mid)   temp[k++] = data[i++];
        while (j <= right) temp[k++] = data[j++];

        for (int idx = 0; idx < k; ++idx) {
            data[left + idx] = temp[idx];
        }
    }
};

/**
 * @brief Generates a reproducible random array using a fixed seed.
 * @param size Number of elements.
 * @return Vector of random integers in [0, 9999].
 */
std::vector<int> generate_random_array(int size) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 9999);
    std::vector<int> data(size);
    for (auto& val : data) val = dist(rng);
    return data;
}

/**
 * @brief Orchestrates the full sorting benchmark across multiple input sizes.
 */
class SortBenchmark {
public:
    /**
     * @brief Opens results file and writes CSV header.
     */
    SortBenchmark() {
        results_file_.open("sort_result.txt");
        results_file_ << "N,Algorithm,Seq_Time,Par_Time,Speedup,Efficiency\n";
    }

    ~SortBenchmark() {
        if (results_file_.is_open()) results_file_.close();
    }

    /**
     * @brief Runs the full benchmark suite across all test sizes.
     */
    void run() {
        int available_cores = omp_get_max_threads();
        const std::vector<int> test_sizes = {100, 1000, 10000, 20000, 30000};

        std::cout << std::fixed << std::setprecision(6);

        for (int n : test_sizes) {
            std::vector<int> original = generate_random_array(n);

            benchmark_bubble_sort(original, n, available_cores);
            benchmark_merge_sort(original, n, available_cores);

            std::cout << "\n";
        }

        std::cout << "Results saved to sort_result.txt\n";
    }

private:
    std::ofstream results_file_;

    /** @brief Benchmarks sequential vs parallel bubble sort for a given input. */
    void benchmark_bubble_sort(const std::vector<int>& original, int n, int cores) {
        std::vector<int> seq_data = original;
        std::vector<int> par_data = original;

        double seq_time = measure([&]{ BubbleSorter::sort_sequential(seq_data); });

        // sort_parallel owns its own #pragma omp parallel for internally
        double par_time = measure([&]{ BubbleSorter::sort_parallel(par_data); });

        print_and_export("BubbleSort", n, seq_time, par_time, cores);
    }

    /** @brief Benchmarks sequential vs parallel merge sort for a given input. */
    void benchmark_merge_sort(const std::vector<int>& original, int n, int cores) {
        std::vector<int> seq_data = original;
        std::vector<int> par_data = original;

        double seq_time = measure([&]{ MergeSorter::sort_sequential(seq_data, 0, n - 1); });

        // sort_parallel manages its own parallel region and auxiliary buffer internally
        double par_time = measure([&]{ MergeSorter::sort_parallel(par_data, 0, n - 1); });

        print_and_export("MergeSort", n, seq_time, par_time, cores);
    }

    /** @brief Times a callable using high-resolution wall-clock time. */
    template <typename Func>
    double measure(Func fn) {
        double start = omp_get_wtime();
        fn();
        return omp_get_wtime() - start;
    }

    /** @brief Prints result to console and appends it to the CSV file. */
    void print_and_export(const std::string& algo, int n, double seq, double par, int cores) {
        double speedup    = seq / par;
        double efficiency = speedup / cores;

        std::cout << "[" << algo << " | N=" << std::setw(6) << n << "]"
                  << "  Seq=" << seq << "s"
                  << "  Par=" << par << "s"
                  << "  Speedup=" << std::setprecision(2) << speedup << "x\n";

        results_file_ << std::fixed << std::setprecision(8)
                      << n        << ","
                      << algo     << ","
                      << seq      << ","
                      << par      << ","
                      << speedup  << ","
                      << efficiency << "\n";
    }
};

/**
 * @brief Application entry point.
 */
int main() {
    display_openmp_info();
    SortBenchmark benchmark;
    benchmark.run();
    return 0;
}
