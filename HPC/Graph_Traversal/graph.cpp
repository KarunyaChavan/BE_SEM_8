#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <memory>
#include <random>
#include <omp.h>

/**
 * @brief Represents an undirected graph using an adjacency list.
 */
class Graph {
public:
    /**
     * @brief Constructs a graph with a fixed number of vertices.
     * @param vertex_count The number of vertices in the graph.
     */
    explicit Graph(int vertex_count) 
        : vertex_count_(vertex_count), 
          adjacency_list_(vertex_count) {}

    /**
     * @brief Adds an undirected edge between two vertices.
     * @param u First vertex.
     * @param v Second vertex.
     */
    void add_undirected_edge(int u, int v) {
        if (u >= 0 && u < vertex_count_ && v >= 0 && v < vertex_count_) {
            adjacency_list_[u].push_back(v);
            adjacency_list_[v].push_back(u);
        }
    }

    /**
     * @brief Gets the total number of vertices.
     * @return Integer count of vertices.
     */
    int get_vertex_count() const { return vertex_count_; }
    
    /**
     * @brief Provides read-only access to neighbors of a vertex.
     * @param vertex The index of the vertex.
     * @return Const reference to the list of neighboring vertex indices.
     */
    const std::vector<int>& get_neighbors(int vertex) const { 
        return adjacency_list_[vertex]; 
    }

private:
    int vertex_count_;
    std::vector<std::vector<int>> adjacency_list_;
};

/**
 * @brief Utility class to generate random graphs based on edge density.
 */
class GraphGenerator {
public:
    /**
     * @brief Generates a random graph using edge probability.
     * @param n Number of vertices.
     * @param density Probability (0 to 1) that an edge exists between any two nodes.
     * @param seed Random seed for reproducibility.
     * @return unique_ptr to the generated Graph object.
     */
    static std::unique_ptr<Graph> generate_random_graph(int n, double density, unsigned int seed = 42) {
        auto graph = std::make_unique<Graph>(n);
        std::mt19937 random_engine(seed);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (distribution(random_engine) < density) {
                    graph->add_undirected_edge(i, j);
                }
            }
        }
        return graph;
    }
};

/**
 * @brief Interface for graph traversal algorithms.
 */
class ITraversalStrategy {
public:
    virtual ~ITraversalStrategy() = default;
    
    /**
     * @brief Executes the traversal algorithm.
     * @param graph The graph to traverse.
     * @param start_node The index of the starting vertex.
     */
    virtual void traverse(const Graph& graph, int start_node) = 0;
    
    /**
     * @brief Returns the name of the algorithm.
     * @return String name.
     */
    virtual std::string get_algorithm_name() const = 0;
};

/**
 * @brief Standard sequential implementation of Breadth-First Search.
 */
class SequentialBFS : public ITraversalStrategy {
public:
    void traverse(const Graph& graph, int start_node) override {
        int n = graph.get_vertex_count();
        if (n == 0) return;

        std::vector<bool> visited(n, false);
        std::queue<int> frontier_queue;

        visited[start_node] = true;
        frontier_queue.push(start_node);

        while (!frontier_queue.empty()) {
            int current_node = frontier_queue.front();
            frontier_queue.pop();

            for (int neighbor : graph.get_neighbors(current_node)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    frontier_queue.push(neighbor);
                }
            }
        }
    }

    std::string get_algorithm_name() const override { return "Sequential BFS"; }
};

/**
 * @brief High-performance parallel BFS using level-synchronized frontiers.
 */
class ParallelBFS : public ITraversalStrategy {
public:
    void traverse(const Graph& graph, int start_node) override {
        int n = graph.get_vertex_count();
        if (n == 0) return;

        std::vector<std::atomic<bool>> visited(n);
        for (int i = 0; i < n; ++i) visited[i] = false;

        std::vector<int> current_frontier;
        visited[start_node] = true;
        current_frontier.push_back(start_node);

        int thread_count = omp_get_max_threads();
        std::vector<std::vector<int>> thread_local_buffers(thread_count);

        while (!current_frontier.empty()) {
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                thread_local_buffers[thread_id].clear();

                #pragma omp for schedule(guided)
                for (int i = 0; i < (int)current_frontier.size(); ++i) {
                    int node = current_frontier[i];
                    for (int neighbor : graph.get_neighbors(node)) {
                        bool expected_visited = false;
                        // Use CAS for lock-free thread-safe discovery
                        if (visited[neighbor].compare_exchange_strong(expected_visited, true)) {
                            thread_local_buffers[thread_id].push_back(neighbor);
                        }
                    }
                }
            }

            // Consolidate level discovery results
            std::vector<int> next_frontier;
            for (int i = 0; i < thread_count; ++i) {
                next_frontier.insert(
                    next_frontier.end(), 
                    thread_local_buffers[i].begin(), 
                    thread_local_buffers[i].end()
                );
            }
            current_frontier.swap(next_frontier);
        }
    }

    std::string get_algorithm_name() const override { return "Parallel BFS"; }
};

/**
 * @brief Standard sequential implementation of Depth-First Search.
 */
class SequentialDFS : public ITraversalStrategy {
public:
    void traverse(const Graph& graph, int start_node) override {
        int n = graph.get_vertex_count();
        if (n == 0) return;

        std::vector<bool> visited(n, false);
        std::stack<int> search_stack;

        search_stack.push(start_node);
        while (!search_stack.empty()) {
            int current_node = search_stack.top();
            search_stack.pop();

            if (!visited[current_node]) {
                visited[current_node] = true;
                for (int neighbor : graph.get_neighbors(current_node)) {
                    if (!visited[neighbor]) {
                        search_stack.push(neighbor);
                    }
                }
            }
        }
    }

    std::string get_algorithm_name() const override { return "Sequential DFS"; }
};

/**
 * @brief Parallel DFS implementation using OpenMP tasks with neighbor-count cutoff.
 */
class ParallelDFS : public ITraversalStrategy {
public:
    void traverse(const Graph& graph, int start_node) override {
        int n = graph.get_vertex_count();
        if (n == 0) return;

        std::vector<std::atomic<bool>> visited(n);
        for (int i = 0; i < n; ++i) visited[i] = false;

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                perform_task_dfs(graph, start_node, visited);
            }
        }
    }

    std::string get_algorithm_name() const override { return "Parallel DFS"; }

private:
    void perform_task_dfs(const Graph& graph, int u, std::vector<std::atomic<bool>>& visited) {
        bool expected_visited = false;
        if (!visited[u].compare_exchange_strong(expected_visited, true)) {
            return;
        }

        const auto& neighbors = graph.get_neighbors(u);
        
        // Parallel discovery threshold
        if (neighbors.size() > 4) {
            for (int v : neighbors) {
                if (!visited[v]) {
                    #pragma omp task shared(graph, visited) firstprivate(v)
                    perform_task_dfs(graph, v, visited);
                }
            }
            #pragma omp taskwait
        } else {
            // Sequential fallback to avoid task overhead on low-degree nodes
            for (int v : neighbors) {
                if (!visited[v]) {
                    perform_task_dfs(graph, v, visited);
                }
            }
        }
    }
};

/**
 * @brief Orchestrates performance benchmarks for graph traversals.
 */
class PerformanceAnalyzer {
public:
    /**
     * @brief Initializes analyzer and creates results file with headers.
     */
    PerformanceAnalyzer() {
        results_file_.open("graph_result.txt");
        results_file_ << "N,Density,BFS_Seq,BFS_Par,BFS_Speedup,BFS_Efficiency,"
                      << "DFS_Seq,DFS_Par,DFS_Speedup,DFS_Efficiency\n";
    }

    /**
     * @brief Ensures the results file is properly closed.
     */
    ~PerformanceAnalyzer() {
        if (results_file_.is_open()) results_file_.close();
    }

    /**
     * @brief Runs the full benchmark suite across all sizes and densities.
     */
    void run_benchmarks() {
        int available_cores = omp_get_max_threads();
        std::cout << "Starting Benchmarks using " << available_cores << " threads.\n\n";

        const std::vector<int> test_sizes = {100, 1000, 5000, 10000, 15000};
        const std::vector<double> test_densities = {0.01, 0.1, 0.5};

        SequentialBFS seq_bfs;
        ParallelBFS   par_bfs;
        SequentialDFS seq_dfs;
        ParallelDFS   par_dfs;

        for (double density : test_densities) {
            std::cout << "--- Testing Edge Density: " << density << " ---\n";
            for (int n : test_sizes) {
                auto graph = GraphGenerator::generate_random_graph(n, density);

                double bfs_seq_time = measure_execution_time(seq_bfs, *graph, 0);
                double bfs_par_time = measure_execution_time(par_bfs, *graph, 0);
                double dfs_seq_time = measure_execution_time(seq_dfs, *graph, 0);
                double dfs_par_time = measure_execution_time(par_dfs, *graph, 0);

                double bfs_speedup = bfs_seq_time / bfs_par_time;
                double dfs_speedup = dfs_seq_time / dfs_par_time;

                std::cout << "  N=" << std::setw(5) << n 
                          << " | BFS Speedup: " << std::fixed << std::setprecision(2) << bfs_speedup << "x"
                          << " | DFS Speedup: " << dfs_speedup << "x\n";

                export_to_csv(n, density, bfs_seq_time, bfs_par_time, bfs_speedup, 
                              dfs_seq_time, dfs_par_time, dfs_speedup, available_cores);
            }
            std::cout << "\n";
        }

        std::cout << "Benchmarks completed. Results saved to graph_result.txt\n";
    }

private:
    std::ofstream results_file_;

    /**
     * @brief Measures timing of a single strategy execution.
     */
    double measure_execution_time(ITraversalStrategy& strategy, const Graph& graph, int start_node) {
        double start_time = omp_get_wtime();
        strategy.traverse(graph, start_node);
        return omp_get_wtime() - start_time;
    }

    /**
     * @brief Appends raw metrics to the CSV file.
     */
    void export_to_csv(int n, double density, double b_seq, double b_par, double b_sp, 
                       double d_seq, double d_par, double d_sp, int cores) {
        results_file_ << std::fixed << std::setprecision(8)
                      << n << "," << density << ","
                      << b_seq << "," << b_par << "," << b_sp << "," << (b_sp / cores) << ","
                      << d_seq << "," << d_par << "," << d_sp << "," << (d_sp / cores) << "\n";
    }
};

/**
 * @brief Entry point for the benchmarking application.
 */
int main() {
    PerformanceAnalyzer analyzer;
    analyzer.run_benchmarks();
    return 0;
}
