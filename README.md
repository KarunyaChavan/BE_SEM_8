# BE Semester 8 - 2019 Pattern: LPV

This repository contains two primary projects used for coursework and experiments:

**DeepLearning** — Jupyter notebooks and small datasets for ML experiments.
- Notebooks:
  - BostonHousePricePrediction.ipynb
  - LetterRecognition.ipynb
  - plant-disease-classification-resnet-99-2.ipynb
- Dataset folder: `datasets/` (contains `boston_housing.csv`, `HousingData.csv`)

**HPC** — High-performance computing examples: CUDA kernels, OpenMP C++ code, and supporting notebooks/docs.
- `docs/` — Explanations for implementations and algorithms.
- `Graph_Traversal/` — `graph.cpp`, `graphs.ipynb` (graph traversal examples).
- `Mandelbrot/` — `MANDELBROT_CUDA.cu` (CUDA Mandelbrot implementation).
- `Reduction/` — `reduction.cu`, `reduction_analysis.ipynb`.
- `Sorting/` — `sort_OpenMp.cpp`, `sort_analysis.ipynb` (OpenMP sorting example).
- `Vector_Matrix/` — `vector_matrix_ops.cu`, `vector_matrix_analysis.ipynb`.

**Quick Build & Run (examples)**

Prerequisites:
- For OpenMP/C++: a GCC toolchain with `g++` (or compatible compiler).
- For CUDA: NVIDIA CUDA Toolkit with `nvcc` on PATH.
- For notebooks: `jupyter lab` or `jupyter notebook`.

OpenMP (C++):
- Compile: `g++ -fopenmp <file>.cpp -o <file-exe>`
- Run: `./<file-exe>`

Example (Sorting):
- `g++ -fopenmp sort_OpenMp.cpp -o sort_OpenMp`
- `./sort_OpenMp`

CUDA (.cu):
- Compile (basic): `nvcc <file>.cu -o <file-exe>`
- Compile (specify arch): `nvcc -arch=sm_60 <file>.cu -o <file-exe>`
- Run: `./<file-exe>`

Examples:
- Mandelbrot: `nvcc -O2 HPC/Mandelbrot/MANDELBROT_CUDA.cu -o HPC/Mandelbrot/mandelbrot`
- Reduction: `nvcc -O2 HPC/Reduction/reduction.cu -o HPC/Reduction/reduction`
- Vector/Matrix ops: `nvcc -O2 HPC/Vector_Matrix/vector_matrix_ops.cu -o HPC/Vector_Matrix/vector_matrix_ops`

Notebooks:
- Launch: `jupyter lab` or `jupyter notebook` from repository root.
- Open the notebooks under `DeepLearning/` or `HPC/` to run analyses interactively.

Notes & Best Practices:
- On Windows, use an environment with `g++` (WSL) for `g++` commands.
- Ensure the CUDA Toolkit version matches your GPU driver; `nvcc --version` verifies installation.
- Use relative paths shown above when building from repository root.