# Warp-Level Primitives for Fast Reduction Operations

This project studies parallel sum reduction on NVIDIA GPUs. It compares a
traditional shared-memory reduction with a warp-shuffle reduction implemented
in CUDA C++. Both approaches calculate the same integer sum and are validated
and benchmarked with the same input sizes and launch configuration.

## What we implemented

- Configured and inspected CUDA environments on:
  - NVIDIA Tesla T4, compute capability 7.5 (`sm_75`)
  - NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9 (`sm_89`)
- Wrote custom CUDA kernels in
  [`src/scalable_reduction.cu`](src/scalable_reduction.cu).
- Implemented `reduceShared`, a block reduction using shared memory and
  `__syncthreads()`.
- Implemented `reduceShuffle`, a block reduction using
  `__shfl_down_sync()` for communication inside 32-thread warps.
- Implemented `finishShared`, a second-stage kernel that combines one partial
  sum from each block.
- Used a grid-stride loop so each thread can process multiple input elements.
- Used 256 threads per block and up to 1024 first-stage blocks.
- Added CUDA error checking after runtime calls and kernel launches.
- Tested input sizes of 32, 256, 1,000, 65,536, and 1,000,000 elements.
- Verified that both methods produce the expected result for every tested size.
- Added warm-up iterations and CUDA-event timing to reduce measurement noise.
- Repeated the benchmark five times and stored the results in
  [`results/`](results/).
- Compared the Tesla T4 and RTX 4050 results in
  [`RTX4050_vs_T4_COMPARISON.md`](RTX4050_vs_T4_COMPARISON.md).
- Documented the CUDA concepts and project status in
  [`../concepts.md`](../concepts.md).

## How the reduction works

The first kernel computes one partial sum per block:

1. Each thread reads input values through a grid-stride loop.
2. Threads combine their local sums inside the block.
3. Thread 0 writes the block result to the `partial` array.

The second-stage `finishShared` kernel reduces the partial array to one final
integer. This two-stage design supports inputs much larger than one block.

The shared-memory version exchanges values through shared memory at every tree
reduction step. The warp-shuffle version exchanges values directly between
lanes in registers within a warp, then uses a small shared-memory array for the
warp totals. This reduces shared-memory traffic and block-wide synchronization.

## Libraries and tools

The implementation uses:

- CUDA C++ and the NVIDIA CUDA Runtime API
- `cuda_runtime.h`
- `nvcc`
- CUDA runtime functions such as `cudaMalloc`, `cudaMemcpy`, `cudaFree`,
  `cudaDeviceSynchronize`, and CUDA events
- The CUDA intrinsic `__shfl_down_sync`

No CUB, Thrust, cuBLAS, cuDNN, PyTorch, or other third-party reduction library
is used. The reduction kernels are intentionally implemented manually so that
the two communication strategies can be studied directly.

## Results

All recorded correctness tests passed. On the Tesla T4 benchmark, warp shuffle
was faster than shared memory for every tested input size. The measured
latency reduction ranged from approximately 9% for small inputs to
approximately 29% for the largest tested input. The RTX 4050 comparison also
showed lower latency for the warp-shuffle method.

The measurements are implementation experiments, not a perfectly controlled
architectural comparison: the GPUs, drivers, CUDA versions, power limits, and
runtime environments differ.

## Educational introductory kernel

The CUDA basics notebook also contains a small `addTen` example. It is an
introductory kernel used to demonstrate indexing and a basic kernel launch; it
is separate from the scalable reduction implementation.
