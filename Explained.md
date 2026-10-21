# Project Explained

## Project title

**Warp-Level Primitives for Fast Reduction Operations**

The project demonstrates how to sum a large array in parallel on an NVIDIA GPU
and compares two within-block communication strategies:

- Shared-memory tree reduction
- Warp-shuffle reduction using `__shfl_down_sync`

Both methods use a two-stage reduction so that large arrays can be processed by
many blocks.

## 1. Are the two GPU architectures different?

Yes. The tested GPUs are from different NVIDIA architecture generations:

| GPU | Architecture | Compute capability |
|---|---|---:|
| Tesla T4 | Turing | 7.5 (`sm_75`) |
| RTX 4050 Laptop GPU | Ada Lovelace | 8.9 (`sm_89`) |

Ada Lovelace is newer than Turing and supports newer instructions and hardware
features. The GPU name alone does not determine exact performance because
laptop power limits, clock speeds, cooling, memory configuration, drivers, and
the workload also matter.

The source code is portable across these GPUs because it uses standard CUDA
operations supported by both architectures. The project was tested on T4 and
RTX 4050; it was **not** executed on a Blackwell GPU.

## 2. Blackwell architecture, code, and approach

### Important clarification

This project is not designed specifically for Blackwell yet. It is written
using portable CUDA C++ and can potentially be compiled for a Blackwell
compute capability with the matching CUDA Toolkit. That does not make the
current implementation a Blackwell-optimized implementation.

To target a specific GPU, the build should use the appropriate architecture
option, for example:

```text
nvcc -arch=sm_75 scalable_reduction.cu -o scalable_reduction
nvcc -arch=sm_89 scalable_reduction.cu -o scalable_reduction
```

The exact Blackwell `sm_XX` value and minimum toolkit should be taken from the
specific Blackwell GPU and CUDA Toolkit documentation. A Blackwell
optimization would require benchmarking and possibly adding architecture
specific paths rather than assuming that a newer GPU automatically gives the
best result.

### Code approach

1. The host creates an integer input array and allocates device buffers.
2. `reduceShared` or `reduceShuffle` is launched with 256 threads per block.
3. Each thread uses a grid-stride loop to sum multiple input positions.
4. Threads within each block combine their local sums.
5. Thread 0 of each block writes one value to the partial-results array.
6. `finishShared` reduces all block partials to one final output value.
7. The host copies the result back and checks it against the expected sum.

### Shared-memory path

`reduceShared` stores one local sum per thread in shared memory. It repeatedly
adds pairs of values with strides 128, 64, 32, 16, 8, 4, 2, and 1. A
`__syncthreads()` call separates each step so that all participating threads
observe completed writes.

### Warp-shuffle path

`reduceShuffle` divides 256 threads into eight 32-thread warps. It reduces each
warp using register-to-register transfers:

```cpp
sum += __shfl_down_sync(mask, sum, offset);
```

The offsets are 16, 8, 4, 2, and 1. One lane from each warp stores its result
in a small shared-memory array. The first warp then reduces those eight warp
totals, again using shuffle operations.

This approach reduces shared-memory traffic and block-wide synchronization,
which is why it generally performs better than the shared-memory baseline.

### Kernels in the project

The main reduction source defines three CUDA kernel functions:

- `reduceShared`
- `reduceShuffle`
- `finishShared`

Each complete reduction uses two launches: one first-stage kernel and the
`finishShared` second-stage kernel. The introductory notebook also contains a
separate `addTen` teaching example.

## 3. Which kernel library is used?

The project uses **custom CUDA C++ kernels** with the **NVIDIA CUDA Runtime
API**. It includes `cuda_runtime.h` and is compiled with `nvcc`.

The project directly uses:

- CUDA kernel syntax and the `__global__` qualifier
- `cudaMalloc`, `cudaMemcpy`, and `cudaFree`
- `cudaDeviceSynchronize`
- CUDA events for timing
- `__shfl_down_sync` for warp-level communication

It does not call a pre-built reduction routine from CUB, Thrust, cuBLAS,
cuDNN, or PyTorch. The manual implementation is intentional: the project is
teaching and measuring the reduction algorithm itself.

### Recommended library for future work

For the next production-oriented version, we recommend **NVIDIA CUB** as the
first library to evaluate. CUB provides optimized device-wide reduction
primitives through `cub::DeviceReduce::Reduce`, is designed for CUDA/NVIDIA
GPUs, and is the closest library equivalent to the custom reduction in this
project.

The planned direction is:

1. Keep the current custom kernels as the educational and algorithm-analysis
   baseline.
2. Add a separate CUB reduction implementation without removing the custom
   kernels.
3. Run the same correctness tests and CUDA-event benchmarks for both.
4. Use CUB for production unless the custom kernel provides a measured,
   workload-specific advantage.

This recommendation should be kept in mind for future changes: do not replace
the current implementation immediately. First add CUB as a separate baseline
so correctness, latency, memory usage, and maintainability can be compared
fairly.

## 4. Were LKL, oneMKL, SKL, or Windows Kernel-Mode Kernel Library used?

No. None of the following is used:

| Library or technology | Used? | Why it is not part of this project |
|---|---|---|
| Linux Kernel Library (LKL) | No | LKL provides a Linux kernel environment/library for running Linux kernel code; it is not a CUDA reduction library. |
| Intel oneAPI Math Kernel Library (oneMKL) | No | oneMKL targets Intel oneAPI math workloads and is not needed for custom NVIDIA CUDA kernels. |
| SiFive Kernel Library (SKL) | No | It is associated with a different processor/software ecosystem and is not required for NVIDIA GPU CUDA execution. |
| Windows Kernel-Mode Kernel Library | No | Windows kernel-mode libraries are for operating-system drivers and kernel components, not application-level CUDA reduction kernels. |

These technologies solve different problems. CUDA kernels execute on NVIDIA
GPU SMs; operating-system kernel libraries do not replace the CUDA runtime or
the GPU reduction code.

## 5. What could be integrated instead?

The right choice depends on the goal:

- **CUB:** Use `cub::DeviceReduce::Reduce` for a highly optimized NVIDIA
  reduction. This is the closest production alternative to the custom code.
- **Thrust:** Use `thrust::reduce` for a higher-level and easier-to-read
  reduction API. It may use optimized CUDA primitives internally.
- **cuBLAS:** Useful for linear algebra such as dot products and norms, but it
  is not a direct replacement for every integer reduction.
- **CUTLASS:** Useful when reduction is part of a larger matrix or tensor
  computation, especially when targeting specialized GPU instructions.
- **oneMKL or SYCL:** Consider these when portability across Intel, NVIDIA,
  and other accelerator backends is more important than a CUDA-specific
  implementation. This would require a different implementation path and
  benchmarking.
- **PyTorch or another framework:** Useful when the reduction is one operation
  inside a larger machine-learning pipeline rather than a standalone CUDA
  study.

### Library decision for this project

**Primary recommendation: CUB.** It is the preferred future integration for
this NVIDIA-focused reduction because it offers a direct device-level
reduction, while still allowing the project to keep its custom kernels for
comparison.

**Secondary option: Thrust.** Use Thrust when a simpler, higher-level API is
more important than low-level control. It is a good usability baseline, but
CUB is the better first choice for directly comparing device reduction
performance.

**Use oneMKL/SYCL only for portability.** Consider oneMKL or SYCL if the
project's future goal changes from NVIDIA-specific optimization to supporting
multiple accelerator vendors. This would be a separate portability path, not
an addition to the current CUDA implementation.

**Do not use LKL, SiFive SKL, or Windows kernel-mode libraries for this
reduction.** They address operating-system or processor-platform concerns and
are not suitable replacements for CUDA reduction primitives.

## 6. Why would someone use this project?

Someone would use this approach when they need to aggregate a large amount of
data efficiently on an NVIDIA GPU, for example:

- Machine-learning loss and statistics calculations
- Image and video pixel aggregation
- Signal and sensor-data processing
- Scientific and engineering simulations
- Database and analytics aggregation
- Vector norms, sums, and error metrics
- High-performance-computing building blocks

The custom version is especially useful when a developer needs to:

- Understand GPU thread, block, and warp execution
- Control memory access and synchronization
- Add a reduction into a larger custom CUDA kernel
- Measure the effect of warp-level primitives
- Tune a kernel for a known GPU and data type
- Avoid a general-purpose library's constraints

For production code, a library such as CUB may be preferable because it is
already optimized and maintained. The custom implementation is valuable when
algorithm control, integration into a larger kernel, education, or detailed
performance investigation is the priority.

## Short review answer

> We compare a Turing Tesla T4 with an Ada Lovelace RTX 4050, so the GPU
> architectures and compute capabilities are different. The code is portable
> CUDA C++, not Blackwell-specific code. It uses custom kernels and the CUDA
> Runtime API: one first-stage reduction kernel followed by `finishShared`.
> The shared-memory version uses shared memory and synchronization, while the
> warp-shuffle version uses `__shfl_down_sync` to exchange values inside
> 32-thread warps. We did not use LKL, oneMKL, SKL, or Windows kernel-mode
> libraries. CUB or Thrust would be sensible future baselines, while CUDA
> remains the most direct choice for this NVIDIA-focused project.
