# CUDA Concepts and Project Study Guide

## Project: Warp-Level Primitives for Fast Reduction Operations

**Purpose:** Beginner-friendly notes for understanding CUDA, the
reduction algorithms implemented so far, and preparing for the first
project review.

**Current environment:** Google Colab with an NVIDIA Tesla T4 GPU (shown
in the screenshot).

**Current status:** The mid-sem implementation covers scalable sum
reduction using both a shared-memory tree baseline and a warp-shuffle
version. Both methods pass correctness tests for five input sizes, and
repeated CUDA-event benchmarks have been saved in `warp_reduction/results`.
On the Tesla T4 run, the warp-shuffle version is faster than the
shared-memory version for every tested size.

## Mid-sem completion summary

This is what has been completed from the original project scope so far:

| Project requirement | Mid-sem status |
| --- | --- |
| CUDA environment setup | Completed on Google Colab with NVIDIA Tesla T4 and CUDA toolkit 12.8. Environment details are saved in `warp_reduction/results/environment.txt`. |
| Repository structure | Partially completed. Current folders include `src/`, `notebooks/`, and `results/`. |
| Shared-memory tree reduction baseline | Completed for INT32 sum in `warp_reduction/src/scalable_reduction.cu`. |
| Warp-shuffle reduction using `__shfl_down_sync` | Completed for INT32 sum in `reduceShuffle`. |
| Grid-stride loop for scalable input sizes | Completed in both shared-memory and shuffle kernels. |
| Block-level two-stage reduction | Completed. First kernel writes one partial result per block; `finishShared` combines block partials. |
| Correctness tests | Completed for input sizes 32, 256, 1,000, 65,536, and 1,000,000. All recorded tests pass. |
| Benchmark harness | Completed using CUDA events, 50 warm-up iterations, and 500 timed iterations per method. |
| Repeated benchmark runs | Completed. Five runs across five input sizes were parsed into raw and summary CSV files. |
| Result documentation | Completed for the current sum-reduction milestone in this study guide and the GPU comparison report. |
| Max reduction | Not completed yet. |
| Argmax with deterministic tie-breaking | Not completed yet. |
| CUB/Thrust baseline | Not completed yet. |
| PyTorch baseline | Not completed yet. |
| `__shfl_xor_sync` butterfly variant | Not completed yet. |
| Vectorized loads and unrolling experiments | Not completed yet. |
| Atomic vs two-kernel finalization comparison | Not completed yet. Current implementation uses the two-kernel approach. |
| FP32/FP16 experiments and numerical-error study | Not completed yet. Current implementation uses INT32, so floating-point error has not been evaluated. |
| Nsight Compute analysis | Not completed yet. |

### Mid-sem benchmark result

The current benchmark compares the shared-memory sum baseline with the
warp-shuffle sum implementation on a Tesla T4. Median timings from five
runs are:

| Input size | Shared memory (us) | Warp shuffle (us) | Speedup from median times | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 8.9210 | 8.0159 | 1.11x | 10.15% |
| 256 | 8.9211 | 8.0404 | 1.11x | 9.87% |
| 1,000 | 8.9661 | 8.1169 | 1.10x | 9.47% |
| 65,536 | 13.6274 | 9.6191 | 1.42x | 29.41% |
| 1,000,000 | 29.5940 | 20.9838 | 1.41x | 29.09% |

So, by mid-sem, the project has already demonstrated the expected main
idea for sum reduction: replacing most within-block shared-memory
exchange with warp shuffles reduces latency. The strongest current
evidence is for medium and large tested arrays, where the measured
latency reduction is about 29% on the T4 run.

------------------------------------------------------------------------

# 1. CUDA: the big picture

CUDA is NVIDIA's programming platform for using an NVIDIA GPU to perform
parallel computation. A CPU is designed for a wide range of tasks and
often excels at low-latency sequential work. A GPU is designed to
execute many similar operations concurrently.

For example, adding 10 to every element of a very large array can be
divided among many GPU threads. This does not mean a GPU is faster for
every task: small jobs can be dominated by launch overhead and data
transfers.

## Host and device

-   **Host:** CPU and its memory. Ordinary C++ code such as `main()`
    runs here.
-   **Device:** GPU and its memory. CUDA kernels execute here.

A common CUDA workflow is:

1.  Prepare data on the CPU.
2.  Allocate memory on the GPU.
3.  Copy input from host to device.
4.  Launch a kernel.
5.  Wait for the GPU when the result is needed.
6.  Copy results back to the CPU.
7.  Free device memory.

## What is a kernel?

A kernel is a function executed by many GPU threads. In our project,
`reduceShared` and `reduceShuffle` are kernels.

``` cpp
__global__ void addTen(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += 10;
    }
}
```

`__global__` marks a kernel callable from host code and executed on the
device. A launch such as:

``` cpp
addTen<<<blocks, threads>>>(a, n);
```

specifies the grid size and threads per block. Kernel launches are
generally asynchronous with respect to the CPU, so synchronization or a
blocking operation is needed when the host must know the GPU work is
complete.

------------------------------------------------------------------------

# 2. CUDA's thread hierarchy

CUDA organizes work as **Grid → Blocks → Threads**.

-   A **thread** executes one instance of the kernel.
-   A **block** is a group of threads that can cooperate through shared
    memory and block synchronization.
-   A **grid** is all blocks launched for one kernel call.

For a 1D launch:

  Built-in        Meaning
  --------------- ---------------------------------
  `threadIdx.x`   Thread's index inside its block
  `blockIdx.x`    Block's index inside the grid
  `blockDim.x`    Threads in each block
  `gridDim.x`     Blocks in the grid

## Global thread index

``` cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

With 256 threads per block: - Block 0 covers global indices 0--255. -
Block 1 covers 256--511. - Block 2 covers 512--767.

The formula turns a local thread index into an index across the whole
grid.

## Why bounds checks matter

If `n=1000` and 4 blocks of 256 threads are launched, there are 1024
thread positions. The final 24 are outside the input array. A condition
such as `if (i < n)` prevents invalid memory access. In a sum reduction,
a thread with no value contributes the neutral value zero.

## Grid-stride loops

Our scalable kernels use:

``` cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
int step = gridDim.x * blockDim.x;

int sum = 0;
for (; i < n; i += step) {
    sum += input[i];
}
```

The stride is the total number of launched threads. If the grid has 4
blocks and each has 256 threads, `step=1024`. A thread processes indices
such as `i`, `i+1024`, `i+2048`, and so on. This lets a fixed-size grid
handle arrays larger than its initial number of thread positions.

------------------------------------------------------------------------

# 3. Memory spaces used in this project

## Host memory

``` cpp
std::vector<int> h_input(n);
```

This is ordinary CPU-side memory. The `h_` prefix is a naming convention
meaning host.

## Device global memory

``` cpp
int* d_input = nullptr;
cudaMalloc(&d_input, n * sizeof(int));
```

This allocates memory on the GPU. A device pointer should be passed to
CUDA APIs or kernels; the CPU cannot use it like a normal host pointer.

Copy input to the GPU:

``` cpp
cudaMemcpy(d_input, h_input.data(),
           n * sizeof(int), cudaMemcpyHostToDevice);
```

Copy a result back:

``` cpp
cudaMemcpy(&result, d_output, sizeof(int),
           cudaMemcpyDeviceToHost);
```

## Shared memory

Shared memory is a fast, block-local memory space. Threads in one block
can use it to exchange intermediate values.

Our baseline declares:

``` cpp
__shared__ int s[THREADS];
```

Each thread writes its local sum to this array, and the block
cooperatively reduces it. Ordinary shared memory is not shared across
different blocks.

## Registers

Local variables such as `int sum = 0;` are typically held in registers
or other thread-local storage. Each thread has its own `sum`; another
thread cannot directly read that variable.

------------------------------------------------------------------------

# 4. Synchronization

Parallel threads do not necessarily reach each instruction at the same
time. Synchronization is needed when threads exchange intermediate data.

## `__syncthreads()`

`__syncthreads()` is a barrier for threads in the same block. Threads
wait until all participating threads in that block reach the barrier.

In our shared reduction:

``` cpp
s[tid] = sum;
__syncthreads();
```

the barrier makes sure the shared-memory writes are ready before the
reduction reads them. We also synchronize between reduction stages
because one stage updates values the next stage will read.

Key points: - It synchronizes a block, not the whole grid. - It must be
used in control flow where the participating threads reach it
consistently. - It does not replace synchronization between separate
kernel stages.

## Why use two kernels?

Our first-stage reduction launches multiple blocks. A normal
`__syncthreads()` cannot make all those blocks wait for each other. We
therefore use one kernel to produce per-block partial sums and a second
kernel to combine those partials. Work in the same stream is ordered, so
the final kernel runs after the first-stage work.

------------------------------------------------------------------------

# 5. Reduction fundamentals

A reduction combines many values into one using an operation such as
sum, maximum, or minimum.

Example:

``` text
Input: [10, 20, 30, 40]
Sum:   100
```

A sequential sum adds one value at a time. A parallel reduction uses a
tree:

``` text
a b c d e f g h
 \ /   \ /   \ /
 a+b   c+d   e+f   g+h
   \   /       \   /
  (a+b)+(c+d)  (e+f)+(g+h)
          \       /
             total
```

A balanced tree combines groups in stages. For a power-of-two number of
values, the number of stages is logarithmic. The total amount of
addition work remains proportional to the number of elements, but many
additions can happen concurrently.

Integer addition is associative provided overflow is not involved.
Floating-point reductions can differ slightly from sequential sums
because floating-point addition is not perfectly associative.

------------------------------------------------------------------------

# 6. Our shared-memory baseline

`reduceShared` produces one partial sum per block.

Its structure:

1.  Each thread accumulates one or more input elements in a local `sum`.
2.  Each thread stores that sum in shared memory.
3.  The block repeatedly combines pairs of shared values.
4.  Thread 0 writes the block total to `partial[blockIdx.x]`.

Core code:

``` cpp
s[tid] = sum;
__syncthreads();

for (int stride = blockDim.x / 2;
     stride > 0; stride /= 2) {
    if (tid < stride) {
        s[tid] += s[tid + stride];
    }
    __syncthreads();
}

if (tid == 0) {
    partial[blockIdx.x] = s[0];
}
```

For a 256-thread block, the stride sequence is:

``` text
128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```

At each stage, the lower-index active threads add values from partners
in the upper half. The final sum ends up in shared-memory position zero.

This is our baseline for comparing the warp-shuffle method. A fair
comparison should use the same input and comparable launch dimensions
and should time equivalent stages.

------------------------------------------------------------------------

# 7. Warps and lanes

A warp is a group of 32 threads used as a scheduling unit on the NVIDIA
architecture targeted by this project. A 256-thread block contains:

``` text
256 / 32 = 8 warps
```

Each thread has a lane ID from 0 to 31 within its warp.

Our code computes:

``` cpp
int lane = tid % 32;
int warp = tid / 32;
```

Examples:

    Thread `tid`   Warp   Lane
  -------------- ------ ------
               0      0      0
               1      0      1
              31      0     31
              32      1      0
              33      1      1
             255      7     31

A lane is the thread's position within its warp. A warp is not the same
thing as a block: one block can contain several warps.

------------------------------------------------------------------------

# 8. Warp shuffle: `__shfl_down_sync`

Our warp reduction uses:

``` cpp
__shfl_down_sync(mask, value, offset)
```

This lets a lane read a value from another lane in the same warp at a
specified offset toward a higher lane ID. In the reduction pattern, the
offsets are:

``` text
16, 8, 4, 2, 1
```

Code:

``` cpp
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
}
```

The stages combine values in a tree. After the sequence, lane 0 holds
the total for that warp.

## The mask

``` cpp
unsigned int mask = 0xffffffff;
```

This is a 32-bit mask with every lane bit set. In our current kernel,
all 32 lanes of every warp execute the shuffle calls. Threads without a
valid input contribute zero, but they still participate.

A full mask is not correct for every possible kernel. If some lanes are
inactive or take a different branch, the participation mask and control
flow must be handled deliberately.

## Why shuffle can reduce overhead

The shared baseline uses shared-memory exchanges and block barriers at
each reduction stage. Warp shuffle lets lanes exchange values within a
warp directly, avoiding those shared-memory exchanges and block-wide
barriers for the within-warp reduction.

Our shuffle kernel still uses shared memory to combine the warp totals.
Its design is: - Shuffle within each warp. - Lane 0 of each warp stores
its warp total. - Synchronize the block. - The first warp loads and
reduces the warp totals using shuffle. - Lane 0 writes the block total.

So this is not a "no shared memory" algorithm. It uses shuffle for
within-warp communication and shared memory for between-warp
communication.

------------------------------------------------------------------------

# 9. `reduceShuffle` step by step

## Step 1: Reserve one shared slot per warp

``` cpp
__shared__ int warpSums[THREADS / 32];
```

With `THREADS=256`, there are 8 slots, one for each warp.

## Step 2: Calculate IDs

``` cpp
int tid = threadIdx.x;
int lane = tid % 32;
int warp = tid / 32;
```

## Step 3: Accumulate inputs

The grid-stride loop accumulates the elements assigned to the thread in
its local `sum`.

## Step 4: Reduce inside each warp

The shuffle loop uses offsets 16, 8, 4, 2, 1. Lane 0 of each warp ends
with that warp's total.

## Step 5: Store warp totals

``` cpp
if (lane == 0) {
    warpSums[warp] = sum;
}
__syncthreads();
```

One thread per warp writes its result. The barrier ensures the first
warp reads only after the warp totals have been stored.

## Step 6: Reduce the eight warp totals

Only the first warp does this part:

``` cpp
if (warp == 0) {
    int total = (lane < THREADS / 32)
        ? warpSums[lane] : 0;
    // Shuffle reduction follows.
}
```

Lanes 0--7 load the eight warp sums. Lanes 8--31 use zero. The first
warp reduces those values, and lane 0 writes the block's partial sum.

------------------------------------------------------------------------

# 10. Scalable reduction: two stages

A single block is not enough for large arrays if we want many blocks to
work in parallel. Our scalable method has two stages.

``` text
Large input array
       |
       v
First-stage kernel: many blocks
       |
       +--> Block 0 writes partial[0]
       +--> Block 1 writes partial[1]
       +--> Block 2 writes partial[2]
       ...
       +--> Block B-1 writes partial[B-1]
                         |
                         v
              Final reduction kernel
                         |
                         v
                   Final sum
```

## First-stage grid size

Our host code calculates:

``` cpp
int blocks = std::min(
    (n + THREADS - 1) / THREADS,
    MAX_BLOCKS
);
```

For positive `n`, `(n + THREADS - 1) / THREADS` is integer ceiling
division. It estimates the number of blocks needed for one element per
thread. `MAX_BLOCKS` caps the grid size. The grid-stride loop handles
any remaining elements when the input is larger than the initial
coverage.

With `THREADS=256` and `MAX_BLOCKS=1024`, the five test sizes use:

          `n`   Blocks
  ----------- --------
           32        1
          256        1
        1,000        4
       65,536      256
    1,000,000    1,024

For one million values, 1,024 blocks provide 262,144 initial thread
positions. Each thread can process additional elements by repeatedly
adding the grid stride to its index.

## Final kernel: `finishShared`

The final kernel reduces the partial array:

``` cpp
for (int i = tid; i < count; i += blockDim.x) {
    sum += partial[i];
}
```

Each thread accumulates one or more partial sums, then the block reduces
those thread-local sums through shared memory. Thread 0 writes the final
answer to `output[0]`.

------------------------------------------------------------------------

# 11. What the host program does

The host-side `main()` coordinates the whole experiment.

## Create test data

``` cpp
std::vector<int> h_input(n);
for (int i = 0; i < n; i++) {
    h_input[i] = (i % 10) + 1;
}
```

This creates a repeating sequence from 1 through 10.

## Calculate a CPU reference

``` cpp
long long expected = std::accumulate(
    h_input.begin(), h_input.end(), 0LL
);
```

This is the reference result used to check the GPU outputs. The `0LL`
makes the CPU accumulator a `long long`.

## Allocate and copy GPU memory

The program allocates: - `d_input`: input array on the GPU. -
`d_partial`: one integer per first-stage block. - `d_output`: one final
integer.

Then it copies `h_input` to `d_input`.

## Run both algorithms

For the shared method: 1. Launch `reduceShared`. 2. Launch
`finishShared`. 3. Synchronize and copy the result to the CPU.

For the shuffle method: 1. Launch `reduceShuffle`. 2. Launch the same
`finishShared`. 3. Synchronize and copy the result to the CPU.

Using the same final kernel helps isolate the comparison to the
first-stage block reduction.

## Check errors and clean up

The project uses a `CHECK` macro to inspect CUDA API return codes.
Kernel launches are checked with `cudaGetLastError()`, and
synchronization helps surface asynchronous execution errors.

At the end of each size, the program destroys timing events and frees
device allocations.

------------------------------------------------------------------------

# 12. What our correctness output proves

The current Colab output is:

``` text
N=32      expected=168     shared=168     shuffle=168     PASS
N=256     expected=1396    shared=1396    shuffle=1396    PASS
N=1000    expected=5500    shared=5500    shuffle=5500    PASS
N=65536   expected=360436  shared=360436  shuffle=360436  PASS
N=1000000 expected=5500000 shared=5500000 shuffle=5500000 PASS
```

For each of these five cases, both GPU outputs match the CPU reference.

This is a good correctness milestone, but it does not prove correctness
for every possible input. Later, add tests for: - Sizes around warp
boundaries: 31, 32, 33. - Sizes around block boundaries: 255, 256,
257. - Other non-multiples of 256, such as 1025. - All zeros, all ones,
random values, and negative values if supported. - Larger values and
overflow limits.

The GPU kernels currently use `int`. Current test sums fit in `int`, but
larger magnitudes or lengths may overflow. If extending the range, use a
wider type consistently for thread sums, shared arrays, partials,
output, and reference comparison.

------------------------------------------------------------------------

# 13. Benchmarking: what we intended to measure

The scalable benchmark is intended to measure the combined GPU time of
the two kernels for each method:

**Shared path:** `reduceShared` + `finishShared`

**Shuffle path:** `reduceShuffle` + `finishShared`

The final stage is the same for both, making this a comparison of the
complete two-stage methods while focusing the difference on the first
stage.

## Warm-up

The code runs both methods several times before timing. This helps
reduce first-use effects, though it cannot remove all variation.

## CUDA events

The code records start and stop events around repeated kernel launches.
`cudaEventElapsedTime` returns elapsed time in milliseconds.

The average in microseconds is calculated as:

``` cpp
averageUs = totalMs * 1000.0f / ITERS;
```

There are 1,000 microseconds in one millisecond.

## Speedup

Once timings are visible:

``` text
speedup = shared_time / shuffle_time
```

-   `1.00x`: equal measured latency.
-   Above `1.00x`: shuffle was faster in that measurement.
-   Below `1.00x`: shared was faster in that measurement.

Percent lower latency for shuffle:

``` text
(shared_time - shuffle_time) / shared_time * 100%
```

For example, shared = 10 μs and shuffle = 8 μs: - Speedup = 10/8 =
1.25x. - Latency reduction = (10−8)/10 × 100% = 20%.

A 1.25× speedup is 20% lower latency, not 25% lower latency.

## Current benchmark status

The latest screenshot shows the five PASS lines but no `Shared avg (us)`
or `Shuffle avg (us)` lines. The `grep` command for `"Shared avg"` is
visible, but no match is shown. Therefore, the scalable timing output
still needs to be confirmed. Do not claim that the scalable shuffle
version is faster until the timing code is present, runs, and produces
repeated measurements.

For a stronger benchmark: 1. Record GPU model, driver, and CUDA toolkit
version. 2. Run multiple independent trials. 3. Consider alternating
which method is timed first. 4. Report median and variation, not only
one average. 5. Keep input sizes, block sizes, and measured stages
identical. 6. Label results by hardware; Colab T4 and a friend's GPU are
separate test environments.

------------------------------------------------------------------------

# 14. CUDA-specific C++ details in the project

## `constexpr`

``` cpp
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;
```

These are compile-time constants. Using one `THREADS` constant helps
keep launch dimensions and shared array sizes consistent.

## `cudaMalloc` and `cudaFree`

`cudaMalloc` reserves device memory; `cudaFree` releases it. Every
successful allocation should be paired with a free when it is no longer
needed.

## `cudaMemcpy`

This API transfers data between host and device. The direction argument
tells CUDA which way the copy goes.

## Error-check macro

The macro evaluates a CUDA API call, checks whether it returned
`cudaSuccess`, and prints the CUDA error string if it failed. This is
important because otherwise a failed allocation or copy could lead to
confusing downstream behavior.

## Standard C++ helpers

-   `std::vector<int>` manages host input storage.
-   `std::accumulate` computes the reference sum.
-   `std::min` caps the number of blocks.

These are ordinary C++ features, not CUDA intrinsics.

------------------------------------------------------------------------

# 15. Common misunderstandings to avoid in the review

1.  **"A warp is a block."** No. A block contains one or more warps. Our
    256-thread block contains eight warps.
2.  **"Shuffle removes shared memory completely."** No. We use shared
    memory to collect and combine warp totals.
3.  **"`__syncthreads()` synchronizes every block."** No. It
    synchronizes threads within one block.
4.  **"PASS means the optimized code is faster."** No. PASS is a
    correctness result. Performance requires timing.
5.  **"One thread handles one element forever."** Not in our scalable
    kernels. The grid-stride loop lets each thread process multiple
    elements.
6.  **"A single timing is definitive."** No. GPU measurements vary; use
    repeated trials and document the setup.
7.  **"The CPU and GPU use the same memory pointer."** Not in this
    program. We allocate separate host and device buffers and copy
    between them.
8.  **"A smaller runtime always means a better general algorithm."** Not
    necessarily. Results depend on size, hardware, compiler, and
    measurement method.

------------------------------------------------------------------------

# 16. First-review questions

Practice answering these aloud without reading the code first.

1.  What is CUDA, and why use a GPU?
2.  What is the host, and what is the device?
3.  What is a kernel?
4.  What do grid, block, and thread mean?
5.  How do `threadIdx.x` and `blockIdx.x` combine into a global index?
6.  Why do we need bounds checks?
7.  What does a grid-stride loop do?
8.  What is global memory? What is shared memory?
9.  What does `__syncthreads()` guarantee?
10. Why can't ordinary `__syncthreads()` combine results across all
    blocks?
11. What is a reduction?
12. How does the shared-memory tree reduction work?
13. What is a warp? What is a lane?
14. What does `__shfl_down_sync` do?
15. Why are the shuffle offsets 16, 8, 4, 2, 1?
16. Why does each warp's lane 0 write a value to `warpSums`?
17. Why does the first warp reduce those warp totals?
18. Why does the scalable version use two kernels?
19. What do the five PASS lines establish?
20. What do we still need before claiming a performance improvement?

Short answer reminders: - Kernel: GPU function executed by many
threads. - Global index: `blockIdx.x * blockDim.x + threadIdx.x`. -
Warp: 32 lanes on the target architecture. - Shuffle: intra-warp value
exchange. - Two-stage reduction: blocks create partials; final kernel
combines them. - PASS: GPU answers match the CPU reference for tested
cases. - Performance claim: requires timing data, repeated trials, and
hardware context.

------------------------------------------------------------------------

# 17. Practical revision exercises

Do these on paper or in a notebook:

1.  For block 2, thread 17, with 256 threads per block, calculate the
    global index.
2.  For `tid=100`, calculate `warp` and `lane`.
3.  With 4 blocks × 256 threads, calculate the grid-stride step.
4.  If a thread starts at index 5 with step 1024, write its first four
    indices.
5.  Write the shared reduction stride sequence for 256 threads.
6.  Write the shuffle offsets used by the warp reduction.
7.  For `n=1000`, calculate how many blocks are launched and how many
    partial sums are produced.
8.  Explain why `finishShared` can reduce more than 256 partials.
9.  Add test sizes 31, 33, 255, 257, and 1025 and compare both GPU
    results with the CPU reference.
10. Explain the data path from host vector to final GPU result and back
    to the host.
11. Explain the difference between correctness testing and benchmarking.
12. Explain why a T4 benchmark should be labeled separately from a
    friend's GPU benchmark.

------------------------------------------------------------------------

# 18. Suggested study order and project next steps

## Study order

1.  Host vs. device and kernel launches.
2.  Grid, block, thread, and indexing.
3.  Global memory, shared memory, and registers.
4.  Synchronization and barriers.
5.  Reduction trees.
6.  Warps, lanes, and shuffle operations.
7.  Grid-stride loops and two-stage reduction.
8.  CUDA events and benchmark interpretation.

For each concept, find the exact line in your `.cu` file that uses it.
Ask four questions: - Which thread executes this line? - Which memory
space does it access? - Does it depend on other threads? - What makes
the operation safe?

## Project next steps

**After the mid-sem milestone:**

-   Study and be able to walk through `reduceShared`, `reduceShuffle`,
    and `finishShared`.
-   Add boundary-size tests such as 31, 33, 255, 257, and 1025.
-   Implement max reduction.
-   Implement argmax with deterministic tie-breaking.
-   Add CUB/Thrust and PyTorch baselines.
-   Add block-size, unroll-factor, and vectorized-load experiments.
-   Compare the current two-kernel finalization with atomics where
    appropriate.
-   Extend the benchmark beyond INT32 sum to FP32/FP16 and report
    numerical behavior honestly.

**When using your friend's GPU PC:** - Check the NVIDIA GPU and driver
with `nvidia-smi`. - Confirm `nvcc` is available. - Compile the `.cu`
source on that PC. - Run correctness tests before timing. - Record that
machine's GPU and software versions separately.

**Later extensions, if in scope:** - Max reduction. - Argmax reduction,
with a clearly defined tie-breaking rule. - More block-size
experiments. - More systematic benchmark reporting.

------------------------------------------------------------------------

# 19. One-minute project explanation

"Our project compares two GPU strategies for reduction. Till mid-sem, we
have completed the INT32 sum part. The baseline uses shared memory and
block synchronization to reduce each block's values. The warp-shuffle
version uses `__shfl_down_sync` to combine values inside each warp, then
shared memory to combine the warp totals. For large arrays, both use a
grid-stride loop and a two-stage design: the first kernel produces one
partial sum per block, and a final kernel combines those partial sums.
We validated both methods against a CPU reference for five input sizes,
from 32 to one million elements, and saved five benchmark runs. On the
Tesla T4 results, the shuffle version is faster for every tested size,
with roughly 1.10x speedup for small inputs and about 1.41x speedup for
one million elements. The next phase is to add max, argmax, CUB/PyTorch
baselines, vectorization/unrolling experiments, and deeper performance
analysis."

------------------------------------------------------------------------

**Core takeaway:** Don't memorize CUDA syntax alone. Understand which
thread runs each line, what it reads and writes, how threads cooperate,
and where synchronization is required.
