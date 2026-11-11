# CUDA Reduction Results: RTX 4050 vs Tesla T4

## Project

This project compares two CUDA reduction implementations:

- Shared-memory tree reduction
- Warp-shuffle reduction using `__shfl_down_sync`

Both implementations use the same scalable reduction program, 256 threads per block, grid-stride loops, and a final reduction of block partials.

## Hardware and Software

| Item | Tesla T4: Colab | RTX 4050: Local PC |
|---|---|---|
| GPU | NVIDIA Tesla T4 | NVIDIA GeForce RTX 4050 Laptop GPU |
| Compute capability | 7.5 | 8.9 |
| GPU memory | 15,360 MiB | 6,141 MiB reported available |
| CUDA driver | 580.82.07 | 535.309.01 |
| CUDA toolkit / `nvcc` | 12.8 | 12.4 |
| Compilation target | Colab build | `-arch=sm_89` |

## Compute Capability Comparison

| Capability | Tesla T4 | RTX 4050 Laptop GPU | Comparison |
|---|---:|---:|---|
| CUDA compute capability | 7.5 (`sm_75`) | 8.9 (`sm_89`) | RTX 4050 supports the newer Ada Lovelace instruction set and CUDA features |
| Architecture | Turing | Ada Lovelace | RTX 4050 is two NVIDIA architecture generations newer |
| CUDA cores | 2,560 | 2,560 | Same nominal CUDA-core count; architecture and clock speed still affect performance |
| Tensor Core generation | 2nd generation | 4th generation | RTX 4050 provides newer Tensor Core capabilities, including improved AI acceleration |
| Ray-tracing cores | 1st generation RT cores | 3rd generation RT cores | RTX 4050 provides newer hardware ray-tracing support |
| FP32 peak throughput* | 8.1 TFLOPS | Model-dependent; typically higher | RTX 4050 generally has higher FP32 throughput, but laptop power and clock limits vary |

* Peak throughput figures depend on boost clocks and power limits. The RTX 4050 Laptop GPU has model-specific performance characteristics, so its exact peak cannot be inferred from the GPU name alone.

For this CUDA reduction, the most relevant difference is the compute capability: `sm_75` is the Tesla T4 target, while `sm_89` is the RTX 4050 target. The code remains portable because both GPUs support the CUDA operations used by the implementation, including warp shuffle instructions.

## Correctness Results

Every tested input size produced the expected result on both GPUs.

| Input size | Expected result | Tesla T4 | RTX 4050 |
|---:|---:|:---:|:---:|
| 32 | 168 | PASS | PASS |
| 256 | 1,396 | PASS | PASS |
| 1,000 | 5,500 | PASS | PASS |
| 65,536 | 360,436 | PASS | PASS |
| 1,000,000 | 5,500,000 | PASS | PASS |

The RTX 4050 test executed 5 runs with 5 input sizes, for 25 total correctness checks. All passed.

## Average Timing Results

Times are average kernel-pipeline latency per measurement in microseconds. The RTX 4050 values are averaged across five program runs. The Tesla T4 values are the recorded medians from the Colab benchmark's five runs.

| Input size | T4 shared (us) | T4 shuffle (us) | T4 speedup* | RTX 4050 shared (us) | RTX 4050 shuffle (us) | RTX 4050 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8.9210 | 8.0159 | 1.11x | 3.5631 | 3.4185 | 1.04x |
| 256 | 8.9211 | 8.0404 | 1.10x | 3.6386 | 3.4922 | 1.04x |
| 1,000 | 8.9661 | 8.1169 | 1.10x | 3.6912 | 3.3053 | 1.12x |
| 65,536 | 13.6274 | 9.6191 | 1.42x* | 6.0927 | 4.7021 | 1.30x |
| 1,000,000 | 29.5940 | 20.9838 | 1.41x | 16.5437 | 10.1134 | 1.64x |

* The T4 summary reports a median speedup of 1.2931x for 65,536 elements when speedup is calculated per run. The ratio of the displayed median times is 1.4167x. The RTX 4050 speedups above use the ratio of average shared time to average shuffle time.

## Latency Reduction from Warp Shuffle

| Input size | T4 latency reduction | RTX 4050 latency reduction |
|---:|---:|---:|
| 32 | 10.15% | 4.06% |
| 256 | 9.87% | 4.02% |
| 1,000 | 9.47% | 10.45% |
| 65,536 | 29.41% | 22.82% |
| 1,000,000 | 29.09% | 38.87% |

## Main Observations

1. The CUDA code is portable between the Tesla T4 and RTX 4050 without source changes.
2. Both GPUs pass all correctness tests.
3. Warp-shuffle reduction is faster than shared-memory reduction for every tested input size on both GPUs.
4. The RTX 4050 is faster in absolute latency for all tested sizes.
5. The performance advantage of warp shuffle becomes more visible for larger inputs, especially at 1,000,000 elements.
6. Small inputs are dominated by kernel-launch and synchronization overhead, so their speedups are relatively small.

## Important Comparison Note

This is not a strictly controlled hardware comparison because the experiments used different GPUs, CUDA toolkit versions, drivers, and runtime environments:

- Tesla T4 benchmark: Google Colab, CUDA 12.8
- RTX 4050 benchmark: local PC, CUDA 12.4

Therefore, the results demonstrate that the implementation works and show the behavior on each GPU, but they should not be interpreted as a precise architectural speedup study.

## Reproduce the RTX 4050 Results

From the project root:

```bash
cd ~/Downloads/warp_reduction-20260928T114225Z-1-001/warp_reduction
./run_benchmark.sh
```

Generated files:

- `results/benchmark_5_runs.txt`: raw output from all five runs
- `results/benchmark_average.csv`: averaged RTX 4050 results
