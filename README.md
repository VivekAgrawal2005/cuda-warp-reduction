# Warp-Level Primitives for Fast Reduction Operations

A CUDA C++ project that implements, validates, and benchmarks **shared-memory** and
**warp-shuffle** reduction algorithms for **SUM**, **MAX**, and **ARGMAX**,
with an interactive Python/Streamlit dashboard for exploration.

---

## Motivation

GPU reductions are a fundamental primitive. Two common approaches differ in how
threads communicate partial results inside a thread block:

- **Shared-memory (baseline):** Tree reduction using `__shared__` memory and
  `__syncthreads()` barriers.
- **Warp-shuffle (optimised):** Intra-warp reduction with `__shfl_down_sync`,
  avoiding shared-memory writes for within-warp communication.

This project measures whether and by how much the shuffle approach outperforms
the shared-memory approach across different operations and input sizes.

---

## Concepts

| Term | Meaning |
|---|---|
| **Warp** | The atomic scheduling unit on an NVIDIA GPU: 32 threads that execute in lockstep (SIMT). |
| **Lane** | A thread's position within its warp (0–31). |
| **Shared memory** | Fast on-chip SRAM shared by all threads in a thread block. |
| **`__shfl_down_sync`** | Warp intrinsic: lane `i` receives the value from lane `i + offset`, staying within the warp. No shared memory needed. |
| **Tree reduction** | Halve the active thread count each step; each active thread combines its value with one neighbour. `O(log N)` steps. |
| **Grid-stride loop** | Each thread processes multiple elements (striding by `gridDim.x × blockDim.x`) so a fixed grid can handle any input size. |

---

## Algorithm Explanations

### Shared-memory reduction (1 block)
```
Stage 1: each thread accumulates its stripe into s[tid]
Stage 2: for stride = blockDim/2 down to 1:
           if tid < stride: s[tid] op= s[tid+stride]
           __syncthreads()
Stage 3: thread 0 writes s[0] to output
```

### Warp-shuffle reduction (1 block, 256 threads = 8 warps)
```
Stage 1: each thread accumulates its stripe into `acc`
Stage 2: for offset = 16, 8, 4, 2, 1:
           acc op= __shfl_down_sync(0xFFFFFFFF, acc, offset)
           → lane 0 of each warp holds the warp result
Stage 3: lane 0 writes warpResult[warp]
Stage 4: __syncthreads()
Stage 5: first warp loads warpResults and repeats Stage 2
Stage 6: lane 0 of warp 0 writes block result
```

### How SUM, MAX, and ARGMAX differ

| Operation | Combination rule | Identity | Extra data |
|---|---|---|---|
| SUM | `a + b` | `0` | – |
| MAX | `max(a, b)` | `-FLT_MAX` | – |
| ARGMAX | `best(a, b)` — higher value, lower index on tie | `{-FLT_MAX, INT_MAX}` | Carries `(value, index)` pair |

ARGMAX shuffles both `val` and `idx` simultaneously so the pair is never inconsistent.

---

## Project Structure

```
warp_reduction/
├── README.md
├── plan.md
├── requirements.txt
├── src/
│   ├── scalable_reduction.cu   # Original SUM benchmark (preserved)
│   └── reduction_bench.cu      # New: SUM + MAX + ARGMAX, JSON CLI
├── app/
│   ├── app.py                  # Streamlit dashboard
│   ├── cuda_runner.py          # Subprocess wrapper + CSV persistence
│   └── README.md
├── scripts/
│   ├── analyze_benchmarks.py   # Generate comparison charts from CSV
│   └── correctness_test.py     # Full test matrix PASS/FAIL report
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_cuda_basics.ipynb
│   ├── 03_shared_memory_reduction.ipynb
│   ├── 04_warp_shuffle_reduction.ipynb
│   ├── 05_correctness_and_benchmark.ipynb
│   └── 06_scalable_reduction.ipynb
└── results/
    ├── benchmark_raw.csv           # Legacy SUM-only benchmark data
    ├── benchmark_summary.csv       # Legacy SUM summary
    ├── benchmark_raw_v2.csv        # Full schema: SUM + MAX + ARGMAX
    ├── correctness_report.json     # Output of correctness_test.py
    ├── environment.txt             # GPU / compiler info
    └── figures/                    # Generated charts
```

---

## Requirements & Setup

### GPU environment (Google Colab / Linux)
```bash
nvidia-smi          # verify GPU
nvcc --version      # verify CUDA compiler
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Compile the CUDA benchmark
```bash
# Tesla T4 (Colab default) → sm_75
nvcc -O2 -arch=sm_75 src/reduction_bench.cu -o src/reduction_bench

# RTX 30xx → sm_86   |   A100 → sm_80   |   H100 → sm_90
```

### Run the Streamlit dashboard
```bash
streamlit run app/app.py
```

---

## Command-Line Interface

```
./src/reduction_bench
  --operation    sum | max | argmax
  --algorithm    shared | shuffle | both
  --size         <N>                        (default: 1000000)
  --seed         <S>                        (default: 42)
  --repetitions  <R>                        (default: 5)
  --warmup       <W>                        (default: 50)
  --pattern      random_signed | positive | all_equal | dup_max | max_at_end
  --output-format json
```

Output: machine-readable JSON to **stdout**.  
Errors: to **stderr** with a non-zero exit code.

---

## Test Matrix & Correctness

Run the full correctness suite:
```bash
python scripts/correctness_test.py --verbose
```

Tested sizes: 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1 000, 65 536, 1 000 000  
Patterns: random signed, positive, all equal, duplicate max, max at end  
Seeds: 42, 7, 123  
Algorithms: both shared and shuffle  
Report: `results/correctness_report.json`

ARGMAX uses **zero-based indexing** and **first-occurrence tie-breaking** (lowest index wins on equal values).

---

## Benchmark Methodology

- CUDA events wrap both kernel stages (first-stage + final reduction) for both algorithms.
- Both algorithms are warmed up on the same input before any timing.
- Per-repetition latencies are recorded; **median** is reported.
- **Speedup** = median_shared_us / median_shuffle_us
- **Latency reduction (%)** = (shared − shuffle) / shared × 100
- A speedup > 1× means shuffle was faster for that case; < 1× means shared was faster.

---

## Generate Charts

```bash
python scripts/analyze_benchmarks.py
# Saves charts to results/figures/
```

Charts produced:
- Grouped bar: shared vs shuffle latency per operation × size
- Speedup vs input size with 1× baseline
- Latency reduction % per size
- Log-log latency vs size (all operations)

---

## Measured Results (Tesla T4, CUDA 12.8 — SUM only, 500 iterations)

| Input size | Shared (µs) | Shuffle (µs) | Speedup |
|---|---|---|---|
| 32 | 8.92 | 8.02 | 1.11× |
| 256 | 8.92 | 8.04 | 1.11× |
| 1 000 | 8.97 | 8.12 | 1.10× |
| 65 536 | 13.63 | 9.62 | 1.42× |
| 1 000 000 | 29.59 | 20.98 | 1.41× |

*All 5 runs × 5 sizes = 25 correctness checks: all PASS.*  
MAX and ARGMAX results will appear after running on a CUDA-capable machine.

---

## Limitations

- Benchmarks require an NVIDIA GPU, CUDA driver, and the compiled executable.
- Small-input timings are dominated by kernel-launch overhead (~8 µs regardless of N).
- Results are hardware-specific; the T4 data above may not generalise to other GPUs.
- Shuffle is not guaranteed to be faster for all sizes; for small N the difference is within noise.
- Integer overflow is not possible with float inputs at the tested sizes, but the data type restricts precision for very large N.

---

## Demo Checklist

1. ✅ SUM shared-memory and shuffle – implemented and preserved from original work.
2. ✅ MAX shared-memory and shuffle – implemented.
3. ✅ ARGMAX shared-memory and shuffle – implemented with correct (value, index) pairs.
4. ✅ JSON CLI for Python ↔ CUDA integration.
5. ✅ Streamlit dashboard with Live Run, History, Charts, and About tabs.
6. ✅ Correctness test script with comprehensive edge-case coverage.
7. ✅ Chart generation script.
8. ✅ requirements.txt and setup instructions.
