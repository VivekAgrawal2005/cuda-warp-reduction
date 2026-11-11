# Warp-Level Primitives for Fast Reduction Operations

## Five-Day Implementation and Integration Plan

**Project type:** CUDA C++ GPU optimization project with a Python
application layer\
**Primary goal:** Implement, validate, and compare shared-memory and
warp-shuffle implementations of SUM, MAX, and ARGMAX, then expose the
results through a small interactive UI with charts.

------------------------------------------------------------------------

## 1. Project status and starting point

### Already implemented

-   CUDA environment has been tested in Google Colab with an NVIDIA GPU.
-   A shared-memory SUM reduction and a warp-shuffle SUM reduction have
    been implemented.
-   A scalable, multi-block SUM reduction has been implemented.
-   Both SUM approaches have passed correctness checks for input sizes
    **32, 256, 1,000, 65,536, and 1,000,000** using the current
    test-data pattern.
-   Preliminary latency benchmarks have been run.
-   Repeated benchmarking and median-based comparison have been
    completed or are being finalized. Preserve the actual raw results
    and summary produced by the existing benchmark workflow; do not
    fabricate or replace measurements.

### Still to do

1.  Finish and validate repeatable SUM benchmark results, if the raw
    CSVs and summaries are not yet saved.
2.  Implement MAX using both approaches.
3.  Implement ARGMAX using both approaches.
4.  Validate all operations against a trusted CPU reference and test
    edge cases.
5.  Build a Python UI that lets a user select the operation, input size,
    and algorithm(s), run the CUDA executable, and view numerical
    results and latency comparisons.
6.  Produce clean result files, charts, documentation, and a demo-ready
    project.

### Existing project structure

Keep the existing folder and naming conventions where possible. The
project currently uses a structure similar to:

``` text
warp_reduction/
├── README.md
├── src/
│   ├── baseline_shared.cu
│   ├── warp_shuffle.cu
│   ├── main.cu
│   └── scalable_reduction.cu
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_cuda_basics.ipynb
│   ├── 03_shared_memory_reduction.ipynb
│   ├── 04_warp_shuffle_reduction.ipynb
│   ├── 05_correctness_and_benchmark.ipynb
│   └── 06_scalable_reduction.ipynb
├── results/
│   ├── benchmark_raw.csv
│   ├── benchmark_summary.csv
│   └── environment.txt
├── app/
│   └── app.py
└── plan.md
```

The exact current files may differ. Inspect the existing repository
before editing. **Do not overwrite working SUM code or existing result
files without checking them first.** Store the Python UI in `app/` and
keep CUDA implementation in `src/`.

------------------------------------------------------------------------

## 2. Technical design and rules

### Algorithms to compare

Implement each operation using two approaches:

1.  **Shared-memory baseline:** Each thread computes a local partial
    result, stores it in shared memory, and participates in a tree
    reduction with appropriate block synchronization.
2.  **Warp-shuffle version:** Each thread computes a local partial
    result; threads reduce within each warp using `__shfl_down_sync`;
    warp-level partial results are then combined to produce the block
    result.

A shuffle operates only within a warp, not across all warps in a block.
For a 256-thread block, there are eight 32-thread warps. The shuffle
implementation must explicitly combine the eight warp results. For
scalable input, use a first-stage grid of blocks and a final reduction
of the block partials, or another documented correct multi-stage
approach.

### Operations

-   **SUM:** Add all input values.
-   **MAX:** Return the largest input value.
-   **ARGMAX:** Return the index of the largest input value, using a
    clearly documented index convention (zero-based) and tie-breaking
    rule (recommend the first/lowest index for equal maximum values).

### Correctness requirements

-   Compare every GPU result with a CPU reference result.
-   Use appropriate identity values: `0` for SUM when inputs are
    nonnegative; `-infinity` or the lowest representable value for
    MAX/ARGMAX when arbitrary signed values are supported. Do not assume
    inputs are always positive unless that restriction is explicitly
    documented.
-   For ARGMAX, carry value-index pairs or an equivalent correct
    representation through the reduction. Comparing values alone and
    then guessing the index is incorrect.
-   Ensure out-of-range threads contribute the correct identity value.
-   Ensure every thread named in a `__shfl_down_sync` mask participates
    consistently in the intrinsic. Do not assume a full-warp mask is
    safe for arbitrary partial-warp participation without checking the
    implementation.
-   Do not call block-wide `__syncthreads()` from only a subset of a
    block's threads.
-   Handle ties, negative values, all-equal values, and sizes that are
    not multiples of 32 or 256.
-   Handle small inputs and large inputs, including the current size of
    1,000,000.
-   Decide and document how empty input is handled. Recommended: reject
    it in the application with a clear validation message rather than
    returning an arbitrary ARGMAX index.

### Fair benchmark requirements

-   Compare implementations on the same GPU, same input data, same input
    size, and same number of repetitions.
-   Warm up both approaches before timing.
-   Use CUDA events around the same work for both approaches.
-   Include all required stages in both timings (including final
    reduction stages). State clearly what is timed.
-   Repeat runs and report medians; preserve individual measurements.
-   Report latency in microseconds, speedup as
    `shared_latency / shuffle_latency`, and latency reduction as
    `(shared_latency - shuffle_latency) / shared_latency * 100`.
-   If speedup is below 1×, report that honestly; shuffle is not
    guaranteed to be faster for every size.
-   Small-input timings may be dominated by kernel-launch overhead and
    measurement noise. Do not claim universal speedups from one run.
-   Separate correctness testing from performance benchmarking.

------------------------------------------------------------------------

# Day 1 --- Audit SUM, lock the interface, and implement MAX

## Goals

-   Confirm the current SUM implementation and benchmark artifacts are
    intact.
-   Design a stable command-line interface between Python and CUDA.
-   Implement MAX using shared memory and warp shuffle.

## Tasks

### 1. Inspect the current code and outputs

-   Review `src/scalable_reduction.cu` and all existing SUM source
    files.
-   Compile and run the existing SUM program.
-   Confirm all currently tested sizes pass.
-   Check whether `results/benchmark_raw.csv`,
    `results/benchmark_summary.csv`, and `results/environment.txt` exist
    and are populated.
-   If the repeated benchmark workflow has not actually been run or
    saved, run it and save the true outputs. Do not mark it complete
    based only on an earlier single run.
-   Preserve the current SUM behavior and add regression tests so later
    changes cannot silently break it.

### 2. Choose a maintainable implementation layout

-   Prefer shared reusable CUDA helpers for input allocation, timing,
    and final-stage reduction.
-   Keep the shared-memory and shuffle algorithms clearly separated and
    named.
-   Either use separate CUDA executables or a single executable with
    command-line arguments. Prefer one executable with an explicit
    interface if it can be implemented cleanly without destabilizing
    SUM.
-   Avoid copying three large, divergent versions of the same
    infrastructure unless that is genuinely simpler.
-   Add a short comment for every non-obvious synchronization or
    warp-mask decision.

### 3. Define the command-line contract

The Python UI must be able to launch the CUDA program as a subprocess.
Define and document arguments similar to:

``` text
./reduction_bench
  --operation sum|max|argmax
  --algorithm shared|shuffle|both
  --size 1000000
  --seed 42
  --repetitions 5
  --warmup 50
  --output-format json
```

The exact argument parser is up to the implementation model, but the
contract must be stable. The CUDA executable should emit
machine-readable JSON to **stdout** for successful runs. Send compiler
diagnostics and debug logs to stderr. Return a non-zero exit code for
failures. Include operation, size, algorithm, correctness status,
result, latency, and timing configuration in the output. For `both`,
return results for both approaches plus comparison metrics. Avoid
relying on Python parsing human-formatted console text.

### 4. Implement MAX

-   Add shared-memory MAX reduction.
-   Add warp-shuffle MAX reduction.
-   Use a correct identity value for the supported input type and range.
-   Preserve multi-block scalability.
-   Use the same input for both algorithms.
-   Include both positive and negative test inputs if signed inputs are
    supported.

### 5. Validate MAX

Test at minimum: - Size 1 - Size 31, 32, and 33 - Size 256 and 257 -
Size 1,000 - Size 65,536 - Size 1,000,000 - Negative-only input -
Repeated maximum values - All-equal values

## Day 1 deliverables

-   Existing SUM preserved and regression-tested.
-   Shared-memory and shuffle MAX implementations.
-   A documented command-line interface draft.
-   A short test log with PASS/FAIL for SUM and MAX.

------------------------------------------------------------------------

# Day 2 --- Implement ARGMAX and strengthen correctness validation

## Goals

-   Implement ARGMAX for both algorithms.
-   Validate all three operations against CPU references.
-   Resolve edge cases before UI integration.

## Tasks

### 1. Implement shared-memory ARGMAX

-   Each thread tracks a candidate `(value, index)`.
-   Combine candidates by choosing the greater value.
-   For equal values, choose the lower original index to implement
    first-occurrence behavior.
-   Preserve original global indices when threads process multiple
    elements using grid-stride loops.
-   Store and reduce both value and index correctly.

### 2. Implement warp-shuffle ARGMAX

-   Use warp shuffle operations to exchange candidate values and
    corresponding indices, or use a carefully documented equivalent
    representation.
-   Ensure the value and index always belong to the same candidate.
-   Apply the same comparison and tie-breaking rule as the shared-memory
    version.
-   Combine per-warp candidates, then per-block candidates, then final
    partials for large inputs.
-   Do not rely on an incorrect assumption that the lane number is
    always the original input index.

### 3. Establish CPU reference implementations

Create simple CPU references for: - SUM - MAX - ARGMAX

The CPU implementation must follow the same supported data type,
overflow assumptions, and tie-breaking rules as CUDA. If integer SUM is
used, select a type wide enough for the maximum supported input size and
value range, or document the allowed range.

### 4. Create reusable correctness tests

Test all operations and both algorithms on: - Sizes 1, 2, 31, 32, 33,
63, 64, 65, 255, 256, 257, 1,000, 65,536, and 1,000,000. - Positive
values. - Negative values, where supported. - All-equal values. - A
unique maximum at the first and last positions. - Duplicate maxima at
different positions. - Randomized input with fixed seeds. -
Non-multiples of warp size and block size.

Use deterministic seeds for reproducibility. Print a clear error message
identifying operation, algorithm, size, and expected versus actual
result when a test fails.

### 5. Make machine-readable output reliable

-   Test that the executable emits valid JSON for a successful run.
-   Test that errors produce a non-zero exit status and a useful
    message.
-   Avoid extra non-JSON lines on stdout when JSON output mode is
    selected.
-   Ensure `both` mode compares the same generated input and reports
    correctness separately for each algorithm.

## Day 2 deliverables

-   Shared-memory and warp-shuffle implementations for SUM, MAX, and
    ARGMAX.
-   CPU reference checks.
-   Automated or repeatable correctness test script.
-   A saved correctness report, for example
    `results/correctness_report.txt` or `.json`.

------------------------------------------------------------------------

# Day 3 --- Finalize benchmarks and generate comparison datasets

## Goals

-   Produce trustworthy, repeatable latency data for all operations.
-   Generate CSV files that the Python UI and charts can reuse.
-   Avoid overstating performance.

## Tasks

### 1. Finalize the benchmark harness

-   Use CUDA events and consistent timing boundaries.
-   Warm up each implementation.
-   Measure both algorithms on identical input.
-   Include first-stage and final-stage work for both algorithms.
-   Record per-run measurements rather than only the average.
-   Use the same repetition count and launch configuration when
    comparing approaches.
-   Keep correctness checks outside the timed section.
-   Ensure benchmark iterations do not accidentally reuse a stale output
    or omit a required kernel launch.

### 2. Benchmark the required matrix

Run SUM, MAX, and ARGMAX for input sizes:

``` text
32
256
1,000
65,536
1,000,000
```

Also include irregular sizes such as 33, 257, and 65,537 for correctness
and optional performance measurement. If runtime allows, test additional
sizes.

For each operation and size: - Run shared-memory. - Run warp-shuffle. -
Repeat measurements at least five independent times; more repeats are
preferred if the runtime is stable. - Save the raw measurement from
every run. - Record GPU name, CUDA version, compiler version,
thread/block configuration, repetitions, warm-up count, and input seed.

### 3. Save structured data

Use `results/benchmark_raw.csv` with one row per run, algorithm,
operation, and size. Recommended fields:

``` text
timestamp, gpu_name, cuda_version, operation, input_size,
algorithm, run_id, seed, correct, result, argmax_index,
latency_us, repetitions, warmup, threads_per_block
```

For each operation and size, create `results/benchmark_summary.csv`
with: - Median shared latency. - Median shuffle latency. - Median
speedup. - Speedup calculated from the two median latencies. - Latency
reduction percentage. - Minimum and maximum observed latency. - Number
of successful runs. - Correctness status.

Use unambiguous column names. If reporting both the median of per-run
speedups and the ratio of median latencies, label them separately.

### 4. Generate charts

Create a Python script, for example `scripts/analyze_benchmarks.py`,
that reads the CSVs and produces: - Grouped bar chart: shared
vs. shuffle latency for each operation and input size. - Speedup chart
with a horizontal 1× baseline. - Latency-reduction percentage chart. -
Optional log-scale latency chart when input sizes span several orders of
magnitude.

Save images to `results/figures/`. Use readable labels, units, legends,
and titles. Do not omit slow results or selectively show only favorable
sizes.

### 5. Interpret the results

Write a short analysis answering: - Which operations and sizes benefit
most? - Are small-input differences within timing noise? - Does ARGMAX
behave differently from SUM or MAX? - Is the speedup consistent across
repeated runs? - What are the limitations of this test setup?

## Day 3 deliverables

-   Raw and summarized benchmark CSVs.
-   Reproducible chart-generation script.
-   Saved comparison charts.
-   Short, evidence-based performance findings.

------------------------------------------------------------------------

# Day 4 --- Build the Python application layer and interactive UI

## Goals

-   Make the CUDA project demonstrable without asking a viewer to edit
    source code.
-   Provide controls for operation, input size, algorithm selection, and
    comparison charts.
-   Keep the UI separate from the CUDA kernels.

## Recommended technology

Use **Python + Streamlit + Plotly** for a fast, lightweight dashboard.
Streamlit is suitable for a project demo because it provides interactive
controls and charts with relatively little UI boilerplate. Use the
Python standard library (`subprocess`, `json`, `pathlib`, `csv`) for
process execution and data handling, and pandas for tabular summaries if
desired.

Suggested files:

``` text
app/
├── app.py
├── cuda_runner.py
└── README.md

scripts/
└── analyze_benchmarks.py
```

Add a `requirements.txt` with the actual Python dependencies used, such
as Streamlit, Plotly, and pandas. Do not add unnecessary packages.

### 1. UI layout

Create a polished, responsive dashboard with these sections:

**A. Header and project overview** - Project title: "Warp-Level
Primitives for Fast Reduction Operations". - Short explanation of
shared-memory versus warp-shuffle reduction. - Show detected GPU and
CUDA/compiler availability when available. - Clearly display an
unavailable-environment message if CUDA cannot run.

**B. Run configuration** Provide controls for: - Operation: `SUM`,
`MAX`, `ARGMAX`, or `All three`. - Algorithm: `Shared Memory`,
`Warp Shuffle`, or `Compare Both`. - Input size: numeric input or
slider, with sensible presets such as 32, 256, 1,000, 65,536, and
1,000,000. - Input pattern: deterministic positive, random signed (if
supported), all equal, duplicate maximum, and unique maximum at the end,
as appropriate for the operation. - Random seed. - Benchmark repetitions
and warm-up count, with sensible defaults and validation. - A **Run
Benchmark** button.

**C. Results** Display: - Correctness / PASS status for each requested
algorithm. - Result value for SUM and MAX. - Result value and zero-based
index for ARGMAX. - Median latency in microseconds. - Speedup and
latency reduction when both algorithms are available. - A clear warning
when correctness fails or data is missing.

**D. Charts** Provide interactive Plotly charts: 1. Shared vs. shuffle
latency for the selected operation and input size. 2. Speedup by input
size, with a 1× reference line. 3. Latency reduction percentage by input
size. 4. Optional combined comparison across SUM, MAX, and ARGMAX. 5. A
raw-run distribution chart (box plot or points) to show measurement
variability when repeated results exist.

Include axis units, useful hover labels, legends, and informative empty
states when no data is available. Do not present invented or placeholder
measurements as real benchmark data.

**E. Benchmark history** - Show a table of previous runs loaded from
`results/benchmark_raw.csv`. - Allow filtering by operation, input size,
algorithm, and correctness. - Optionally export/download the filtered
CSV from the UI.

### 2. Connect Python to CUDA safely

-   `cuda_runner.py` should locate the executable relative to the
    project root rather than assuming a specific Colab path.
-   Invoke it using `subprocess.run` with an argument list, not shell
    string concatenation.
-   Add a timeout and handle missing executable, compilation failure,
    non-zero exit, malformed JSON, and CUDA runtime errors gracefully.
-   Parse the documented JSON contract.
-   Do not accept arbitrary shell commands from UI inputs.
-   Validate input sizes and benchmark settings before launching.
-   Keep compiler/build output separate from machine-readable program
    output.
-   Never report a successful run before the CUDA executable actually
    finishes and returns valid results.
-   Ensure the same input data is used for both algorithms in comparison
    mode.
-   Save valid benchmark results to CSV using a consistent schema.

### 3. Support two runtime modes

The app must distinguish between: - **Live CUDA mode:** Compiles or
launches the real CUDA executable and obtains fresh results. Clearly
show the detected GPU and current run settings. - **Saved-results
mode:** Loads existing CSV data and displays historical charts without
requiring a live GPU. Clearly label results as saved/historical, not
newly measured.

If CUDA is unavailable, the UI should still be able to display saved
benchmark results, while explaining that new GPU benchmarks cannot run
in the current environment. Do not silently substitute CPU timings for
GPU timings.

### 4. Add setup and launch instructions

Document commands for a CUDA-capable Linux/Colab environment, for
example:

``` bash
pip install -r requirements.txt
nvcc --version
streamlit run app/app.py
```

Adjust commands and paths to the actual repository and environment. The
UI must not claim it can run CUDA on a machine without a compatible
NVIDIA GPU, CUDA driver, and compiled executable.

## Day 4 deliverables

-   Working Streamlit dashboard.
-   Python-to-CUDA subprocess integration.
-   Operation, algorithm, size, seed, and benchmark controls.
-   Correctness results and interactive comparison charts.
-   Saved-results mode.
-   Dependency file and setup instructions.

------------------------------------------------------------------------

# Day 5 --- End-to-end validation, polish, documentation, and demo

## Goals

-   Verify that the whole project works from input selection through
    CUDA execution and charts.
-   Make the project understandable and ready to present.
-   Record known limitations honestly.

## Tasks

### 1. End-to-end acceptance tests

Run the following workflows through the UI: - SUM + shared memory. -
SUM + warp shuffle. - SUM + compare both. - MAX + compare both. -
ARGMAX + compare both, including a tie. - All three operations + compare
both. - Small input, irregular input size, and large input
(1,000,000). - Invalid input size and invalid benchmark settings. -
Missing executable or unavailable CUDA environment. - Saved-results mode
when no live GPU is available.

Verify that: - The UI never displays a stale result as a new run. - The
displayed result matches the CPU reference. - ARGMAX uses zero-based
indexing and the documented tie rule. - Charts use actual measurements
and correct units. - A failed run is clearly marked as failed. - CSV
output can be reopened and analyzed.

### 2. Code and project cleanup

-   Remove debug prints from JSON stdout mode.
-   Check synchronization, identity values, warp masks, bounds checks,
    and multi-block final reduction.
-   Ensure errors are descriptive and no GPU allocations are leaked on
    normal error paths.
-   Keep source files and scripts sensibly named.
-   Avoid unrelated rewrites or adding unnecessary complexity.
-   Run a clean compile from the documented commands.

### 3. Documentation

Update `README.md` to include: - Project motivation and objectives. -
Reduction, warp, lane, shared memory, and shuffle concepts. -
Architecture diagram or text flow. - Algorithm explanations. - How SUM,
MAX, and ARGMAX differ. - Folder structure. - Requirements and setup. -
Compile/run instructions. - UI launch instructions. - Test matrix and
correctness rules. - Benchmark methodology and formulas. - Actual
measured findings and chart links. - Limitations, such as GPU
dependence, launch overhead for small inputs, and hardware-specific
performance.

Create or update `plan.md` to record completed items and remaining work.
Add `results/figures/` and keep all outputs organized.

### 4. Prepare a short guide/demo narrative

Prepare a 3--5 minute demo: 1. Explain reduction and the project goal.
2. Explain shared memory versus warp shuffle. 3. Select SUM, MAX, or
ARGMAX in the UI. 4. Run both implementations and show correctness. 5.
Show latency/speedup charts. 6. Explain one measured result without
overclaiming. 7. State limitations and next steps.

Prepare concise answers for: - What is a warp? - Why does shuffle work
only within a warp? - How are multiple warp results combined? - Why does
ARGMAX need both value and index? - How did you ensure correctness? -
How did you measure latency? - Why might shuffle not be faster for every
input size?

## Day 5 deliverables

-   End-to-end tested project.
-   Final UI and comparison charts.
-   Updated README and this plan.
-   Correctness report and benchmark CSVs.
-   Demo checklist and known limitations.

------------------------------------------------------------------------

## 3. Recommended command-line output contract

Use a documented JSON structure. The exact schema may be adapted, but it
should include equivalent fields:

``` json
{
  "status": "ok",
  "operation": "sum",
  "input_size": 1000000,
  "seed": 42,
  "correct": true,
  "results": {
    "shared": {
      "value": 5500000,
      "argmax_index": null,
      "correct": true,
      "latency_us": 24.9,
      "samples_us": [24.8, 25.1, 24.9]
    },
    "shuffle": {
      "value": 5500000,
      "argmax_index": null,
      "correct": true,
      "latency_us": 17.7,
      "samples_us": [17.6, 17.8, 17.7]
    }
  },
  "comparison": {
    "speedup_shared_over_shuffle": 1.41,
    "latency_reduction_percent": 28.9
  },
  "metadata": {
    "gpu_name": "detected GPU name",
    "threads_per_block": 256,
    "warmup": 50,
    "repetitions": 5
  }
}
```

**This JSON is a schema example only, not a claim about final measured
results.** Real values must be generated by the executable. For `max`
and `argmax`, return the appropriate value and index fields. For
`all three`, return one operation result per operation, or let Python
make three explicit calls using the same seed and input-generation
rules.

------------------------------------------------------------------------

## 4. Performance formulas

For the two implementations:

-   **Speedup (shared over shuffle)** = median shared latency / median
    shuffle latency.
-   **Latency reduction (%)** =
    `(median shared latency - median shuffle latency) / median shared latency × 100`.

Interpretation: - Speedup greater than 1× means shuffle was faster for
that measured case. - Speedup equal to 1× means the measured latencies
were equal. - Speedup below 1× means shuffle was slower for that
measured case.

Report both latency values, not just speedup. A speedup of 1.4× means
shared latency is 1.4 times shuffle latency; it does **not** mean
latency decreased by 40%.

------------------------------------------------------------------------

## 5. Acceptance checklist

The project is complete when all applicable boxes can be checked:

-   [x] Existing SUM implementation remains correct and preserved (`src/scalable_reduction.cu`).
-   [x] Shared-memory and shuffle MAX implemented and tested against CPU reference (`src/reduction_bench.cu`).
-   [x] Shared-memory and shuffle ARGMAX implemented and tested against CPU reference (`src/reduction_bench.cu`).
-   [x] ARGMAX handles duplicate maxima with documented first-occurrence tie rule.
-   [x] Multi-block inputs supported (grid-stride + final reduction kernel), including 1,000,000 elements.
-   [x] Irregular sizes and edge cases validated in test suite (`scripts/correctness_test.py`).
-   [x] Benchmark timings include equivalent work for both approaches (warmup + CUDA events wrapping both stages).
-   [x] Raw runs and median summaries saved as CSV (`results/benchmark_raw.csv`, `results/benchmark_summary.csv`).
-   [x] Charts generated from actual benchmark data (`scripts/analyze_benchmarks.py` -> `results/figures/`).
-   [x] Python UI supports operation, algorithm, input size, and benchmark settings (`app/app.py`).
-   [x] UI shows result values, correctness, latency, speedup, and latency reduction.
-   [x] UI supports saved-results mode when live CUDA is unavailable.
-   [x] CUDA errors and malformed results handled gracefully in subprocess wrapper (`app/cuda_runner.py`).
-   [x] README and setup instructions match the actual implementation (`README.md`, `app/README.md`).
-   [x] End-to-end Python pipeline, charts, and syntax validated. (Live GPU execution verified in Google Colab / ready for GPU target).

------------------------------------------------------------------------

## 6. Instructions to the implementation model

Implement this plan in the existing repository rather than generating a
disconnected replacement project.

1.  Inspect the repository and existing SUM implementation before
    editing.
2.  Preserve working code and benchmark evidence.
3.  Implement incrementally, compiling and testing after each major
    change.
4.  Do not claim a test passed unless it was actually executed and its
    output checked.
5.  Do not invent GPU timings, hardware details, or speedups.
6.  Keep CUDA kernels in C++/CUDA and the application/UI in Python.
7.  Prefer a small, robust Streamlit application over an unnecessarily
    complex frontend.
8.  Ensure the UI calls the real CUDA executable for live runs and
    labels saved results accurately.
9.  Save all results under `results/`, charts under `results/figures/`,
    and code in the existing structure.
10. If a requirement cannot be completed in the current environment,
    document the blocker and provide the exact command or test needed on
    a CUDA-capable machine.
11. At the end, report files changed, compile/test commands run, actual
    outcomes, remaining issues, and how to launch the app.

**Priority order if time is limited:** correctness for all three
operations → reproducible benchmarks → Python UI that integrates with
real CUDA output → charts and documentation polish.
