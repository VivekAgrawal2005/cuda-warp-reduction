"""
cuda_runner.py
==============
Locates, invokes, and parses the CUDA reduction_bench executable.
All subprocess calls use argument lists (no shell=True, no string concat).
"""

import subprocess
import json
import pathlib
import csv
import datetime
import sys

# Project root is one directory above this file (app/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Executable paths (Linux/Colab vs Windows)
_EXE_CANDIDATES = [
    PROJECT_ROOT / "src" / "reduction_bench",
    PROJECT_ROOT / "src" / "reduction_bench.exe",
]

RAW_CSV     = PROJECT_ROOT / "results" / "benchmark_raw_v2.csv"
SUM_RAW_CSV = PROJECT_ROOT / "results" / "benchmark_raw.csv"   # legacy SUM CSV

# CSV columns for new full schema
RAW_COLS = [
    "timestamp","gpu_name","operation","input_size","algorithm",
    "pattern","seed","correct","value","argmax_index",
    "latency_us","repetitions","warmup","blocks","threads_per_block",
    "samples_us","speedup_shared_over_shuffle","latency_reduction_percent",
]


def find_executable() -> pathlib.Path | None:
    for p in _EXE_CANDIDATES:
        if p.exists():
            return p
    return None


def is_cuda_available() -> bool:
    exe = find_executable()
    return exe is not None


def get_gpu_info() -> str:
    """Return GPU name from nvidia-smi, or empty string on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return ""


def validate_inputs(n: int, reps: int, warmup: int):
    if n < 1:
        raise ValueError("Input size must be >= 1.")
    if n > 50_000_000:
        raise ValueError("Input size must be <= 50,000,000.")
    if reps < 1 or reps > 500:
        raise ValueError("Repetitions must be between 1 and 500.")
    if warmup < 0 or warmup > 1000:
        raise ValueError("Warmup must be between 0 and 1000.")


def run_benchmark(
    operation: str,
    algorithm: str,
    size: int,
    seed: int = 42,
    repetitions: int = 5,
    warmup: int = 50,
    pattern: str = "random_signed",
    timeout: int = 300,
) -> dict:
    """
    Invoke the CUDA executable and return parsed JSON result dict.
    Raises RuntimeError with a descriptive message on any failure.
    """
    exe = find_executable()
    if exe is None:
        raise RuntimeError(
            "CUDA executable not found. "
            "Compile with: nvcc -O2 -arch=sm_75 src/reduction_bench.cu -o src/reduction_bench"
        )

    validate_inputs(size, repetitions, warmup)

    cmd = [
        str(exe),
        "--operation",   operation,
        "--algorithm",   algorithm,
        "--size",        str(size),
        "--seed",        str(seed),
        "--repetitions", str(repetitions),
        "--warmup",      str(warmup),
        "--pattern",     pattern,
        "--output-format", "json",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"CUDA benchmark timed out after {timeout}s.")
    except FileNotFoundError:
        raise RuntimeError(f"Executable not found at: {exe}")

    if proc.returncode != 0:
        stderr_snippet = proc.stderr.strip()[:500] if proc.stderr else "(no stderr)"
        raise RuntimeError(
            f"CUDA executable exited with code {proc.returncode}.\n"
            f"Stderr: {stderr_snippet}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("CUDA executable produced no output.")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        snippet = stdout[:200]
        raise RuntimeError(
            f"Failed to parse JSON from CUDA executable.\n"
            f"JSON error: {e}\nOutput snippet: {snippet}"
        )

    if data.get("status") != "ok":
        msg = data.get("message", "Unknown error from CUDA executable.")
        raise RuntimeError(f"CUDA executable reported an error: {msg}")

    # Persist results to CSV
    _append_to_csv(data)

    return data


def _append_to_csv(data: dict):
    """Append one or two rows (per algorithm) to the raw benchmark CSV."""
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RAW_CSV.exists()

    ts        = datetime.datetime.now().isoformat(timespec="seconds")
    meta      = data.get("metadata", {})
    comp      = data.get("comparison") or {}
    results   = data.get("results", {})

    rows = []
    for alg, res in results.items():
        row = {
            "timestamp":            ts,
            "gpu_name":             meta.get("gpu_name", ""),
            "operation":            data.get("operation", ""),
            "input_size":           data.get("input_size", ""),
            "algorithm":            alg,
            "pattern":              data.get("pattern", ""),
            "seed":                 data.get("seed", ""),
            "correct":              res.get("correct", ""),
            "value":                res.get("value", ""),
            "argmax_index":         res.get("argmax_index", ""),
            "latency_us":           res.get("latency_us", ""),
            "repetitions":          meta.get("repetitions", ""),
            "warmup":               meta.get("warmup", ""),
            "blocks":               meta.get("blocks", ""),
            "threads_per_block":    meta.get("threads_per_block", ""),
            "samples_us":           json.dumps(res.get("samples_us", [])),
            "speedup_shared_over_shuffle":  comp.get("speedup_shared_over_shuffle", ""),
            "latency_reduction_percent":    comp.get("latency_reduction_percent", ""),
        }
        rows.append(row)

    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_raw_csv() -> list[dict]:
    """Load all rows from the new raw CSV; return list of dicts."""
    if not RAW_CSV.exists():
        return []
    with open(RAW_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_legacy_sum_csv() -> list[dict]:
    """Load the legacy SUM benchmark_raw.csv for historical display."""
    if not SUM_RAW_CSV.exists():
        return []
    with open(SUM_RAW_CSV, newline="") as f:
        return list(csv.DictReader(f))
