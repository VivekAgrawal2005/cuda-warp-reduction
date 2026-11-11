"""
correctness_test.py
===================
Runs the CUDA executable over a comprehensive test matrix and reports
PASS/FAIL for each (operation, algorithm, size, pattern) combination.

Usage:
    python scripts/correctness_test.py
    python scripts/correctness_test.py --verbose

Exit code: 0 if all tests pass, 1 if any fail.
"""

import subprocess
import json
import pathlib
import sys
import argparse
import datetime

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXE_CANDIDATES = [
    PROJECT_ROOT / "src" / "reduction_bench",
    PROJECT_ROOT / "src" / "reduction_bench.exe",
]
REPORT_PATH = PROJECT_ROOT / "results" / "correctness_report.json"

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# Locate executable
exe = None
for c in EXE_CANDIDATES:
    if c.exists():
        exe = c; break
if exe is None:
    print("ERROR: reduction_bench executable not found. Compile first.", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Test matrix
# ──────────────────────────────────────────────────────────────

SIZES = [1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1_000, 65_536, 1_000_000]
OPS   = ["sum", "max", "argmax"]
PATTERNS = {
    "sum":    ["random_signed", "positive", "all_equal"],
    "max":    ["random_signed", "positive", "all_equal", "dup_max", "max_at_end"],
    "argmax": ["random_signed", "positive", "all_equal", "dup_max", "max_at_end"],
}
SEEDS = [42, 7, 123]

results = []
total = passed = failed = 0

print(f"Correctness test – {datetime.datetime.now().isoformat()}")
print(f"Executable: {exe}")
print(f"Test matrix: {len(OPS)} ops × {len(SIZES)} sizes × patterns × {len(SEEDS)} seeds")
print("-" * 70)

for op in OPS:
    for pattern in PATTERNS[op]:
        for sz in SIZES:
            for seed in SEEDS:
                total += 1
                cmd = [
                    str(exe),
                    "--operation",   op,
                    "--algorithm",   "both",
                    "--size",        str(sz),
                    "--seed",        str(seed),
                    "--repetitions", "1",
                    "--warmup",      "2",
                    "--pattern",     pattern,
                    "--output-format", "json",
                ]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT  op={op} sz={sz} pat={pattern} seed={seed}")
                    results.append({"op":op,"sz":sz,"pattern":pattern,"seed":seed,"status":"TIMEOUT"})
                    failed += 1
                    continue

                if proc.returncode != 0:
                    stderr = proc.stderr.strip()[:200]
                    print(f"CRASH    op={op} sz={sz} pat={pattern} seed={seed}  stderr={stderr}")
                    results.append({"op":op,"sz":sz,"pattern":pattern,"seed":seed,"status":"CRASH","stderr":stderr})
                    failed += 1
                    continue

                try:
                    data = json.loads(proc.stdout)
                except Exception as e:
                    print(f"BAD_JSON op={op} sz={sz} pat={pattern} seed={seed}  {e}")
                    results.append({"op":op,"sz":sz,"pattern":pattern,"seed":seed,"status":"BAD_JSON"})
                    failed += 1
                    continue

                ok = data.get("correct", False)
                alg_details = {}
                for alg, res in data.get("results", {}).items():
                    alg_details[alg] = {
                        "correct": res.get("correct"),
                        "value":   res.get("value"),
                        "argmax_index": res.get("argmax_index"),
                    }

                status = "PASS" if ok else "FAIL"
                record = {"op":op,"sz":sz,"pattern":pattern,"seed":seed,"status":status,"alg_details":alg_details}
                results.append(record)

                if ok:
                    passed += 1
                    if args.verbose:
                        print(f"PASS     op={op:6s} sz={sz:8,} pat={pattern:15s} seed={seed}")
                else:
                    failed += 1
                    print(f"FAIL     op={op:6s} sz={sz:8,} pat={pattern:15s} seed={seed}  details={alg_details}")

print("-" * 70)
print(f"Results: {passed}/{total} PASS  |  {failed} FAIL")

# Save report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
report = {
    "timestamp": datetime.datetime.now().isoformat(),
    "executable": str(exe),
    "total": total, "passed": passed, "failed": failed,
    "tests": results,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report saved: {REPORT_PATH}")

sys.exit(0 if failed == 0 else 1)
