"""
analyze_benchmarks.py
=====================
Reads results/benchmark_raw_v2.csv (or the legacy benchmark_raw.csv for SUM-only)
and produces comparison charts saved under results/figures/.

Usage:
    python scripts/analyze_benchmarks.py
"""

import pathlib
import csv
import json
import math
import sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV      = PROJECT_ROOT / "results" / "benchmark_raw_v2.csv"
LEGACY_CSV   = PROJECT_ROOT / "results" / "benchmark_raw.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"shared": "#60a5fa", "shuffle": "#4ade80"}

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────

def load_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

rows = load_csv(RAW_CSV)

# Fallback to legacy SUM CSV if new CSV is empty
if not rows:
    legacy = load_csv(LEGACY_CSV)
    if not legacy:
        print("No benchmark data found. Run benchmarks first.", file=sys.stderr)
        sys.exit(0)
    # Reshape legacy CSV
    for r in legacy:
        for alg in ("shared", "shuffle"):
            rows.append({
                "operation":  "sum",
                "algorithm":  alg,
                "input_size": r.get("input_size", "0"),
                "latency_us": r.get(f"{alg}_us", "0"),
                "correct":    "True" if r.get("correctness") == "PASS" else "False",
                "speedup_shared_over_shuffle": r.get("speedup_shared_over_shuffle", ""),
                "latency_reduction_percent": "",
            })
    print("Using legacy SUM-only data.")

# Parse numeric fields
for r in rows:
    r["input_size"] = int(float(r.get("input_size") or 0))
    r["latency_us"] = float(r.get("latency_us") or 0)
    r["speedup_shared_over_shuffle"] = float(r["speedup_shared_over_shuffle"]) if r.get("speedup_shared_over_shuffle") else None

# Group by (operation, input_size, algorithm) → list of latencies
from collections import defaultdict
groups: dict = defaultdict(list)
for r in rows:
    key = (r.get("operation","sum"), r["input_size"], r["algorithm"])
    groups[key].append(r["latency_us"])

def median(lst):
    s = sorted(lst)
    m = len(s)
    if not m: return 0
    return s[m//2] if m & 1 else (s[m//2-1]+s[m//2])/2

# Build summary table
ops      = sorted(set(r.get("operation","sum") for r in rows))
algs     = ["shared", "shuffle"]
sizes    = sorted(set(r["input_size"] for r in rows))

summary = {}
for op in ops:
    summary[op] = {}
    for sz in sizes:
        summary[op][sz] = {}
        for alg in algs:
            lats = groups.get((op, sz, alg), [])
            summary[op][sz][alg] = median(lats) if lats else None

# ──────────────────────────────────────────────────────────────
# Chart 1 – Grouped bar: Latency by operation × size
# ──────────────────────────────────────────────────────────────

plt.style.use("dark_background")

for op in ops:
    valid_sizes = [s for s in sizes if any(summary[op][s][a] for a in algs)]
    if not valid_sizes:
        continue

    x = np.arange(len(valid_sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i, alg in enumerate(algs):
        vals = [summary[op][s][alg] or 0 for s in valid_sizes]
        bars = ax.bar(x + i*width - width/2, vals, width, label=alg.capitalize(),
                      color=COLORS[alg], alpha=0.85, edgecolor="none")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#94a3b8")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in valid_sizes], fontsize=9, color="#94a3b8")
    ax.set_xlabel("Input Size", color="#94a3b8")
    ax.set_ylabel("Latency (µs)", color="#94a3b8")
    ax.set_title(f"{op.upper()} – Shared vs Shuffle Median Latency", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", color="#334155", linewidth=0.5)
    ax.spines[:].set_color("#334155")

    out = FIG_DIR / f"latency_bar_{op}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")

# ──────────────────────────────────────────────────────────────
# Chart 2 – Speedup by size (per operation)
# ──────────────────────────────────────────────────────────────

for op in ops:
    speedups, x_labels = [], []
    for sz in sizes:
        sh = summary[op][sz].get("shared")
        sf = summary[op][sz].get("shuffle")
        if sh and sf and sf > 0:
            speedups.append(sh / sf)
            x_labels.append(f"{sz:,}")

    if not speedups:
        continue

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(x_labels, speedups, marker="o", color="#a78bfa", linewidth=2, markersize=8)
    for xl, sp in zip(x_labels, speedups):
        ax.annotate(f"{sp:.2f}×", (xl, sp), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#e2e8f0")
    ax.axhline(1.0, linestyle="--", color="#f97316", linewidth=1.2, label="1× baseline")
    ax.set_xlabel("Input Size", color="#94a3b8")
    ax.set_ylabel("Speedup (×)", color="#94a3b8")
    ax.set_title(f"{op.upper()} – Speedup (shared latency / shuffle latency)", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#334155", linewidth=0.5)
    ax.spines[:].set_color("#334155")

    out = FIG_DIR / f"speedup_{op}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")

# ──────────────────────────────────────────────────────────────
# Chart 3 – Latency reduction % by size
# ──────────────────────────────────────────────────────────────

for op in ops:
    lrs, x_labels = [], []
    for sz in sizes:
        sh = summary[op][sz].get("shared")
        sf = summary[op][sz].get("shuffle")
        if sh and sf and sh > 0:
            lrs.append((sh - sf) / sh * 100)
            x_labels.append(f"{sz:,}")

    if not lrs:
        continue

    colors = [COLORS["shuffle"] if v > 0 else "#f87171" for v in lrs]
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.bar(x_labels, lrs, color=colors, alpha=0.85, edgecolor="none")
    for bar, v in zip(bars, lrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5 if v >= 0 else bar.get_height() - 1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Input Size", color="#94a3b8")
    ax.set_ylabel("Latency Reduction (%)", color="#94a3b8")
    ax.set_title(f"{op.upper()} – Latency Reduction (positive = shuffle faster)", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", color="#334155", linewidth=0.5)
    ax.spines[:].set_color("#334155")

    out = FIG_DIR / f"latency_reduction_{op}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")

# ──────────────────────────────────────────────────────────────
# Chart 4 – Log-scale latency vs size (all ops combined)
# ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

op_styles = {"sum": "o-", "max": "s--", "argmax": "^:"}
op_colors = {"sum": "#60a5fa", "max": "#4ade80", "argmax": "#f59e0b"}

for op in ops:
    for alg in algs:
        xs, ys = [], []
        for sz in sizes:
            v = summary[op][sz].get(alg)
            if v and sz > 0:
                xs.append(sz); ys.append(v)
        if xs:
            ls = "-" if alg == "shared" else "--"
            ax.plot(xs, ys, linestyle=ls, marker="o", markersize=6,
                    color=op_colors.get(op, "#aaa"), alpha=0.8 if alg == "shared" else 0.5,
                    label=f"{op}/{alg}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Input Size (log scale)", color="#94a3b8")
ax.set_ylabel("Latency µs (log scale)", color="#94a3b8")
ax.set_title("All Operations – Log-Log Latency vs Input Size", color="#e2e8f0", fontsize=13, fontweight="bold")
ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", ncol=2, fontsize=8)
ax.tick_params(colors="#94a3b8")
ax.grid(color="#334155", linewidth=0.5, which="both")
ax.spines[:].set_color("#334155")

out = FIG_DIR / "logscale_all.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")

print("\nAll charts generated in", FIG_DIR)
