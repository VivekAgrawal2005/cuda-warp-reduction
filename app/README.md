# Warp Reduction – Python Application Layer

## Files
- `app.py` – Streamlit dashboard (Live Run, History, Charts, About tabs)
- `cuda_runner.py` – subprocess wrapper for the CUDA executable; CSV persistence

## Launch

```bash
# From the project root:
pip install -r requirements.txt
streamlit run app/app.py
```

## Runtime modes

| Mode | Condition | Behaviour |
|---|---|---|
| **Live CUDA** | `src/reduction_bench` exists | Runs real GPU benchmarks; results saved to `results/benchmark_raw_v2.csv` |
| **Saved-results** | Executable not found | Shows historical CSVs; clearly labelled as saved data |

## Compile the executable (CUDA-capable Linux / Colab)

```bash
nvcc -O2 -arch=sm_75 src/reduction_bench.cu -o src/reduction_bench
```

Adjust `-arch=sm_XX` to match your GPU (T4 → sm_75, RTX 30xx → sm_86, A100 → sm_80).
