# SolidAttention Evaluation Project

This project implements an evaluation framework for **SolidAttention**, exploring the trade-offs between NVMe SSD read throughput and Large Language Model (LLM) accuracy when serving context from SSDs instead of RAM. 

Due to the memory constraints of modern PCs, caching long sequences in VRAM or RAM can lead to Out-Of-Memory (OOM) errors. SolidAttention mitigates this by dynamically loading Attention KV cache blocks directly from an SSD.

## Directory Structure

*   `scripts/`: Contains the main execution scripts for Phase 1 (SSD Throughput), Phase 2 (Model Accuracy), and Phase 3 (Simulation).
    * `scripts/mlx/`: Optimized Apple Silicon natively compatible MLX scripts.
    * `scripts/pytorch/`: Universal PyTorch scripts that run on CPU/CUDA.
*   `utils/`: Contains helper scripts, such as plotting utilities.
*   `outputs/`: Stores all generated artifacts, such as throughput plots, accuracy benchmarks, and the final trade-off simulation charts.

## Getting Started

Ensure you have your Conda environment activated before running the Python scripts:
```bash
conda activate storage-systems
```

### Phase 1: Storage Profiling
Measures the NVMe random read throughput of your machine across varying block sizes using `fio`.
```bash
./scripts/phase1_fio_profiler.sh
```

### Phase 2: Model Accuracy Profiling
Evaluates the model's performance on Needle-In-A-Haystack and LongBench when dropping chunks of its attention context based on different block sizes.

For Apple Silicon (MLX):
```bash
python scripts/mlx/phase2_accuracy_profiler.py --model mlx-community/Meta-Llama-3.1-8B-4bit --trials 20 --haystacks us_haystack.txt ww2_haystack.txt --verbose
python scripts/mlx/phase2_longbench_profiler.py --model mlx-community/Meta-Llama-3.1-8B-4bit --samples 3
```

For **PyTorch**:
```bash
python scripts/pytorch/phase2_accuracy_profiler.py --model meta-llama/Meta-Llama-3.1-8B --trials 20 --haystacks us_haystack.txt ww2_haystack.txt --verbose
python scripts/pytorch/phase2_longbench_profiler.py --model meta-llama/Meta-Llama-3.1-8B --samples 3
```

### 3. Simulation & Trade-off Plotting (Phase 3)
Combine the results to plot the Pareto front of the SolidAttention tradeoff:

```bash
python scripts/phase3_simulation.py --model Meta-Llama-3.1-8B-4bit
```
*(Optionally pass `--dummy` to test the simulation using dummy data without running the 30-sample evaluations)*
