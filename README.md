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
Measures the NVMe random read throughput of your machine across varying block sizes using `fio` and plots the results.
```bash
./scripts/run_phase1.sh
```

### Phase 2: Model Accuracy Profiling
Evaluates the model's performance on Needle-In-A-Haystack when dropping chunks of its attention context based on different block sizes.

For Apple Silicon (MLX):
```bash
python scripts/mlx/phase2_accuracy_profiler.py --model mlx-community/Meta-Llama-3.1-8B-4bit --trials 20 --haystacks us_haystack.txt ww2_haystack.txt --verbose
```

For **PyTorch**:
```bash
python scripts/pytorch/phase2_accuracy_profiler.py --model meta-llama/Meta-Llama-3.1-8B --trials 20 --haystacks us_haystack.txt ww2_haystack.txt --verbose
```

### 3. Simulation & Trade-off Plotting (Phase 3)
Run simulations and combine the results across multiple models to plot the Pareto front of the SolidAttention tradeoff:

```bash
./scripts/run_phase3.sh
```

You can also run the underlying python script manually for a single configuration:
```bash
python scripts/phase3_simulation.py --throughput <path-to-json> --accuracy <path-to-json> --output <path-to-json>
```

**FDP Simulation Mode**
You can also simulate the FDP (Flexible Data Placement) hardware improvement, which models cross-channel striping and zero Garbage Collection (GC) overhead:
```bash
python scripts/phase3_simulation.py --throughput <path-to-json> --accuracy <path-to-json> --output <path-to-json> --enable-fdp
```
*Optional standard simulation flags:*
* `--gc-spike-prob 0.05` (5% chance of a GC spike)
* `--gc-penalty-ms 50.0` (Stall penalty during GC)
* `--stripe-multiplier 4.0` (FDP throughput multiplier simulating 4-channel striping)
