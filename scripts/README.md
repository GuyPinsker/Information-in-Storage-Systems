# Scripts Directory

This directory contains the core execution scripts representing the three phases of the SolidAttention evaluation.

## Phase 1: SSD Profiling
*   `run_phase1.sh`: **(Primary Entry Point)** A wrapper script that automates running `phase1_fio_profiler.sh` across multiple block sizes and plotting the results.
*   `phase1_fio_profiler.sh`: A Bash script that uses `fio` to measure random read throughput on the host's NVMe SSD across various block sizes.
    *   **Configurable Parameters (via CLI or environment variables)**:
        *   `TOKEN_SIZE_KB`: KV cache size in KB per token. (Default: `4`, CLI flag `-t` or `--token-size`, or positional `$1`)
        *   `numjobs`: Number of parallel jobs for `fio`. (Default: `4`, CLI flag `-j` or `--numjobs`, or positional `$2`)
        *   `iodepth`: I/O depth queue size per job for `fio`. (Default: `32`, CLI flag `-d` or `--iodepth`, or positional `$3`)
        *   `OUTPUT_JSON`: Output JSON file path. (Default: `outputs/phase1_throughput-${TOKEN_SIZE_KB}KB.json`, CLI flag `-o` or `--output`, or positional `$4`)

## Phase 2: Accuracy Profiling
Due to limitations with PyTorch's `bitsandbytes` library on Apple Silicon (which forces slow GPU-to-CPU weight transfers), we maintain two separate implementations of the Phase 2 scripts.

### 1. Apple MLX Implementation (Recommended for Mac)
Located in `scripts/mlx/`. These scripts use the Apple-native `mlx` and `mlx-lm` frameworks to run 4-bit block-sparse attention directly and flawlessly on the Mac GPU. **This is exponentially faster on Apple Silicon.**

*   `mlx/phase2_accuracy_profiler.py`: Runs the generative "Needle-in-a-Haystack" accuracy test.
    *   **Configurable Parameters (via CLI)**:
        *   `--model`: The Hugging Face or MLX Community model ID. (Default: `mlx-community/Meta-Llama-3.1-8B-4bit`)
        *   `--trials`: Number of trials to run per block size configuration. (Default: `20`)
        *   `--haystacks`: List of haystack filenames located in `utils/` to distribute the trials across. (Default: `us_haystack.txt ww2_haystack.txt`)
        *   `--verbose`: Flag to print 1-2 lines of context surrounding the needle injection.

### 2. PyTorch Implementation (Cross-Platform / Reference)
Located in `scripts/pytorch/`. These are the standard cross-platform scripts. **Note:** On Mac, these are hardcoded to run on the CPU (`device_map="cpu"`) to avoid system crashes. They will take a long time to run.

*   `pytorch/phase2_accuracy_profiler.py`: Runs the generative "Needle-in-a-Haystack" accuracy test.
    *   **Configurable Parameters (via CLI)**:
        *   `--model`: The Hugging Face model ID. (Default: `meta-llama/Meta-Llama-3.1-8B`)
        *   `--trials`: Number of trials to run per block size configuration. (Default: `20`)
        *   `--haystacks`: List of haystack filenames located in `utils/` to distribute the trials across. (Default: `us_haystack.txt ww2_haystack.txt`)
        *   `--verbose`: Flag to print 1-2 lines of context surrounding the needle injection.
### Common Configurable Parameters (in scripts)
The following internal parameters can be modified directly within both the MLX and PyTorch scripts:
*   `block_sizes` list: Defines the token block sizes to evaluate.
*   `needle_fact` (in Accuracy Profiler): The fact to hide in the context.

## Phase 3: Trade-off Simulation
*   `run_phase3.sh`: **(Primary Entry Point)** A wrapper script that automates running `phase3_simulation.py` for multiple configurations/models and generates the combined Pareto charts.
*   `phase3_simulation.py`: Combines the empirical throughput from Phase 1 and the empirical accuracy from Phase 2 to simulate I/O stalls and generate trade-off results.
    *   **Command-Line Arguments**:
        *   `--throughput`: Path to the throughput JSON file.
        *   `--accuracy`: Path to the accuracy JSON file.
        *   `--output`: Path to save the output simulation results JSON.
        *   `--selected-budget-tokens`: Selected budget tokens (Top-K) (default: `500`).
        *   `--generation-steps`: Number of generation steps (default: `1000`).
        *   `--token-size-bytes`: Token size in bytes per layer (default: `4096`).
        *   `--num-layers`: Number of layers in model (default: `32`).
        *   `--ssd-base-latency-ms`: Base NVMe latency in ms (default: `0.05`).
        *   `--miss-rate`: Miss rate for speculative prefetcher (default: `0.19`).
        *   `--compute-time-ms-per-step`: Compute time in ms per step (default: `20.0`).
        *   `--gc-spike-prob`: Probability of GC spike (default: `0.05`).
        *   `--gc-penalty-ms`: Latency penalty for GC spike in ms (default: `50.0`).
        *   `--enable-fdp`: Enable FDP mode (zero GC, multiplied throughput).
        *   `--stripe-multiplier`: Throughput multiplier for FDP striping (default: `4.0`).
        *   `--pcie-max-mbps`: Max PCIe bandwidth in MB/s (default: `7000.0`).
