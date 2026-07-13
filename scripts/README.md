# Scripts Directory

This directory contains the core execution scripts representing the three phases of the SolidAttention evaluation.

## Phase 1: SSD Profiling
*   `phase1_fio_profiler.sh`: A Bash script that uses `fio` to measure random read throughput on the host's NVMe SSD across various block sizes.
    *   **Configurable Parameters**:
        *   `FILE_SIZE`: Size of the dummy test file (e.g., `10G`, `1G`). Default: `10G`.
        *   `NUM_RUNS`: Number of profiling runs to average results. Default: `5`.
        *   `TOKEN_SIZE_KB`: KV cache size in KB per token. Default: `128`.

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
*   `mlx/phase2_longbench_profiler.py`: Evaluates the model on the LongBench (HotpotQA) dataset.
    *   **Configurable Parameters (via CLI)**:
        *   `--model`: The Hugging Face or MLX Community model ID.
        *   `--samples`: The number of dataset samples to evaluate. (Default: `3`)

### 2. PyTorch Implementation (Cross-Platform / Reference)
Located in `scripts/pytorch/`. These are the standard cross-platform scripts. **Note:** On Mac, these are hardcoded to run on the CPU (`device_map="cpu"`) to avoid system crashes. They will take a long time to run.

*   `pytorch/phase2_accuracy_profiler.py`: Runs the generative "Needle-in-a-Haystack" accuracy test.
    *   **Configurable Parameters (via CLI)**:
        *   `--model`: The Hugging Face model ID. (Default: `meta-llama/Meta-Llama-3.1-8B`)
        *   `--trials`: Number of trials to run per block size configuration. (Default: `20`)
        *   `--haystacks`: List of haystack filenames located in `utils/` to distribute the trials across. (Default: `us_haystack.txt ww2_haystack.txt`)
        *   `--verbose`: Flag to print 1-2 lines of context surrounding the needle injection.
*   `pytorch/phase2_longbench_profiler.py`: Evaluates the model on the LongBench (HotpotQA) dataset.
    *   **Configurable Parameters (via CLI)**:
        *   `--model`: The Hugging Face model ID.
        *   `--samples`: The number of dataset samples to evaluate. (Default: `3`)).
### Common Configurable Parameters (in scripts)
The following internal parameters can be modified directly within both the MLX and PyTorch scripts:
*   `block_sizes` list: Defines the token block sizes to evaluate.
*   `needle_fact` (in Accuracy Profiler): The fact to hide in the context.

## Phase 3: Trade-off Simulation
*   `phase3_simulation.py`: Combines the empirical throughput from Phase 1 and the empirical accuracy from Phase 2 to simulate I/O stalls and plot the final SSD Throughput vs Model Accuracy trade-off curve.
    *   **Command-Line Arguments**:
        *   `--model`: Hugging Face model ID to graph against (e.g. `meta-llama/Meta-Llama-3.1-8B`).
    *   **Configurable Parameters (in script)**:
        *   `context_length`: The length of the context in tokens (default `128000`).
        *   `vram_budget_tokens`: The max budget of VRAM in tokens (default `1000`).
        *   `selected_budget_tokens`: The token budget allocated for dynamically fetched blocks (default `500`).
        *   `generation_steps`: The number of generation steps to simulate (default `1000`).
        *   `token_size_bytes`: Byte size per token KV (default `131072`).
        *   `ssd_base_latency_ms`: Base IO latency per fetch in ms (default `0.05`).
        *   `miss_rate`: Projected speculative miss rate (default `0.19`).
