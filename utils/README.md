# Utils Directory

This directory contains utility and helper scripts to support the main evaluation phases.

## Contents
*   `us_haystack.txt` & `ww2_haystack.txt`: Natural language text corpus files used as background noise (the "haystack") for the Phase 2 Needle-In-A-Haystack evaluation.
*   `plot_phase1.py`: Reads the output data from Phase 1 and generates a logarithmic Matplotlib chart (`phase1_throughput.png`) visualizing how SSD throughput scales with block size.
    *   **Configurable Parameters**: `--json_path`, `--token_size`, `--output_filename` (aliases: `--output`, `-o`).
*   `plot_phase3.py`: Reads simulation output JSON files from Phase 3 and generates combined Pareto and dual-axis charts (`solidattention_tradeoff_combined.png` and `pareto_combined.png`).
    *   **Configurable Parameters**: `--inputs` (list of JSON paths), `--output_dir`, `--labels` (optional custom labels).
