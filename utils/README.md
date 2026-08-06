# Utils Directory

This directory contains utility and helper scripts to support the main evaluation phases.

## Contents
*   `us_haystack.txt` & `ww2_haystack.txt`: Natural language text corpus files used as background noise (the "haystack") for the Phase 2 Needle-In-A-Haystack evaluation.
*   `plot_phase1.py`: Reads the output data from Phase 1 and generates a logarithmic Matplotlib chart (`phase1_throughput.png`) visualizing how SSD throughput scales with block size. All outputs are saved to the `outputs/` directory by default.
    *   **Configurable Parameters**: `--json_path`, `--token_size`, `--output_filename` (aliases: `--output`, `-o`).
