#!/bin/bash

# Meta-Llama-3.1-8B-4bit-baseline
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit-baseline/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json
    
# Meta-Llama-3.1-8B-4bit-high_miss_rate
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit-high_miss_rate/simulation_results_Meta-Llama-3.1-8B-4bit-high_miss_rate.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json \
    --miss-rate 0.25

# Meta-Llama-3.1-8B-4bit-low_miss_rate
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit-low_miss_rate/simulation_results_Meta-Llama-3.1-8B-4bit-low_miss_rate.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json \
    --miss-rate 0.13

# Qwen2.5-7B-4bit
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_2KB_1j_1d.json \
    --output outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit.json \
    --accuracy outputs/phase2/accuracy_Qwen2.5-7B-4bit.json \
    --token-size-bytes 2048 \
    --num-layers 28

# Meta-Llama-3.2-3B-8bit
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.2-3B-8bit.json \
    --num-layers 28

# plot miss rate results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Meta-Llama-3.1-8B-4bit-low_miss_rate/simulation_results_Meta-Llama-3.1-8B-4bit-low_miss_rate.json outputs/phase3/Meta-Llama-3.1-8B-4bit-baseline/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json outputs/phase3/Meta-Llama-3.1-8B-4bit-high_miss_rate/simulation_results_Meta-Llama-3.1-8B-4bit-high_miss_rate.json \
    --output_dir outputs/phase3/miss_rate_results \
    --labels "13% Miss Rate" "19% Miss Rate" "25% Miss Rate"

# plot pareto results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Meta-Llama-3.1-8B-4bit-baseline/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit.json outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit.json \
    --output_dir outputs/phase3/pareto_results \
    --labels "Meta-Llama-3.1-8B-4bit" "Qwen2.5-7B-4bit" "Meta-Llama-3.2-3B-8bit"