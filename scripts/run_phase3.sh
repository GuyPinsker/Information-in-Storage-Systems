#!/bin/bash

# usage: phase3_simulation.py [-h] --throughput THROUGHPUT --accuracy ACCURACY [--output OUTPUT] [--selected-budget-tokens SELECTED_BUDGET_TOKENS]
#                             [--generation-steps GENERATION_STEPS] [--token-size-bytes TOKEN_SIZE_BYTES] [--num-layers NUM_LAYERS]
#                             [--ssd-base-latency-ms SSD_BASE_LATENCY_MS] [--miss-rate MISS_RATE] [--compute-time-ms-per-step COMPUTE_TIME_MS_PER_STEP]
#                             [--gc-spike-prob GC_SPIKE_PROB] [--gc-penalty-ms GC_PENALTY_MS] [--enable-fdp] [--stripe-multiplier STRIPE_MULTIPLIER]
#                             [--pcie-max-mbps PCIE_MAX_MBPS]
# 
# Run Phase 3 SolidAttention Simulation
# 
# options:
#   -h, --help            show this help message and exit
#   --throughput THROUGHPUT
#                         Path to the throughput JSON file
#   --accuracy ACCURACY   Path to the accuracy JSON file
#   --output OUTPUT       Path to save the output simulation results JSON
#   --selected-budget-tokens SELECTED_BUDGET_TOKENS
#                         Selected budget tokens (Top-K) (default: 500)
#   --generation-steps GENERATION_STEPS
#                         Number of generation steps (default: 1000)
#   --token-size-bytes TOKEN_SIZE_BYTES
#                         Token size in bytes per layer (default: 4096)
#   --num-layers NUM_LAYERS
#                         Number of layers in model (default: 32)
#   --ssd-base-latency-ms SSD_BASE_LATENCY_MS
#                         Base NVMe latency in ms (default: 0.05)
#   --miss-rate MISS_RATE
#                         Miss rate for speculative prefetcher (default: 0.19)
#   --compute-time-ms-per-step COMPUTE_TIME_MS_PER_STEP
#                         Compute time in ms per step (default: 20.0)
#   --gc-spike-prob GC_SPIKE_PROB
#                         Probability of GC spike (default: 0.05)
#   --gc-penalty-ms GC_PENALTY_MS
#                         Latency penalty for GC spike in ms (default: 50.0)
#   --enable-fdp          Enable FDP mode (zero GC, multiplied throughput)
#   --stripe-multiplier STRIPE_MULTIPLIER
#                         Throughput multiplier for FDP striping (default: 4.0)
#   --pcie-max-mbps PCIE_MAX_MBPS
#                         Max PCIe bandwidth in MB/s (default: 7000.0)

# Meta-Llama-3.1-8B-4bit-baseline
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json
    
# Meta-Llama-3.1-8B-4bit-high_miss_rate
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-high_miss_rate.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json \
    --miss-rate 0.25

# Meta-Llama-3.1-8B-4bit-low_miss_rate
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-low_miss_rate.json \
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
    --inputs outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-low_miss_rate.json outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-high_miss_rate.json \
    --output_dir outputs/phase3/miss_rate_results \
    --labels "13% Miss Rate" "19% Miss Rate" "25% Miss Rate"

# plot pareto results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit.json outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit.json \
    --output_dir outputs/phase3/pareto_results \
    --labels "Meta-Llama-3.1-8B-4bit" "Qwen2.5-7B-4bit" "Meta-Llama-3.2-3B-8bit"


##### FDP Improvement results #####
# Meta-Llama-3.1-8B-4bit-fdp
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-fdp.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.1-8B-4bit.json \
    --enable-fdp

# Qwen2.5-7B-4bit-fdp
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_2KB_1j_1d.json \
    --output outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit-fdp.json \
    --accuracy outputs/phase2/accuracy_Qwen2.5-7B-4bit.json \
    --token-size-bytes 2048 \
    --num-layers 28 \
    --enable-fdp

# Meta-Llama-3.2-3B-8bit-fdp
python scripts/phase3_simulation.py \
    --throughput outputs/phase1/throughput_4KB_1j_1d.json \
    --output outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit-fdp.json \
    --accuracy outputs/phase2/accuracy_Meta-Llama-3.2-3B-8bit.json \
    --num-layers 28 \
    --enable-fdp

# plot FDP improvement results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-baseline.json outputs/phase3/Meta-Llama-3.1-8B-4bit/simulation_results_Meta-Llama-3.1-8B-4bit-fdp.json \
    --output_dir outputs/phase3/fdp_results/Meta-Llama-3.1-8B-4bit \
    --labels "Baseline" "FDP Improvement"

# plot FDP improvement results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit.json outputs/phase3/Qwen2.5-7B-4bit/simulation_results_Qwen2.5-7B-4bit-fdp.json \
    --output_dir outputs/phase3/fdp_results/Qwen2.5-7B-4bit \
    --labels "Baseline" "FDP Improvement"

# plot FDP improvement results
python utils/plot_phase3.py \
    --inputs outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit.json outputs/phase3/Meta-Llama-3.2-3B-8bit/simulation_results_Meta-Llama-3.2-3B-8bit-fdp.json \
    --output_dir outputs/phase3/fdp_results/Meta-Llama-3.2-3B-8bit \
    --labels "Baseline" "FDP Improvement"