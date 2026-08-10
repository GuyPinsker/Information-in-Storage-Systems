import os
import math
import json
import argparse
import random
import matplotlib.pyplot as plt

def run_simulation(
    throughput_path,
    accuracy_path,
    output_path=None,
    selected_budget_tokens=500,    # 500 tokens for Selected Blocks (Top-K)
    generation_steps=1000,         # Generating 1000 new tokens
    token_size_bytes=4*1024,       # 4 KB per token per layer (FP16 K+V)
    num_layers=32,                 # 32 layers in Llama-3-8B
    ssd_base_latency_ms=0.05,      # Base NVMe latency
    miss_rate=0.19,                # 19% miss rate for speculative prefetcher
    compute_time_ms_per_step=20.0, # Assumed dummy compute time (20ms/token)
    gc_spike_prob=0.05,            # Probability of a GC spike per step
    gc_penalty_ms=50.0,            # Latency penalty when a GC spike occurs
    enable_fdp=False,              # Enable FDP mode (zero GC + striping multiplier)
    stripe_multiplier=4.0,         # Multiplier for FDP striping
    pcie_max_mbps=7000.0           # PCIe bandwidth cap
):
    """
    Simulates the SolidAttention framework performance using the profiled NVMe
    throughputs (Phase 1) and empirical accuracies (Phase 2).
    
    Args:
        throughput_path (str): Path to the empirical throughput JSON file.
        accuracy_path (str): Path to the empirical accuracy JSON file.
        output_path (str): Path to save the simulation results directory.
        selected_budget_tokens (int): Budget tokens for Selected Blocks (Top-K).
        generation_steps (int): Number of tokens to generate.
        token_size_bytes (int): Size per token per layer in bytes.
        num_layers (int): Number of layers in the model.
        ssd_base_latency_ms (float): Base NVMe latency in ms.
        miss_rate (float): Miss rate for speculative prefetcher.
        compute_time_ms_per_step (float): Compute time in ms per step.
    """
    # ---------------------------------------------------------
    # 1. Empirical Throughput (Phase 1)
    # ---------------------------------------------------------
    if not os.path.exists(throughput_path):
        print(f"Error: Could not find empirical throughput results at {throughput_path}")
        print("Please run phase1_fio_profiler.sh first.")
        return
        
    with open(throughput_path, 'r') as f:
        throughput_str = json.load(f)
        
    empirical_throughput_mbps = {int(k): v for k, v in throughput_str.items()}
    block_sizes_tokens = sorted(list(empirical_throughput_mbps.keys()))
    
    # ---------------------------------------------------------
    # 2. Empirical Accuracy (Phase 2)
    # ---------------------------------------------------------
    if not os.path.exists(accuracy_path):
        print(f"Error: Could not find empirical accuracy results at {accuracy_path}")
        print("Please run phase2 accuracy scripts for this model first.")
        return
        
    with open(accuracy_path, 'r') as f:
        empirical_accuracy_str = json.load(f)
        
    # Convert string keys back to int
    empirical_accuracy = {int(k): v for k, v in empirical_accuracy_str.items()}
    
    # ---------------------------------------------------------
    # 3. Simulation Parameters
    # ---------------------------------------------------------
    context_length = 128000       # 128k tokens context
    vram_budget_tokens = 1000     # 1000 tokens VRAM budget
    
    if enable_fdp:
        gc_spike_prob = 0.0
        print(f"\nFDP Mode Enabled: GC overhead eliminated. Throughput multiplier: {stripe_multiplier}x (Max: {pcie_max_mbps} MB/s)")
    else:
        print(f"\nStandard Mode: GC spike probability: {gc_spike_prob*100}% ({gc_penalty_ms}ms penalty)")

    
    # ---------------------------------------------------------
    # 4. The Workload Loop
    # ---------------------------------------------------------
    results_throughput_tps = []
    
    print("\nSimulation Results")
    print(f"{'Block Size':<15} | {'K':<10} | {'Missed Blocks/Step':<20} | {'Total Stall (ms)':<20} | {'Tokens/sec':<15}")
    print("-" * 90)
    
    for bs in block_sizes_tokens:
        k = math.floor(selected_budget_tokens / bs)
        if k == 0:
            k = 1
            
        block_size_mb = (bs * token_size_bytes) / (1024 * 1024)
        throughput = empirical_throughput_mbps[bs]
        if enable_fdp:
            throughput = min(throughput * stripe_multiplier, pcie_max_mbps)

        
        total_stall_penalty_ms = 0.0
        total_compute_ms = 0.0
        
        total_discrete_misses = 0
        
        for step in range(generation_steps):
            # Probabilistic Roll for Missed Blocks
            discrete_missed_blocks = 0
            for _ in range(k):
                if random.random() > (1.0 - miss_rate):
                    discrete_missed_blocks += 1
            
            total_discrete_misses += discrete_missed_blocks
            missed_data_mb = discrete_missed_blocks * block_size_mb
            
            gc_penalty_this_step = gc_penalty_ms if random.random() < gc_spike_prob else 0.0
            
            # Stall = Fixed IO latency (incurred per layer) + Transfer time (per layer)
            # Since the forward pass is sequential, this happens `num_layers` times
            stall_penalty_per_layer = (ssd_base_latency_ms if discrete_missed_blocks > 0 else 0.0) + ((missed_data_mb / throughput) * 1000)
            stall_penalty_ms = stall_penalty_per_layer * num_layers + gc_penalty_this_step

            
            total_stall_penalty_ms += stall_penalty_ms
            total_compute_ms += compute_time_ms_per_step
            
        total_time_s = (total_compute_ms + total_stall_penalty_ms) / 1000.0
        tokens_per_sec = generation_steps / total_time_s
        results_throughput_tps.append(tokens_per_sec)
        
        avg_missed_blocks = total_discrete_misses / generation_steps
        
        print(f"{bs:<15} | {k:<10} | {avg_missed_blocks:<20.2f} | {total_stall_penalty_ms:<20.2f} | {tokens_per_sec:<15.2f}")
        
    # ---------------------------------------------------------
    # 5. Export Results
    # ---------------------------------------------------------
    accuracies = [empirical_accuracy[bs] for bs in block_sizes_tokens]
    if len(accuracies) > 0 and max(accuracies) <= 1.0:
        accuracies = [acc * 100.0 for acc in accuracies]
        
    output_data = {
        "block_sizes": block_sizes_tokens,
        "throughputs": results_throughput_tps,
        "accuracies": accuracies
    }
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        output_path = os.path.join(output_dir, 'simulation_results.json')
    else:
        output_dir = os.path.dirname(os.path.abspath(output_path))
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved simulation results to '{output_path}'")

def parse_int_expr(val):
    try:
        return int(val)
    except ValueError:
        try:
            return int(eval(val, {"__builtins__": None}, {}))
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid integer expression: '{val}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 SolidAttention Simulation")
    parser.add_argument("--throughput", type=str, required=True, help="Path to the throughput JSON file")
    parser.add_argument("--accuracy", type=str, required=True, help="Path to the accuracy JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output simulation results JSON")
    parser.add_argument("--selected-budget-tokens", type=int, default=500, help="Selected budget tokens (Top-K) (default: 500)")
    parser.add_argument("--generation-steps", type=int, default=1000, help="Number of generation steps (default: 1000)")
    parser.add_argument("--token-size-bytes", type=parse_int_expr, default=4 * 1024, help="Token size in bytes per layer (default: 4096)")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of layers in model (default: 32)")
    parser.add_argument("--ssd-base-latency-ms", type=float, default=0.05, help="Base NVMe latency in ms (default: 0.05)")
    parser.add_argument("--miss-rate", type=float, default=0.19, help="Miss rate for speculative prefetcher (default: 0.19)")
    parser.add_argument("--compute-time-ms-per-step", type=float, default=20.0, help="Compute time in ms per step (default: 20.0)")
    parser.add_argument("--gc-spike-prob", type=float, default=0.05, help="Probability of GC spike (default: 0.05)")
    parser.add_argument("--gc-penalty-ms", type=float, default=50.0, help="Latency penalty for GC spike in ms (default: 50.0)")
    parser.add_argument("--enable-fdp", action="store_true", help="Enable FDP mode (zero GC, multiplied throughput)")
    parser.add_argument("--stripe-multiplier", type=float, default=4.0, help="Throughput multiplier for FDP striping (default: 4.0)")
    parser.add_argument("--pcie-max-mbps", type=float, default=7000.0, help="Max PCIe bandwidth in MB/s (default: 7000.0)")
    args = parser.parse_args()
    
    run_simulation(
        throughput_path=args.throughput,
        accuracy_path=args.accuracy,
        output_path=args.output,
        selected_budget_tokens=args.selected_budget_tokens,
        generation_steps=args.generation_steps,
        token_size_bytes=args.token_size_bytes,
        num_layers=args.num_layers,
        ssd_base_latency_ms=args.ssd_base_latency_ms,
        miss_rate=args.miss_rate,
        compute_time_ms_per_step=args.compute_time_ms_per_step,
        gc_spike_prob=args.gc_spike_prob,
        gc_penalty_ms=args.gc_penalty_ms,
        enable_fdp=args.enable_fdp,
        stripe_multiplier=args.stripe_multiplier,
        pcie_max_mbps=args.pcie_max_mbps
    )
