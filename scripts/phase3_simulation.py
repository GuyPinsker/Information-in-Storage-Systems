import os
import math
import json
import argparse
import random
import matplotlib.pyplot as plt

def run_simulation(model_name, use_dummy=False, task='niah'):
    """
    Simulates the SolidAttention framework performance using the profiled NVMe
    throughputs (Phase 1) and empirical accuracies (Phase 2).
    
    Args:
        model_name (str): The Hugging Face model ID to run the simulation for.
                          Used to locate the correct accuracy JSON file.
        use_dummy (bool): Whether to use dummy data from outputs/dummy_data/
        task (str): The task to load accuracy for ('niah' or 'longbench').
    """
    safe_model_name = model_name.split("/")[-1]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    data_dir = os.path.join(output_dir, 'dummy_data') if use_dummy else output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Empirical Throughput (Phase 1)
    # ---------------------------------------------------------
    throughput_path = os.path.join(data_dir, 'phase1_throughput.json')
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
    if task == 'longbench':
        accuracy_file = f'longbench_accuracy_{safe_model_name}.json'
    else:
        accuracy_file = f'accuracy_{safe_model_name}.json'
        
    accuracy_path = os.path.join(data_dir, accuracy_file)
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
    selected_budget_tokens = 500  # 500 tokens for Selected Blocks (Top-K)
    generation_steps = 1000       # Generating 1000 new tokens
    
    token_size_bytes = 16 *1024   # 16 KB per token per layer (FP16 K+V)
    num_layers = 32               # 32 layers in Llama-3-8B
    ssd_base_latency_ms = 0.05    # Base NVMe latency
    miss_rate = 0.19              # 19% miss rate for speculative prefetcher
    compute_time_ms_per_step = 20.0 # Assumed dummy compute time (20ms/token)
    
    # ---------------------------------------------------------
    # 4. The Workload Loop
    # ---------------------------------------------------------
    results_throughput_tps = []
    
    print(f"\nSimulation Results for {safe_model_name}")
    print(f"{'Block Size':<15} | {'K':<10} | {'Missed Blocks/Step':<20} | {'Total Stall (ms)':<20} | {'Tokens/sec':<15}")
    print("-" * 90)
    
    for bs in block_sizes_tokens:
        k = math.floor(selected_budget_tokens / bs)
        if k == 0:
            k = 1
            
        block_size_mb = (bs * token_size_bytes) / (1024 * 1024)
        throughput = empirical_throughput_mbps[bs]
        
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
            
            # Stall = Fixed IO latency (incurred per layer) + Transfer time (per layer)
            # Since the forward pass is sequential, this happens `num_layers` times
            stall_penalty_per_layer = (ssd_base_latency_ms if discrete_missed_blocks > 0 else 0.0) + ((missed_data_mb / throughput) * 1000)
            stall_penalty_ms = stall_penalty_per_layer * num_layers
            
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
    
    prefix = "dummy_" if use_dummy else ""
    json_output_path = os.path.join(output_dir, f'{prefix}simulation_results_{task}_{safe_model_name}.json')
    
    with open(json_output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved simulation results to '{json_output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 SolidAttention Simulation")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-4bit",
                        help="Hugging Face model ID used in Phase 2")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--task", type=str, choices=['niah', 'longbench'], default='niah', help="Which task accuracy to plot")
    args = parser.parse_args()
    
    run_simulation(args.model, args.dummy, args.task)
