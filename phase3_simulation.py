import math
import matplotlib.pyplot as plt

def run_simulation():
    # ---------------------------------------------------------
    # 1. Hardcoded Dictionaries (To be populated by Phase 1 & 2)
    # ---------------------------------------------------------
    block_sizes_tokens = [1, 8, 16, 32, 64, 128]
    
    # Replace these values with actual profiled SSD throughputs (MB/s) from Phase 1
    empirical_throughput_mbps = {
        1: 20.0,
        8: 150.0,
        16: 300.0,
        32: 600.0,
        64: 1200.0,
        128: 2000.0
    }
    
    # Replace these values with actual profiled recall rates (%) from Phase 2
    empirical_accuracy = {
        1: 100.0,
        8: 99.0,
        16: 97.5,
        32: 95.0,
        64: 85.0,
        128: 70.0
    }
    
    # ---------------------------------------------------------
    # 2. Simulation Parameters
    # ---------------------------------------------------------
    context_length = 128000       # 128k tokens context
    vram_budget_tokens = 1000     # 1000 tokens VRAM budget
    selected_budget_tokens = 500  # 500 tokens for Selected Blocks (Top-K)
    generation_steps = 1000       # Generating 1000 new tokens
    
    token_size_bytes = 16384      # 16 KB per token (FP16 K+V)
    ssd_base_latency_ms = 0.05    # Base NVMe latency
    miss_rate = 0.19              # 19% miss rate for speculative prefetcher
    
    # ---------------------------------------------------------
    # 3. The Workload Loop
    # ---------------------------------------------------------
    results_stall_ms = []
    
    print(f"{'Block Size':<15} | {'K':<10} | {'Missed Blocks/Step':<20} | {'Total Stall Penalty (ms)':<25}")
    print("-" * 75)
    
    for bs in block_sizes_tokens:
        # Calculate dynamic K for the Selected Blocks
        k = math.floor(selected_budget_tokens / bs)
        
        # Calculate misses based on 19% miss rate strictly applied to K
        missed_blocks_per_step = k * miss_rate
        
        # Data size of missed blocks in MB
        block_size_mb = (bs * token_size_bytes) / (1024 * 1024)
        missed_data_mb = missed_blocks_per_step * block_size_mb
        
        # Empirical throughput for this block size
        throughput = empirical_throughput_mbps[bs]
        
        # Total stall penalty for the 1000 generation steps
        total_stall_penalty_ms = 0.0
        
        for step in range(generation_steps):
            # Stall penalty = SSD base latency + Transfer time for missed data
            if missed_data_mb > 0:
                stall_penalty_ms = ssd_base_latency_ms + ((missed_data_mb / throughput) * 1000)
            else:
                stall_penalty_ms = 0.0
            
            total_stall_penalty_ms += stall_penalty_ms
            
        results_stall_ms.append(total_stall_penalty_ms)
        
        print(f"{bs:<15} | {k:<10} | {missed_blocks_per_step:<20.2f} | {total_stall_penalty_ms:<25.2f}")
        
    # ---------------------------------------------------------
    # 4. Plotting Dual Y-Axis Chart
    # ---------------------------------------------------------
    x_labels = [str(bs) for bs in block_sizes_tokens]
    throughputs = [empirical_throughput_mbps[bs] for bs in block_sizes_tokens]
    accuracies = [empirical_accuracy[bs] for bs in block_sizes_tokens]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Y-Axis: SSD Throughput (Blue)
    ax1.set_xlabel('Block Size (Tokens)', fontweight='bold')
    ax1.set_ylabel('SSD Throughput (MB/s)', color='tab:blue', fontweight='bold')
    ax1.plot(x_labels, throughputs, marker='o', color='tab:blue', linewidth=2, label='Throughput')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Right Y-Axis: Accuracy/Recall (Red)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Profiled Accuracy (%)', color='tab:red', fontweight='bold')
    ax2.plot(x_labels, accuracies, marker='s', color='tab:red', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title and Legend
    plt.title('SolidAttention: Tradeoff Between SSD Throughput and Model Accuracy', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('solidattention_tradeoff.png', dpi=300)
    print("\nSaved chart to 'solidattention_tradeoff.png'")

if __name__ == "__main__":
    run_simulation()
