import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_results(model_name, use_dummy=False, task='niah'):
    safe_model_name = model_name.split("/")[-1]
    # Go up one directory from utils to root, then into outputs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    prefix = "dummy_" if use_dummy else ""
    
    json_input_path = os.path.join(output_dir, f'{prefix}simulation_results_{task}_{safe_model_name}.json')
    
    if not os.path.exists(json_input_path):
        print(f"Error: Could not find simulation results at {json_input_path}")
        print("Please run phase3_simulation.py first to generate the JSON results.")
        return
        
    with open(json_input_path, 'r') as f:
        data = json.load(f)
        
    block_sizes_tokens = data["block_sizes"]
    system_throughputs = data["throughputs"]
    accuracies = data["accuracies"]
    
    x_labels = [str(bs) for bs in block_sizes_tokens]
    
    # ---------------------------------------------------------
    # 1. Plotting Dual Y-Axis Chart
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Y-Axis: System Throughput (Blue)
    ax1.set_xlabel('Block Size (Tokens)', fontweight='bold')
    ax1.set_ylabel('Effective Throughput (Tokens/sec)', color='tab:blue', fontweight='bold')
    ax1.plot(x_labels, system_throughputs, marker='o', color='tab:blue', linewidth=2, label='Tokens/sec')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Right Y-Axis: Accuracy/Recall (Red)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Profiled Accuracy (%)', color='tab:red', fontweight='bold')
    ax2.plot(x_labels, accuracies, marker='s', color='tab:red', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title and Legend
    plt.title(f'SolidAttention Tradeoff: {safe_model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    dual_output_path = os.path.join(output_dir, f'{prefix}solidattention_tradeoff_{task}_{safe_model_name}.png')
    plt.savefig(dual_output_path, dpi=300)
    print(f"Saved chart to '{dual_output_path}'")
    
    # ---------------------------------------------------------
    # 2. Plotting Pareto Curve (Throughput vs Accuracy)
    # ---------------------------------------------------------
    fig2, ax_pareto = plt.subplots(figsize=(10, 6))
    ax_pareto.plot(system_throughputs, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Annotate points with the block size
    for bs, t, a in zip(block_sizes_tokens, system_throughputs, accuracies):
        ax_pareto.annotate(f"{bs}", (t, a), textcoords="offset points", xytext=(0,10), ha='center')

    ax_pareto.set_xlabel('Effective Throughput (Tokens/sec)', fontweight='bold', fontsize=12)
    ax_pareto.set_ylabel('Profiled Accuracy (%)', fontweight='bold', fontsize=12)
    ax_pareto.set_title(f'Throughput vs Accuracy Pareto: {safe_model_name}', fontsize=14, fontweight='bold')
    ax_pareto.grid(True, linestyle='--', alpha=0.7)
    fig2.tight_layout()

    pareto_output_path = os.path.join(output_dir, f'{prefix}pareto_{task}_{safe_model_name}.png')
    fig2.savefig(pareto_output_path, dpi=300)
    print(f"Saved Pareto chart to '{pareto_output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Phase 3 SolidAttention Simulation Results")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-4bit",
                        help="Hugging Face model ID used in Phase 3")
    parser.add_argument("--dummy", action="store_true", help="Use dummy prefix for output files")
    parser.add_argument("--task", type=str, choices=['niah', 'longbench'], default='niah', help="Which task results to plot")
    args = parser.parse_args()
    
    plot_results(args.model, args.dummy, args.task)
