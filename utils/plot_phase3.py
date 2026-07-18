import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_results(input_paths, output_dir_arg, custom_labels=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, output_dir_arg) if not os.path.isabs(output_dir_arg) else output_dir_arg
    os.makedirs(output_dir, exist_ok=True)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    fig2, ax_pareto = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors
    
    for idx, json_input_path in enumerate(input_paths):
        if not os.path.exists(json_input_path):
            print(f"Error: Could not find simulation results at {json_input_path}")
            continue
            
        with open(json_input_path, 'r') as f:
            data = json.load(f)
            
        if custom_labels and idx < len(custom_labels):
            label = custom_labels[idx]
        else:
            label = os.path.basename(json_input_path).replace('.json', '').replace('simulation_results_', '')
        
        block_sizes_tokens = data["block_sizes"]
        system_throughputs = data["throughputs"]
        accuracies = data["accuracies"]
        
        x_labels = [str(bs) for bs in block_sizes_tokens]
        color = colors[idx % len(colors)]
        
        # Dual Y-Axis Plot
        ax1.plot(x_labels, system_throughputs, marker='o', linestyle='-', color=color, linewidth=2, label=f'{label} (Throughput)')
        ax2.plot(x_labels, accuracies, marker='s', linestyle='--', color=color, linewidth=2, label=f'{label} (Accuracy)')
        
        # Pareto Plot
        ax_pareto.plot(system_throughputs, accuracies, marker='o', linestyle='-', color=color, linewidth=2, markersize=8, label=label)
        
        # Annotate points with the block size
        for bs, t, a in zip(block_sizes_tokens, system_throughputs, accuracies):
            ax_pareto.annotate(f"{bs}", (t, a), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # ---------------------------------------------------------
    # 1. Formatting Dual Y-Axis Chart
    # ---------------------------------------------------------
    ax1.set_xlabel('Block Size (Tokens)', fontweight='bold')
    ax1.set_ylabel('Effective Throughput (Tokens/sec)', fontweight='bold')
    ax2.set_ylabel('Profiled Accuracy (%)', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.15, 1))
    ax1.set_title('SolidAttention Accuracy vs. Throughput Tradeoff', fontsize=14, fontweight='bold')
    
    fig1.tight_layout()
    
    dual_output_path = os.path.join(output_dir, 'solidattention_tradeoff_combined.png')
    fig1.savefig(dual_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined chart to '{dual_output_path}'")
    
    # ---------------------------------------------------------
    # 2. Formatting Pareto Curve
    # ---------------------------------------------------------
    ax_pareto.set_xlabel('Effective Throughput (Tokens/sec)', fontweight='bold', fontsize=12)
    ax_pareto.set_ylabel('Profiled Accuracy (%)', fontweight='bold', fontsize=12)
    ax_pareto.set_title('Throughput vs Accuracy Pareto', fontsize=14, fontweight='bold')
    ax_pareto.grid(True, linestyle='--', alpha=0.7)
    ax_pareto.legend()
    fig2.tight_layout()

    pareto_output_path = os.path.join(output_dir, 'pareto_combined.png')
    fig2.savefig(pareto_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined Pareto chart to '{pareto_output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Phase 3 SolidAttention Simulation Results")
    parser.add_argument("--inputs", type=str, nargs='+', required=True, help="List of input simulation results JSON files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the combined plots")
    parser.add_argument("--labels", type=str, nargs='+', help="Optional list of custom labels/miss rates to use in the legend")
    args = parser.parse_args()
    
    plot_results(args.inputs, args.output_dir, args.labels)
