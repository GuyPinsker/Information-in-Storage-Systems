import os
import json
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
json_path = os.path.join(output_dir, 'phase1_throughput.json')

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found. Please run scripts/phase1_fio_profiler.sh first.")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

# Convert string keys to int and sort them
block_sizes = sorted([int(k) for k in data.keys()])
throughputs = [data[str(bs)] for bs in block_sizes]

plt.figure(figsize=(10, 6))
plt.plot(block_sizes, throughputs, marker='o', linewidth=2, color='tab:blue')
plt.title('MacBook Air M4 (512GB SSD) Random Read Throughput')
plt.xlabel('Block Size (Tokens)')
plt.ylabel('Throughput (MB/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(block_sizes, [str(bs) for bs in block_sizes])
plt.tight_layout()

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'phase1_throughput.png')
plt.savefig(output_path, dpi=300)
print(f"Saved chart to {output_path}")
