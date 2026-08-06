import os
import json
import argparse
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
default_json_path = os.path.join(output_dir, 'phase1_throughput.json')

parser = argparse.ArgumentParser(description="Plot Phase 1 Random Read Throughput")
parser.add_argument("--json_path", type=str, nargs="?", default=default_json_path, help="Path to the JSON file containing throughput data")
parser.add_argument("--token_size", type=int, nargs="?", default=4, help="Token size in KB")
parser.add_argument("--output_filename", "--output", "-o", type=str, nargs="?", default=None, help="Output image filename or path")
args = parser.parse_args()

json_path = args.json_path

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
plt.title(f'MacBook Air M4 (512GB SSD) Random Read Throughput | {args.token_size}KB Tokens')
plt.xlabel('Block Size (Tokens)')
plt.ylabel('Throughput (MB/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(block_sizes, [str(bs) for bs in block_sizes])
plt.tight_layout()

if args.output_filename:
    output_path = args.output_filename if os.path.dirname(args.output_filename) else os.path.join(output_dir, args.output_filename)
else:
    json_basename = os.path.basename(json_path)
    output_filename = os.path.splitext(json_basename)[0] + '.png'
    output_path = os.path.join(output_dir, output_filename)

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Saved chart to {output_path}")
