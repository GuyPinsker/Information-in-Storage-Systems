#!/bin/bash

# Phase 1: Throughput Profiler (SSD Speed)
# Profiles the NVMe SSD throughput across different block sizes using fio.

DUMMY_FILE="fio_test_dummy.dat"
FILE_SIZE="10G"
NUM_RUNS=5

# 1 token = 16 KB of FP16 K+V data per layer.
TOKEN_SIZE_KB=16
BLOCK_TOKEN_SIZES=("1" "2" "4" "8" "16" "32" "64" "128" "256")

# Prepare output file
# Use native directory queries instead of realpath to ensure macOS compatibility
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$(dirname "$SCRIPT_DIR")/outputs"
mkdir -p "$OUTPUT_DIR"
OUTPUT_JSON="$OUTPUT_DIR/phase1_throughput-${TOKEN_SIZE_KB}KB.json"

if [ ! -f "$DUMMY_FILE" ]; then
    echo "Creating dummy file of size $FILE_SIZE..."
    dd if=/dev/zero of="$DUMMY_FILE" bs=1M count=10240 status=progress
fi

# We will collect JSON lines in an array to prevent trailing comma corruption on failure
results_json=()

echo "Starting FIO SSD Profiling on Dummy File ($FILE_SIZE)..."
echo "Running $NUM_RUNS times for each size and calculating averages."
echo "Block Size (Tokens) | Transfer Size (KB) | Avg Throughput (MB/s) | Avg IOPS"
echo "------------------------------------------------------------------------------"

for i in "${!BLOCK_TOKEN_SIZES[@]}"; do
    tokens="${BLOCK_TOKEN_SIZES[$i]}"
    transfer_kb=$(( tokens * TOKEN_SIZE_KB ))
    bs="${transfer_kb}k"
    
    total_bw_bytes=0
    total_iops=0
    
    for run in $(seq 1 $NUM_RUNS); do
        # Run fio for random read, direct IO, 10 seconds per test
        result=$(fio --name=ssd_test \
            --filename=$DUMMY_FILE \
            --size=$FILE_SIZE \
            --ioengine=posixaio \
            --rw=randread \
            --bs=$bs \
            --direct=1 \
            --numjobs=1 \
            --iodepth=1 \
            --runtime=10 \
            --time_based \
            --output-format=json)
            
        # Extract the bandwidth (in bytes) and IOPS using Python's json module
        bw_bytes=$(echo "$result" | python3 -c "import sys, json;
try:
    print(json.load(sys.stdin)['jobs'][0]['read']['bw_bytes'])
except Exception:
    pass" 2>/dev/null)
        iops=$(echo "$result" | python3 -c "import sys, json;
try:
    print(json.load(sys.stdin)['jobs'][0]['read']['iops'])
except Exception:
    pass" 2>/dev/null)
        
        if [ -z "$bw_bytes" ]; then bw_bytes=0; fi
        if [ -z "$iops" ]; then iops=0; fi
        
        total_bw_bytes=$(echo "$total_bw_bytes + $bw_bytes" | bc)
        total_iops=$(echo "$total_iops + $iops" | bc)
    done
    
    # Calculate averages
    avg_bw_bytes=$(echo "scale=2; $total_bw_bytes / $NUM_RUNS" | bc 2>/dev/null)
    avg_iops=$(echo "scale=2; $total_iops / $NUM_RUNS" | bc 2>/dev/null)
    
    # Convert average bw_bytes to MB/s
    if [[ -n "$avg_bw_bytes" ]] && (( $(echo "$avg_bw_bytes > 0" | bc -l) )); then
        avg_bw_mb=$(echo "scale=2; $avg_bw_bytes / 1024 / 1024" | bc 2>/dev/null)
        if [[ -n "$avg_bw_mb" ]]; then
            echo "$(printf '%-19s' "$tokens") | $(printf '%-18s' "$bs") | $(printf '%-21s' "$avg_bw_mb") | $avg_iops"
            results_json+=("\"$tokens\": $avg_bw_mb")
        else
            echo "Error parsing output for block size $bs"
        fi
    else
        echo "Error parsing output for block size $bs"
    fi
done

# Write the final JSON file safely to avoid trailing comma corruption
echo "{" > "$OUTPUT_JSON"
num_results=${#results_json[@]}
for idx in "${!results_json[@]}"; do
    if [ $idx -eq $((num_results - 1)) ]; then
        echo "  ${results_json[$idx]}" >> "$OUTPUT_JSON"
    else
        echo "  ${results_json[$idx]}," >> "$OUTPUT_JSON"
    fi
done
echo "}" >> "$OUTPUT_JSON"

# Cleanup
echo "Cleaning up dummy file..."
rm -f $DUMMY_FILE
echo "Done. Results saved to $OUTPUT_JSON"
