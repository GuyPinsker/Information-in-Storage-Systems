#!/bin/bash

# Phase 1: Throughput Profiler (SSD Speed)
# Profiles the NVMe SSD throughput across different block sizes using fio.

# 1 token = 16 KB of FP16 K+V data.
# Block sizes (in tokens) to test: 1, 8, 16, 32, 64, 128
# Corresponding block sizes (in KB): 16, 128, 256, 512, 1024, 2048

DUMMY_FILE="fio_test_dummy.dat"
FILE_SIZE="10G"

TRANSFER_SIZES=("16k" "128k" "256k" "512k" "1024k" "2048k")
BLOCK_TOKEN_SIZES=("1" "8" "16" "32" "64" "128")

echo "Starting FIO SSD Profiling on Dummy File ($FILE_SIZE)..."
echo "Block Size (Tokens) | Transfer Size (KB) | Throughput (MB/s) | IOPS"
echo "----------------------------------------------------------------------"

for i in "${!TRANSFER_SIZES[@]}"; do
    bs="${TRANSFER_SIZES[$i]}"
    tokens="${BLOCK_TOKEN_SIZES[$i]}"
    
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
        
    # Extract the bandwidth (in bytes) using grep and awk to avoid jq dependency
    bw_bytes=$(echo "$result" | grep -A 20 '"read": {' | grep '"bw_bytes":' | head -n 1 | awk '{print $2}' | tr -d ',')
    iops=$(echo "$result" | grep -A 20 '"read": {' | grep '"iops":' | head -n 1 | awk '{print $2}' | tr -d ',')
    
    # Convert bw_bytes to MB/s
    if [[ -n "$bw_bytes" ]]; then
        bw_mb=$(echo "scale=2; $bw_bytes / 1024 / 1024" | bc)
        echo "$(printf '%-19s' "$tokens") | $(printf '%-18s' "$bs") | $(printf '%-17s' "$bw_mb") | $iops"
    else
        echo "Error parsing output for block size $bs"
    fi
done

# Cleanup
echo "Cleaning up dummy file..."
rm -f $DUMMY_FILE
echo "Done."
