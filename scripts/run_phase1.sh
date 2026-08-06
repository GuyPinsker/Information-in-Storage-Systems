#!/bin/bash

# 4KB with 1 thread, 1 queue depth
./scripts/phase1_fio_profiler.sh -t 4 -j 1 -d 1 -o outputs/phase1/throughput_4KB_1j_1d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_4KB_1j_1d.json \
  --output_filename outputs/phase1/throughput_4KB_1j_1d.png \
  --token_size 4
sleep 60

# 4KB with 4 threads, 32 queue depth
./scripts/phase1_fio_profiler.sh -t 4 -j 4 -d 32 -o outputs/phase1/throughput_4KB_4j_32d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_4KB_4j_32d.json \
  --output_filename outputs/phase1/throughput_4KB_4j_32d.png \
  --token_size 4
sleep 60

# 2KB with 1 thread, 1 queue depth
./scripts/phase1_fio_profiler.sh -t 2 -j 1 -d 1 -o outputs/phase1/throughput_2KB_1j_1d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_2KB_1j_1d.json \
  --output_filename outputs/phase1/throughput_2KB_1j_1d.png \
  --token_size 2
sleep 60

# 2KB with 4 threads, 32 queue depth
./scripts/phase1_fio_profiler.sh -t 2 -j 4 -d 32 -o outputs/phase1/throughput_2KB_4j_32d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_2KB_4j_32d.json \
  --output_filename outputs/phase1/throughput_2KB_4j_32d.png \
  --token_size 2
sleep 60

# 8KB with 1 thread, 1 queue depth
./scripts/phase1_fio_profiler.sh -t 8 -j 1 -d 1 -o outputs/phase1/throughput_8KB_1j_1d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_8KB_1j_1d.json \
  --output_filename outputs/phase1/throughput_8KB_1j_1d.png \
  --token_size 8
sleep 60

# 8KB with 4 threads, 32 queue depth
./scripts/phase1_fio_profiler.sh -t 8 -j 4 -d 32 -o outputs/phase1/throughput_8KB_4j_32d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_8KB_4j_32d.json \
  --output_filename outputs/phase1/throughput_8KB_4j_32d.png \
  --token_size 8
sleep 60

# 16KB with 1 thread, 1 queue depth
./scripts/phase1_fio_profiler.sh -t 16 -j 1 -d 1 -o outputs/phase1/throughput_16KB_1j_1d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_16KB_1j_1d.json \
  --output_filename outputs/phase1/throughput_16KB_1j_1d.png \
  --token_size 16
sleep 60

# 16KB with 4 threads, 32 queue depth
./scripts/phase1_fio_profiler.sh -t 16 -j 4 -d 32 -o outputs/phase1/throughput_16KB_4j_32d.json
conda run -n storage-systems python utils/plot_phase1.py \
  --json_path outputs/phase1/throughput_16KB_4j_32d.json \
  --output_filename outputs/phase1/throughput_16KB_4j_32d.png \
  --token_size 16
