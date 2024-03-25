#!/bin/bash

# Variables
USERNAME="al343729"  # Replace with your username
CUDA_FILE="pi_par_loop.cu"  # Replace with your CUDA file path

# Array of blocks and threads
BLOCKS=(1 2 3 4)
THREADS=(256 512 1024)
RENCTANGLES=(10000000 50000000 100000000 500000000)

# Loop over blocks and threads
for block in "${BLOCKS[@]}"; do
  for thread in "${THREADS[@]}"; do
    for rectangle in "${RENCTANGLES[@]}"; do
      # Call patan-run to execute CUDA code
      patan-run $USERNAME cuda $CUDA_FILE $block $thread $rectangle >> ouput.txt
    done
  done
done