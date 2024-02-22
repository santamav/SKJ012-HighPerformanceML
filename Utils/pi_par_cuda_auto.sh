#!/bin/bash

# Array of process counts
blocks_start=1
blocks_end=6
threads_start=1
threads_end=1024

rectangles=(10000000 50000000 100000000 500000000)

for ((b=blocks_start; b<=blocks_end; b++))
do
    for ((t=threads_start; t<=threads_end; t*=2))
    do
        for r in "${rectangles[@]}"
        do
            echo "Running cuda program with $b blocks, $t threads and $r rectangles"
            command="patan-run cuda pi_par_loop.cu $b $t $r"

            # Use system() to execute the command
            ret=$($command)

            # Check if the command execution was successful
            if [ $? -ne 0 ]; then
                echo "Error running patan-run cuda program with $b blocks, $t threads and $r rectangles."
                exit 1
            fi
        done
    done
done