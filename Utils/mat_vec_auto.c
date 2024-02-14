#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    if (argc < 1) {
        printf("Wrong number of parameters\n");
        printf("no parameters execution expected\n");
        exit(-1);
    }  

    char *mpi_program = argv[1];

    // Array of process counts
    int process_counts[] = {2, 4, 8};
    int num_processes = sizeof(process_counts) / sizeof(process_counts[0]);

    // Array Sizes
    int array_sizes[] = {80, 800, 8000};
    int array_sizes_size = sizeof(array_sizes) / sizeof(array_sizes[0]);

    for (int i = 0; i < num_processes; i++) {
        for(int j = 0; j < array_sizes_size; j++) {
            int num_procs = process_counts[i];
            int array_size = array_sizes[j];
            printf("Running MPI program with %d processes and %d array size\n", num_procs, array_size);

            char command[256];
            snprintf(command, sizeof(command), "mpirun -np %d %s %d %d", num_procs, mpi_program, array_size, array_size);

            int ret = system(command);

            if (ret != 0) {
                fprintf(stderr, "Error running MPI program with %d processes.\n", num_procs);
                return 1;
            }
        }
    }
}
