#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    // MPI program executable
    if (argc < 1) {
        printf("Wrong number of parameters\n");
        printf("no parameters execution expected\n");
        exit(-1);
    }  
    char *mpi_program = argv[1];

    // Array of process counts
    int process_counts[] = {1, 2, 4, 8};
    int num_processes = sizeof(process_counts) / sizeof(process_counts[0]);

    int rectangles[] = {1e7, 5e7, 1e8, 5e8};
    int rectangles_size = sizeof(rectangles) / sizeof(rectangles[0]);

    // Loop over each process count and run the MPI program
    for (int i = 0; i < num_processes; i++) {
        for(int j = 0; j < rectangles_size; j++) {
            int num_procs = process_counts[i];
            int num_rectangles = rectangles[j];
            printf("Running MPI program with %d processes and %d rectangles\n", num_procs, num_rectangles);

            // Construct the command to run mpiexec with the specified number of processes
            char command[256];
            snprintf(command, sizeof(command), "mpirun -np %d %s %d", num_procs, mpi_program, num_rectangles);

            // Use system() to execute the command
            int ret = system(command);

            // Check if the command execution was successful
            if (ret != 0) {
                fprintf(stderr, "Error running MPI program with %d processes.\n", num_procs);
                return 1;
            }
        }
    }

    return 0;
}