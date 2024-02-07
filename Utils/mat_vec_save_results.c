#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    if (argc > 1) {
        printf("Wrong number of parameters\n");
        printf("no parameters execution expected\n");
        exit(-1);
    }  

    double sp, ep;
    int nums[] = {80, 800, 8000};
    for (int size = 1; size <= 16; size*=2){
        for (int i = 0; i < sizeof(nums) / sizeof(nums[0]); i++) {
            // Construct the command string with parameters
            char command[100];  // Adjust the size based on your needs
            printf("Executing with %d threads, M = %d and N=%d\n", size, nums[i], nums[i]);
            snprintf(command, sizeof(command), "../2_2_Codigos_Iniciales/a.out %d %d %d", size, nums[i], nums[i]);

            // Use the system function to execute the command
            int result = system(command);
        }
    }
}
