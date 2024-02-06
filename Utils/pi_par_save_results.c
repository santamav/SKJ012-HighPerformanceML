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
    int nums[] = {1e7, 5e7, 1e8, 5e8};
    for (int size = 1; size <= 16; size*=2){
        for (int i = 0; i < sizeof(nums) / sizeof(nums[0]); i++) {
            // Construct the command string with parameters
            char command[100];  // Adjust the size based on your needs
            snprintf(command, sizeof(command), "../2_2_Codigos_Iniciales/a.out %d %d", size, nums[i]);

            // Use the system function to execute the command
            int result = system(command);
            if (result == 0) {
                printf("Command executed successfully.\n");
            } else {
                printf("Error executing command.\n");
            }
        }
    }
}
