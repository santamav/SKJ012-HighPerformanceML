#include <stdio.h>
#include <stdlib.h>

// Definition of CUDA kernels
// ... 

int main(int argc, char *argv[]) {
  float t_seq, t_par, sp, ep;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Adjust the number of rectangules
  int num_steps = 100000;
  if (argc == 2) {
    num_steps = atoi(argv[1]);
  } else if (argc > 2) {
    printf("Wrong number of parameters\n");
    printf("./a.out [ num_steps ]\n");
    exit(-1);
  }

/*************************************/
/******** Computation of pi **********/
/*************************************/

  int i;
  double step = 1.0 / (double) num_steps;  
  double pi = 0.0;

  //
  // Sequential implementation
  //
  double x, sum = 0.0;
  cudaEventRecord(start);
  step = 1.0 / (double) num_steps;  
  for (i=0; i<num_steps; i++){
     x = (i+0.5)*step;
     sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  t_seq = 0.0;
  cudaEventElapsedTime(&t_seq, start, stop);

  printf(" pi_seq = %20.15f\n", pi);
  printf(" time_seq = %20.15f\n", t_seq);

  //
  // Parallel implementation
  //
  
  // Defining the number of active threads
  int size = 0;
  
  cudaEventRecord(start);
  // Call to the CUDA 
  // ...
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  t_par = 0.0; 
  cudaEventElapsedTime(&t_par, start, stop);
  sp = t_seq / t_par;
  ep = sp / size;

  printf(" pi_par = %20.15f\n", pi);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
