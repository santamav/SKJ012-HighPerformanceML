#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Definition of CUDA kernels
// Atomic add for double precision otherwise could not compile on Patan
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Pi accumulation kernel
__global__ void accumulate_pi(double* sum, int num_steps, double step){
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int window = num_steps / (gridDim.x * blockDim.x);
  int start = index * window;
  int end = (index + 1) * window;
  if (index == (gridDim.x * blockDim.x) -1) end = num_steps;

  double local_sum = 0.0;
  for (int i = start; i < end; i++){
    double x = (i + 0.5) * step;
    local_sum += 4.0 / (1.0 + x * x);
  }

  atomicAddDouble(sum, local_sum);
}

int main(int argc, char *argv[]) {
  float t_seq, t_par, sp, ep;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Adjust the number of rectangules
  int num_steps = 100000;
  int num_blocks = 2;
  int num_threads = 1024;
  if (argc == 2) {
      num_steps = atoi(argv[1]);
  } else if (argc == 4) {
    num_blocks = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    num_steps = atoi(argv[3]);
    printf("Using %d blocks, %d threads and %d steps\n", num_blocks, num_threads, num_steps);
  }
  else if (argc > 4) {
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
  int size = num_blocks * num_threads;
  sum = 0.0;
  cudaEventRecord(start);
  // Call to the CUDA 
  //Allocate memory on the device
  double *d_sum;
  cudaMalloc((void**)&d_sum, sizeof(double));
  //Initialize the device memory
  cudaMemset(d_sum, 0, sizeof(double));
  //Launch the kernel
  accumulate_pi<<<num_blocks, num_threads>>>(d_sum, num_steps, step);
  //Copy the result from device to host
  cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  //Free the device memory
  cudaFree(d_sum);
  pi = step * sum;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  t_par = 0.0;
  cudaEventElapsedTime(&t_par, start, stop);
  sp = t_seq / t_par;
  ep = sp / size;

  printf(" pi_par = %20.15f\n", pi);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
