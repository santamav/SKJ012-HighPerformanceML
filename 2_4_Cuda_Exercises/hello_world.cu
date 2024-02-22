#include <stdio.h> 
#include <cuda_runtime.h>

__global__ void mykernel( void ) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello world from! %d\n", index);
}

int main( int argc, char *argv[] ) {  
  mykernel <<<2,2>>> ();
  cudaDeviceSynchronize();

  return 0;
}

