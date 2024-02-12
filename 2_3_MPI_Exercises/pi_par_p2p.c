#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int i;
  double t1, t2, t_seq, t_par, sp, ep;

  int rank, size, rc;
  MPI_Status st;

  // MPI Initialization
  rc = ...
  if (rc != ...) {
    printf ("Error starting MPI program. Terminating.\n");
    // Aborting the execution
  }
  // Get the rank of the process
  // ...
  // Get the number of processes
  // ...

  // Adjust the number of rectangules
  int num_steps = 100000;
  if (rank == 0) {
    if (argc == 2) {
      num_steps = atoi(argv[1]);
    } else if (argc > 2) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ num_steps ]\n");
      // Aborting the execution 
    }
  }
  // Sending num_steps from process 0 to the other processes
  // ...
  printf ("num_steps(%d) = %d\n", rank, num_steps);

/*************************************/
/******** Computation of pi **********/
/*************************************/

  double step = 1.0 / (double) num_steps;  
  double x, pi, sum = 0.0;

  //
  // Sequential implementation
  //
  if (rank == 0) {
    // Computation without getting time
    pi = 0.0;
    step = 1.0 / (double) num_steps;  
    for (i=0; i<num_steps; i++){
      x = (i+0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    // Computation getting time
    t1 = MPI_Wtime( );
    sum = 0.0;
    step = 1.0 / (double) num_steps;  
    for (i=0; i<num_steps; i++){
      x = (i+0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    t2 = MPI_Wtime( );
    t_seq = (t2-t1);
  
    printf(" pi_seq = %20.15f\n", pi);
    printf(" time_seq = %20.15f\n", t_seq);
  }

  //
  // Parallel implementation
  //

  // Getting the first tick
  // Synchronization of processes
  t1 = ...

  // Local computation of pi
  // ...

  // Sending local computations to process 0,
  // which accumulates them and obtains pi
  // ...

  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  t2 = ...
  t_par = ...
  sp = ...
  ep = ...

  if (rank == 0) {
    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
  }

  // MPI Finalization
  // ...
}
