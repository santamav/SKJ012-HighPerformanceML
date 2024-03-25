#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int i;
  double t1, t2, t_seq, t_par, sp, ep;

  int rank, size, rc;
  MPI_Status st;

  // MPI Initialization
  rc = MPI_Init (&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    // Aborting the execution
    MPI_Abort (MPI_COMM_WORLD, rc);
  }
  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Adjust the number of rectangules
  int num_steps = 100000;
  if (rank == 0) {
    if (argc == 2) {
      num_steps = atoi(argv[1]);
    } else if (argc > 2) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ num_steps ]\n");
      // Aborting the execution 
      MPI_Abort (MPI_COMM_WORLD, rc);
    }
  }
  // Sending num_steps from process 0 to the other processes
  MPI_Bcast(&num_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //printf ("num_steps(%d) = %d\n", rank, num_steps);

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
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  // Local computation of pi
  double local_sum = 0.0;
  sum = 0.0;
  for(int i = rank; i < num_steps; i+=size) {
    x = (i + 0.5) * step;
    local_sum += 4.0 / (1.0 + x * x);
  }

  // Sending local computations to process 0,
  // which accumulates them and obtains pi
  pi = 0.0;

  MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  //printf(" rank(%d) sum = %20.15f\n", rank, local_sum);
  if(rank == 0){
    pi = sum * step;
  }
  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_par = t2 - t1;
  sp = t_seq / t_par;
  ep = sp / size;

  if (rank == 0) {
    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
  }

  // MPI Finalization
  MPI_Finalize();
}
