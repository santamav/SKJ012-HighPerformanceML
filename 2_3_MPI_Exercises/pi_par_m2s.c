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
  int num_steps = 100000, loc_num_steps = 1000;
  if (rank == 0) {
    if (argc == 2) {
      num_steps = atoi(argv[1]);
    } else if (argc == 3) {
      num_steps = atoi(argv[1]);
      loc_num_steps = atoi(argv[2]);
    } else if (argc > 3) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ num_steps [ loc_num_steps ] ]\n");
      // Aborting the execution
      MPI_Abort (MPI_COMM_WORLD, rc);
    }
  }
  // Sending num_steps and loc_num_steps from process 0 to the other processes
  int v[2] ={num_steps,loc_num_steps};
  if (rank == 0) {
    for (i=1; i<size; i++) {
      // ... sending v to Pi
      MPI_Ssend(v, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    // ... receiving v from P0
    MPI_Recv(v, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
    num_steps = v[0];
    loc_num_steps = v[1];
  }
  /*printf ("num_steps(%d) = %d , loc_num_steps(%d) = %d\n", 
          rank, num_steps, rank, loc_num_steps);*/

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

  // Computation of pi
  if (rank == 0) {
    int frs_step = 0, cnt_steps = 0;
    
    // Process 0 send an initial interval to each process
    for (i=1; i<size; i++) {
      cnt_steps = ((frs_step + loc_num_steps) > num_steps)? 
                             (num_steps-frs_step): loc_num_steps;
      // ... sending frs_step
      MPI_Ssend(&frs_step, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      frs_step += cnt_steps;
    }
    pi = 0.0;
    // Process 0 receives the calculations and sends the rest of the intervals 
    while (frs_step < num_steps) {
      // ... receiving sum from Px
      MPI_Recv(&sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
      pi += sum;
      cnt_steps = ((frs_step + loc_num_steps) > num_steps)? 
                             (num_steps-frs_step): loc_num_steps;
      // ... sending frs_step to Px
      MPI_Ssend(&frs_step, 1, MPI_INT, st.MPI_SOURCE, 0, MPI_COMM_WORLD);
      frs_step += cnt_steps;
    }
    // Process 0 receives the calculations and sends poisons to the workers
    for (i=1; i<size; i++) {
      // ... receiving sum from Px
      MPI_Recv(&sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &st);
      pi += sum;
      // ... sending poison to Px
      MPI_Ssend(&frs_step, 1, MPI_INT, i, 99, MPI_COMM_WORLD);
    }
    pi *= step;
  } else {
    int frs_step = 0, cnt_steps = 0;
    step = 1.0 / (double) num_steps;  
    // Each process receives the initial interval 
    // ... receiving frs_step from P0
    MPI_Recv(&frs_step, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
    // Each process sends the calculations and receive a new interval 
    do {
      cnt_steps = ((frs_step + loc_num_steps) > num_steps)? 
                             (num_steps-frs_step): loc_num_steps;
      // Local computation of pi
      // ... computing interval from frs_step snd executig cnt_steps iterations
      for(i=frs_step, sum=0.0; i<frs_step+cnt_steps; i++){
        x = (i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
      }
      // ... sending sum to P0
      MPI_Ssend(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      // ... receiving frs_step from P0
      MPI_Recv(&frs_step, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      // The loop finalizes when a poison arrives
    } while (st.MPI_TAG != 99);
  }
  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_par = t2-t1;
  sp = t_seq / t_par;
  ep = sp / size;

  if (rank == 0) {
    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
  }

  // MPI Finalization
  MPI_Finalize();
}
