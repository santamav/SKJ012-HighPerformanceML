#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int i, j;
  double t1, t2, t_seq, t_dis, t_par, t_rec, sp, ep;

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
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  // Adjust the matrix dimensions
  int M = 100, N = 100, ML = 0, NL = 0;
  if (rank == 0) {
    if (argc == 2) {
      M = atoi(argv[1]);
    } else if (argc == 3) {
      M = atoi(argv[1]);
      N = atoi(argv[2]);
    } else if (argc > 3) {
      printf("Wrong number of parameters\n");
      printf("mpirun -np ?? ./a.out [ M [ N ] ]\n");
      exit(-1);
    }
    if (((M % size) > 0) || ((N % size) > 0)) {
      printf ("The number of rows (%d) and columns (%d) have ", M, N);
      printf ("to be multiple of number of processors (%d)\n", size);
      // Aborting the execution
      MPI_Abort (MPI_COMM_WORLD, rc);
    }
  }
  // Sending M and N from process 0 to the other processes
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int v[2] ={M,N};
  // ...
  ML = M / size;
  NL = N / size;
  printf ("M(%d) = %d , N(%d) = %d , ML(%d) = %d , NL(%d) = %d\n", 
          rank, M, rank, N, rank, ML, rank, NL);

/*************************************/
/****** Computation of mat_mult ******/
/*************************************/

  double A[M][N], x[N], y[M], sum, sum_seq, sum_par;

  if (rank == 0) {
    for (i=0; i<M; i++) {
      y[i] = 0;
      for (j=0; j<N; j++) {
        A[i][j] = ( ( rand( ) % 10) * 1.0 ) / 
                    ( ( rand( ) % 1000 ) + 1 );
      }
    }
    for (j=0; j<N; j++) {
      x[j] = ( ( rand( ) % 10) * 1.0 ) / 
               ( ( rand( ) % 1000 ) + 1 );
    }
  }

  //
  // Sequential implementation
  //
  if (rank == 0) {
    t1 = MPI_Wtime( );
    for (i=0; i<M; i++) {
      y[i] = 0;
      for (j=0; j<N; j++) {
        y[i] += A[i][j] * x[j];
      }
    }
    t2 = MPI_Wtime( );
    t_seq = (t2-t1);

    sum_seq = 0.0;
    for (i=0; i<M; i++) {
      sum_seq += y[i];
    }
    printf(" sum_seq = %20.15f\n", sum_seq);
    printf(" time_seq = %20.15f\n", t_seq);
  }

  //
  // Parallel implementation
  //

  double AL[M][NL], xG[N], xL[NL], yG[M], yL[ML];
  
  // Defining the datatype for column distribution
  MPI_Datatype Type_Send_Col_A, Type_Send_Col_B, Type_Recv_Col_A, Type_Recv_Col_B;
  // Datatype for sender
  MPI_Type_vector(M, 1, N, MPI_DOUBLE, &Type_Send_Col_A);
  MPI_Type_commit(&Type_Send_Col_A);
  MPI_Type_create_resized(Type_Send_Col_A, 0, 1*sizeof(double), &Type_Send_Col_B);
  MPI_Type_commit(&Type_Send_Col_B);
  // Datatype for receiver
  MPI_Type_vector(M, 1, NL, MPI_DOUBLE, &Type_Recv_Col_A);
  MPI_Type_commit(&Type_Recv_Col_A);
  MPI_Type_create_resized(Type_Recv_Col_A, 0, 1*sizeof(double), &Type_Recv_Col_B);
  MPI_Type_commit(&Type_Recv_Col_B);

  // Getting the first tick
  MPI_Barrier(MPI_COMM_WORLD);
  // Synchronization of processes
  t1 = MPI_Wtime();

  // Distributing A to Al, x to xL, y to YL
  // ...
      MPI_Scatter(A[i], N, Type_Send_Col_A, AL[i], NL, Type_Send_Col_B, 0, MPI_COMM_WORLD);


  // Getting the final tick and calculating the distribution time
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  


  t2 = MPI_Wtime();
  t_dis = t2 - t1;

  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();


  // Computing the local product
  // ...
  sum_par = 0.0;
  for (i=0; i<M; i++) {
    yL[i] = 0;
    for (j=0; j<NL; j++) {
      yL[i] += AL[i][j] * xG[j];
    }
    sum_par += yL[i];
  }
  
  // Accumulate the local products and distribute the solution
  // ...
  MPI_Allgather(yL, ML, MPI_DOUBLE, yG, ML, MPI_DOUBLE, MPI_COMM_WORLD);
  
  // Getting the final tick and calculating performance indices
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_par = t2-t1;
  sp = t_seq / t_par;
  ep = sp / size;

  // Getting the first tick
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  // Gathering yL to yG on processor 0
  // ...
  MPI_Gather(yL, ML, MPI_DOUBLE, yG, ML, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Getting the final tick and calculating the joining time
  // Synchronization of processes
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  t_rec = t2-t1;

  if (rank == 0) {
    sum_par = 0.0;
    for (i=0; i<M; i++) {
      sum_par += yG[i];
    }
    double sp_glb = t_seq / (t_par + t_dis + t_rec);
    printf(" sum_par = %20.15f , diff = %20.15e\n", sum_par, sum_seq-sum_par);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
    printf(" time_dis = %20.15f, time_rec = %20.15f , Sp = %20.15f\n", 
              t_dis, t_rec, sp_glb);
  }

  MPI_Finalize();
}
