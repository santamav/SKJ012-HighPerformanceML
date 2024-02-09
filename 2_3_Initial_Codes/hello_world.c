#include <stdio.h>
#include <mpi.h>  

int main (int argc, char *argv[]){
  int rank, size, rc;

	// MPI Initialization
  rc = MPI_Init (&argc, &argv);  
  if (rc != ...) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort (MPI_COMM_WORLD, rc);
  }

  // Get the rank of the process
  // ...
  // Get the number of processes
  // ...

  printf ("Hello World!!, from rank %d out of %d\n", rank, size);  

	// MPI Finalization
  MPI_Finalize ();
}

